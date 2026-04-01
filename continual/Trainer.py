import datetime
import logging
import os
import numpy as np
import torch.distributed as dist
from datasets import voc as voc
import os.path as osp
import tasks
from model.losses import get_masked_ptc_loss, get_seg_loss
from model.model_seg_neg import network
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import *
from utils.pyutils import AverageMeter, cal_eta, format_tabs
from utils.modification import *

import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F


# --- 辅助函数保持不变 ---
def denormalize_image(img_tensor, mean, std):
    """反归一化：(Tensor) -> (H, W, 3) uint8 numpy array"""
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def create_palette(num_classes=256):
    """生成 PASCAL VOC 风格的调色板"""
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        label = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r |= ((label >> 0) & 1) << (7 - j)
            g |= ((label >> 1) & 1) << (7 - j)
            b |= ((label >> 2) & 1) << (7 - j)
            label >>= 3
        palette[i] = [r, g, b]
    # 将 ignore_index (255) 设置为白色或黑色以便区分
    palette[255] = [255, 255, 255]
    return palette


def colorize_mask(mask, palette):
    """给 Mask 上色"""
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
    new_mask.putpalette(palette)
    return new_mask.convert("RGB")


def colorize_sam(mask):
    """给 SAM Mask 上随机颜色"""
    unique_ids = np.unique(mask)
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    rng = np.random.default_rng(seed=42)  # 固定种子保证颜色一致

    for uid in unique_ids:
        if uid == -1: continue
        color = rng.integers(50, 255, size=3)
        color_mask[mask == uid] = color

    return Image.fromarray(color_mask)


# --- 核心更新函数 ---
def save_debug_images(save_dir, batch_idx, img_tensor, cam_lbl, old_lbl,
                      old_bg_prob,  # [新增] 旧模型背景概率图
                      sam_mask, final_lbl,
                      mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    保存调试图像，增加旧模型背景概率热力图。
    Args:
        old_bg_prob: [B, H, W] float tensor, 范围 0~1, 表示旧模型预测为背景的概率
    """
    B = img_tensor.shape[0]
    palette = create_palette(256).flatten()

    # 使用 'jet' 颜色映射来显示概率热力图 (红高蓝低)
    cmap = plt.cm.jet

    for b in range(B):
        sample_dir = os.path.join(save_dir, f"batch_{batch_idx}_sample_{b}")
        os.makedirs(sample_dir, exist_ok=True)

        # 1. 数据准备
        img_np = denormalize_image(img_tensor[b], mean, std)
        cam_np = cam_lbl[b].cpu().numpy().astype(np.uint8)
        old_np = old_lbl[b].cpu().numpy().astype(np.uint8)
        sam_np = sam_mask[b].cpu().numpy().astype(np.int32)
        final_np = final_lbl[b].cpu().numpy().astype(np.uint8)
        # 新增: 背景概率
        bg_prob_np = old_bg_prob[b].cpu().numpy()  # (H, W) float 0-1

        # 2. 保存基础图像
        Image.fromarray(img_np).save(os.path.join(sample_dir, "0_original.png"))
        colorize_mask(cam_np, palette).save(os.path.join(sample_dir, "1_cam_refined.png"))
        colorize_mask(old_np, palette).save(os.path.join(sample_dir, "2a_old_label.png"))
        colorize_sam(sam_np).save(os.path.join(sample_dir, "3_sam_mask.png"))
        colorize_mask(final_np, palette).save(os.path.join(sample_dir, "4_final_mixed.png"))

        # 3. [新增] 保存背景概率热力图
        # 将 0-1 的概率映射到 colormap，再转为 0-255 的 RGB 图像
        heatmap_rgba = cmap(bg_prob_np)
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        Image.fromarray(heatmap_rgb).save(os.path.join(sample_dir, "2b_old_bg_prob_heatmap.png"))

        # 4. 保存组合对比图 (增加了一列)
        fig, axs = plt.subplots(1, 6, figsize=(24, 4))
        titles = ["Original", "CAM Refined", "Old Label", "Old BG Prob (Heatmap)", "SAM Mask", "Final Mixed"]

        axs[0].imshow(img_np)
        axs[1].imshow(colorize_mask(cam_np, palette))
        axs[2].imshow(colorize_mask(old_np, palette))
        # 显示热力图，设置范围 0-1
        im3 = axs[3].imshow(bg_prob_np, cmap='jet', vmin=0, vmax=1)
        axs[4].imshow(colorize_sam(sam_np))
        axs[5].imshow(colorize_mask(final_np, palette))

        # 添加 colorbar
        fig.colorbar(im3, ax=axs[3], fraction=0.046, pad=0.04)

        for ax, title in zip(axs, titles):
            ax.set_title(title)
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "combined_comparison.jpg"))
        plt.close(fig)



class Trainer:
    def __init__(self, args):
        self.args = args
        self.step = args.step
        self.task = args.task
        self.model = network(
            backbone=args.backbone,
            num_classes=sum(tasks.get_per_task_classes(args.dataset, args.task, args.step)),
            classes_list=tasks.get_per_task_classes(args.dataset, args.task, args.step),
            pretrained=args.pretrained,
            init_momentum=args.momentum,
            aux_layer=args.aux_layer,
            step = args.step + 1
        )
        self.device = torch.device(args.local_rank)
        self.total_classes = sum(tasks.get_per_task_classes(args.dataset, args.task, args.step)) - 1
        self.new_classes = tasks.get_per_task_classes(args.dataset, args.task, args.step)[-1]
        self.old_classes = self.total_classes - self.new_classes
        self.new_classes_origin_weight = 0
        self.new_classes_origin_idx = 0

        if args.step == 0:  # if step 0, we don't need to instance the model_old
            self.model_old = None
        else:  # instance model_old
            self.model_old = network(
                backbone=args.backbone,
                num_classes=sum(tasks.get_per_task_classes(args.dataset, args.task, args.step - 1)),
                classes_list=tasks.get_per_task_classes(args.dataset, args.task, args.step - 1),
                pretrained=args.pretrained,
                init_momentum=args.momentum,
                aux_layer=args.aux_layer,
                step = args.step
            )
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.eval()

        param_groups = self.model.get_param_groups()
        self.optimizer = self.get_optimizer(args, param_groups)

    def get_optimizer(self, args, param_groups):

        optim = getattr(optimizer, args.optimizer)(
            params=[
                {
                    "params": param_groups[0],
                    "lr": args.lr,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[1],
                    "lr": args.lr,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[2],
                    "lr": args.lr * 10,
                    "weight_decay": args.wt_decay,
                },
                {
                    "params": param_groups[3],
                    "lr": args.lr * 10,
                    "weight_decay": args.wt_decay,
                },
            ],
            lr=args.lr,
            weight_decay=args.wt_decay,
            betas=args.betas,
            warmup_iter=args.warmup_iters,
            max_iter=args.max_iters,
            warmup_ratio=args.warmup_lr,
            power=args.power)

        return optim

    def load_step_ckpt(self, path, step_0_checkpoints=None):
        # generate model from path
        if osp.exists(path):
            step_checkpoint = torch.load(path, map_location="cpu")
            # step_checkpoint1 = torch.load('/home/fangkai/code/ICME_final/output_voc/10-5/high_value_filter/step2/checkpoints/model_final.pth', map_location="cpu")
            self.model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False for incr. classifiers
            self.model_old.load_state_dict(step_checkpoint['model_state'], strict=True)  # Load also here old parameters

            logging.info(f"[!] Previous model loaded from {path}")
            # clean memory
            del step_checkpoint['model_state']
            # del step_checkpoint1['model_state']
        elif osp.exists(step_0_checkpoints):
            step_checkpoint = torch.load(step_0_checkpoints, map_location="cpu")
            self.model.load_state_dict(step_checkpoint['model_state'], strict=False)  # False for incr. classifiers
            self.model_old.load_state_dict(step_checkpoint['model_state'], strict=True)  # Load also here old parameters

            logging.info(f"[!] step_0 model loaded from {step_0_checkpoints}")
            del step_checkpoint['model_state']

        else:
            logging.info(f"[!] WARNING: Unable to find of step {self.args.step - 1}! "
                         f"Do you really want to do from scratch?")

    def validate(self, model=None, data_loader=None, args=None):
        preds, gts, cams, cams_aux, type_preds = [], [], [], [], []
        model.eval()
        avg_meter = AverageMeter()

        # [新增] 定义未来类别的数量，需要与训练时保持一致
        NUM_FUTURE = 3

        with torch.no_grad():
            for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):
                name, inputs, labels, cls_label, depth, normal = data

                # 备份 inputs
                raw_inputs = inputs[:, :3, :, :].clone()

                inputs = inputs[:, :3, :, :]
                inputs = inputs.cuda()
                labels = labels.cuda()
                cls_label = cls_label.cuda()
                cls_label = cls_label[:, :self.total_classes]

                # Resize inputs for model
                inputs = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear',
                                       align_corners=False)
                depth = F.interpolate(depth, size=[args.crop_size, args.crop_size], mode='bilinear',
                                      align_corners=False)

                # --- 模型前向传播 ---
                # type_seg 输出形状: [B, Total_Classes + 1 + NUM_FUTURE, H, W]
                cls, segs, _, _, type_seg, _, _, _ = model(inputs, depth)

                # 1. 计算分类 F1 Score (只计算一次即可)
                cls_pred = (cls > 0).type(torch.int16)
                _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
                avg_meter.add({"cls_score": _f1})

                # 2. CAM 生成与评估
                _cams, _cams_aux = multi_scale_cam2(model, inputs, depth, args.cam_scales)

                # Resize CAM 到原图大小
                resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
                cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                         low_thre=args.low_thre, ignore_index=args.ignore_index)

                resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
                cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre,
                                             high_thre=args.high_thre, low_thre=args.low_thre,
                                             ignore_index=args.ignore_index)

                # 3. Seg 分割头评估 (常规分割)
                resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)
                preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))

                # 4. [关键修改] Type Seg (原型分割) 评估
                # resize 到原图大小
                type_segs_full = F.interpolate(type_seg, size=labels.shape[1:], mode='bilinear', align_corners=False)

                # --- 核心逻辑 ---
                # type_segs_full 包含了 [背景, 已知类..., 未来类1, 未来类2, 未来类3]
                # 但是 GT (labels) 里面只有 [背景, 已知类...]
                # 如果我们让模型预测出了未来类，evaluate.scores 会报错 (IndexError)

                # 策略：在评估 mIoU 时，只取前 (self.total_classes + 1) 个通道
                # 这样强迫模型在“已知类别”和“背景”中选一个最大的，忽略掉未来类通道的影响
                known_type_segs = type_segs_full[:, :self.total_classes + 1, :, :]

                type_preds += list(torch.argmax(known_type_segs, dim=1).cpu().numpy().astype(np.int16))

                # 收集 GT 和 CAM
                cams += list(cam_label.cpu().numpy().astype(np.int16))
                gts += list(labels.cpu().numpy().astype(np.int16))
                cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))

        cls_score = avg_meter.pop('cls_score')

        # 计算 mIoU
        # 注意：这里 num_class 传入的是 self.total_classes + 1 (即 21 类)
        seg_score = evaluate.scores(gts, preds, self.total_classes + 1)
        cam_score = evaluate.scores(gts, cams, self.total_classes + 1)
        cam_aux_score = evaluate.scores(gts, cams_aux, self.total_classes + 1)
        type_seg_score = evaluate.scores(gts, type_preds, self.total_classes + 1)

        model.train()

        tab_results = format_tabs([cam_score, cam_aux_score, seg_score, type_seg_score],
                                  name_list=["CAM", "aux_CAM", "Seg_Pred", "Type_Pred"],
                                  cat_list=voc.class_list)

        return cls_score, tab_results

    def train(self, args):
        torch.cuda.set_device(args.local_rank)
        logging.info("Total gpus: %d, samples per gpu: %d..." % (dist.get_world_size(), args.spg))

        time0 = datetime.datetime.now()
        time0 = time0.replace(microsecond=0)
        train_dataset = voc.VOC12ClsDataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.train_set,
            stage='train',
            aug=True,
            # resize_range=cfg.dataset.resize_range,
            rescale_range=args.scales,
            crop_size=args.crop_size,
            img_fliplr=True,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
            tasks=args.task,
            step=args.step,
        )

        train_step0_dataset = voc.VOC12Step0Dataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.train_set,
            stage='train',
            aug=True,
            crop_size=args.crop_size,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
            tasks=args.task,
            step=args.step,
        )

        val_dataset = voc.VOC12SegDataset(
            root_dir=args.data_folder,
            name_list_dir=args.list_folder,
            split=args.val_set,
            stage='val',
            aug=False,
            ignore_index=args.ignore_index,
            num_classes=args.num_classes,
            tasks=args.task,
            step=args.step,
        )

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.spg,
            # shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler,
            prefetch_factor=4)

        train_step0_loader = DataLoader(
            train_step0_dataset,
            batch_size=args.spg,
            # shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=True,
            sampler=train_sampler,
            prefetch_factor=4)

        val_loader = DataLoader(val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers,
                                pin_memory=False,
                                drop_last=False)

        device = self.device
        model = self.model.to(device)
        optim = self.optimizer
        logging.info('\nOptimizer: \n%s' % optim)
        model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
        train_sampler.set_epoch(np.random.randint(args.max_iters))
        train_loader_iter = iter(train_loader)
        avg_meter = AverageMeter()

        train_step0_loader_iter = iter(train_step0_loader)

        NUM_FUTURE = 3

        if self.step == 0:
            # step_checkpoint1 = torch.load('/home/fangkai/code/ICME_final/output_voc/10-5/high_value_filter/step0/checkpoints/model_final.pth', map_location="cpu")
            # self.model.load_state_dict(step_checkpoint1['model_state'], strict=False)  # False for incr. classifiers
            # del step_checkpoint1['model_state']
            for n_iter in range(args.max_iters):
                try:
                    # [修改 1] 务必确保你的 DataLoader 能返回 sam_masks
                    # sam_masks: [B, H, W], 每个像素是一个整数 ID (0是背景/无Mask, 1,2,3...是物体ID)
                    name, inputs, labels, cls_label, depth, normal, sam_masks = next(train_step0_loader_iter)
                except:
                    train_sampler.set_epoch(np.random.randint(args.max_iters))
                    train_step0_loader_iter = iter(train_step0_loader)
                    name, inputs, labels, cls_label, depth, normal, sam_masks = next(train_step0_loader_iter)

                inputs = inputs.to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True).cuda()
                inputs = inputs.cuda()
                labels = labels.cuda()  # [B, H, W] (Pseudo labels from CAM)
                cls_label = cls_label.cuda()
                cls_label = cls_label[:, :self.total_classes]
                sam_masks = sam_masks.cuda()  # [B, H, W]

                # 过滤 ignore index
                filter_idx = labels >= self.total_classes + 1
                labels[filter_idx] = 255

                # 模型前向传播
                # P: [Total+3, Dim] (包含未来原型)
                # type_seg: [B, Total+3, H, W] (包含未来类预测)
                cls, segs, fmap, cls_aux, type_seg, P, delta_p, p_final = model(
                    inputs, depth
                )

                # --- [常规 Loss] 保持不变 ---
                segs = F.interpolate(segs, size=[448, 448], mode='bilinear', align_corners=False)
                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)
                seg_loss = get_seg_loss(segs, labels.type(torch.long), ignore_index=args.ignore_index)

                # --- [关键修改 2] 拆分原型预测 (Known vs Future) ---
                T = 0.1
                # 插值到原图大小以便计算 Loss
                type_seg_interp = F.interpolate(type_seg, size=labels.shape[1:], mode='bilinear', align_corners=False)

                # 拆分:
                # known_seg: [B, Total_Classes, H, W]
                # future_seg: [B, 3, H, W]
                known_type_seg = type_seg_interp[:, :-NUM_FUTURE, :, :]
                future_type_seg = type_seg_interp[:, -NUM_FUTURE:, :, :]

                # 计算已知类的 Prototype Loss (只针对 Labels 里已知的区域)
                proto_seg_loss = get_type_seg_loss(known_type_seg / T, labels.type(torch.long),
                                                   ignore_index=args.ignore_index)

                # 正交 Loss (保持不变)
                proto_sep_loss = compute_comprehensive_ortho_loss(P, p_final, delta_p, mode='final_only')

                # --- [关键修改 3] 未来类别建模的核心逻辑 (Future Logic) ---
                # 目标：找到背景里的 SAM Mask，计算它属于哪个未来原型，并生成监督信号

                # vis_dir = os.path.join(args.ckpt_dir, "visualizations", f"iter_{n_iter + 1}")
                # os.makedirs(vis_dir, exist_ok=True)
                #
                # print(f"Saving future activation maps to {vis_dir}...")
                #
                # # 2. 准备数据 (detach, cpu, numpy)
                # # future_type_seg shape: [B, 3, H, W]
                # activations_np = future_type_seg.detach().cpu().numpy()
                #
                # # 也可以顺便保存一下原图以便对比
                # # 假设 inputs 已经被 normalize 过了，这里简单反 normalize 一下以便肉眼观看
                # # 这里假设 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                # mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
                # std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
                # inputs_np = inputs.detach().cpu().numpy() * std + mean
                # inputs_np = np.clip(inputs_np * 255, 0, 255).astype(np.uint8)
                #
                # bs, num_future, h, w = activations_np.shape
                #
                # # 遍历 batch 中的每一张图
                # for b in range(bs):
                #     # 保存原图
                #     orig_img = inputs_np[b].transpose(1, 2, 0)  # [H, W, 3] RGB
                #     # matplotlib 保存 RGB，cv2 保存 BGR
                #     plt.imsave(os.path.join(vis_dir, f"batch_{b}_original.png"), orig_img)
                #
                #     # 遍历每一个未来通道 (例如 3 个)
                #     for c in range(num_future):
                #         # 获取单张激活图 [H, W]
                #         act_map = activations_np[b, c, :, :]
                #
                #         # 3. 归一化到 [0, 1]
                #         # 使用 Min-Max Normalization，加一个极小值防止除零
                #         min_val = act_map.min()
                #         max_val = act_map.max()
                #         if max_val - min_val > 1e-5:
                #             norm_map = (act_map - min_val) / (max_val - min_val)
                #         else:
                #             norm_map = np.zeros_like(act_map)  # 如果全图是常数，就全黑
                #
                #         # 4. 应用伪彩色映射 (Colormap)
                #         # 'jet' 或 'viridis' 是常用的热力图颜色
                #         # plt.cm.jet(norm_map) 返回的是 [H, W, 4] 的 RGBA 数组，范围 [0, 1]
                #         heatmap = plt.cm.jet(norm_map)
                #
                #         # 去掉 Alpha 通道，保留 RGB
                #         heatmap_rgb = heatmap[:, :, :3]
                #
                #         # 5. 保存图像
                #         # 文件名格式: batch_{b}_future_{c}.png
                #         save_path = os.path.join(vis_dir, f"batch_{b}_future_proto_{c}.png")
                #
                #         # 使用 matplotlib 保存 (接收 [0, 1] 的 float 数组)
                #         plt.imsave(save_path, heatmap_rgb)
                #
                #         # [可选] 将热力图叠加在原图上 (看起来更酷炫)
                #         # 需要把原图转成 float [0, 1]
                #         # overlay = orig_img.astype(np.float32) / 255.0 * 0.5 + heatmap_rgb * 0.5
                #         # plt.imsave(os.path.join(vis_dir, f"batch_{b}_future_proto_{c}_overlay.png"), overlay)
                #
                # print("Visualization saved.")

                # --- [关键修改 3] 未来类别建模的核心逻辑 (Future Logic) [适配 SAM背景=-1 版] ---
                # ... (前面的代码保持不变，直到进入循环) ...

                # --- [关键修改 3] 未来类别建模的核心逻辑 (Future Logic: Robust Version) ---
                future_loss = torch.tensor(0.0, device=device)

                # 只有当存在有效的 SAM Mask 时才计算
                if sam_masks.max() > 0:
                    # 1. 准备未来原型
                    # P: [Total+3, Dim] -> 取最后 NUM_FUTURE 个
                    future_prototypes = P[-NUM_FUTURE:, :].detach()  # detach 很重要，避免单纯拉近原型而不更新特征
                    future_prototypes = F.normalize(future_prototypes, dim=1, p=2)

                    h_feat, w_feat = fmap.shape[-2:]

                    # 2. 下采样 SAM Mask 和 Label 到特征图大小
                    # 注意：Label 下采样要用 nearest 保持类别整数
                    sam_masks_small = F.interpolate(sam_masks.unsqueeze(1).float(),
                                                    size=(h_feat, w_feat),
                                                    mode='nearest').squeeze(1).long()
                    labels_small = F.interpolate(labels.unsqueeze(1).float(),
                                                 size=(h_feat, w_feat),
                                                 mode='nearest').squeeze(1).long()  # [B, H, W]

                    # 初始化 Target，默认 255 (Ignore)
                    future_targets = torch.full((8, h_feat, w_feat), 255, dtype=torch.long, device=device)

                    # --- 参数设置 ---
                    # 过滤太小的碎片 (例如小于 1% 面积)
                    MIN_AREA_THRESH = 0.01 * h_feat * w_feat
                    # 过滤太大的背景块 (例如大于 90% 面积，通常是天空或地面)
                    MAX_AREA_THRESH = 0.80 * h_feat * w_feat

                    # 动态阈值：前期低(0.15)以便启动学习，后期高(0.6)保证质量
                    progress = n_iter / args.max_iters
                    curr_sim_thresh = 0.15 + (0.6 - 0.15) * progress

                    # 统计 batch 中有多少像素被激活了 (Debug用)
                    activated_pixels_count = 0

                    for b in range(8):
                        curr_sam = sam_masks_small[b]  # [H, W]
                        curr_label = labels_small[b]  # [H, W]
                        curr_feat = fmap[b]  # [C, H, W]

                        # 获取该图片中所有的 Mask ID (排除背景 0 和 -1)
                        unique_ids = torch.unique(curr_sam)
                        unique_ids = unique_ids[unique_ids > 0]

                        if len(unique_ids) == 0: continue

                        # 预先计算特征图的 Permute，方便提取
                        # [C, H, W] -> [H, W, C]
                        curr_feat_perm = curr_feat.permute(1, 2, 0)

                        for uid in unique_ids:
                            # 创建当前 mask 的 bool 索引
                            mask_bool = (curr_sam == uid)
                            mask_area = mask_bool.sum()

                            # --- [Filter 1] 面积过滤 ---
                            if mask_area < MIN_AREA_THRESH or mask_area > MAX_AREA_THRESH:
                                continue

                            # --- [Filter 2] 已知类重叠过滤 (核心逻辑) ---
                            # 找出该 mask 区域内，属于“已知前景类”的像素数量
                            # 假设 self.total_classes 是当前 Step 的已知类数量 (不含背景)
                            # labels: 0=bg, 1~Total=known, 255=ignore

                            # 提取该 Mask 区域对应的 CAM Label
                            roi_labels = curr_label[mask_bool]

                            # 计算该区域内也是“已知类(1~Total)”的像素占比
                            # 注意：我们要避开 label=255 的区域，因为那是 CAM 不确定的地方
                            valid_known_pixels = (roi_labels > 0) & (roi_labels <= self.total_classes)
                            known_overlap_ratio = valid_known_pixels.float().mean()

                            # 如果这个 SAM Mask 超过 20% 的区域已经是已知类了，那它就不是未来类
                            if known_overlap_ratio > 0.2:
                                continue

                            # --- [Mining] 这是一个潜在的未来物体 ---
                            with torch.no_grad():
                                # 提取特征: [Area, C] -> Mean -> [C]
                                # 这种写法比 masked_select 更快且维度清晰
                                region_feat = curr_feat_perm[mask_bool].mean(dim=0)
                                region_feat = F.normalize(region_feat, dim=0, p=2)

                                # 计算与未来原型的相似度 [Num_Future]
                                sim_logits = torch.matmul(future_prototypes, region_feat)
                                max_sim, assigned_proto_idx = torch.max(sim_logits, dim=0)

                                # --- [Assignment] 只有相似度够高才分配 ---
                                if max_sim > curr_sim_thresh:
                                    # 填入 Target
                                    # assigned_proto_idx 是 0, 1, 2... 对应 future channel 的索引
                                    future_targets[b][mask_bool] = assigned_proto_idx.item()
                                    activated_pixels_count += mask_area.item()

                    # 3. 计算 Loss
                    # 只有当构建了有效的 target 时才计算
                    if activated_pixels_count > 0:
                        # future_type_seg: [B, 3, H, W]
                        # 这里的 target 已经在 feature map 尺寸了，所以我们要把 prediction 插值缩小，
                        # 或者把 target 插值放大。通常把 prediction 缩小计算量更小，
                        # 但为了精度，我们在上面已经把 labels_small 弄成了 feature map 大小。

                        # 确保尺寸对齐 (future_type_seg 通常是原图大小或者 stride=8)
                        if future_type_seg.shape[-2:] != future_targets.shape[-2:]:
                            # 将预测下采样到 target 大小 (feature map size)
                            pred_future = F.interpolate(future_type_seg,
                                                        size=future_targets.shape[-2:],
                                                        mode='bilinear',
                                                        align_corners=False)
                        else:
                            pred_future = future_type_seg

                        # Cross Entropy Loss
                        # pred: [B, 3, H, W], target: [B, H, W]
                        future_loss = F.cross_entropy(pred_future / T, future_targets, ignore_index=255)

                    # [Debug Info] 偶尔打印一下，看看是否一直为0
                    # if n_iter % 50 == 0:
                    #     print(f"Iter {n_iter}: Activated Pixels={activated_pixels_count}, Thresh={curr_sim_thresh:.3f}, Loss={future_loss.item():.4f}")

                # --- 总 Loss ---
                # 建议提高一点 future_loss 的权重，因为它比较稀疏
                w_future = 0.5

                loss = 1 * cls_loss + 1 * cls_loss_aux + args.w_seg * seg_loss + \
                       0.1 * proto_sep_loss + 0.2 * proto_seg_loss + \
                       w_future * future_loss

                # ... (后续反向传播代码保持不变) ...

                optim.zero_grad()
                loss.backward()
                optim.step()

                # 记录
                avg_meter.add({
                    'cls_loss': cls_loss.item(),
                    'seg_loss': seg_loss.item(),
                    'proto_seg_loss': proto_seg_loss.item(),
                    'proto_sep_loss': proto_sep_loss.item(),
                    'future_loss': future_loss.item()  # 新增记录
                })

                if (n_iter + 1) % args.log_iters == 0:
                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']
                    # print(model.module.prototype_module())
                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f ,seg_loss: %.4f, proto_seg_loss: %.4f..., proto_sep_loss: %.4f..., future_loss: %.4f..." % (
                                n_iter + 1, delta, eta, cur_lr,
                                avg_meter.pop('cls_loss'), avg_meter.pop('seg_loss'),avg_meter.pop('proto_seg_loss'),avg_meter.pop('proto_sep_loss'),avg_meter.pop('future_loss'),
                            ))

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save(model.state_dict(), ckpt_name)
                    val_cls_score, tab_results = self.validate(model=model, data_loader=val_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("val cls score: %.6f" % (val_cls_score))
                        logging.info("\n" + tab_results)

        else:
            model_old = self.model_old.to(device)

            par = PAR(num_iter=10, dilations=[1, 2, 4, 8, 12, 24]).cuda()

            for n_iter in range(args.max_iters):
                try:
                    img_name, inputs, cls_label, img_box, depth, normal, sam_mask = next(train_loader_iter)
                except:
                    train_sampler.set_epoch(np.random.randint(args.max_iters))
                    train_loader_iter = iter(train_loader)
                    img_name, inputs, cls_label, img_box, depth, normal, sam_mask = next(train_loader_iter)

                inputs = inputs.to(device, non_blocking=True)
                depth = depth.to(device, non_blocking=True)
                depth = depth.cuda()
                inputs_denorm = imutils.denormalize_img2(inputs.clone())

                cls_label = cls_label.to(device, non_blocking=True)
                cls_label = cls_label[:, :self.total_classes]
                old_cls, old_segs, _x4, old_cls_aux, type_seg_old, P_old, delta_p_old, p_final_old = model_old(inputs, depth=depth)
                cls_label_old_pred = (old_cls > 2.0).long()

                cls_label_gt_new = cls_label[:, -self.new_classes:]
                cls_label = torch.cat((cls_label_old_pred, cls_label_gt_new), dim=1)

                cams, cams_aux = multi_scale_cam2(model, inputs=inputs, depth=depth, scales=args.cam_scales)

                cls, segs, fmap, cls_aux, type_seg_new, P_new, delta_p_new, p_final_new = model(inputs, depth=depth)

                old_segs = F.interpolate(type_seg_old, size=[448, 448], mode='bilinear', align_corners=False)
                old_pixel_label = torch.argmax(old_segs, dim=1)
                # old_pixel_label_filter = filter_old_dense_labels(old_pixel_label, cls_label = cls_label, ignore_index=args.ignore_index)


                cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
                cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)



                valid_cam, _ = cam_to_label(cams.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True,
                                            bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre,
                                            ignore_index=args.ignore_index)

            

                refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label,
                                                               high_thre=args.high_thre, low_thre=args.low_thre,
                                                               ignore_index=0, img_box=img_box, )

                # old_probs_highres = F.softmax(old_segs, dim=1)
                # # 取出第 0 个通道 (背景通道) 的概率
                # # shape: [B, H, W], 范围 0.0 ~ 1.0
                # old_bg_prob_map = old_probs_highres[:, 0, :, :]

                mixed_pseudo_label = get_mixed_label_with_sam(
                    cam_label=refined_pseudo_label,  # PAR 结果
                    old_segs_label=old_pixel_label,
                    sam_mask=sam_mask,
                    total_classes=self.total_classes,  # 16
                    new_classes=self.new_classes,
                    ignore_index=args.ignore_index
                )


                # debug_save_dir = "./debug_vis_results"  # 结果保存在这里
                #
                # # 策略：只在第0个epoch，或者是想debug的时候运行
                # # 且每个epoch只保存前3个batch，避免刷屏
                # if n_iter < 10:
                #     print(f"Saving debug visualization for batch {n_iter}...")
                #     save_debug_images(
                #         save_dir=debug_save_dir,
                #         batch_idx=n_iter,
                #         img_tensor=inputs,
                #         cam_lbl=refined_pseudo_label,
                #         old_lbl=old_pixel_label,
                #         # [新增] 传入刚才计算的背景概率图
                #         old_bg_prob=old_bg_prob_map,
                #         sam_mask=sam_mask,
                #         final_lbl=mixed_pseudo_label,
                #         # 请确保这里的 mean/std 和你数据集预处理一致
                #         mean=np.array([0.485, 0.456, 0.406]),
                #         std=np.array([0.229, 0.224, 0.225])
                #     )

                

                segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                seg_loss = get_seg_loss(segs, mixed_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

                resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
                _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box,
                                                   ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre,
                                                   low_thre=args.low_thre, ignore_index=args.ignore_index)
                aff_mask = label_to_aff_mask(pseudo_label_aux)
                ptc_loss = get_masked_ptc_loss(fmap, aff_mask)



                T = 0.1
                type_seg_new = F.interpolate(type_seg_new, size=mixed_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
                proto_seg_loss = get_type_seg_loss(type_seg_new / T, mixed_pseudo_label.type(torch.long), ignore_index=args.ignore_index)
                # print(proto_seg_loss)
                
                proto_kd_loss = prototype_kd_loss(P_new, P_old, cls_label, self.new_classes)
                # proto_sep_loss = prototype_sep_loss(prototype_new, cls_label, self.new_classes)

                proto_sep_loss = compute_comprehensive_ortho_loss(P_new, p_final_new, delta_p_new, mode= 'cross')

                # prototype_peak_loss = margin_triplet_peaky_loss(type_seg_new)
                # print(prototype_peak_loss)
                prototype_loss = 2 * proto_seg_loss + 1 * proto_kd_loss + 1 * proto_sep_loss

            

                # warmup
                if n_iter <= 2000:
                    loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + 0.0 * seg_loss 
                else:
                    loss = 1.0 * cls_loss + 1.0 * cls_loss_aux + args.w_ptc * ptc_loss + args.w_seg * seg_loss + 0.1* prototype_loss

                if n_iter % 2000 == 0 and n_iter != 0:
                    if args.local_rank == 0:  # save model at the eval iteration
                        state = {
                            "model_state": self.model.state_dict(),
                        }
                        path = os.path.join(args.ckpt_dir, f"model_{n_iter}.pth")
                        torch.save(state, path)
                        logging.info("[!] Checkpoint saved.")

                cls_pred = (cls > 0).type(torch.int16)
                cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
                avg_meter.add({
                    'cls_loss': cls_loss,
                    'ptc_loss': ptc_loss,
                    'cls_loss_aux': cls_loss_aux,
                    'seg_loss': seg_loss,
                    'cls_score': cls_score,
                    'proto_seg_loss':proto_seg_loss.item(),
                    'proto_kd_loss': proto_kd_loss.item(),
                    'proto_sep_loss': proto_sep_loss.item(),
                })
                optim.zero_grad()
                loss.backward()
                optim.step()
                # model.module.prototype_module.update_prototype_with_mask(p_iter, momentum=0.98)
                if (n_iter + 1) % args.log_iters == 0:

                    delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
                    cur_lr = optim.param_groups[0]['lr']

                    if args.local_rank == 0:
                        logging.info(
                            "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, seg_loss: %.4f, proto_seg_loss: %.4f..., proto_kd_loss: %.4f, proto_sep_loss: %.4f..." % (
                                n_iter + 1, delta, eta, cur_lr, avg_meter.pop('cls_loss'),
                                avg_meter.pop('cls_loss_aux'),
                                avg_meter.pop('ptc_loss'), avg_meter.pop('seg_loss'),avg_meter.pop('proto_seg_loss'),avg_meter.pop('proto_kd_loss'),avg_meter.pop('proto_sep_loss')))

                if (n_iter + 1) % args.eval_iters == 0:
                    ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
                    if args.local_rank == 0:
                        logging.info('Validating...')
                        if args.save_ckpt:
                            torch.save(model.state_dict(), ckpt_name)
                    val_cls_score, tab_results = self.validate(model=model, data_loader=val_loader, args=args)
                    if args.local_rank == 0:
                        logging.info("val cls score: %.6f" % (val_cls_score))
                        logging.info("\n" + tab_results)

        return True