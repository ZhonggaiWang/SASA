from torch.utils.data import Dataset
import imageio
from . import transforms
from PIL import Image
import torchvision.transforms as T
import os
import numpy as np
import cv2

class_list = ["_background_", 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
              'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
              'tvmonitor']

# —— PASCAL VOC 调色板（0~255），背景=0，忽略=255 用灰色 ——
def voc_colorize(mask, ignore_index=255):
    """
    mask: HxW (uint8/int)，语义标签
    return: HxWx3 (uint8) 伪彩
    """
    # Pascal VOC 21 类调色板（可按需替换/扩展到你的类数）
    palette = [
        (0, 0, 0),        # 0 background
        (128, 0, 0),      # 1 aeroplane
        (0, 128, 0),      # 2 bicycle
        (128, 128, 0),    # 3 bird
        (0, 0, 128),      # 4 boat
        (128, 0, 128),    # 5 bottle
        (0, 128, 128),    # 6 bus
        (128, 128, 128),  # 7 car
        (64, 0, 0),       # 8 cat
        (192, 0, 0),      # 9 chair
        (64, 128, 0),     # 10 cow
        (192, 128, 0),    # 11 diningtable
        (64, 0, 128),     # 12 dog
        (192, 0, 128),    # 13 horse
        (64, 128, 128),   # 14 motorbike
        (192, 128, 128),  # 15 person
        (0, 64, 0),       # 16 potted plant
        (128, 64, 0),     # 17 sheep
        (0, 192, 0),      # 18 sofa
        (128, 192, 0),    # 19 train
        (0, 64, 128),     # 20 tv/monitor
    ]
    H, W = mask.shape
    color = np.zeros((H, W, 3), dtype=np.uint8)

    # 忽略区先涂灰
    if ignore_index is not None:
        ign = (mask == ignore_index)
        color[ign] = (128, 128, 128)

    # 其它类上色
    uniq = np.unique(mask)
    for cls in uniq:
        if ignore_index is not None and cls == ignore_index:
            continue
        cls_int = int(cls)
        if 0 <= cls_int < len(palette):
            color[mask == cls_int] = palette[cls_int]
        else:
            # 超出调色板范围的类，给一个可见颜色
            color[mask == cls_int] = (255, 0, 255)
    return color  # BGR 格式（按 OpenCV 保存需求）


def depth_to_color(depth_hw_or_hwc, q_low=1, q_high=99):
    """
    depth: HxW 或 HxWx1 (float32)
    return: HxWx3 uint8 (BGR, INFERNO)
    """
    d = depth_hw_or_hwc
    if d.ndim == 3 and d.shape[2] == 1:
        d = d[..., 0]
    d = d.astype(np.float32)
    d = np.nan_to_num(d, 0.0, 0.0, 0.0)

    # 分位数拉伸，抗极值
    vmin = np.percentile(d, q_low)
    vmax = np.percentile(d, q_high)
    if vmax <= vmin:
        vmax = vmin + 1e-6
    dn = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)
    d8 = (dn * 255).astype(np.uint8)

    color = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)  # BGR
    return color


def save_rgb_label_depth(rgb_hwc, label_hw, depth_hw_or_hwc,
                         out_dir, name, ignore_index=255, rgb_is_rgb=True):
    """
    把 RGB、label、depth 都保存下来（PNG）。
    - rgb_hwc: HxWx3 uint8（未标准化的原图）
    - label_hw: HxW uint8/int（语义标签）
    - depth_hw_or_hwc: HxW 或 HxWx1 float32（深度）
    - rgb_is_rgb: 如果你的 rgb 是 RGB 顺序，保存时会转 BGR 给 OpenCV
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) RGB
    rgb = rgb_hwc
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb_is_rgb:
        rgb_bgr = rgb[..., ::-1]
    else:
        rgb_bgr = rgb
    cv2.imwrite(os.path.join(out_dir, f"{name}_rgb.png"), rgb_bgr)

    # 2) Label 伪彩
    if label_hw is not None:
        lab_vis = voc_colorize(label_hw.astype(np.int32), ignore_index)
        cv2.imwrite(os.path.join(out_dir, f"{name}_label.png"), lab_vis)

    # 3) Depth 伪彩
    if depth_hw_or_hwc is not None:
        dep_vis = depth_to_color(depth_hw_or_hwc)
        cv2.imwrite(os.path.join(out_dir, f"{name}_depth.png"), dep_vis)



def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list


def load_cls_label_list(name_list_dir):
    return np.load(os.path.join(name_list_dir, 'cls_labels_onehot.npy'), allow_pickle=True).item()


class VOC12Dataset(Dataset):
    def __init__(
            self,
            root_dir=None,
            name_list_dir=None,
            split='train',
            stage='train',
            tasks=None,
            step=0,
            # [新增] 接收 sam_mask 路径，默认为 None
            sam_mask_dir='./default'
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.tasks = tasks
        self.img_dir = os.path.join(root_dir, 'JPEGImages')
        self.depth_dir = '/data/fangkai/VOC_depth/depth'
        self.normal_dir = '/data/fangkai/VOC_normal/normal'

        # [新增] SAM Mask 路径
        self.sam_mask_dir = sam_mask_dir

        self.label_dir = os.path.join(root_dir, 'SegmentationClassAug')
        self.name_list_dir = os.path.join(name_list_dir, 'incremental_split',
                                          split + '_' + tasks + '_step_' + str(step + 1) + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name + '.jpg')
        depth_name = os.path.join(self.depth_dir, _img_name + '.npy')
        normal_name = os.path.join(self.normal_dir, _img_name + '.npy')

        # [新增] 加载 SAM Mask (.npy)
        # 假设文件名就是编号，例如 2007_000032.npy
        sam_name = os.path.join(self.sam_mask_dir, _img_name + '.npy')

        image = np.asarray(imageio.imread(img_name))
        depth = np.load(depth_name, allow_pickle=False)
        depth = depth[..., None]
        normal = np.load(normal_name, allow_pickle=False)

        # 加载 SAM Mask (如果文件存在)
        if os.path.exists(sam_name):
            sam_mask = np.load(sam_name, allow_pickle=True)
            # 确保是 HxW 格式
            if sam_mask.ndim == 3:
                sam_mask = sam_mask.squeeze()
        else:
            # 兜底：如果没有找到，生成一个全0的mask (或者报错，看你需求)
            sam_mask = np.zeros(image.shape[:2], dtype=np.int32)

        if self.stage == "train":
            label_dir = os.path.join(self.label_dir, _img_name + '.png')
            label = np.asarray(imageio.imread(label_dir))
        elif self.stage == "val":
            label_dir = os.path.join(self.label_dir, _img_name + '.png')
            label = np.asarray(imageio.imread(label_dir))
        elif self.stage == "test":
            label = image[:, :, 0]

        # [修改] 返回值增加 sam_mask
        return _img_name, image, label, depth, normal, sam_mask

class VOC12ClsDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 tasks=None,
                 step=0,
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage, tasks, step)
        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.local_crop_size = 96
        self.step = step
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()
        self.gaussian_blur = transforms.GaussianBlur
        self.solarization = transforms.Solarization(p=0.2)

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

        self.global_view1 = T.Compose([
            # T.RandomResizedCrop(224, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=1.0),
            # self.normalize,
        ])
        self.global_view2 = T.Compose([
            T.RandomResizedCrop(self.crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.1),
            self.solarization,
            self.normalize,
        ])
        self.local_view = T.Compose([
            # T.RandomResizedCrop(self.local_crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.5),
            self.normalize,
        ])

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label, depth, normal, sam_mask=None):
        img_box = None
        local_image = None
        if self.aug:
            if self.img_fliplr:
                # 直接传入所有参数，函数内部会自动处理翻转
                image, label, depth, normal, sam_mask = transforms.random_fliplr(
                    image, label, depth, normal, sam_mask
                )
            image = self.color_jittor(image)  # 只对 RGB 做光照扰动
            if self.crop_size:
                image, label, depth, normal, sam_mask, img_box = transforms.random_crop_rgbd(
                    image, label, depth, normal, sam_mask,
                    crop_size=self.crop_size,
                    mean_rgb=(123.675, 116.28, 103.53),
                    ignore_index=self.ignore_index,
                    depth_pad_strategy='edge'
                )
            local_image = self.local_view(Image.fromarray(image))

        image = transforms.normalize_img(image)  # (H,W,3) -> float 标准化 (按你原来的实现)
        image = np.transpose(image, (2, 0, 1))  # -> (3,H,W)
        normal = normal.astype(np.float32)
        normal = normal * 2.0 - 1.0  # 映射回物理向量区间
        normal = np.transpose(normal, (2, 0, 1))  # (3, H, W)
        depth = transforms.preprocess_depth(depth)
        depth = np.transpose(depth, (2, 0, 1))  # -> (3,H,W)
        return image, local_image, img_box, depth, normal, sam_mask

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        # label_onehot = F.one_hot(label, num_classes)

        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):
        # [修改] 解包 6 个返回值
        img_name, image, label, depth, normal, sam_mask = super().__getitem__(idx)

        pil_image = Image.fromarray(image)

        # [修改] 传入 sam_mask
        image, local_image, img_box, depth, normal, sam_mask = self.__transforms(
            image=image, label=label, depth=depth, normal=normal, sam_mask=sam_mask
        )

        cls_label = self.label_list[img_name]

        if self.aug:
            crops = []
            crops.append(image)
            crops.append(self.global_view2(pil_image))
            crops.append(local_image)

            # [修改] 返回 sam_mask
            return img_name, image, cls_label, img_box, depth, normal, sam_mask
        else:
            # [修改] 返回 sam_mask
            return img_name, image, cls_label, depth, normal, sam_mask


class VOC12SegDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 tasks=None,
                 step=0,
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage, tasks, step)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()
        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label, depth, normal):
        # —— 在归一化和转 CHW 之前可视化保存（此时 image 仍是 HWC、uint8）——
        # 同步几何增强
        if self.aug:
            if self.img_fliplr:
                image, label, depth, normal = transforms.random_fliplr(image, label, depth, normal)
            image = self.color_jittor(image)  # 只对 RGB 做光照扰动
            if self.crop_size:
                image, label, depth, normal, _ = transforms.random_crop_rgbd(
                    image, label, depth, normal,
                    crop_size=self.crop_size,
                    mean_rgb=(123.675, 116.28, 103.53),
                    ignore_index=self.ignore_index,
                    depth_pad_strategy='edge'
                )
        # 归一化 + 转 CHW（给模型用）
        image = transforms.normalize_img(image)  # (H,W,3) -> float 标准化 (按你原来的实现)
        depth = transforms.preprocess_depth(depth)
        image = np.transpose(image, (2, 0, 1))  # -> (3,H,W)
        normal = normal.astype(np.float32)
        normal = normal * 2.0 - 1.0  # 映射回物理向量区间
        normal = np.transpose(normal, (2, 0, 1))  # (3, H, W)

        # depth -> CHW
        if depth.ndim == 2:
            depth = depth[..., None]  # (H,W,1)
        depth = depth.astype(np.float32)
        depth = np.transpose(depth, (2, 0, 1))  # -> (1,H,W)

        return image, label, depth, normal

    def __getitem__(self, idx):
        img_name, image, label, depth, normal, _ = super().__getitem__(idx)

        image, label, depth, normal = self.__transforms(image=image, label=label, depth=depth, normal=normal)

        if self.stage == "test":
            cls_label = 0
        else:
            cls_label = self.label_list[img_name]

        return img_name, image, label, cls_label, depth, normal


class VOC12Step0Dataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 tasks=None,
                 step=0,
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage, tasks, step)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.img_fliplr = img_fliplr
        self.color_jittor = transforms.PhotoMetricDistortion()
        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

    def __len__(self):
        return len(self.name_list)


    def __transforms(self, image, label, depth, normal, sam_mask):
        # —— 在归一化和转 CHW 之前可视化保存（此时 image 仍是 HWC、uint8）——
        # 同步几何增强
        if self.aug:
            image, label, depth, normal, sam_mask = transforms.random_fliplr(
                image, label, depth, normal, sam_mask
            )
            image = self.color_jittor(image)  # 只对 RGB 做光照扰动
            if self.crop_size:
                image, label, depth, normal, sam_mask, img_box = transforms.random_crop_rgbd(
                    image, label, depth, normal, sam_mask,
                    crop_size=self.crop_size,
                    mean_rgb=(123.675, 116.28, 103.53),
                    ignore_index=self.ignore_index,
                    depth_pad_strategy='edge'
                )
        # 归一化 + 转 CHW（给模型用）
        image = transforms.normalize_img(image)  # (H,W,3) -> float 标准化 (按你原来的实现)
        depth = transforms.preprocess_depth(depth)
        image = np.transpose(image, (2, 0, 1))  # -> (3,H,W)
        normal = normal.astype(np.float32)
        normal = normal * 2.0 - 1.0  # 映射回物理向量区间
        normal = np.transpose(normal, (2, 0, 1))  # (3, H, W)

        # depth -> CHW
        if depth.ndim == 2:
            depth = depth[..., None]  # (H,W,1)
        depth = depth.astype(np.float32)
        depth = np.transpose(depth, (2, 0, 1))  # -> (1,H,W)

        return image, label, depth, normal, sam_mask

    def __getitem__(self, idx):
        img_name, image, label, depth, normal, sam_mask = super().__getitem__(idx)

        image, label, depth, normal, sam_mask = self.__transforms(
            image=image, label=label, depth=depth, normal=normal, sam_mask=sam_mask
        )

        if self.stage == "test":
            cls_label = 0
        else:
            cls_label = self.label_list[img_name]

        return img_name, image, label, cls_label, depth, normal, sam_mask


class Coco2VocClsDataset(VOC12Dataset):
    def __init__(self,
                 root_dir=None,
                 name_list_dir=None,
                 split='train',
                 stage='train',
                 tasks=None,
                 step=0,
                 resize_range=[512, 640],
                 rescale_range=[0.5, 2.0],
                 crop_size=512,
                 img_fliplr=True,
                 ignore_index=255,
                 num_classes=21,
                 aug=False,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage, tasks, step)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.local_crop_size = 96
        self.step = step
        self.img_fliplr = img_fliplr
        self.num_classes = num_classes
        self.color_jittor = transforms.PhotoMetricDistortion()

        self.gaussian_blur = transforms.GaussianBlur
        self.solarization = transforms.Solarization(p=0.2)

        self.label_list = load_cls_label_list(name_list_dir=name_list_dir)

        self.normalize = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.0),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            T.RandomGrayscale(p=0.2),
        ])

        self.global_view1 = T.Compose([
            # T.RandomResizedCrop(224, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=1.0),
            # self.normalize,
        ])
        self.global_view2 = T.Compose([
            T.RandomResizedCrop(self.crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.1),
            self.solarization,
            self.normalize,
        ])
        self.local_view = T.Compose([
            # T.RandomResizedCrop(self.local_crop_size, scale=[0.4, 1], interpolation=Image.BICUBIC),
            self.flip_and_color_jitter,
            self.gaussian_blur(p=0.5),
            self.normalize,
        ])

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label, depth, normal, sam_mask=None):
        img_box = None
        local_image = None
        if self.aug:
            if self.img_fliplr:
                # 直接传入所有参数，函数内部会自动处理翻转
                image, label, depth, normal, sam_mask = transforms.random_fliplr(
                    image, label, depth, normal, sam_mask
                )
            image = self.color_jittor(image)  # 只对 RGB 做光照扰动
            if self.crop_size:
                image, label, depth, normal, sam_mask, img_box = transforms.random_crop_rgbd(
                    image, label, depth, normal, sam_mask,
                    crop_size=self.crop_size,
                    mean_rgb=(123.675, 116.28, 103.53),
                    ignore_index=self.ignore_index,
                    depth_pad_strategy='edge'
                )
            local_image = self.local_view(Image.fromarray(image))

        image = transforms.normalize_img(image)  # (H,W,3) -> float 标准化 (按你原来的实现)
        image = np.transpose(image, (2, 0, 1))  # -> (3,H,W)
        normal = normal.astype(np.float32)
        normal = normal * 2.0 - 1.0  # 映射回物理向量区间
        normal = np.transpose(normal, (2, 0, 1))  # (3, H, W)
        depth = transforms.preprocess_depth(depth)
        depth = np.transpose(depth, (2, 0, 1))  # -> (3,H,W)
        return image, local_image, img_box, depth, normal, sam_mask

    @staticmethod
    def _to_onehot(label_mask, num_classes, ignore_index):
        # label_onehot = F.one_hot(label, num_classes)

        _label = np.unique(label_mask).astype(np.int16)
        # exclude ignore index
        _label = _label[_label != ignore_index]
        # exclude background
        _label = _label[_label != 0]

        label_onehot = np.zeros(shape=(num_classes), dtype=np.uint8)
        label_onehot[_label] = 1
        return label_onehot

    def __getitem__(self, idx):

        # [修改] 解包 6 个返回值
        img_name, image, label, depth, normal, sam_mask = super().__getitem__(idx)

        pil_image = Image.fromarray(image)

        # [修改] 传入 sam_mask
        image, local_image, img_box, depth, normal, sam_mask = self.__transforms(
            image=image, label=label, depth=depth, normal=normal, sam_mask=sam_mask
        )

        cls_label = self.label_list[img_name]

        return img_name, image, cls_label, sam_mask, img_box

