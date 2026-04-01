import os
import torch
import numpy as np
import torch.nn.functional as F
import glob
import cv2
from tqdm import tqdm

# ================= Configuration =================
CONFIG = {
    # 输入路径 (全是 .npy)
    "mask_dir": "/data/fangkai/SAM_MASK",
    "depth_dir": "/data/fangkai/VOC_depth/depth",
    "normal_dir": "/data/fangkai/VOC_normal/normal",

    # 输出路径 (.npy 数据保存路径)
    "save_dir": "/data/fangkai/SAM_MASK_REFINED_V2",

    # --- [新增] 可视化图片保存路径 ---
    # 这里会保存前 100 张对比图 (png)
    "vis_dir": "/data/fangkai/SAM_MASK_REFINED_V2/visual_debug",

    # --- 合并阈值 ---
    "thresh_depth": 0.03,
    "thresh_normal": 0.9,
    "thresh_boundary_edge": 0.1,

    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# =================================================

class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_i] = root_j


def compute_depth_edges(depth_tensor):
    """Sobel 算子计算深度边缘"""
    img = depth_tensor.unsqueeze(0).unsqueeze(0)
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=img.device, dtype=torch.float32).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=img.device, dtype=torch.float32).view(1, 1, 3, 3)
    gx = F.conv2d(img, kx, padding=1)
    gy = F.conv2d(img, ky, padding=1)
    edge_magnitude = torch.sqrt(gx ** 2 + gy ** 2).squeeze()
    return edge_magnitude


def is_boundary_strong(mask_a_bool, mask_b_bool, edge_map, threshold):
    """检查边界强度"""
    mask_a_float = mask_a_bool.float().unsqueeze(0).unsqueeze(0)
    dilated_a = F.max_pool2d(mask_a_float, kernel_size=3, stride=1, padding=1).squeeze() > 0
    boundary_region = dilated_a & mask_b_bool
    num_boundary_pixels = boundary_region.sum()

    if num_boundary_pixels == 0: return False

    avg_edge_strength = edge_map[boundary_region].mean()
    if avg_edge_strength > threshold:
        return True
    return False


def get_mask_properties(masks, depth, normal, unique_ids):
    props = {}
    for uid in unique_ids:
        mask_bool = (masks == uid)
        d_val = depth[mask_bool].mean()
        n_vals = normal[:, mask_bool].mean(dim=1)
        n_vals = F.normalize(n_vals, dim=0)
        ys, xs = torch.where(mask_bool)
        if len(ys) == 0: continue
        bbox = (ys.min().item(), ys.max().item(), xs.min().item(), xs.max().item())
        props[uid.item()] = {
            'depth': d_val, 'normal': n_vals, 'bbox': bbox, 'mask_bool': mask_bool
        }
    return props


def is_bbox_overlap(bbox1, bbox2, padding=5):
    y1_min, y1_max, x1_min, x1_max = bbox1
    y2_min, y2_max, x2_min, x2_max = bbox2
    if (x1_max + padding < x2_min) or (x2_max + padding < x1_min) or \
            (y1_max + padding < y2_min) or (y2_max + padding < y1_min):
        return False
    return True


def merge_masks_logic(mask_tensor, depth_tensor, normal_tensor):
    unique_ids = torch.unique(mask_tensor)
    unique_ids = unique_ids[unique_ids > 0]  # 排除背景 (0 或 -1)

    if len(unique_ids) <= 1: return mask_tensor

    props = get_mask_properties(mask_tensor, depth_tensor, normal_tensor, unique_ids)
    if not props: return mask_tensor

    depth_edge_map = compute_depth_edges(depth_tensor)
    uf = UnionFind(props.keys())
    ids = list(props.keys())

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            id_a, id_b = ids[i], ids[j]
            p_a, p_b = props[id_a], props[id_b]

            # 1. BBox
            if not is_bbox_overlap(p_a['bbox'], p_b['bbox'], padding=10): continue
            # 2. Boundary Edge Check
            if is_boundary_strong(p_a['mask_bool'], p_b['mask_bool'], depth_edge_map,
                                  CONFIG['thresh_boundary_edge']): continue
            # 3. Depth Diff
            if torch.abs(p_a['depth'] - p_b['depth']) > CONFIG['thresh_depth']: continue
            # 4. Normal Sim
            if torch.dot(p_a['normal'], p_b['normal']) < CONFIG['thresh_normal']: continue

            uf.union(id_a, id_b)

    new_mask = torch.zeros_like(mask_tensor)
    # 保留背景
    new_mask[mask_tensor <= 0] = mask_tensor[mask_tensor <= 0]

    for uid in ids:
        root = uf.find(uid)
        mask_bool = (mask_tensor == uid)
        new_mask[mask_bool] = float(root)

    return new_mask


# ================= [新增] 可视化工具函数 =================

def mask_to_color_img(mask_np):
    """将整数 ID mask 转换为彩色图片以便观察"""
    h, w = mask_np.shape
    img_color = np.zeros((h, w, 3), dtype=np.uint8)

    unique_ids = np.unique(mask_np)
    # 排除背景 (<=0)
    unique_ids = unique_ids[unique_ids > 0]

    for uid in unique_ids:
        # 使用 uid 作为种子，保证同一 ID 颜色固定
        np.random.seed(int(uid) * 123)
        color = np.random.randint(50, 255, size=3)
        img_color[mask_np == uid] = color

    return img_color


def save_debug_visualization(filename, orig_mask_t, refined_mask_t, save_dir):
    """保存对比图：左边原始，右边合并后"""
    # 转 Numpy
    orig_np = orig_mask_t.cpu().numpy().astype(np.int32)
    refined_np = refined_mask_t.cpu().numpy().astype(np.int32)

    # 转彩色
    vis_orig = mask_to_color_img(orig_np)
    vis_refined = mask_to_color_img(refined_np)

    # 添加文字标签
    cv2.putText(vis_orig, "Original SAM", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_refined, "Merged (Refined)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 左右拼接
    combined = cv2.hconcat([vis_orig, vis_refined])

    # 保存
    save_path = os.path.join(save_dir, filename + "_vis.png")
    cv2.imwrite(save_path, combined)


# =======================================================

def process_dataset():
    # 路径检查
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    os.makedirs(CONFIG['vis_dir'], exist_ok=True)  # 创建可视化目录

    search_path = os.path.join(CONFIG['mask_dir'], "*.npy")
    file_list = glob.glob(search_path)
    print(f"Found {len(file_list)} .npy files.")
    device = torch.device(CONFIG['device'])

    # 计数器
    vis_count = 0
    VIS_LIMIT = 100  # 只保存前 100 张

    for mask_path in tqdm(file_list):
        try:
            filename = os.path.basename(mask_path)
            basename = os.path.splitext(filename)[0]  # 去掉后缀

            # --- Load Data ---
            mask_np = np.load(mask_path)
            mask_t = torch.from_numpy(mask_np.astype(np.int64)).to(device)
            if len(mask_t.shape) == 3: mask_t = mask_t.squeeze()

            depth_path = os.path.join(CONFIG['depth_dir'], filename)
            if not os.path.exists(depth_path):
                np.save(os.path.join(CONFIG['save_dir'], filename), mask_np)
                continue
            depth_np = np.load(depth_path)
            depth_t = torch.from_numpy(depth_np.astype(np.float32)).to(device)
            if len(depth_t.shape) == 3: depth_t = depth_t.squeeze()
            if depth_t.max() > 1.0: depth_t = depth_t / (depth_t.max() + 1e-6)

            normal_path = os.path.join(CONFIG['normal_dir'], filename)
            if not os.path.exists(normal_path):
                np.save(os.path.join(CONFIG['save_dir'], filename), mask_np)
                continue
            normal_np = np.load(normal_path)
            normal_t = torch.from_numpy(normal_np.astype(np.float32)).to(device)
            if normal_t.shape[-1] == 3: normal_t = normal_t.permute(2, 0, 1)
            if normal_t.min() >= 0 and normal_t.max() <= 1.0:
                normal_t = (normal_t - 0.5) * 2.0
            elif normal_t.max() > 1.0:
                normal_t = (normal_t / 255.0 - 0.5) * 2.0
            normal_t = F.normalize(normal_t, dim=0)

            # --- Merge Logic V2 ---
            refined_mask_t = merge_masks_logic(mask_t, depth_t, normal_t)

            # --- [新增] 可视化保存 (仅前100张) ---
            if vis_count < VIS_LIMIT:
                save_debug_visualization(basename, mask_t, refined_mask_t, CONFIG['vis_dir'])
                vis_count += 1
                if vis_count == VIS_LIMIT:
                    print(
                        f"\n[Info] Reached {VIS_LIMIT} visualization limit. Continuing processing without saving images...")

            # --- Save NPY ---
            refined_mask_np = refined_mask_t.cpu().numpy().astype(mask_np.dtype)
            np.save(os.path.join(CONFIG['save_dir'], filename), refined_mask_np)

        except Exception as e:
            print(f"Error processing {mask_path}: {e}")
            continue


if __name__ == "__main__":
    process_dataset()
    print(f"\nProcessing complete.")
    print(f"Refined masks saved to: {CONFIG['save_dir']}")
    print(f"Visualizations saved to: {CONFIG['vis_dir']}")