import random
import numpy as np
from PIL import Image
# from scipy import misc
import mmcv
import imageio

from PIL import ImageFilter, ImageOps, Image


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

def pad_depth_edge(depth, H, W):
    # depth: H0xW0 (float32)，返回 HxW，四周用边缘复制
    d = depth.astype(np.float32)
    ph, pw = max(0, H - d.shape[0]), max(0, W - d.shape[1])
    # 先右下补齐成目标大小的左上角，再整体做 edge 填充（简化：直接 np.pad 到目标尺寸）
    pad = ((0, H - d.shape[0]), (0, W - d.shape[1]))
    return np.pad(d, pad, mode='edge')

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    imgarr = np.asarray(img)
    proc_img = np.empty_like(imgarr, np.float32)

    proc_img[..., 0] = (imgarr[..., 0] - mean[0]) / std[0]
    proc_img[..., 1] = (imgarr[..., 1] - mean[1]) / std[1]
    proc_img[..., 2] = (imgarr[..., 2] - mean[2]) / std[2]
    return proc_img


def random_scaling(image, depth, label=None, scale_range=None):
    min_ratio, max_ratio = scale_range
    assert min_ratio <= max_ratio

    ratio = random.uniform(min_ratio, max_ratio)

    return _img_rescaling(image, depth, label, scale=ratio)


def _img_rescaling(image, depth, label=None, scale=None):
    assert scale is not None
    h, w = image.shape[:2]
    new_size = (int(round(scale * w)), int(round(scale * h)))  # (new_w, new_h)

    # --- image: BILINEAR ---
    img8 = image.astype(np.uint8) if image.dtype != np.uint8 else image
    new_image = Image.fromarray(img8).resize(new_size, resample=Image.BILINEAR)
    new_image = np.asarray(new_image, dtype=np.float32)

    # --- depth: BILINEAR，同步缩放 ---
    # 支持 HxW 或 HxWx1，保持返回形状与输入一致
    keep_channel = (depth.ndim == 3 and depth.shape[2] == 1)
    d = depth.squeeze() if keep_channel else depth
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    d_img = Image.fromarray(d, mode='F')                   # 32-bit float
    d_rs  = d_img.resize(new_size, resample=Image.BILINEAR)
    new_depth = np.asarray(d_rs, dtype=np.float32)
    if keep_channel:
        new_depth = new_depth[..., None]

    if label is None:
        return new_image, new_depth

    # --- label: NEAREST ---
    new_label = Image.fromarray(label.astype(np.int32)).resize(new_size, resample=Image.NEAREST)
    new_label = np.asarray(new_label, dtype=label.dtype)

    return new_image, new_depth, new_label



def img_resize_short(image, min_size=512):
    h, w, _ = image.shape
    if min(h, w) >= min_size:
        return image

    scale = float(min_size) / min(h, w)
    new_scale = [int(scale * w), int(scale * h)]

    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_image = np.asarray(new_image).astype(np.float32)

    return new_image


def random_resize(image, label=None, size_range=None):
    _new_size = random.randint(size_range[0], size_range[1])

    h, w, _ = image.shape
    scale = _new_size / float(max(h, w))
    new_scale = [int(scale * w), int(scale * h)]

    return _img_rescaling(image, label, scale=new_scale)


def random_fliplr(image, label=None, depth=None, normal=None, sam_mask=None):
    """
    随机水平翻转增强 (Random Horizontal Flip)
    同步翻转 Image, Label, Depth, Normal 和 SAM Mask
    """
    p = random.random()

    if p > 0.5:
        # 1. 图像翻转 (增加 .copy() 以解决负步长问题)
        image = np.fliplr(image).copy()

        if label is not None:
            label = np.fliplr(label).copy()

        if depth is not None:
            depth = np.fliplr(depth).copy()

        if normal is not None:
            normal = np.fliplr(normal).copy()
            # 法线翻转修正：
            # 假设 normal 范围是 [0, 1]，水平翻转后 x 分量 (index 0) 需要关于 0.5 对称翻转
            # 即: new_x = 1.0 - old_x
            # (如果您的 normal 是 [-1, 1] 范围，请改为 normal[:, :, 0] *= -1)
            normal[:, :, 0] = 1.0 - normal[:, :, 0]

        if sam_mask is not None:
            sam_mask = np.fliplr(sam_mask).copy()

    # =========================================================
    # 动态返回值 (根据输入参数决定返回内容)
    # =========================================================

    # 场景 A: 有 SAM Mask
    if sam_mask is not None:
        if label is not None:
            # 完整返回: image, label, depth, normal, sam_mask
            return image, label, depth, normal, sam_mask
        else:
            # 无 Label (例如 Cls Dataset): image, depth, normal, sam_mask
            return image, depth, normal, sam_mask

    # 场景 B: 无 SAM Mask (兼容旧代码)
    else:
        if label is None:
            return image, depth, normal
        else:
            return image, label, depth, normal


def random_flipud(image, label=None):
    p = random.random()

    if label is None:
        if p > 0.5:
            image = np.flipud(image)
        return image
    else:
        if p > 0.5:
            image = np.flipud(image)
            label = np.flipud(label)

        return image, label


def random_rot(image, label):
    k = random.randrange(3) + 1

    image = np.rot90(image, k).copy()

    if label is None:
        return image

    label = np.rot90(label, k).copy()

    return image, label

def preprocess_depth(depth_hw, q_low=1, q_high=99):
    """
    depth_hw: HxW float 或整型（npy/png 读入）
    返回: HxW float32, 范围[0,1]
    """
    d = depth_hw.astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)  # 处理NaN/Inf

    vmin = np.percentile(d, q_low)
    vmax = np.percentile(d, q_high)
    if not np.isfinite(vmin): vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin: vmax = vmin + 1e-6

    d01 = np.clip((d - vmin) / (vmax - vmin), 0.0, 1.0)    # 线性拉伸到[0,1]
    return d01


import numpy as np


def random_crop_rgbd(image, label, depth, normal, sam_mask=None,
                     crop_size=512,
                     mean_rgb=(123.675, 116.28, 103.53),
                     ignore_index=255,
                     depth_pad_strategy='edge'):
    """
    同步对 image/label/depth/normal/sam_mask 做随机裁剪
    增加 sam_mask 支持，填充默认为 0
    """
    # --- 1. 统一 depth 形状 (去掉最后一维) ---
    if depth.ndim == 3 and depth.shape[2] == 1:
        d = depth[..., 0].astype(np.float32)
    elif depth.ndim == 2:
        d = depth.astype(np.float32)
    else:
        raise ValueError(f"Unexpected depth shape: {depth.shape}")

    h, w = image.shape[:2]

    # --- 2. 确定画布大小 ---
    if isinstance(crop_size, (tuple, list)):
        ch, cw = int(crop_size[0]), int(crop_size[1])
    else:
        ch = cw = int(crop_size)
    H, W = max(ch, h), max(cw, w)

    # --- 3. 生成统一偏移量 (所有图公用) ---
    Hp = int(np.random.randint(0, H - h + 1))
    Wp = int(np.random.randint(0, W - w + 1))

    # 计算 Padding 的上下左右距离
    pad_top = Hp
    pad_bottom = H - h - Hp
    pad_left = Wp
    pad_right = W - w - Wp

    # ---------- Pad Image (RGB用均值填补) ----------
    C = image.shape[2]
    pad_image = np.zeros((H, W, C), dtype=np.uint8)
    pad_image[..., 0] = mean_rgb[0]
    pad_image[..., 1] = mean_rgb[1]
    pad_image[..., 2] = mean_rgb[2]
    pad_image[Hp:Hp + h, Wp:Wp + w, :C] = image

    # ---------- Pad Label (用ignore_index填补) ----------
    pad_label = None
    if label is not None:
        pad_label = np.full((H, W), ignore_index, dtype=np.uint8)
        pad_label[Hp:Hp + h, Wp:Wp + w] = label

    # ---------- Pad SAM Mask (新增，用 0 填补) ----------
    pad_sam_mask = None
    if sam_mask is not None:
        # SAM Mask 通常是 int 类型，0 表示背景
        pad_sam_mask = np.full((H, W), -1, dtype=sam_mask.dtype)
        pad_sam_mask[Hp:Hp + h, Wp:Wp + w] = sam_mask

    # ---------- Pad Depth (几何图推荐用 edge) ----------
    if depth_pad_strategy == 'edge':
        pad_depth = np.pad(d, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='edge')
    elif depth_pad_strategy == 'lowq':
        q1 = float(np.quantile(d, 0.01))
        pad_depth = np.ones((H, W), dtype=np.float32) * q1
        pad_depth[Hp:Hp + h, Wp:Wp + w] = d
    elif depth_pad_strategy == 'constant0':
        pad_depth = np.zeros((H, W), dtype=np.float32)
        pad_depth[Hp:Hp + h, Wp:Wp + w] = d
    else:
        raise ValueError(f"Unknown depth_pad_strategy: {depth_pad_strategy}")

    pad_normal = np.pad(normal,
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode='edge')

    # --- 4. 采样裁剪窗口 ---
    def get_random_cropbox(_label, cat_max_ratio=0.75):
        # 如果没有 label，直接随机裁
        if _label is None:
            H_start = np.random.randint(0, H - ch + 1)
            W_start = np.random.randint(0, W - cw + 1)
            return H_start, H_start + ch, W_start, W_start + cw

        # 尝试 10 次，寻找包含前景且单一类别占比不过高的区域
        for _ in range(10):
            H_start = np.random.randint(0, H - ch + 1)
            W_start = np.random.randint(0, W - cw + 1)
            H_end, W_end = H_start + ch, W_start + cw

            tmp = _label[H_start:H_end, W_start:W_end]
            idx, cnt = np.unique(tmp, return_counts=True)
            if ignore_index is not None:
                cnt = cnt[idx != ignore_index]

            if len(cnt) > 1 and (np.max(cnt) / (np.sum(cnt) + 1e-6) < cat_max_ratio):
                return H_start, H_end, W_start, W_end

        return H_start, H_end, W_start, W_end

    Hs, He, Ws, We = get_random_cropbox(pad_label)

    # --- 5. 执行裁剪 ---
    crop_image = pad_image[Hs:He, Ws:We, :]
    crop_label = None if pad_label is None else pad_label[Hs:He, Ws:We]
    crop_depth = pad_depth[Hs:He, Ws:We][..., None]
    crop_normal = pad_normal[Hs:He, Ws:We, :]

    # 裁剪 SAM Mask
    crop_sam_mask = None
    if pad_sam_mask is not None:
        crop_sam_mask = pad_sam_mask[Hs:He, Ws:We]

    # 记录有效区域框
    img_H_start = max(Hp - Hs, 0)
    img_W_start = max(Wp - Ws, 0)
    img_H_end = min(ch, h + Hp - Hs)
    img_W_end = min(cw, w + Wp - Ws)
    img_box = np.asarray([img_H_start, img_H_end, img_W_start, img_W_end], dtype=np.int16)

    # --- 6. 动态返回 ---
    if sam_mask is not None:
        return crop_image, crop_label, crop_depth, crop_normal, crop_sam_mask, img_box
    else:
        return crop_image, crop_label, crop_depth, crop_normal, img_box



def random_crop(image, label=None, crop_size=None, mean_rgb=[0, 0, 0], ignore_index=255):
    h, w, _ = image.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_image = np.zeros((H, W, 3), dtype=np.uint8)

    pad_image[:, :, 0] = mean_rgb[0]
    pad_image[:, :, 1] = mean_rgb[1]
    pad_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = image

    def get_random_cropbox(_label, cat_max_ratio=0.75):

        for i in range(10):

            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            if _label is None:
                return H_start, H_end, W_start, W_end,

            temp_label = _label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]

            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break

        return H_start, H_end, W_start, W_end,

    H_start, H_end, W_start, W_end = get_random_cropbox(label)

    crop_image = pad_image[H_start:H_end, W_start:W_end, :]

    img_H_start = max(H_pad - H_start, 0)
    img_W_start = max(W_pad - W_start, 0)
    img_H_end = min(crop_size, h + H_pad - H_start)
    img_W_end = min(crop_size, w + W_pad - W_start)
    img_box = np.asarray([img_H_start, img_H_end, img_W_start, img_W_end], dtype=np.int16)

    if label is None:
        return crop_image, img_box

    pad_label = np.ones((H, W), dtype=np.uint8) * ignore_index
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label
    label = pad_label[H_start:H_end, W_start:W_end]

    return crop_image, label, img_box


class PhotoMetricDistortion(object):
    """ from mmseg """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):

        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if np.random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if np.random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if np.random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
            0] = (img[:, :, 0].astype(int) +
                  np.random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, img):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        # img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        # results['img'] = img
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str