import pdb
import torch
import torch.nn.functional as F


def cam_to_label(cam, cls_label, img_box=None, bkg_thre=None, high_thre=None, low_thre=None, ignore_mid=False,
                 ignore_index=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _pseudo_label = valid_cam.max(dim=1, keepdim=False)
    _pseudo_label += 1
    _pseudo_label[cam_value <= bkg_thre] = 0

    if img_box is None:
        return _pseudo_label

    if ignore_mid:
        _pseudo_label[cam_value <= high_thre] = ignore_index
        _pseudo_label[cam_value <= low_thre] = 0
    pseudo_label = torch.ones_like(_pseudo_label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = _pseudo_label[idx, coord[0]:coord[1],
                                                                  coord[2]:coord[3]]

    return valid_cam, pseudo_label


def cam_to_roi_mask2(cam, cls_label, hig_thre=None, low_thre=None):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam
    cam_value, _ = valid_cam.max(dim=1, keepdim=False)
    # _pseudo_label += 1
    roi_mask = torch.ones_like(cam_value, dtype=torch.int16)
    roi_mask[cam_value <= low_thre] = 0
    roi_mask[cam_value >= hig_thre] = 2

    return roi_mask


def get_valid_cam(cam, cls_label):
    b, c, h, w = cam.shape
    # pseudo_label = torch.zeros((b,h,w))
    cls_label_rep = cls_label.unsqueeze(-1).unsqueeze(-1).repeat([1, 1, h, w])
    valid_cam = cls_label_rep * cam

    return valid_cam


def ignore_img_box(label, img_box, ignore_index):
    pseudo_label = torch.ones_like(label) * ignore_index

    for idx, coord in enumerate(img_box):
        pseudo_label[idx, coord[0]:coord[1], coord[2]:coord[3]] = label[idx, coord[0]:coord[1], coord[2]:coord[3]]

    return pseudo_label


def crop_from_roi_neg(images, roi_mask=None, crop_num=8, crop_size=96):
    crops = []

    b, c, h, w = images.shape

    temp_crops = torch.zeros(size=(b, crop_num, c, crop_size, crop_size)).to(images.device)
    flags = torch.ones(size=(b, crop_num + 2)).to(images.device)
    margin = crop_size // 2

    for i1 in range(b):
        roi_index = (roi_mask[i1, margin:(h - margin), margin:(w - margin)] <= 1).nonzero()
        if roi_index.shape[0] < crop_num:
            roi_index = (roi_mask[i1, margin:(h - margin),
                         margin:(w - margin)] >= 0).nonzero()  ## if NULL then random crop
        rand_index = torch.randperm(roi_index.shape[0])
        crop_index = roi_index[rand_index[:crop_num], :]

        for i2 in range(crop_num):
            h0, w0 = crop_index[i2, 0], crop_index[i2, 1]  # centered at (h0, w0)
            temp_crops[i1, i2, ...] = images[i1, :, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            temp_mask = roi_mask[i1, h0:(h0 + crop_size), w0:(w0 + crop_size)]
            if temp_mask.sum() / (crop_size * crop_size) <= 0.2:
                ## if ratio of uncertain regions < 0.2 then negative
                flags[i1, i2 + 2] = 0

    _crops = torch.chunk(temp_crops, chunks=crop_num, dim=1, )
    crops = [c[:, 0] for c in _crops]

    return crops, flags


def multi_scale_cam2(model, inputs, depth, scales):
    '''process cam and aux-cam'''
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)
        depth_cat = torch.cat([depth, depth.flip(-1)], dim=0)

        _cam_aux, _cam = model(inputs_cat, depth_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                _depth = F.interpolate(depth, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                _inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)
                _depth_cat = torch.cat([_depth, _depth.flip(-1)], dim=0)

                _cam_aux, _cam = model(_inputs_cat, _depth_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5

        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux

def multi_scale_cam2_filter(model, inputs, scales):
    # cam_list, tscam_list = [], []
    b, c, h, w = inputs.shape
    with torch.no_grad():
        inputs_cat = torch.cat([inputs, inputs.flip(-1)], dim=0)

        _cam_aux, _cam = model(inputs_cat, cam_only=True)

        _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
        _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
        _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
        _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

        cam_list = [F.relu(_cam)]
        cam_aux_list = [F.relu(_cam_aux)]

        for s in scales:
            if s != 1.0:
                _inputs = F.interpolate(inputs, size=(int(s * h), int(s * w)), mode='bilinear', align_corners=False)
                inputs_cat = torch.cat([_inputs, _inputs.flip(-1)], dim=0)

                _cam_aux, _cam = model(inputs_cat, cam_only=True)

                _cam = F.interpolate(_cam, size=(h, w), mode='bilinear', align_corners=False)
                _cam = torch.max(_cam[:b, ...], _cam[b:, ...].flip(-1))
                _cam_aux = F.interpolate(_cam_aux, size=(h, w), mode='bilinear', align_corners=False)
                _cam_aux = torch.max(_cam_aux[:b, ...], _cam_aux[b:, ...].flip(-1))

                cam_list.append(F.relu(_cam))
                cam_aux_list.append(F.relu(_cam_aux))

        cam = torch.sum(torch.stack(cam_list, dim=0), dim=0)
        cam = cam + F.adaptive_max_pool2d(-cam, (1, 1))
        cam_max = F.adaptive_max_pool2d(cam, (1, 1)).squeeze(-1).squeeze(-1) #[b,c]
        cam /= F.adaptive_max_pool2d(cam, (1, 1)) + 1e-5


        cam_aux = torch.sum(torch.stack(cam_aux_list, dim=0), dim=0)
        cam_aux = cam_aux + F.adaptive_max_pool2d(-cam_aux, (1, 1))
        cam_aux /= F.adaptive_max_pool2d(cam_aux, (1, 1)) + 1e-5

    return cam, cam_aux, cam_max


def label_to_aff_mask(cam_label, ignore_index=255):
    b, h, w = cam_label.shape

    _cam_label = cam_label.reshape(b, 1, -1)
    _cam_label_rep = _cam_label.repeat([1, _cam_label.shape[-1], 1])
    _cam_label_rep_t = _cam_label_rep.permute(0, 2, 1)
    aff_label = (_cam_label_rep == _cam_label_rep_t).type(torch.long)

    for i in range(b):
        aff_label[i, :, _cam_label_rep[i, 0, :] == ignore_index] = ignore_index
        aff_label[i, _cam_label_rep[i, 0, :] == ignore_index, :] = ignore_index
    aff_label[:, range(h * w), range(h * w)] = ignore_index
    return aff_label


def refine_cams_with_bkg_v2(ref_mod=None, images=None, cams=None, cls_labels=None, high_thre=None, low_thre=None,
                            ignore_index=False, img_box=None, down_scale=2):
    b, _, h, w = images.shape
    _images = F.interpolate(images, size=[h // down_scale, w // down_scale], mode="bilinear", align_corners=False)

    bkg_h = torch.ones(size=(b, 1, h, w)) * high_thre
    bkg_h = bkg_h.to(cams.device)
    bkg_l = torch.ones(size=(b, 1, h, w)) * low_thre
    bkg_l = bkg_l.to(cams.device)

    bkg_cls = torch.ones(size=(b, 1))
    bkg_cls = bkg_cls.to(cams.device)
    cls_labels = torch.cat((bkg_cls, cls_labels), dim=1)

    refined_label = torch.ones(size=(b, h, w)) * ignore_index
    refined_label = refined_label.to(cams.device)
    refined_label_h = refined_label.clone()
    refined_label_l = refined_label.clone()

    cams_with_bkg_h = torch.cat((bkg_h, cams), dim=1)
    _cams_with_bkg_h = F.interpolate(cams_with_bkg_h, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)
    cams_with_bkg_l = torch.cat((bkg_l, cams), dim=1)
    _cams_with_bkg_l = F.interpolate(cams_with_bkg_l, size=[h // down_scale, w // down_scale], mode="bilinear",
                                     align_corners=False)  # .softmax(dim=1)

    for idx, coord in enumerate(img_box):
        valid_key = torch.nonzero(cls_labels[idx, ...])[:, 0]
        valid_cams_h = _cams_with_bkg_h[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)
        valid_cams_l = _cams_with_bkg_l[idx, valid_key, ...].unsqueeze(0).softmax(dim=1)

        _refined_label_h = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_h,
                                        valid_key=valid_key, orig_size=(h, w))
        _refined_label_l = _refine_cams(ref_mod=ref_mod, images=_images[[idx], ...], cams=valid_cams_l,
                                        valid_key=valid_key, orig_size=(h, w))

        refined_label_h[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_h[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]
        refined_label_l[idx, coord[0]:coord[1], coord[2]:coord[3]] = _refined_label_l[0, coord[0]:coord[1],
                                                                     coord[2]:coord[3]]

    refined_label = refined_label_h.clone()
    refined_label[refined_label_h == 0] = ignore_index
    refined_label[(refined_label_h + refined_label_l) == 0] = 0

    return refined_label


def _refine_cams(ref_mod, images, cams, valid_key, orig_size):
    refined_cams = ref_mod(images, cams)
    refined_cams = F.interpolate(refined_cams, size=orig_size, mode="bilinear", align_corners=False)
    refined_label = refined_cams.argmax(dim=1)
    refined_label = valid_key[refined_label]

    return refined_label


def get_mixed_label(cam_label, old_segs_label, total_classes, new_classes):
    # b H W
    cam_label = cam_label.long()
    old_segs_label = old_segs_label.long()

    new_segs_label = torch.full_like(cam_label, 255).long()
    cam_old_classes = total_classes - new_classes
    cam_new_label = torch.full_like(cam_label, 255).long()
    cam_new_label[cam_label >= cam_old_classes + 1] = cam_label[cam_label >= cam_old_classes + 1]

    new_segs_label = old_segs_label
    new_segs_label[cam_new_label != 255] = cam_new_label[cam_new_label != 255]

    return new_segs_label



def cam_high_pass_filter(cam_channel_max, cls_label, cam_value_thre=20):
    low_activation_channel = cam_channel_max < cam_value_thre
    cls_label_filter = cls_label.clone().cuda()
    cls_label_filter[low_activation_channel] = 0

    b, c = cls_label.shape
    for i in range(b):
        if torch.sum(cls_label_filter[i]) == 0:
            valid_channels = cls_label[i].nonzero(as_tuple=False).squeeze(1)
            if valid_channels.numel() > 0:
                max_idx = torch.argmax(cam_channel_max[i, valid_channels])
                max_channel = valid_channels[max_idx]
                cls_label_filter[i, max_channel] = 1

    return cls_label_filter

def filter_cam_pesudo_label(cam_label, cam_filter_class, ignore_idx=255):

    B, H, W = cam_label.shape
    bg_class = torch.ones((B, 1), dtype=cam_filter_class.dtype, device=cam_filter_class.device)
    cam_filter_class = torch.cat((bg_class,cam_filter_class),dim=1)
    
    cam_label_filtered = cam_label.clone()

    for i in range(B):

        class_mask = cam_filter_class[i].bool()  # [C + 1]
        valid_classes = torch.arange(class_mask.size(0), device=cam_label.device)[class_mask]
        keep_mask = torch.isin(cam_label_filtered[i], valid_classes)  
        cam_label_filtered[i][~keep_mask] = ignore_idx

    return cam_label_filtered

def _refine_boundaries_bilateral(self, label, depth, normal, kernel_size=5, alpha_d=15.0, beta_n=5.0):
    b, h, w = label.shape
    device = label.device

    # 临时处理 ignore
    ignore_mask = (label == 255)
    label_clean = label.clone()
    label_clean[ignore_mask] = self.bg_class_id

    label_onehot = F.one_hot(label_clean, num_classes=self.total_classes).permute(0, 3, 1, 2).float()
    pad = kernel_size // 2

    # Unfold
    label_unf = F.unfold(label_onehot, kernel_size, padding=pad)
    depth_unf = F.unfold(depth, kernel_size, padding=pad)
    normal_unf = F.unfold(normal, kernel_size, padding=pad)

    # Reshape
    label_unf = label_unf.view(b, self.total_classes, -1, h * w)
    depth_unf = depth_unf.view(b, 1, -1, h * w)
    normal_unf = normal_unf.view(b, 3, -1, h * w)

    depth_center = depth.view(b, 1, 1, h * w)
    normal_center = normal.view(b, 3, 1, h * w)

    # Weights
    diff_depth = torch.abs(depth_unf - depth_center)
    w_depth = torch.exp(-diff_depth * alpha_d)

    dot_normal = torch.sum(normal_unf * normal_center, dim=1, keepdim=True)
    w_normal = torch.pow(torch.clamp(dot_normal, min=0), beta_n)

    total_weight = w_depth * w_normal

    # Vote
    weighted_votes = torch.sum(label_unf * total_weight, dim=2)
    refined_flat = torch.argmax(weighted_votes, dim=1)
    refined_label = refined_flat.view(b, h, w)

    # Restore ignore
    refined_label = torch.where(ignore_mask, torch.tensor(255, device=device), refined_label)

    return refined_label


def get_mixed_label_with_sam(cam_label, old_segs_label, sam_mask, total_classes, new_classes, ignore_index=255):
    """
    [魔改版 - 兼容 total_classes=15]
    即使你传入 total_classes=15 (最大ID)，这个版本也会自动在内部 +1，
    从而支持 <= total_classes 的判定，解决 Person mIoU=0 的问题。
    """
    b, h, w = cam_label.shape
    device = cam_label.device

    # ============================================================
    # 1. [关键修改] 自动修正总类别数
    # 如果你习惯传 15 (MaxID)，这里我们强制 +1 变成 16 (Count)，用于申请显存
    # ============================================================
    real_total_classes = total_classes + 1  # 内部使用 16

    # 重新计算分界线：16 - 5 = 11. (索引 0-10 是旧, 11-15 是新)
    cam_old_classes = real_total_classes - new_classes

    SAM_TRUST_THRESH = 0.6

    final_label = old_segs_label.clone().to(device)
    sam_mask = sam_mask.to(device)

    # 构建网格
    y_grid, x_grid = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
    y_grid = y_grid.unsqueeze(0).expand(b, -1, -1).float()
    x_grid = x_grid.unsqueeze(0).expand(b, -1, -1).float()

    for i in range(b):
        curr_sam = sam_mask[i]
        curr_cam = cam_label[i]
        curr_old = old_segs_label[i]
        flat_sam = curr_sam.view(-1)

        valid_pixel_mask = (flat_sam != -1)
        if not valid_pixel_mask.any(): continue

        active_sam = flat_sam[valid_pixel_mask]
        active_y = y_grid[i].view(-1)[valid_pixel_mask]
        active_x = x_grid[i].view(-1)[valid_pixel_mask]

        num_regions = int(active_sam.max().item()) + 1

        # --- 计算中心和权重 (保持不变) ---
        region_counts = torch.zeros(num_regions, device=device)
        region_counts.scatter_add_(0, active_sam, torch.ones_like(active_sam, dtype=torch.float))
        region_counts = region_counts.clamp(min=1e-5)

        sum_y = torch.zeros(num_regions, device=device)
        sum_x = torch.zeros(num_regions, device=device)
        sum_y.scatter_add_(0, active_sam, active_y)
        sum_x.scatter_add_(0, active_sam, active_x)

        cent_y = sum_y / region_counts
        cent_x = sum_x / region_counts

        radius_proxy = torch.sqrt(region_counts) / 2.0
        sigma = torch.max(radius_proxy, torch.tensor(1.0, device=device))

        pixel_cy = cent_y[active_sam]
        pixel_cx = cent_x[active_sam]
        pixel_sigma = sigma[active_sam]
        dist_sq = (active_y - pixel_cy) ** 2 + (active_x - pixel_cx) ** 2
        weights = torch.exp(-dist_sq / (2 * pixel_sigma ** 2 + 1e-5))

        # --- 仲裁逻辑 ---
        active_cam = curr_cam.view(-1)[valid_pixel_mask]

        # 注意：这里逻辑基于 cam_old_classes (11)
        # ID 11 >= 11 (True, 新类)
        is_new_class_pixel = (active_cam >= cam_old_classes) & (active_cam != ignore_index)

        sum_weights = torch.zeros(num_regions, device=device)
        sum_weights.scatter_add_(0, active_sam, weights)
        weighted_new_score = torch.zeros(num_regions, device=device)
        weighted_new_score.scatter_add_(0, active_sam, weights * is_new_class_pixel.float())
        ratios = weighted_new_score / (sum_weights + 1e-5)
        region_decision_is_new = (ratios > SAM_TRUST_THRESH)

        # --- 投票逻辑 ---
        pixel_decision_is_new = region_decision_is_new[active_sam]
        active_old = curr_old.view(-1)[valid_pixel_mask]

        vote_labels = torch.full_like(active_sam, -1, dtype=torch.long)

        mask_branch_new = pixel_decision_is_new & is_new_class_pixel
        vote_labels[mask_branch_new] = active_cam[mask_branch_new].long()

        mask_branch_old = (~pixel_decision_is_new) & (active_old != ignore_index)
        vote_labels[mask_branch_old] = active_old[mask_branch_old].long()

        # ============================================================
        # 2. [关键修改] 这里的判断逻辑改成 <=
        # 因为我们内部扩容了，所以 index=15 是安全的
        # total_classes (15) < real_total_classes (16)
        # ============================================================
        valid_votes_mask = (vote_labels != -1) & (vote_labels <= total_classes) & (vote_labels >= 0)

        if not valid_votes_mask.any(): continue

        final_vote_regions = active_sam[valid_votes_mask]
        final_vote_classes = vote_labels[valid_votes_mask]
        final_vote_weights = weights[valid_votes_mask]

        # ============================================================
        # 3. [关键修改] 展平索引必须乘 real_total_classes (16)
        # 否则数据会错位，把 Region A 的 Person 票投给 Region B 的背景
        # ============================================================
        flat_vote_indices = final_vote_regions * real_total_classes + final_vote_classes

        # 申请显存也要用 real_total_classes
        vote_bins = torch.zeros(num_regions * real_total_classes, device=device)
        vote_bins.scatter_add_(0, flat_vote_indices, final_vote_weights)

        # reshape 也要用 real_total_classes
        winner_classes = torch.argmax(vote_bins.view(num_regions, real_total_classes), dim=1)

        region_has_votes = (vote_bins.view(num_regions, real_total_classes).sum(dim=1) > 0)

        # --- 赋值 ---
        pixel_winners = winner_classes[active_sam]
        pixel_valid_update = region_has_votes[active_sam]

        curr_final_flat = final_label[i].view(-1)
        original_vals = curr_final_flat[valid_pixel_mask]
        new_vals = torch.where(pixel_valid_update, pixel_winners.long(), original_vals)
        curr_final_flat[valid_pixel_mask] = new_vals
        final_label[i] = curr_final_flat.view(h, w)

    return final_label