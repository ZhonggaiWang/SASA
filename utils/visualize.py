import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def save_prototype_similarity(decoder, classes_list=None, save_path='prototype_similarity.png'):
    """
    提取 LargeFOV 解码器的原型，计算余弦相似度矩阵并保存为图片。

    Args:
        decoder: 你的 LargeFOV 实例
        classes_list: (可选) 任务类别列表，例如 [15, 5]，用于画分界线
        save_path: 图片保存的具体路径，例如 'results/sim_epoch_10.png'
    """
    # 确保模型在评估模式（虽然对权重提取没影响，但好习惯）

    # --- 1. 提取并拼接所有原型 ---
    all_prototypes = []

    # 遍历 IncrementalConv (conv8) 的子模块
    for sub_conv in decoder.conv8.children():
        # 获取权重: [Out, In, 1, 1] -> [Out, In]
        weights = sub_conv.weight.data
        weights = weights.view(weights.size(0), -1)
        all_prototypes.append(weights)

    # 拼接: [Total_Classes, 512]
    prototypes = torch.cat(all_prototypes, dim=0)

    # --- 2. 计算余弦相似度 ---
    # L2 归一化
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    # 矩阵乘法得到相似度 [-1, 1]
    similarity_matrix = torch.mm(prototypes_norm, prototypes_norm.t())
    sim_mat_np = similarity_matrix.cpu().numpy()

    # --- 3. 绘图并保存 ---
    plt.figure(figsize=(10, 8))

    # cmap='Blues': 颜色越深代表值越大（越相似）
    # vmin=0, vmax=1: 锁定颜色范围，便于不同 Epoch 之间对比
    ax = sns.heatmap(sim_mat_np, cmap='Blues', square=True,
                     xticklabels=False, yticklabels=False,  # 类别太多时隐藏坐标轴文字更整洁
                     vmin=0, vmax=1)

    plt.title("Prototype Cosine Similarity", fontsize=16)
    plt.xlabel("Class Index", fontsize=12)
    plt.ylabel("Class Index", fontsize=12)

    # --- 4. 画出增量任务分界线 (红色虚线) ---
    if classes_list is not None and len(classes_list) > 1:
        current_idx = 0
        # 只需要画 n-1 条线
        for num_cls in classes_list[:-1]:
            current_idx += num_cls
            # 画线：颜色红，虚线，线宽1.5
            plt.axhline(current_idx, color='red', linestyle='--', linewidth=1.5)
            plt.axvline(current_idx, color='red', linestyle='--', linewidth=1.5)

    # --- 5. 保存图片 ---
    # bbox_inches='tight' 可以去除周围多余的白边
    print(f"正在保存相似度矩阵到: {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # --- 6. 清理内存 ---
    # 非常重要：在循环中如果不关闭 figure，内存会爆掉
    plt.close()