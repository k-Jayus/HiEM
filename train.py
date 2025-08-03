import os
import time
import argparse
import torch
from torch.utils.data import DataLoader
# from model.cnn import Net as Model
# from model.resnet3d import resnet50 as Model
from model.explainmodel import ExplainablesMRIModel as Model
# from model.efficientnet3D import efficientnetv2_m as Model
# from model.vgg3d import vgg as Model
from dataset import MyDataset
from utils import get_params_groups, train_one_epoch_freeze_masks, validate_freeze_masks
from utils import cosine_lr_scheduler as create_lr_scheduler
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import numpy as np
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.ndimage import binary_dilation, generate_binary_structure

import scipy.stats as stats
import torch.nn.functional as F
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def check_gradient_stability(gradient_history, model, window_size=3, threshold=0.95):
    """
    检查Mask梯度在最近几个epoch是否稳定。

    Args:
        gradient_history (list): 存储每个epoch的`epoch_gradient_stats`的列表。
        model: 包含masks的模型对象
        window_size (int): 用于比较的连续epoch数量。
        threshold (float): 梯度符号一致性比例的阈值，高于此值被认为稳定。

    Returns:
        bool: 如果在窗口期内所有相邻epoch的平均一致性都高于阈值，则返回True。
    """
    # 至少需要窗口大小的梯度历史记录才能进行比较
    if len(gradient_history) < window_size:
        return False

    # 获取当前的masks
    current_masks = [mask.data.clone().to('cpu') for mask in model.masks]

    # 获取最近的梯度记录
    recent_grads = gradient_history[-window_size:]

    all_pair_consistencies = []
    all_union_ratios = []

    # 比较窗口内的每一对相邻的epoch梯度
    for i in range(window_size - 1):
        epoch_grads_1 = recent_grads[i]
        epoch_grads_2 = recent_grads[i + 1]

        # 计算这一对epoch中所有mask的平均一致性
        current_pair_consistencies = []
        current_union_ratios = []

        # 遍历所有masks
        for j in range(len(current_masks)):
            mask_name = f'mask_{j}'

            # 确保两个epoch都有该mask的梯度
            if mask_name not in epoch_grads_1 or mask_name not in epoch_grads_2:
                continue

            grad1 = epoch_grads_1[mask_name]['mean_gradient']
            grad2 = epoch_grads_2[mask_name]['mean_gradient']

            # 获取对应的mask
            current_mask = current_masks[j]

            # 获取每个点的梯度符号 (+1, -1, or 0)
            signs1 = torch.sign(grad1)
            signs2 = torch.sign(grad2)

            # 计算总点数
            total_points = grad1.numel()

            # 计算一致性比例
            if total_points == 0:
                consistency_ratio = 1.0
                union_ratio = 1.0
            else:
                # 分别计算两个梯度的前90%重要点
                abs_grad1 = torch.abs(grad1).flatten()
                abs_grad2 = torch.abs(grad2).flatten()

                # 对grad1找前90%点
                sorted_abs1, indices1 = torch.sort(abs_grad1, descending=True)
                cumsum1 = torch.cumsum(sorted_abs1, dim=0)
                total_sum1 = cumsum1[-1]
                threshold_90_1 = total_sum1 * 1
                mask_90_1 = cumsum1 <= threshold_90_1
                if not mask_90_1.any():
                    num_selected1 = 1
                else:
                    num_selected1 = mask_90_1.sum().item() + 1
                    num_selected1 = min(num_selected1, len(indices1))
                important_indices1 = set(indices1[:num_selected1].tolist())

                # 对grad2找前90%点
                sorted_abs2, indices2 = torch.sort(abs_grad2, descending=True)
                cumsum2 = torch.cumsum(sorted_abs2, dim=0)
                total_sum2 = cumsum2[-1]
                threshold_90_2 = total_sum2 * 1
                mask_90_2 = cumsum2 <= threshold_90_2
                if not mask_90_2.any():
                    num_selected2 = 1
                else:
                    num_selected2 = mask_90_2.sum().item() + 1
                    num_selected2 = min(num_selected2, len(indices2))
                important_indices2 = set(indices2[:num_selected2].tolist())

                # 取两个集合的并集
                union_indices = important_indices1.union(important_indices2)
                union_indices = torch.tensor(list(union_indices))

                # 计算并集占总点数的比例
                union_ratio = len(union_indices) / total_points

                # 只在这些重要点上计算符号一致性
                signs1_flat = signs1.flatten()
                signs2_flat = signs2.flatten()

                signs1_important = signs1_flat[union_indices]
                signs2_important = signs2_flat[union_indices]

                # --- 修改：只在mask为1的位置计算一致性比例 ---
                # 获取mask并展平
                mask_flat = current_mask.flatten()
                mask_important = mask_flat[union_indices]

                # 找出mask为1的位置
                mask_ones_positions = mask_important == 1
                num_mask_ones = mask_ones_positions.sum().item()

                if num_mask_ones == 0:
                    # 如果重要点中没有mask为1的位置，认为是一致的
                    consistency_ratio = 1.0
                else:
                    # 只在mask为1的位置计算符号一致性
                    signs1_mask_ones = signs1_important[mask_ones_positions]
                    signs2_mask_ones = signs2_important[mask_ones_positions]

                    num_consistent_points = (signs1_mask_ones == signs2_mask_ones).sum().item()
                    consistency_ratio = num_consistent_points / num_mask_ones

            current_pair_consistencies.append(consistency_ratio)
            current_union_ratios.append(union_ratio)

        if not current_pair_consistencies:
            continue

        # 计算这对epoch的平均一致性
        avg_consistency = sum(current_pair_consistencies) / len(current_pair_consistencies)
        avg_union_ratio = sum(current_union_ratios) / len(current_union_ratios)
        all_pair_consistencies.append(avg_consistency)
        all_union_ratios.append(avg_union_ratio)

    # 如果所有相邻对的平均一致性都高于阈值，则认为梯度已稳定
    if not all_pair_consistencies:  # 如果没有任何可比较的梯度
        return False

    # 检查一致性和并集占比两个条件
    consistency_stable = all(consist >= threshold for consist in all_pair_consistencies)
    union_stable = all(union_ratio >= 0.95 for union_ratio in all_union_ratios)
    is_stable = consistency_stable and union_stable

    # 打印信息
    if is_stable:
        print(f"📈 梯度符号已稳定! 最近 {window_size - 1} 次epoch间平均一致性: " +
              ", ".join([f"{c:.4f}" for c in all_pair_consistencies]) +
              f" (阈值 > {threshold}) | 并集占比: " +
              ", ".join([f"{r:.4f}" for r in all_union_ratios]) +
              " (阈值 >0.95)")
    else:
        print(f"📉 梯度符号未稳定。最近 {window_size - 1} 次epoch间平均一致性: " +
              ", ".join([f"{c:.4f}" for c in all_pair_consistencies]) +
              f" (阈值 > {threshold}) | 并集占比: " +
              ", ".join([f"{r:.4f}" for r in all_union_ratios]) +
              " (阈值 > 0.95)")

    return is_stable
def load_two_class_data(root, subject, data_csv, max_samples_per_class=30):
    """
    快速加载两类数据（减少样本数，随机采样）
    """
    print(f"🗂️ 快速加载两类数据...")

    # 创建数据集
    train_dataset = MyDataset(root, subject, data_csv, train=True)

    # 预先收集各类样本的索引
    class1_indices = []
    class2_indices = []

    print("📋 扫描数据集标签...")
    for i in range(len(train_dataset)):
        # 只读取标签，不读取数据
        _, label = train_dataset.sample[i], train_dataset.label[i]
        if label == 0:
            class1_indices.append(i)
        elif label == 1:
            class2_indices.append(i)

    # 随机采样
    class1_indices = random.sample(class1_indices, min(max_samples_per_class, len(class1_indices)))
    class2_indices = random.sample(class2_indices, min(max_samples_per_class, len(class2_indices)))

    print(f"📥 加载 AD: {len(class1_indices)} 样本, MCI: {len(class2_indices)} 样本")

    # 批量加载数据
    class1_data = []
    class2_data = []

    # 加载class1数据
    for i in class1_indices:
        try:
            arr, _ ,_= train_dataset[i]
            class1_data.append(arr)
        except Exception as e:
            print(f"跳过样本{i}: {e}")

    # 加载class2数据
    for i in class2_indices:
        try:
            arr, _,_ = train_dataset[i]
            class2_data.append(arr)
        except Exception as e:
            print(f"跳过样本{i}: {e}")

    if len(class1_data) > 0 and len(class2_data) > 0:
        class1_data = torch.stack(class1_data)
        class2_data = torch.stack(class2_data)
        print(f"✅ 加载完成: AD {class1_data.shape}, MCI {class2_data.shape}")
        return class1_data, class2_data
    else:
        raise ValueError("没有收集到足够的数据！")


def generate_initial_masks_ttest(root, subject, data_csv):
    """
    快速生成初始mask（向量化t-test）
    """
    print("🚀 开始快速生成初始mask...")

    # 1. 加载数据（减少样本数）
    class1_data, class2_data = load_two_class_data(root, subject, data_csv, max_samples_per_class=100)

    # 2. 移除channel维度
    if class1_data.dim() == 5 and class1_data.shape[1] == 1:
        class1_data = class1_data.squeeze(1).numpy()  # (n, D, H, W)
        class2_data = class2_data.squeeze(1).numpy()  # (n, D, H, W)

    print("🔍 计算voxel-wise t-test (向量化)...")

    # 3. 向量化计算t值
    shape = class1_data.shape[1:]  # (D, H, W)

    # reshape为 (n_samples, n_voxels)
    class1_flat = class1_data.reshape(class1_data.shape[0], -1)  # (n1, D*H*W)
    class2_flat = class2_data.reshape(class2_data.shape[0], -1)  # (n2, D*H*W)

    # 向量化t-test
    t_stats, p_values = stats.ttest_ind(class1_flat, class2_flat, axis=0)
    t_values = np.abs(t_stats)

    # 处理NaN
    t_values = np.nan_to_num(t_values, nan=0.0)

    print(f"📊 t值范围: [{np.min(t_values):.4f}, {np.max(t_values):.4f}]")

    # 4. 生成mask
    sorted_indices = np.argsort(t_values)[::-1]  # 降序
    total_voxels = len(t_values)

    # 5. 生成5个mask: 50%, 40%, 30%, 20%, 10%
    percentages = [0.5, 0.4, 0.3, 0.2, 0.1]
    masks = []

    for i, p in enumerate(percentages):
        n_voxels = int(total_voxels * p)
        mask = np.zeros_like(t_values)
        mask[sorted_indices[:n_voxels]] = 1.0
        mask = mask.reshape(shape)  # 恢复原始形状
        masks.append(torch.tensor(mask, dtype=torch.float32))
        print(f"   Mask{i + 1} (top {p * 100:.0f}%): {n_voxels} voxels")

    print("✅ 快速mask生成完成!")
    return masks
def update_masks_from_gradients_no_expand(model, gradient_stats):
    """
    根据梯度更新mask，但不进行扩展操作
    """
    print("🔄 根据梯度更新masks...")

    old_masks = [mask.data.clone() for mask in model.masks]
    new_masks = []
    mask_changes = []

    for i in range(5):
        mask_name = f'mask_{i}'
        if mask_name in gradient_stats:
            # 获取平均梯度
            mean_gradient = gradient_stats[mask_name]['mean_gradient']

            # 根据梯度符号更新mask：正值→1，负值→0
            new_mask = torch.zeros_like(old_masks[i])
            new_mask[mean_gradient > 0] = 1.0
            new_mask[mean_gradient <= 0] = 0.0

            # 计算变化率 - 修改为除以旧mask中为1的位置数量
            active_positions = torch.sum(old_masks[i]).item()  # 旧mask中为1的位置数
            if active_positions > 0:
                change_ratio = torch.sum(torch.abs(new_mask - old_masks[i])) / active_positions
            else:
                change_ratio = 0.0  # 如果原来没有激活位置，变化率为0

            new_masks.append(new_mask)
            mask_changes.append({
                'mask_idx': i + 1,
                'change_ratio': change_ratio.item() if isinstance(change_ratio, torch.Tensor) else change_ratio,
                'active_voxels_before': torch.sum(old_masks[i]).item(),
                'active_voxels_after': torch.sum(new_mask).item(),
            })

            print(f"   Mask{i + 1}: 变化率={change_ratio:.4f}, "
                  f"活跃体素: {torch.sum(old_masks[i]).item()} → {torch.sum(new_mask).item()}")
        else:
            # 如果没有梯度信息，保持原mask
            new_masks.append(old_masks[i])
            mask_changes.append({
                'mask_idx': i + 1,
                'change_ratio': 0.0,
                'active_voxels_before': torch.sum(old_masks[i]).item(),
                'active_voxels_after': torch.sum(old_masks[i]).item(),
            })

    new_masks = enforce_subset_constraint(new_masks, mode='shrink')
    # 更新模型中的masks
    for i, new_mask in enumerate(new_masks):
        model.masks[i].data.copy_(new_mask)

    return mask_changes


def adaptive_mask_expansion(model, gradient_stats, decay_rate=1.0, max_distance=3.0, gradient_weight=2.0):
    """
    自适应的Mask扩展：根据梯度信息指导扩展方向。

    Args:
        model (nn.Module): 你的模型实例。
        gradient_stats (dict): 包含每个mask平均梯度的字典。
        decay_rate (float): 距离衰减率，越大则距离惩罚越重。
        max_distance (float): 最大考虑的扩展距离。
        gradient_weight (float): 梯度信息的权重，越大则梯度对扩展方向的影响越大。
    """
    print("🔧 进行自适应Mask扩展 (Adaptive Mask Expansion)...")

    # 获取当前的mask和平均梯度
    current_masks = [mask.data.clone() for mask in model.masks]

    expanded_masks = []

    for i in range(len(current_masks)):

        mask_name = f'mask_{i}'
        if mask_name not in gradient_stats or gradient_stats[mask_name] is None:
            print(f"   Mask{i + 1}: 缺少梯度信息，跳过扩展。")
            expanded_masks.append(current_masks[i])
            continue

        current_mask = current_masks[i]
        mean_gradient = gradient_stats[mask_name]['mean_gradient']
        if i == 4:
            if torch.sum(current_mask)<300:
                decay_rate = 0.8
                max_distance = 3.0
                gradient_weight = 2.0
        # 1. 识别前沿区域 (Find the frontier)
        # 将mask转换为numpy进行膨胀操作
        mask_np = current_mask.cpu().numpy().astype(bool)
        # 3D的结构元素 (6-连通)
        struct = generate_binary_structure(3, 1)
        # 膨胀一层
        dilated_mask_np = binary_dilation(mask_np, structure=struct)
        # 转换回tensor
        dilated_mask = torch.from_numpy(dilated_mask_np).float().to(current_mask.device)
        # 前沿就是膨胀后的区域减去原始区域
        frontier = F.relu(dilated_mask - current_mask)

        # 如果没有前沿（整个空间都满了），则不扩展
        if torch.sum(frontier) == 0:
            expanded_masks.append(current_mask)
            continue

        # 2. 计算扩展概率
        # a) 基于距离的概率
        distance_field = compute_distance_field(current_mask)
        # 只在最大距离内考虑
        valid_region = (distance_field <= max_distance)
        proximity_prob = torch.exp(-decay_rate * distance_field) * valid_region.float()

        # b) 基于梯度的增益
        # 只考虑正梯度（有潜力的区域），并进行归一化处理
        positive_gradient = F.relu(mean_gradient)
        # 归一化梯度，使其影响更稳定
        if positive_gradient.max() > 0:
            normalized_gradient = positive_gradient / positive_gradient.max()
        else:
            normalized_gradient = torch.zeros_like(positive_gradient)

        # 梯度增益因子：基础是1，有正梯度的区域会得到加成
        gradient_gain = 1.0 + gradient_weight * normalized_gradient

        # c) 计算最终扩展概率（只在前沿区域计算）
        final_expansion_prob = proximity_prob.to("cpu") * gradient_gain.to("cpu") * frontier.to("cpu")

        # 3. 概率采样
        random_field = torch.rand_like(final_expansion_prob)
        newly_activated_voxels = (random_field < final_expansion_prob).float()

        # 4. 更新Mask
        expanded_mask = current_mask.to("cpu") + newly_activated_voxels.to("cpu")
        expanded_masks.append(expanded_mask)

        print(
            f"   Mask{i + 1} 扩展: {torch.sum(current_mask).item():.0f} → {torch.sum(expanded_mask).item():.0f} 个活跃体素")

    # 确保扩展后仍满足子集关系
    enforced_expanded_masks = enforce_subset_constraint(expanded_masks, mode='expand')

    # 更新模型中的masks
    for i, new_mask in enumerate(enforced_expanded_masks):
        model.masks[i].data.copy_(new_mask)

    print("✅ 自适应扩展完成!")

def expand_masks(model, decay_rate=0.8, max_distance=3.0,mode=None):
    """
    对模型中的所有mask进行概率场扩展操作
    """
    print("🔧 进行概率场扩展操作...")

    expanded_masks = []
    for i, mask in enumerate(model.masks):
        expanded_mask = probability_field_expansion(mask.data, decay_rate, max_distance)
        expanded_masks.append(expanded_mask)
        print(f"   Mask{i + 1}扩展: {torch.sum(mask.data).item()} → {torch.sum(expanded_mask).item()}个活跃体素")

    # 确保扩展后仍满足子集关系
    enforced_expanded_masks = enforce_subset_constraint(expanded_masks,mode=mode)

    # 更新模型中的masks
    for i, expanded_mask in enumerate(enforced_expanded_masks):
        model.masks[i].data.copy_(expanded_mask)


def probability_field_expansion(mask, decay_rate=0.8, max_distance=3.0):
    """
    概率场扩展：基于距离的概率衰减来决定扩展

    Args:
        mask: 原始mask
        decay_rate: 概率衰减率，越大衰减越快
        max_distance: 最大扩展距离，超过此距离概率为0
    """
    # 计算距离场
    distance_field = compute_distance_field(mask)

    # 只考虑在max_distance范围内的点
    valid_region = distance_field <= max_distance

    # 生成概率场：P(x) = exp(-decay_rate * distance(x))
    probability_field = torch.exp(-decay_rate * distance_field) * valid_region.float()

    # 生成随机场进行概率抽样
    random_field = torch.rand_like(probability_field)

    # 决定扩展：只对非mask区域进行扩展判断
    non_mask_region = (1 - mask)
    expansion_decisions = (random_field < probability_field) * non_mask_region

    # 合并：原始mask + 概率扩展的区域
    expanded_mask = mask + expansion_decisions

    return expanded_mask


def compute_distance_field(mask):
    """
    计算距离场：每个点到最近mask点的距离
    """
    from scipy.ndimage import distance_transform_edt

    # 转换为numpy
    mask_np = mask.cpu().numpy()

    # 计算距离变换（到最近mask点的距离）
    distance_np = distance_transform_edt(1 - mask_np)

    # 转换回tensor
    distance_field = torch.from_numpy(distance_np).float().to(mask.device)

    return distance_field


# 保持你原有的函数
def expand_mask_3d(mask, expand_size=2):
    """
    3D mask扩展：值为1的点向外扩展expand_size个voxel
    """
    # 转换为numpy进行处理
    mask_np = mask.cpu().numpy()

    # 创建结构元素（6-连通）
    struct = generate_binary_structure(3, 1)

    # 进行多次膨胀
    expanded_np = mask_np.copy()
    for _ in range(expand_size):
        expanded_np = binary_dilation(expanded_np, structure=struct)

    # 转换回tensor
    expanded_mask = torch.from_numpy(expanded_np.astype(np.float32)).to(mask.device)

    return expanded_mask


def enforce_subset_constraint(masks, mode='shrink'):
    """
    确保mask满足子集关系

    Args:
        masks: 原始mask列表
        mode: 'shrink' 收缩模式 或 'expand' 扩张模式
    """
    if len(masks) <= 1:
        return masks

    if mode == 'shrink':
        if len(masks) <= 1:
            return masks

        print("   🔗 执行逐步收缩约束...")
        enforced_masks = []

        # 第一个mask保持不变（主导mask）
        enforced_masks.append(masks[0])
        print(f"      Mask1: 保持不变，活跃体素数 = {torch.sum(masks[0]).item()}")

        # 后续mask与前一个mask求交集
        for i in range(1, len(masks)):
            # 当前mask与前一个enforced mask求交集
            enforced_mask = masks[i] * enforced_masks[i - 1]
            enforced_masks.append(enforced_mask)

            # 统计变化
            original_count = torch.sum(masks[i]).item()
            enforced_count = torch.sum(enforced_mask).item()
            removed_count = original_count - enforced_count

            if removed_count > 0:
                print(f"      Mask{i + 1}: 求交集后，移除了{removed_count}个点 ({original_count} → {enforced_count})")
            else:
                print(f"      Mask{i + 1}: 无变化，活跃体素数 = {enforced_count}")

    elif mode == 'expand':
        print("   🔗 执行逐步扩张约束...")
        enforced_masks = [mask.clone() for mask in masks]

        # 从最细的mask(mask5, index=4)开始，向前确保子集关系
        for i in range(4, 0, -1):  # i: 4,3,2,1 对应mask5,4,3,2
            mask_fine = enforced_masks[i]  # 更细的mask
            mask_coarse = enforced_masks[i - 1]  # 更粗的mask

            # 确保细mask为1的地方，粗mask也为1
            enforced_masks[i - 1] = torch.maximum(mask_coarse, mask_fine)

        print(f"      子集约束后活跃体素数: " +
              " → ".join([f"M{i + 1}:{torch.sum(mask).item()}" for i, mask in enumerate(enforced_masks)]))

    return enforced_masks


def check_convergence(mask_changes, threshold=0.01):
    """
    检查mask是否收敛
    """
    if not mask_changes:
        return False

    avg_change = sum([change['change_ratio'] for change in mask_changes]) / len(mask_changes)
    return avg_change < threshold


def main(args):
    # 设置随机种子
    seed = 1121
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 创建保存目录
    os.makedirs(args.save, exist_ok=True)

    # 准备数据
    train_set = MyDataset(args.root, args.subject, args.data_csv, train=True)
    test_set = MyDataset(args.root, args.subject, args.data_csv, train=False)
    test_set = test_set+train_set
    train_loader = DataLoader(train_set, batch_size=args.train_batch, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=args.test_batch, shuffle=False)
    aal = sitk.ReadImage('aal113.nii')
    aal = sitk.GetArrayFromImage(aal)
    aal = torch.tensor(aal, dtype=torch.float32).to(args.device)
    aal[aal>0] = 1
    # 获取数据形状（假设数据是3D的）
    sample_data, _ ,_= train_set[0]
    if len(sample_data.shape) == 4:  # (C, D, H, W)
        input_shape = sample_data.shape[1:]  # (D, H, W)
    else:  # (D, H, W)
        input_shape = sample_data.shape

    print(f"数据形状: {input_shape}")

    # 生成初始mask
    initial_masks = generate_initial_masks_ttest(args.root, args.subject, args.data_csv)
    # 创建模型
    model = Model(input_shape=input_shape, initial_masks=initial_masks).to(args.device)
    with torch.no_grad():  # 如果不需要梯度
        for i in range(5):
            model.masks[i].data = model.masks[i].data * aal
    '''state_dict = torch.load('save/explainable_mri/cn_mci.pt', weights_only=True,map_location='cpu')
    #model.masks.load_state_dict(state_dict)
    for param, saved_mask in zip(model.masks, state_dict):
        param.data.copy_(saved_mask)'''
    '''if "cuda" in args.device and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)'''


    state_dict = torch.load('save/explainable_mri/best_cn_mci_kaiming_uniform_1.532_bn_2.pt', weights_only=True,map_location='cpu')
    model.load_state_dict(state_dict)
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    if args.amp:
        print("启用混合精度训练 (AMP)")
    else:
        print("使用常规精度训练")
    # 优化器、学习率调度器和损失函数


    print("🚀 开始逐epoch渐进式Mask学习...")
    print("=" * 80)

    converged = False
    mask_change = 0
    while 1:
        model=model.to(args.device)
        gradient_history = []
        if converged:
            break
        params_group = get_params_groups(model, weight_decay=args.wd)
        optimizer = torch.optim.SGD(params_group, lr=args.lr, weight_decay=args.wd,momentum=0.9)
        lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epoch, args.warmup)
        criterion = torch.nn.CrossEntropyLoss()

        # 日志记录
        start_time = time.time()
        logger = {
            "train": {"loss": [], "acc": [], "sen": [], "spec": [], "f1": [], "auc": []},
            "test": {"loss": [], "acc": [], "sen": [], "spec": [], "f1": [], "auc": []},
            "epoch_mask_updates": [],  # 记录每个epoch的mask更新
            "convergence_info": {}
        }
        for epoch in range(args.epoch):

            test_metrics = validate_freeze_masks(
                model, test_loader, criterion, args.device,
                scaler if args.amp else None  # 传入scaler
            )
            print(f"\n--- Epoch {epoch + 1}/{args.epoch} ---")

            # 1. 训练一个完整的epoch
            train_loss, epoch_gradient_stats, train_metrics = train_one_epoch_freeze_masks(
                model, train_loader, optimizer, criterion, args.device,
                scaler if args.amp else None,
                lr_scheduler
            )
            epoch_gradient_stats = {k: {'mean_gradient': -v['mean_gradient']} for k, v in epoch_gradient_stats.items()}
            gradient_history.append(epoch_gradient_stats)


            # 记录训练指标
            logger["train"]["loss"].append(train_loss)
            logger["train"]["acc"].append(train_metrics["accuracy"])
            logger["train"]["sen"].append(train_metrics["recall"])
            logger["train"]["spec"].append(train_metrics["precision"])
            logger["train"]["f1"].append(train_metrics["f1_score"])
            logger["train"]["auc"].append(train_metrics["auc"])

            # 2. 测试当前epoch的性能
            test_metrics = validate_freeze_masks(
                model, test_loader, criterion, args.device,
                scaler if args.amp else None  # 传入scaler
            )
            logger["test"]["loss"].append(test_metrics["loss"])
            logger["test"]["acc"].append(test_metrics["accuracy"])
            logger["test"]["sen"].append(test_metrics["recall"])
            logger["test"]["spec"].append(test_metrics["precision"])
            logger["test"]["f1"].append(test_metrics["f1_score"])
            logger["test"]["auc"].append(test_metrics["auc"])
            should_update_mask = False
            if epoch >= 1500:  # 至少5个epoch后才开始判断
                # 条件2：检查梯度是否稳定
                gradient_is_stable = check_gradient_stability(
                    gradient_history,
                    model,
                    window_size=3,  # 比较最近3个epoch
                    threshold=0.99  # 相似度阈值设为95%
                )

                if  gradient_is_stable:
                    should_update_mask = True

                    print(f"🎯 触发Mask更新")
            if should_update_mask:
                # 3. 根据梯度更新mask (不进行扩展)
                if epoch_gradient_stats:
                    mask_changes = update_masks_from_gradients_no_expand(
                        model, epoch_gradient_stats
                    )
                else:
                    print("⚠️ 没有梯度信息，跳过mask更新")
                    mask_changes = []

                # 4. 检查收敛
                if check_convergence(mask_changes, args.convergence_threshold):
                    print(f"🎯 Mask已收敛！训练停止于第{epoch + 1}个epoch")
                    converged = True
                    logger["convergence_info"] = {
                        "converged": True,
                        "final_epoch": epoch + 1,
                        "reason": "mask_convergence"
                    }
                    break
                else:

                    # 5. 不收敛 → 进行扩展操作
                    if mask_changes:  # 只有当有mask变化时才输出变化率
                        avg_change = sum([c['change_ratio'] for c in mask_changes]) / len(mask_changes)
                        print(f"Mask平均变化率: {avg_change:.4f} > {args.convergence_threshold}, 未收敛")
                    # 进行扩展操作
                    #expand_masks(model, decay_rate=2, max_distance=3.0,mode='expand')
                    '''adaptive_mask_expansion(
                        model,
                        epoch_gradient_stats,  # <--- 传入梯度统计信息
                        decay_rate=1.5,  # 可调超参数
                        max_distance=3,  # 可调超参数
                        gradient_weight=2 # <--- 关键超参数：控制梯度的影响力
                    )'''
                    with torch.no_grad():  # 如果不需要梯度
                        for i in range(5):
                            model.masks[i].data = model.masks[i].data * aal
                    model.init_classify_module('kaiming_uniform')
                    print("更新mask，开始新的一轮训练...")
                    mask_change += 1
                    current_round_best_acc = max(logger["test"]["acc"]) if logger["test"]["acc"] else 0
                    print(f"第{mask_change}次mask更新，当前轮次最佳准确率: {current_round_best_acc:.4f}")
                    if mask_change == 2:
                        torch.save(current_masks, args.save + f"cn_mci.pt")
                    break

            # 6. 记录这次epoch的信息
            logger["epoch_mask_updates"].append({
                "epoch": epoch + 1,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "converged": converged,
                "expanded": not converged  # 记录是否进行了扩展
            })

            # 7. 保存模型和日志
            #torch.save(model.state_dict(), args.save + f'epoch_{epoch + 1}_cn_mci_kaiming_uniform_1.532_bn_2.pt')
            if test_metrics["accuracy"] >= max(logger["test"]["acc"]):
                torch.save(model.state_dict(), args.save + 'best_cn_mci_kaiming_uniform_1.532_bn_2.pt')
                print(f"🎯 新的最佳准确率: {test_metrics['accuracy']:.4f}")

            # 保存当前mask状态
            current_masks = model.get_masks()
            torch.save(current_masks, args.save + f"masks_cn_mci_kaiming_uniform_1.532_bn_2.pt")
            torch.save(logger, args.save + "logger_cn_mci_kaiming_uniform_1.532_bn_2.pt")

        # 训练完成
        if not converged:
            logger["convergence_info"] = {
                "converged": False,
                "reason": "max_epochs_reached"
            }

        logger["time"] = time.time() - start_time
        torch.save(logger, args.save + "logger_cn_mci_kaiming_uniform_1.532_bn_2.pt")

    print("\n" + "=" * 80)
    print("✅ 训练完成!")
    print(f"总训练时间: {logger['time']:.2f}秒")
    print(f"最终最佳准确率: {max(logger['test']['acc']):.4f}")
    if logger["convergence_info"]["converged"]:
        print(f"收敛于第 {logger['convergence_info']['final_epoch']} 个epoch")
    else:
        print("未达到收敛，已完成所有epoch")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", default=500, type=int, help="总epoch数")
    parser.add_argument("--warmup", default=1, type=int, help="warmup epochs")
    parser.add_argument("--train_batch", default=128, type=int, help="train batch size")
    parser.add_argument("--test_batch", default=16, type=int, help="test batch size")
    parser.add_argument("--num_classes", default=2, type=int, help="class number")
    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--wd", default=5e-3, type=float, help="weight decay")
    parser.add_argument("--amp", default=True, type=bool, help="use amp")
    parser.add_argument("--device", default="cuda:5", type=str, help="device")
    parser.add_argument("--root", default="./dataset/BrainClass/", type=str, help="dataset root")
    parser.add_argument("--subject", default="./files/subject_all.pt", type=str, help="subject file")
    parser.add_argument("--data_csv", default="./files/data.csv", type=str, help="data csv")
    parser.add_argument("--save", default="./save/explainable_mri/", type=str, help="save path")

    # Mask相关参数
    parser.add_argument("--convergence_threshold", default=0.001, type=float, help="mask收敛阈值")
    parser.add_argument("--expand_size", default=2, type=int, help="mask扩展大小")

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    main(args)