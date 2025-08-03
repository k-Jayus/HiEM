import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SparseConv3dFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias, mask):
        """
        超稀疏优化：只对非零位置计算
        """
        # 🚀 找到真正的有效位置
        effective_positions = ((input.abs() > 1e-8) * mask).nonzero(as_tuple=False)

        if len(effective_positions) == 0:
            # 没有有效位置，直接返回零
            batch_size = input.size(0)
            output = torch.zeros(batch_size, weight.size(0), 1, 1, 1,
                                 device=input.device, dtype=input.dtype)
            ctx.save_for_backward(input, weight, mask)
            ctx.effective_positions = effective_positions
            return output

        # 🚀 只提取有效位置的值
        effective_values = input[effective_positions[:, 0],
        effective_positions[:, 1],
        effective_positions[:, 2],
        effective_positions[:, 3],
        effective_positions[:, 4]]

        effective_weights = weight[:, 0,
                            effective_positions[:, 2],
                            effective_positions[:, 3],
                            effective_positions[:, 4]]

        # 🚀 稀疏矩阵乘法
        batch_size = input.size(0)
        n_channels = weight.size(0)
        output = torch.zeros(batch_size, n_channels, 1, 1, 1,
                             device=input.device, dtype=input.dtype)

        # 按batch分组计算
        for b in range(batch_size):
            batch_mask = effective_positions[:, 0] == b
            if batch_mask.any():
                batch_values = effective_values[batch_mask]
                batch_weights = effective_weights[:, batch_mask]

                # 矩阵乘法：[n_channels, n_positions] × [n_positions] = [n_channels]
                output[b, :, 0, 0, 0] = torch.mv(batch_weights, batch_values)

        if bias is not None:
            output += bias.view(1, -1, 1, 1, 1)

        ctx.save_for_backward(input, weight, bias, mask)
        ctx.effective_positions = effective_positions
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        超稀疏反向传播：只对有效位置计算
        """
        input, weight, bias, mask = ctx.saved_tensors
        effective_positions = ctx.effective_positions

        grad_input = grad_weight = grad_bias = grad_mask = None

        if len(effective_positions) == 0:
            # 没有有效位置，返回零梯度
            if ctx.needs_input_grad[0]:
                grad_input = torch.zeros_like(input)
            if ctx.needs_input_grad[1]:
                grad_weight = torch.zeros_like(weight)
            if ctx.needs_input_grad[2]:
                grad_bias = torch.zeros_like(bias) if bias is not None else None
            if ctx.needs_input_grad[3]:
                grad_mask = torch.zeros_like(mask)
            return grad_input, grad_weight, grad_bias, grad_mask

        # 🚀 只对有效位置计算梯度
        batch_size = input.size(0)
        n_channels = weight.size(0)

        # 输入梯度
        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(input)

            for b in range(batch_size):
                batch_mask = effective_positions[:, 0] == b
                if batch_mask.any():
                    batch_positions = effective_positions[batch_mask]
                    batch_weights = weight[:, 0,
                                    batch_positions[:, 2],
                                    batch_positions[:, 3],
                                    batch_positions[:, 4]]

                    # 反向传播：[n_channels] × [n_channels, n_positions] = [n_positions]
                    grad_values = torch.mv(batch_weights.t(), grad_output[b, :, 0, 0, 0])

                    # 散布回原始位置
                    grad_input[batch_positions[:, 0],
                    batch_positions[:, 1],
                    batch_positions[:, 2],
                    batch_positions[:, 3],
                    batch_positions[:, 4]] = grad_values

        # 权重梯度
        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)

            for b in range(batch_size):
                batch_mask = effective_positions[:, 0] == b
                if batch_mask.any():
                    batch_positions = effective_positions[batch_mask]
                    batch_values = input[batch_positions[:, 0],
                    batch_positions[:, 1],
                    batch_positions[:, 2],
                    batch_positions[:, 3],
                    batch_positions[:, 4]]

                    # 外积：[n_channels, 1] × [1, n_positions] = [n_channels, n_positions]
                    grad_batch = torch.outer(grad_output[b, :, 0, 0, 0], batch_values)

                    # 散布回原始位置
                    grad_weight[:, 0,
                    batch_positions[:, 2],
                    batch_positions[:, 3],
                    batch_positions[:, 4]] += grad_batch

        # bias梯度
        if ctx.needs_input_grad[2] and bias is not None:
            grad_bias = grad_output.sum(dim=(0, 2, 3, 4))

        # mask梯度
        if ctx.needs_input_grad[3]:
            grad_mask = torch.zeros_like(mask)

            for b in range(batch_size):
                batch_mask = effective_positions[:, 0] == b
                if batch_mask.any():
                    batch_positions = effective_positions[batch_mask]

                    # 计算mask梯度
                    input_values = input[batch_positions[:, 0],
                    batch_positions[:, 1],
                    batch_positions[:, 2],
                    batch_positions[:, 3],
                    batch_positions[:, 4]]

                    weight_values = weight[:, 0,
                                    batch_positions[:, 2],
                                    batch_positions[:, 3],
                                    batch_positions[:, 4]]

                    # 梯度计算
                    grad_values = torch.mv(weight_values.t(), grad_output[b, :, 0, 0, 0]) * input_values

                    # 散布回原始位置
                    grad_mask[0, 0,
                    batch_positions[:, 2],
                    batch_positions[:, 3],
                    batch_positions[:, 4]] += grad_values

        return grad_input, grad_weight, grad_bias, grad_mask

class OptimizedClassifyModule(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        D, H, W = input_shape

        # 卷积权重和偏置
        self.weight = nn.Parameter(torch.randn(2, 1, D, H, W) * 0.01)
        self.bias = nn.Parameter(torch.zeros(2))

        # MLP分类器
        self.mlp = nn.Sequential(
            nn.Flatten(),
            #nn.BatchNorm1d(2),

            #nn.Linear(2, 2)
        )

    def forward(self, x, mask):
        """
        Args:
            x: (batch_size, 1, D, H, W)
            mask: (1, 1, D, H, W)
        """
        # 使用带梯度的稀疏卷积
        conv_output = SparseConv3dFunction.apply(x, self.weight, self.bias, mask)
        return self.mlp(conv_output)


class ExplainablesMRIModel(nn.Module):
    def __init__(self, input_shape, initial_masks=None):
        super(ExplainablesMRIModel, self).__init__()

        self.input_shape = input_shape  # (D, H, W)

        # 定义5个可训练的mask - 确保梯度计算
        if initial_masks is not None:
            self.masks = nn.ParameterList([
                nn.Parameter(mask.clone().float(), requires_grad=True) for mask in initial_masks
            ])
            print(f"✅ 使用提供的初始mask，共{len(initial_masks)}个")
        else:
            self.masks = nn.ParameterList([
                nn.Parameter(torch.ones(input_shape), requires_grad=True) for _ in range(5)
            ])
            print("✅ 使用默认全1初始mask")

        # 打印mask信息
        for i, mask in enumerate(self.masks):
            print(f"   Mask{i + 1}: shape={mask.shape}, requires_grad={mask.requires_grad}")

        # 定义分类模块
        self.classify = self._make_classify_module(input_shape)

        print(f"✅ 模型初始化完成，输入形状: {input_shape}")

    def _make_classify_module(self, input_shape):
        return OptimizedClassifyModule(input_shape)

    def init_classify_module(self, init_method='xavier_normal'):
        """
        初始化classify模块的权重

        Args:
            init_method (str): 初始化方法，可选：
                - 'xavier_normal': Xavier正态分布初始化
                - 'xavier_uniform': Xavier均匀分布初始化
                - 'kaiming_normal': Kaiming正态分布初始化
                - 'kaiming_uniform': Kaiming均匀分布初始化
                - 'default': 默认初始化(原有方法)
        """
        with torch.no_grad():
            if init_method == 'xavier_normal':
                # Xavier正态分布初始化 - 适合tanh/sigmoid激活函数
                nn.init.xavier_normal_(self.classify.weight)
                nn.init.zeros_(self.classify.bias)

            elif init_method == 'xavier_uniform':
                # Xavier均匀分布初始化
                nn.init.xavier_uniform_(self.classify.weight)
                nn.init.zeros_(self.classify.bias)

            elif init_method == 'kaiming_normal':
                # Kaiming正态分布初始化 - 适合ReLU激活函数
                nn.init.kaiming_normal_(self.classify.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(self.classify.bias)

            elif init_method == 'kaiming_uniform':
                # Kaiming均匀分布初始化
                nn.init.kaiming_uniform_(self.classify.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(self.classify.bias)

            elif init_method == 'default':
                # 保持原有的初始化方法
                D, H, W = self.input_shape
                self.classify.weight.data = torch.randn(2, 1, D, H, W) * 0.01
                self.classify.bias.data = torch.zeros(2)

            else:
                raise ValueError(f"不支持的初始化方法: {init_method}")

        # 初始化MLP层
        for module in self.classify.mlp.modules():
            if isinstance(module, nn.Linear):
                if init_method.startswith('xavier'):
                    nn.init.xavier_normal_(module.weight)
                elif init_method.startswith('kaiming'):
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                else:
                    nn.init.normal_(module.weight, 0, 0.01)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        print(f"✅ Classify模块权重初始化完成，使用方法: {init_method}")
    def forward(self, x):
        """
        前向传播：依次使用5个mask进行分类

        Args:
            x: (batch_size, 1, D, H, W)

        Returns:
            List[Tensor]: 5个分类结果，每个shape为(batch_size, 2)
        """
        results = []
        current = x
        for i in range(5):
            # 获取mask并扩展维度以匹配batch
            mask = self.masks[i].unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

            # 确保mask在正确设备上
            if mask.device != x.device:
                mask = mask.to(x.device)

            # 使用当前mask进行分类
            result = self.classify(current, mask)
            results.append(result)
            current = current*mask

        return results

    def get_masks(self):
        """返回当前的mask参数，用于可视化"""
        return [mask.data.clone() for mask in self.masks]

    def set_mask(self, mask_idx, new_mask):
        """设置特定的mask"""
        if 0 <= mask_idx < 5:
            with torch.no_grad():
                self.masks[mask_idx].copy_(new_mask)
        else:
            raise ValueError(f"mask_idx应该在0-4之间，得到{mask_idx}")

    def freeze_masks(self):
        """冻结所有mask参数"""
        for mask in self.masks:
            mask.requires_grad_(False)
        print("🔒 所有mask已冻结")

    def unfreeze_masks(self):
        """解冻所有mask参数"""
        for mask in self.masks:
            mask.requires_grad_(True)
        print("🔓 所有mask已解冻")

    def get_mask_gradients(self):
        """获取mask梯度信息"""
        grad_info = {}
        for i, mask in enumerate(self.masks):
            if mask.grad is not None:
                grad_info[f'mask_{i + 1}'] = {
                    'norm': mask.grad.norm().item(),
                    'mean': mask.grad.mean().item(),
                    'std': mask.grad.std().item(),
                    'max': mask.grad.max().item(),
                    'min': mask.grad.min().item()
                }
            else:
                grad_info[f'mask_{i + 1}'] = None
        return grad_info

    def print_model_info(self):
        """打印模型信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"📊 模型信息:")
        print(f"   总参数: {total_params:,}")
        print(f"   可训练参数: {trainable_params:,}")
        print(f"   输入形状: {self.input_shape}")
        print(f"   Mask数量: {len(self.masks)}")

        # 分别统计mask和分类器参数
        mask_params = sum(mask.numel() for mask in self.masks)
        classify_params = sum(p.numel() for p in self.classify.parameters())

        print(f"   Mask参数: {mask_params:,}")
        print(f"   分类器参数: {classify_params:,}")


# 使用示例和测试
if __name__ == "__main__":
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 假设sMRI的形状
    input_shape = (91, 109, 91)  # 典型的MNI空间大小
    batch_size = 2

    # 创建模型（不使用初始mask）
    model = ExplainablesMRIModel(input_shape).to(device)
    model.print_model_info()

    # 创建示例输入
    x = torch.randn(batch_size, 1, *input_shape).to(device)
    print(f"\n📥 输入数据形状: {x.shape}")

    # 前向传播
    print("\n🔄 开始前向传播...")
    results = model(x)

    print(f"✅ 模型输出了 {len(results)} 个结果")
    for i, result in enumerate(results):
        print(f"   结果 {i + 1} 形状: {result.shape}")

    # 测试梯度计算
    print("\n🧮 测试梯度计算...")
    criterion = nn.CrossEntropyLoss()
    targets = torch.randint(0, 2, (batch_size,)).to(device)

    total_loss = 0
    for i, result in enumerate(results):
        loss = criterion(result, targets)
        total_loss += loss

    print(f"总损失: {total_loss.item():.4f}")

    # 反向传播
    total_loss.backward()

    # 检查mask梯度
    print("\n📈 Mask梯度信息:")
    grad_info = model.get_mask_gradients()
    for mask_name, info in grad_info.items():
        if info is not None:
            print(f"   {mask_name}: 范数={info['norm']:.6f}, 均值={info['mean']:.6f}")
        else:
            print(f"   {mask_name}: 无梯度")

    # 测试mask操作
    print("\n🔧 测试mask操作...")
    masks = model.get_masks()
    print(f"获取到 {len(masks)} 个mask，每个形状: {masks[0].shape}")

    # 测试冻结/解冻
    model.freeze_masks()
    model.unfreeze_masks()

    print("\n✅ 所有测试完成！")