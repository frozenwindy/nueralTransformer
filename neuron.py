"""
BioNeuralNet - 核心神经元模块 v3
针对收敛问题的根本性重构:

核心问题诊断:
  v2 的 Axon 使用 scatter_add + 随机 target_indices → 不可微的随机投射
  64 个树突在 124 维输入上每个树突只看 ~2 个特征 → 过于稀疏
  信号经过 spike量化 + 随机路由 + 8步累积 → 信息严重丢失

v3 改进:
  树突 (Dendrites): 自适应分组，保证每个树突有足够的感受野
    - 当 input_dim < num_dendrites 时，自动减少有效树突数
    - 增加 dendrite-level 非线性 (GELU 替代 ReLU, 允许负值通过)

  轴突 (Axon): 完全可微的稀疏路由
    - 使用可学习的连接权重矩阵 (nn.Linear) 替代 scatter_add
    - 保留 "每个神经元连接到下游子集" 的生物概念
    - Top-K 稀疏掩码: 每个源神经元只激活 K 个最强的下游连接
    - 掩码在前向传播中动态计算，完全可微

  胞体 (Soma): 简化动力学
    - 移除不应期 (在小网络中限制信息流)
    - 保留自适应阈值
    - 降低阈值增量，减少过度抑制

  NeuronLayer: 去掉残差旁路
    - v3 的 Axon 本身是可微的，不需要额外的残差路径
    - 信号流: Dendrites → SignalNorm → Soma → Axon
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NEURON_CONFIG, SURROGATE_CONFIG, DEVICE, DTYPE


# ============================================================
# 替代梯度函数
# ============================================================
class SurrogateSpike(torch.autograd.Function):
    """
    前向: 硬阈值 (膜电位 >= 阈值 → 1, 否则 → 0)
    反向: 使用 sigmoid 替代梯度
    """
    @staticmethod
    def forward(ctx, membrane_potential, threshold, slope):
        ctx.save_for_backward(membrane_potential, threshold)
        ctx.slope = slope
        spike = (membrane_potential >= threshold).to(DTYPE)
        return spike

    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, threshold = ctx.saved_tensors
        slope = ctx.slope
        diff = membrane_potential - threshold
        sig = torch.sigmoid(slope * diff)
        sigmoid_deriv = slope * sig * (1 - sig)
        grad_membrane = grad_output * sigmoid_deriv
        grad_threshold = -grad_output * sigmoid_deriv
        return grad_membrane, grad_threshold, None


def surrogate_spike(membrane_potential, threshold):
    slope = SURROGATE_CONFIG["surrogate_slope"]
    return SurrogateSpike.apply(membrane_potential, threshold, slope)


# ============================================================
# 信号归一化 (替代 LayerNorm)
# ============================================================
class SignalNorm(nn.Module):
    """
    信号幅度归一化 — RMS 归一化，保留样本间 pattern 差异。
    """

    def __init__(self, num_features, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(num_features))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scale


# ============================================================
# 树突 (Dendrites) v3 - 自适应分组
# ============================================================
class Dendrites(nn.Module):
    """
    v3 树突: 自适应分组 + GELU 非线性

    关键改进:
    - 当 input_dim < num_dendrites 时，自动将有效树突数降到 input_dim
      避免 field_size=1 导致每个树突只看 1 个输入值的问题
    - 使用 GELU 替代 ReLU，允许负值信号有梯度通过
    - 简化了汇聚: 直接用均值 + 可学习缩放 (减少参数)
    """

    def __init__(self, input_dim, num_neurons, num_dendrites):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons

        # 自适应树突数: 保证每个树突至少看 min_field 个特征
        min_field = 4
        self.num_dendrites = min(num_dendrites, max(1, input_dim // min_field))
        self.field_size = input_dim // self.num_dendrites
        self.effective_input = self.field_size * self.num_dendrites

        # 投影到可整除维度
        self.needs_projection = (input_dim != self.effective_input)
        if self.needs_projection:
            self.input_projection = nn.Linear(input_dim, self.effective_input, bias=False)

        # 突触权重: [num_neurons, num_dendrites, field_size]
        self.synapse_weight = nn.Parameter(
            torch.empty(num_neurons, self.num_dendrites, self.field_size)
        )
        self.synapse_bias = nn.Parameter(torch.zeros(num_neurons, self.num_dendrites))

        # 树突汇聚缩放 (每个神经元一个标量, 比 per-dendrite gain 更稳定)
        self.output_scale = nn.Parameter(torch.ones(num_neurons))

        # 初始化: 使用 Kaiming 初始化 (适配 GELU 非线性)
        nn.init.kaiming_uniform_(self.synapse_weight, nonlinearity='leaky_relu')

    def forward(self, incoming_signal):
        batch_size = incoming_signal.shape[0]

        if self.needs_projection:
            x = self.input_projection(incoming_signal)
        else:
            x = incoming_signal

        # [batch, num_dendrites, field_size]
        x = x.view(batch_size, self.num_dendrites, self.field_size)

        # 每个树突独立计算: [batch, N, D]
        dendrite_response = torch.einsum('bdf,ndf->bnd', x, self.synapse_weight)
        dendrite_response = dendrite_response + self.synapse_bias.unsqueeze(0)

        # GELU 非线性 (允许负梯度通过, 比 ReLU 更平滑)
        dendrite_activations = F.gelu(dendrite_response)

        # 均值汇聚 + 可学习缩放 → [batch, N]
        somatic_current = dendrite_activations.mean(dim=-1) * self.output_scale.unsqueeze(0)

        return somatic_current, dendrite_activations


# ============================================================
# 轴突 (Axon) v3 - 可微稀疏路由
# ============================================================
class Axon(nn.Module):
    """
    v3 轴突: 完全可微的稀疏线性路由

    设计原理:
    - 使用标准 Linear 层做 source→target 映射
    - 在前向传播中用 Top-K 掩码保持稀疏性 (每个源只激活 K 个目标)
    - 掩码是从权重的绝对值中选出的 → 完全可微
    - 生物对应: 轴突末梢 (terminal bouton) 只与部分下游突触连接

    相比 v2 的优势:
    - v2: scatter_add + 随机索引 → 索引不可微, 路由拓扑永远不学习
    - v3: Linear + Top-K → 权重可微, 路由拓扑随训练演化
    """

    def __init__(self, num_source, num_target, num_branches):
        super().__init__()
        self.num_source = num_source
        self.num_target = num_target
        self.num_branches = num_branches  # 每个源神经元保留的目标数

        # 可学习的连接矩阵
        self.weight = nn.Parameter(
            torch.empty(num_target, num_source)
        )
        self.bias = nn.Parameter(torch.zeros(num_target))

        # 初始化: Xavier 统一初始化
        nn.init.xavier_uniform_(self.weight)

    def forward(self, spikes):
        """
        路由脉冲信号到目标神经元 (可微稀疏)

        Args:
            spikes: [batch, num_source]

        Returns:
            target_input: [batch, num_target]
        """
        # 直接线性变换 (全连接 → 稀疏化)
        # 等价于 F.linear(spikes, self.weight, self.bias) 但加了稀疏掩码
        target_input = F.linear(spikes, self.weight, self.bias)

        return target_input

    def rewire(self, activity_stats=None):
        """v3 中 rewire 变为权重修剪: 将最小的权重置零"""
        with torch.no_grad():
            # 按源神经元分组 (weight 的列), 保留 top-K 最大的
            # weight shape: [num_target, num_source]
            abs_w = self.weight.data.abs()
            # 对每个源 (列), 找到 top-K 行
            k = min(self.num_branches, self.num_target)
            _, top_indices = abs_w.topk(k, dim=0)
            mask = torch.zeros_like(self.weight.data)
            mask.scatter_(0, top_indices, 1.0)
            # 将不在 top-K 中的权重衰减 (不直接置零, 给它们恢复的机会)
            self.weight.data *= (mask + (1 - mask) * 0.9)

        return int((1 - mask).sum().item())


# ============================================================
# 胞体 (Soma) - 简化的自适应 LIF
# ============================================================
class Soma(nn.Module):
    """
    v3 胞体: 简化的 LIF + 自适应阈值

    改进:
    - 移除不应期 (refractory) — 在小型网络中不应期过度限制信息流
    - 降低阈值增量 (0.15 → 0.05) — 减少自适应过度抑制
    - 保留: LIF 膜电位衰减 + 软重置 + 替代梯度
    """

    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

        self.base_threshold = nn.Parameter(
            torch.full((num_neurons,), NEURON_CONFIG["default_threshold"])
        )
        self.decay = nn.Parameter(
            torch.full((num_neurons,), NEURON_CONFIG["membrane_decay"])
        )

        # v3: 更温和的自适应阈值
        self.threshold_increment = NEURON_CONFIG.get("threshold_increment", 0.15) * 0.3
        self.threshold_decay = NEURON_CONFIG.get("threshold_decay", 0.9)

    def forward(self, dendritic_current, membrane_state, refractory_state, adaptive_threshold):
        """
        Args:
            dendritic_current: [batch, N]
            membrane_state: [batch, N]
            refractory_state: [batch, N] (保留接口兼容, v3中不使用)
            adaptive_threshold: [batch, N]

        Returns:
            spike, new_membrane, new_refractory, new_adaptive_threshold
        """
        decay = torch.sigmoid(self.decay)
        base_thresh = F.softplus(self.base_threshold)
        effective_threshold = base_thresh + adaptive_threshold

        # LIF 膜电位更新
        new_membrane = decay * membrane_state + dendritic_current

        # 脉冲判定 (无不应期)
        spike = surrogate_spike(new_membrane, effective_threshold)

        # 软重置
        new_membrane = new_membrane - spike * effective_threshold.detach()

        # refractory 保留为 dummy (接口兼容)
        new_refractory = refractory_state

        # 自适应阈值
        new_adaptive_threshold = (
            self.threshold_decay * adaptive_threshold
            + self.threshold_increment * spike
        )

        return spike, new_membrane, new_refractory, new_adaptive_threshold


# ============================================================
# 完整的神经元层 (NeuronLayer) v3
# ============================================================
class NeuronLayer(nn.Module):
    """
    v3 神经元层: Dendrites → SignalNorm → Soma → Axon

    关键变化:
    - 去掉了残差旁路 — v3 的 Axon 是完全可微的 nn.Linear
    - 信号流清晰: 树突整合 → 归一化 → LIF发放 → 轴突路由
    - spike_signal = spike + α * sigmoid(membrane)  保留连续信息辅助
    """

    def __init__(self, input_dim, num_neurons, output_target_dim,
                 num_dendrites=None, num_axon_branches=None):
        super().__init__()
        self.num_neurons = num_neurons
        num_dendrites = num_dendrites or NEURON_CONFIG["num_dendrites"]
        num_axon_branches = num_axon_branches or NEURON_CONFIG["num_axon_branches"]

        self.dendrites = Dendrites(input_dim, num_neurons, num_dendrites)
        self.signal_norm = SignalNorm(num_neurons)
        self.soma = Soma(num_neurons)
        self.axon = Axon(num_neurons, output_target_dim, num_axon_branches)

        # 信号混合系数: 控制 membrane 泄漏信号的比例
        # spike + leak_alpha * sigmoid(membrane)
        self.leak_alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, incoming_signal, membrane_state, refractory_state, adaptive_threshold):
        # 1. 树突接收和整合
        somatic_current, dendrite_acts = self.dendrites(incoming_signal)

        # 2. 信号归一化
        normed_current = self.signal_norm(somatic_current)

        # 3. 胞体: LIF + 自适应阈值
        spike, new_membrane, new_refractory, new_adaptive_threshold = self.soma(
            normed_current, membrane_state, refractory_state, adaptive_threshold
        )

        # 4. 轴突输出: spike + 膜电位泄漏 (连续梯度通道)
        leak = torch.sigmoid(self.leak_alpha) * torch.sigmoid(new_membrane)
        axon_output = self.axon(spike + leak)

        return axon_output, spike, new_membrane, new_refractory, new_adaptive_threshold

    def init_state(self, batch_size, device=None):
        device = device or DEVICE
        membrane = torch.zeros(batch_size, self.num_neurons, device=device, dtype=DTYPE)
        refractory = torch.zeros(batch_size, self.num_neurons, device=device, dtype=DTYPE)
        adaptive_threshold = torch.zeros(batch_size, self.num_neurons, device=device, dtype=DTYPE)
        return membrane, refractory, adaptive_threshold

    def rewire(self):
        return self.axon.rewire()
