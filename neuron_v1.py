"""
BioNeuralNet - 核心神经元模块
实现仿生神经元的数据结构:
- 胞体 (Soma): 激活标识 + 阈值 + 膜电位
- 树突 (Dendrites): 64个输入连接，每个有独立突触权重
- 轴突 (Axon): 8个输出分支，动态连接，有传导延迟和信号衰减
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NEURON_CONFIG, SURROGATE_CONFIG, DEVICE, DTYPE


# ============================================================
# 替代梯度函数 - 使脉冲函数可微分
# ============================================================
class SurrogateSpike(torch.autograd.Function):
    """
    前向: 硬阈值 (膜电位 >= 阈值 → 1, 否则 → 0)
    反向: 使用替代梯度 (sigmoid近似) 使其可微分
    """
    @staticmethod
    def forward(ctx, membrane_potential, threshold, slope):
        ctx.save_for_backward(membrane_potential, threshold)
        ctx.slope = slope
        # 硬阈值判定
        spike = (membrane_potential >= threshold).to(DTYPE)
        return spike

    @staticmethod
    def backward(ctx, grad_output):
        membrane_potential, threshold = ctx.saved_tensors
        slope = ctx.slope
        # 替代梯度: sigmoid导数的缩放版本
        diff = membrane_potential - threshold
        sigmoid_deriv = slope * torch.sigmoid(slope * diff) * (1 - torch.sigmoid(slope * diff))
        grad_membrane = grad_output * sigmoid_deriv
        grad_threshold = -grad_output * sigmoid_deriv  # 阈值梯度方向相反
        return grad_membrane, grad_threshold, None


def surrogate_spike(membrane_potential, threshold):
    """替代梯度脉冲函数的便捷调用"""
    slope = SURROGATE_CONFIG["surrogate_slope"]
    return SurrogateSpike.apply(membrane_potential, threshold, slope)


# ============================================================
# 轴突 (Axon) - 信号传导通道
# ============================================================
class Axon(nn.Module):
    """
    轴突分支管理器
    管理从一组神经元到目标神经元的输出连接，包括:
    - 传导延迟 (不同轴突到达时间不同)
    - 信号衰减 (距离越远信号越弱)
    - 动态连接 (可以重新连接到不同的目标)
    """

    def __init__(self, num_source, num_target, num_branches):
        """
        Args:
            num_source: 源神经元数量
            num_target: 可选目标神经元数量
            num_branches: 每个源神经元的轴突分支数
        """
        super().__init__()
        self.num_source = num_source
        self.num_target = num_target
        self.num_branches = num_branches

        # 每个源神经元的每个轴突分支连接到的目标神经元索引
        # shape: [num_source, num_branches]
        initial_targets = torch.randint(0, num_target, (num_source, num_branches))
        self.register_buffer("target_indices", initial_targets)

        # 传导延迟 (整数时间步), shape: [num_source, num_branches]
        delay_min = NEURON_CONFIG.get("refractory_period", 1)
        from config import AXON_CONFIG
        initial_delays = torch.randint(
            AXON_CONFIG["delay_min"],
            AXON_CONFIG["delay_max"] + 1,
            (num_source, num_branches),
        )
        self.register_buffer("delays", initial_delays)

        # 信号衰减系数 (可学习), shape: [num_source, num_branches]
        self.attenuation = nn.Parameter(
            torch.FloatTensor(num_source, num_branches).uniform_(
                AXON_CONFIG["attenuation_min"], AXON_CONFIG["attenuation_max"]
            )
        )

        # 延迟缓冲区: 存储历史脉冲信号
        # shape: [max_delay, num_source, num_branches]
        max_delay = AXON_CONFIG["delay_max"]
        self.register_buffer(
            "delay_buffer",
            torch.zeros(max_delay + 1, num_source, num_branches),
        )
        self.register_buffer("buffer_ptr", torch.tensor(0, dtype=torch.long))

    def forward(self, spikes):
        """
        接收源神经元的脉冲信号，通过延迟缓冲区传导到目标神经元
        向量化实现，无 Python 循环

        Args:
            spikes: [batch, num_source] 源神经元的脉冲信号

        Returns:
            target_input: [batch, num_target] 传导到各目标神经元的输入电流
        """
        batch_size = spikes.shape[0]

        # 将当前脉冲扩展到所有轴突分支 [batch, num_source, num_branches]
        spike_branches = spikes.unsqueeze(-1).expand(-1, -1, self.num_branches)

        # 应用衰减
        attenuated = spike_branches * torch.clamp(self.attenuation, 0.0, 1.0)

        # 向量化延迟处理:
        # 预计算每个延迟值的衰减因子 decay_factor = smoothing ^ delay
        max_delay = self.delay_buffer.shape[0]
        smoothing = NEURON_CONFIG["signal_smoothing"]

        # delays: [num_source, num_branches] -> 用作衰减指数
        delay_float = self.delays.float()
        decay_factors = smoothing ** delay_float  # [num_source, num_branches]

        # 应用延迟衰减
        delayed_signals = attenuated * decay_factors.unsqueeze(0)

        # 将信号汇聚到目标神经元 (scatter_add)
        target_input = torch.zeros(batch_size, self.num_target,
                                    device=spikes.device, dtype=DTYPE)

        flat_targets = self.target_indices.unsqueeze(0).expand(batch_size, -1, -1)
        flat_targets = flat_targets.reshape(batch_size, -1)
        flat_signals = delayed_signals.reshape(batch_size, -1)

        target_input.scatter_add_(1, flat_targets, flat_signals)

        return target_input

    def rewire(self, activity_stats=None):
        """
        动态重连: 修剪弱连接，随机重新连接
        在训练中定期调用
        """
        from config import AXON_CONFIG
        with torch.no_grad():
            # 找出权重绝对值低于阈值的连接
            weak_mask = self.attenuation.abs() < AXON_CONFIG["prune_threshold"]
            num_weak = weak_mask.sum().item()

            if num_weak > 0:
                # 重新随机连接
                new_targets = torch.randint(
                    0, self.num_target, (self.num_source, self.num_branches),
                    device=self.target_indices.device,
                )
                self.target_indices[weak_mask] = new_targets[weak_mask]

                # 重置衰减系数
                self.attenuation.data[weak_mask] = torch.FloatTensor(
                    [num_weak]
                ).uniform_(
                    AXON_CONFIG["attenuation_min"],
                    AXON_CONFIG["attenuation_max"],
                ).to(self.attenuation.device).expand(num_weak)

        return num_weak


# ============================================================
# 树突 (Dendrites) - 输入整合器
# ============================================================
class Dendrites(nn.Module):
    """
    树突组 - 线性整合
    每个神经元有 num_dendrites 个树突，每个树突有独立的突触权重
    将输入信号加权求和后传递给胞体
    """

    def __init__(self, input_dim, num_neurons, num_dendrites):
        """
        Args:
            input_dim: 输入信号维度（来自上游轴突传导的信号维度）
            num_neurons: 本层神经元数量
            num_dendrites: 每个神经元的树突数量
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.num_dendrites = num_dendrites

        # 突触权重: 线性整合
        # 将 input_dim 维的输入映射到 num_neurons 维
        self.synaptic_weights = nn.Linear(input_dim, num_neurons, bias=True)

        # Kaiming 初始化: 保持信号方差稳定
        nn.init.kaiming_normal_(self.synaptic_weights.weight, nonlinearity='relu')
        nn.init.constant_(self.synaptic_weights.bias, 0.0)

    def forward(self, incoming_signal):
        """
        线性整合: 对输入信号进行加权求和

        Args:
            incoming_signal: [batch, input_dim] 来自上游轴突的输入电流

        Returns:
            integrated: [batch, num_neurons] 整合后的树突电流
        """
        return self.synaptic_weights(incoming_signal)


# ============================================================
# 胞体 (Soma) - 神经元核心状态机
# ============================================================
class Soma(nn.Module):
    """
    胞体 - 管理神经元的核心状态:
    - 膜电位: 累积输入信号，衰减
    - 阈值: 可学习，动态调整
    - 激活判定: 膜电位 >= 阈值 → 发放脉冲
    - 不应期: 脉冲发放后一段时间不能再发放
    - 信号平滑: 膜电位连续变化而非跳变
    """

    def __init__(self, num_neurons):
        super().__init__()
        self.num_neurons = num_neurons

        # 可学习的激活阈值 [num_neurons]
        self.threshold = nn.Parameter(
            torch.full((num_neurons,), NEURON_CONFIG["default_threshold"])
        )

        # 膜电位衰减系数 (可学习)
        self.decay = nn.Parameter(
            torch.full((num_neurons,), NEURON_CONFIG["membrane_decay"])
        )

    def forward(self, dendritic_current, membrane_state, refractory_state):
        """
        更新神经元状态

        Args:
            dendritic_current: [batch, num_neurons] 树突整合后的输入电流
            membrane_state: [batch, num_neurons] 上一时间步的膜电位
            refractory_state: [batch, num_neurons] 不应期倒计时

        Returns:
            spike: [batch, num_neurons] 脉冲输出 (0或1)
            new_membrane: [batch, num_neurons] 更新后的膜电位
            new_refractory: [batch, num_neurons] 更新后的不应期状态
        """
        # 限制衰减系数在合理范围
        decay = torch.clamp(self.decay, 0.0, 1.0)

        # 限制阈值在合理范围
        threshold = torch.clamp(
            self.threshold,
            NEURON_CONFIG["threshold_min"],
            NEURON_CONFIG["threshold_max"],
        )

        # 膜电位更新: 漏积分 (LIF)
        # 简化为: new = decay * old + input (去掉双重平滑)
        new_membrane = decay * membrane_state + dendritic_current

        # 不应期检查: 不应期中的神经元不能发放
        can_fire = (refractory_state <= 0).float()

        # 脉冲判定 (使用替代梯度)
        spike = surrogate_spike(new_membrane, threshold) * can_fire

        # 发放后重置膜电位 (软重置: 减去阈值而非归零)
        new_membrane = new_membrane - spike * threshold.unsqueeze(0)

        # 更新不应期
        new_refractory = refractory_state - 1
        new_refractory = torch.where(
            spike > 0.5,
            torch.full_like(new_refractory, NEURON_CONFIG["refractory_period"]),
            new_refractory,
        )
        new_refractory = torch.clamp(new_refractory, min=0)

        return spike, new_membrane, new_refractory


# ============================================================
# 完整的神经元层 (NeuronLayer) - 整合树突 + 胞体 + 轴突
# ============================================================
class NeuronLayer(nn.Module):
    """
    一层仿生神经元，包含:
    - Dendrites: 接收上游信号，线性整合
    - Soma: 膜电位更新 + 脉冲发放
    - Axon: 向下游传导信号（带延迟和衰减）
    """

    def __init__(self, input_dim, num_neurons, output_target_dim, num_dendrites=None, num_axon_branches=None):
        """
        Args:
            input_dim: 来自上游的输入维度
            num_neurons: 本层神经元数量
            output_target_dim: 下游目标神经元数量 (轴突要连接到的)
            num_dendrites: 树突数量（默认用配置）
            num_axon_branches: 轴突分支数（默认用配置）
        """
        super().__init__()
        self.num_neurons = num_neurons
        num_dendrites = num_dendrites or NEURON_CONFIG["num_dendrites"]
        num_axon_branches = num_axon_branches or NEURON_CONFIG["num_axon_branches"]

        self.dendrites = Dendrites(input_dim, num_neurons, num_dendrites)
        self.layer_norm = nn.LayerNorm(num_neurons)
        self.soma = Soma(num_neurons)
        self.axon = Axon(num_neurons, output_target_dim, num_axon_branches)

    def forward(self, incoming_signal, membrane_state, refractory_state):
        """
        单时间步的前向传播

        Args:
            incoming_signal: [batch, input_dim] 来自上游的信号
            membrane_state: [batch, num_neurons] 膜电位
            refractory_state: [batch, num_neurons] 不应期状态

        Returns:
            axon_output: [batch, output_target_dim] 传导到下游的信号
            spike: [batch, num_neurons] 本层脉冲
            new_membrane: [batch, num_neurons] 更新后膜电位
            new_refractory: [batch, num_neurons] 更新后不应期
        """
        # 1. 树突整合 + 层归一化
        dendritic_current = self.dendrites(incoming_signal)
        dendritic_current = self.layer_norm(dendritic_current)

        # 2. 胞体更新
        spike, new_membrane, new_refractory = self.soma(
            dendritic_current, membrane_state, refractory_state
        )

        # 3. 轴突传导 (传递脉冲 + 膜电位残差作为连续信号)
        # 这给下游提供了更丰富的信号，而不仅仅是0/1脉冲
        signal_to_send = spike + 0.1 * torch.sigmoid(new_membrane)
        axon_output = self.axon(signal_to_send)

        return axon_output, spike, new_membrane, new_refractory

    def init_state(self, batch_size, device=None):
        """初始化神经元状态"""
        device = device or DEVICE
        membrane = torch.zeros(batch_size, self.num_neurons, device=device, dtype=DTYPE)
        refractory = torch.zeros(batch_size, self.num_neurons, device=device, dtype=DTYPE)
        return membrane, refractory

    def rewire(self):
        """动态重连轴突"""
        return self.axon.rewire()
