"""
BioNeuralNet - 完整网络模型
组装 Encoder → Processing Layers → Decoder 的完整仿生神经网络
"""

import torch
import torch.nn as nn
from neuron import NeuronLayer
from event_engine import EventEngine
from config import NETWORK_CONFIG, NEURON_CONFIG, AXON_CONFIG, DEVICE, DTYPE


class BioNeuralNetwork(nn.Module):
    """
    仿生神经网络

    架构:
        Encoder (128 neurons)
            ↓ (axon → dendrite)
        Processing Layer 0 (512 neurons)
            ↓
        Processing Layer 1 (512 neurons)
            ↓
        Processing Layer 2 (512 neurons)
            ↓
        Decoder (32 neurons)
            ↓
        Readout (Linear → 4 actions)

    特性:
        - 事件驱动: 只有激活的神经元参与计算
        - 脉冲传播: 二值脉冲 + 替代梯度
        - 动态连接: 轴突定期重连
        - 信号延迟: 轴突传导有延迟
        - 连续膜电位: 非跳变的平滑变化
    """

    def __init__(self, input_dim=None, num_actions=None):
        super().__init__()

        input_dim = input_dim or NETWORK_CONFIG.get("input_dim", 400)
        num_actions = num_actions or NETWORK_CONFIG["num_actions"]

        enc_n = NETWORK_CONFIG["encoder_neurons"]
        proc_n = NETWORK_CONFIG["processing_neurons_per_layer"]
        dec_n = NETWORK_CONFIG["decoder_neurons"]
        num_proc = NETWORK_CONFIG["processing_layers"]

        # ---- Encoder Layer ----
        # 输入: 游戏状态 (input_dim)
        # 输出: 传导到第一个处理层 (proc_n)
        self.encoder = NeuronLayer(
            input_dim=input_dim,
            num_neurons=enc_n,
            output_target_dim=proc_n,
        )

        # ---- Processing Layers ----
        self.processing_layers = nn.ModuleList()
        for i in range(num_proc):
            # 每层输入来自上一层的轴突输出
            # 最后一层输出到 Decoder
            if i < num_proc - 1:
                output_dim = proc_n  # 到下一个处理层
            else:
                output_dim = dec_n  # 到 Decoder
            self.processing_layers.append(
                NeuronLayer(
                    input_dim=proc_n,
                    num_neurons=proc_n,
                    output_target_dim=output_dim,
                )
            )

        # ---- Decoder Layer ----
        # 输出: 32个神经元的脉冲被readout层读取
        # Decoder的轴突不需要连接到下游，设output_target_dim=dec_n（自身）
        self.decoder = NeuronLayer(
            input_dim=dec_n,
            num_neurons=dec_n,
            output_target_dim=dec_n,
        )

        # ---- Readout Layer ----
        # 将 Decoder 的累积脉冲映射为动作 Q值
        self.readout = nn.Sequential(
            nn.Linear(dec_n, num_actions),
        )

        # ---- Event Engine ----
        self.event_engine = EventEngine()

        # 训练步计数（用于动态重连调度）
        self.train_step_count = 0

    def forward(self, x):
        """
        前向传播

        Args:
            x: [batch, input_dim] 游戏状态输入

        Returns:
            q_values: [batch, num_actions] 各动作的 Q 值
        """
        q_values, activity_stats = self.event_engine.run(self, x)
        self._last_activity_stats = activity_stats
        return q_values

    def init_states(self, batch_size, device=None):
        """初始化所有层的神经元状态 (含 adaptive_threshold)"""
        device = device or DEVICE
        states = {}

        # Encoder
        m, r, a = self.encoder.init_state(batch_size, device)
        states["encoder_membrane"] = m
        states["encoder_refractory"] = r
        states["encoder_adaptive_thresh"] = a

        # Processing layers
        for i, layer in enumerate(self.processing_layers):
            m, r, a = layer.init_state(batch_size, device)
            states[f"proc_{i}_membrane"] = m
            states[f"proc_{i}_refractory"] = r
            states[f"proc_{i}_adaptive_thresh"] = a

        # Decoder
        m, r, a = self.decoder.init_state(batch_size, device)
        states["decoder_membrane"] = m
        states["decoder_refractory"] = r
        states["decoder_adaptive_thresh"] = a

        return states

    def rewire_all(self):
        """
        动态重连所有层的轴突
        返回总共重连的连接数
        """
        total_rewired = 0
        total_rewired += self.encoder.rewire()
        for layer in self.processing_layers:
            total_rewired += layer.rewire()
        total_rewired += self.decoder.rewire()
        return total_rewired

    def maybe_rewire(self):
        """根据训练步数决定是否执行动态重连"""
        self.train_step_count += 1
        if self.train_step_count % AXON_CONFIG["reconnect_interval"] == 0:
            return self.rewire_all()
        return 0

    def get_activity_stats(self):
        """获取最近一次前向传播的活动统计"""
        return getattr(self, "_last_activity_stats", None)

    def get_network_info(self):
        """获取网络结构信息"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        total_neurons = (
            NETWORK_CONFIG["encoder_neurons"]
            + NETWORK_CONFIG["processing_neurons_per_layer"] * NETWORK_CONFIG["processing_layers"]
            + NETWORK_CONFIG["decoder_neurons"]
        )

        return {
            "total_neurons": total_neurons,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "encoder_neurons": NETWORK_CONFIG["encoder_neurons"],
            "processing_layers": NETWORK_CONFIG["processing_layers"],
            "processing_neurons_per_layer": NETWORK_CONFIG["processing_neurons_per_layer"],
            "decoder_neurons": NETWORK_CONFIG["decoder_neurons"],
            "dendrites_per_neuron": NEURON_CONFIG["num_dendrites"],
            "axon_branches_per_neuron": NEURON_CONFIG["num_axon_branches"],
            "max_timesteps": NETWORK_CONFIG["max_timesteps"],
        }
