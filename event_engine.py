"""
BioNeuralNet - 事件驱动引擎
管理神经元层间的事件传播:
- 事件队列管理
- 并行事件分发
- 时间步调度
"""

import torch
from config import NETWORK_CONFIG, DEVICE, DTYPE


class EventEngine:
    """
    事件驱动引擎
    在离散时间步内模拟并行事件传播:
    1. 每个时间步收集所有激活的神经元产生的脉冲事件
    2. 并行将事件分发到下游层
    3. 记录每个时间步的活动统计（用于监控和动态重连）
    """

    def __init__(self):
        self.max_timesteps = NETWORK_CONFIG["max_timesteps"]
        self.activity_log = []

    def run(self, network, input_signal):
        """
        运行完整的事件驱动传播周期

        Args:
            network: BioNeuralNetwork 实例
            input_signal: [batch, input_dim] 输入信号

        Returns:
            output: [batch, num_actions] 输出动作概率
            activity_stats: dict 活动统计信息
        """
        batch_size = input_signal.shape[0]
        device = input_signal.device

        # 初始化所有层的状态
        states = network.init_states(batch_size, device)

        # 收集所有时间步的输出层脉冲
        output_spikes_accumulator = torch.zeros(
            batch_size, NETWORK_CONFIG["decoder_neurons"],
            device=device, dtype=DTYPE,
        )

        # 活动统计
        total_spikes = 0
        layer_activities = []

        for t in range(self.max_timesteps):
            step_activity = {}

            # ---- Encoder ----
            # 直接传入连续值信号，让树突权重学习如何处理
            enc_out, enc_spike, states["encoder_membrane"], states["encoder_refractory"], states["encoder_adaptive_thresh"] = (
                network.encoder(
                    input_signal,
                    states["encoder_membrane"],
                    states["encoder_refractory"],
                    states["encoder_adaptive_thresh"],
                )
            )
            step_activity["encoder"] = enc_spike.sum().item()

            # ---- Processing Layers ----
            layer_input = enc_out
            for i, proc_layer in enumerate(network.processing_layers):
                layer_out, layer_spike, states[f"proc_{i}_membrane"], states[f"proc_{i}_refractory"], states[f"proc_{i}_adaptive_thresh"] = (
                    proc_layer(
                        layer_input,
                        states[f"proc_{i}_membrane"],
                        states[f"proc_{i}_refractory"],
                        states[f"proc_{i}_adaptive_thresh"],
                    )
                )
                step_activity[f"processing_{i}"] = layer_spike.sum().item()
                layer_input = layer_out

            # ---- Decoder ----
            dec_out, dec_spike, states["decoder_membrane"], states["decoder_refractory"], states["decoder_adaptive_thresh"] = (
                network.decoder(
                    layer_input,
                    states["decoder_membrane"],
                    states["decoder_refractory"],
                    states["decoder_adaptive_thresh"],
                )
            )
            step_activity["decoder"] = dec_spike.sum().item()

            # 累积输出层信号: v3 使用完整的 axon 输出 (已经是可微的)
            # dec_out = Axon(spike + leak) 通过 nn.Linear → 完全可微
            output_spikes_accumulator += dec_out

            # 统计
            total_spikes += sum(step_activity.values())
            layer_activities.append(step_activity)

        # 将累积脉冲转换为动作输出
        output = network.readout(output_spikes_accumulator)

        activity_stats = {
            "total_spikes": total_spikes,
            "avg_spikes_per_step": total_spikes / self.max_timesteps,
            "layer_activities": layer_activities,
        }

        return output, activity_stats

    def _rate_encode(self, signal, timestep):
        """
        速率编码: 将连续值转换为脉冲概率
        信号值越大，产生脉冲的概率越高

        Args:
            signal: [batch, input_dim] 连续值输入 (0~1范围)
            timestep: 当前时间步

        Returns:
            encoded: [batch, input_dim] 脉冲编码后的输入
        """
        # 使用确定性编码 + 少量随机性
        # 这样同一个输入在不同时间步会产生不同的脉冲模式
        # 但整体速率与输入值成正比
        threshold = torch.rand_like(signal)
        spikes = (signal > threshold).to(DTYPE)
        return spikes
