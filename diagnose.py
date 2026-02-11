"""
诊断脚本 v2 - 分析重新设计后的神经元架构的信号传播
"""
import sys
import config_fast
sys.modules["config"] = config_fast

import torch
import numpy as np
from network import BioNeuralNetwork
from snake_env import SnakeGame

net = BioNeuralNetwork(input_dim=100)
env = SnakeGame()
state = env.reset()

print("=" * 60)
print("信号传播诊断 v2 (重设计后)")
print("=" * 60)

# 1. 逐层信号检查 (单时间步)
with torch.no_grad():
    x = torch.FloatTensor(state).unsqueeze(0)
    batch_size = 1
    states = net.init_states(batch_size, x.device)

    print(f"\n[Input]          维度: {x.shape}  非零率: {(x != 0).float().mean():.3f}  均值: {x.mean():.4f}")

    # Encoder
    enc_out, enc_spike, _, _, _ = net.encoder(
        x, states["encoder_membrane"], states["encoder_refractory"], states["encoder_adaptive_thresh"]
    )
    print(f"[Encoder Spike]  激活率: {enc_spike.mean():.4f}  激活数: {enc_spike.sum():.0f}/{enc_spike.shape[1]}")
    print(f"[Encoder→Proc]   输出均值: {enc_out.mean():.6f}  最大: {enc_out.max():.4f}  非零: {(enc_out.abs()>1e-6).sum():.0f}/{enc_out.shape[1]}")

    # Processing layers
    layer_input = enc_out
    for i, proc in enumerate(net.processing_layers):
        proc_out, proc_spike, _, _, _ = proc(
            layer_input, states[f"proc_{i}_membrane"], states[f"proc_{i}_refractory"], states[f"proc_{i}_adaptive_thresh"]
        )
        print(f"[Proc Layer {i}]   激活率: {proc_spike.mean():.4f}  激活数: {proc_spike.sum():.0f}/{proc_spike.shape[1]}")
        print(f"[Proc {i}→Next]   输出均值: {proc_out.mean():.6f}  最大: {proc_out.max():.4f}  非零: {(proc_out.abs()>1e-6).sum():.0f}/{proc_out.shape[1]}")
        layer_input = proc_out

    # Decoder
    dec_out, dec_spike, _, _, _ = net.decoder(
        layer_input, states["decoder_membrane"], states["decoder_refractory"], states["decoder_adaptive_thresh"]
    )
    print(f"[Decoder Spike]  激活率: {dec_spike.mean():.4f}  激活数: {dec_spike.sum():.0f}/{dec_spike.shape[1]}")

# 2. 多时间步 - 检查自适应阈值效果
print(f"\n{'='*60}")
print("多时间步自适应阈值动态")
print("=" * 60)
with torch.no_grad():
    x = torch.FloatTensor(state).unsqueeze(0)
    states = net.init_states(1, x.device)

    for t in range(8):
        enc_out, enc_spike, states["encoder_membrane"], states["encoder_refractory"], states["encoder_adaptive_thresh"] = (
            net.encoder(x, states["encoder_membrane"], states["encoder_refractory"], states["encoder_adaptive_thresh"])
        )
        at = states["encoder_adaptive_thresh"]
        print(f"  t={t}  Enc激活率: {enc_spike.mean():.4f}  自适应阈值均值: {at.mean():.4f}  max: {at.max():.4f}")

# 3. 树突分析
print(f"\n{'='*60}")
print("树突结构分析")
print("=" * 60)
for name, layer in [("Encoder", net.encoder), *[(f"Proc_{i}", l) for i, l in enumerate(net.processing_layers)], ("Decoder", net.decoder)]:
    d = layer.dendrites
    print(f"[{name:10s}]  neurons={d.num_neurons}  dendrites={d.num_dendrites}  field_size={d.field_size}  needs_proj={d.needs_projection}")
    print(f"             synapse_weight shape: {d.synapse_weight.shape}  gain shape: {d.dendrite_gain.shape}")
    sw = d.synapse_weight.data
    print(f"             synapse weight: mean={sw.mean():.4f}  std={sw.std():.4f}  |min|={sw.abs().min():.4f}  |max|={sw.abs().max():.4f}")

# 4. 轴突连接覆盖率
print(f"\n{'='*60}")
print("轴突连接覆盖率")
print("=" * 60)
for name, layer in [("Encoder", net.encoder), *[(f"Proc_{i}", l) for i, l in enumerate(net.processing_layers)], ("Decoder", net.decoder)]:
    targets = layer.axon.target_indices
    unique_targets = targets.unique().numel()
    total_targets = layer.axon.num_target
    cond = layer.axon.conductance.data
    print(f"[{name:10s}]  覆盖: {unique_targets}/{total_targets} ({100*unique_targets/total_targets:.1f}%)  conductance: mean={cond.mean():.3f}")

# 5. 胞体参数
print(f"\n{'='*60}")
print("胞体参数")
print("=" * 60)
for name, layer in [("Encoder", net.encoder), *[(f"Proc_{i}", l) for i, l in enumerate(net.processing_layers)], ("Decoder", net.decoder)]:
    bt = layer.soma.base_threshold.data
    dc = layer.soma.decay.data
    print(f"[{name:10s}]  base_thresh: mean={bt.mean():.3f}  decay(raw): mean={dc.mean():.3f}  decay(sigmoid): mean={torch.sigmoid(dc).mean():.3f}")

# 6. 完整前向传播测试
print(f"\n{'='*60}")
print("完整前向传播测试")
print("=" * 60)
with torch.no_grad():
    x = torch.FloatTensor(state).unsqueeze(0)
    q_values = net(x)
    print(f"Q-values: {q_values.squeeze().tolist()}")
    print(f"Q-values std: {q_values.std():.6f}")
    act = net.get_activity_stats()
    if act:
        print(f"Total spikes: {act['total_spikes']:.0f}  Avg/step: {act['avg_spikes_per_step']:.1f}")

# 7. 参数统计
print(f"\n{'='*60}")
print("参数统计")
print("=" * 60)
info = net.get_network_info()
for k, v in info.items():
    print(f"  {k}: {v}")
