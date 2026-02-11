"""
深度诊断脚本 - 分析为什么模型评估得0分
检查: Q值分布、动作多样性、输入信号区分度、梯度流
"""
import sys
import config_fast
sys.modules["config"] = config_fast

import torch
import numpy as np
from network import BioNeuralNetwork
from snake_env import SnakeGame
import os

# 尝试加载训练好的模型
net = BioNeuralNetwork(input_dim=124)
ckpt_path = "checkpoints_fast/best.pth"
if os.path.exists(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(ckpt["policy_net"])
        print(f"✅ 加载了训练好的模型: {ckpt_path}")
    except Exception as e:
        print(f"⚠️ 模型结构已变更, 使用未训练模型测试信号传播: {e}")
else:
    print(f"⚠️ 未找到 {ckpt_path}，使用未训练模型")

net.eval()
env = SnakeGame()

print("\n" + "=" * 60)
print("诊断1: Q值分布 - 模型能否区分不同动作?")
print("=" * 60)

# 在多个不同状态上检查Q值
action_names = ["UP", "DOWN", "LEFT", "RIGHT"]
action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
q_value_stats = []

with torch.no_grad():
    for game in range(20):
        state = env.reset()
        step = 0
        while not env.done and step < 50:
            x = torch.FloatTensor(state).unsqueeze(0)
            q = net(x).squeeze()
            chosen = q.argmax().item()
            action_counts[chosen] += 1
            q_value_stats.append(q.numpy().copy())
            
            state, _, _, _ = env.step(chosen)
            step += 1

q_arr = np.array(q_value_stats)
print(f"\n动作选择分布 (20局 × 最多50步):")
total_steps = sum(action_counts.values())
for a, name in enumerate(action_names):
    pct = action_counts[a] / total_steps * 100
    print(f"  {name:6s}: {action_counts[a]:4d} ({pct:5.1f}%)")

print(f"\nQ值统计:")
print(f"  Q均值 per action: {q_arr.mean(axis=0)}")
print(f"  Q标准差 per action: {q_arr.std(axis=0)}")
print(f"  Q总体标准差: {q_arr.std():.6f}")
print(f"  Q值范围: [{q_arr.min():.4f}, {q_arr.max():.4f}]")

# 关键指标: Q值方差
q_range_per_sample = q_arr.max(axis=1) - q_arr.min(axis=1)
print(f"  单样本内Q值差异(max-min)均值: {q_range_per_sample.mean():.6f}")
print(f"  如果这个值接近0，说明模型对所有动作给出相同的Q值 → 无法区分")

print("\n" + "=" * 60)
print("诊断2: 输入信号区分度 - 不同状态是否产生不同输出?")
print("=" * 60)

with torch.no_grad():
    states_list = []
    env.reset()
    # 收集不同状态
    for i in range(10):
        state = env.reset()
        states_list.append(state)
    
    # 同一状态重复2次
    states_list.append(states_list[0])
    
    all_q = []
    for s in states_list:
        x = torch.FloatTensor(s).unsqueeze(0)
        q = net(x).squeeze().numpy()
        all_q.append(q)
    
    all_q = np.array(all_q)
    print(f"10个不同初始状态的Q值:")
    for i, q in enumerate(all_q[:10]):
        print(f"  State {i}: {q}")
    print(f"  State 0 重复: {all_q[10]}")
    
    # 检查是否所有状态给出相同的Q值
    q_var_across_states = np.var(all_q[:10], axis=0)
    print(f"\n跨状态Q值方差: {q_var_across_states}")
    print(f"如果方差接近0 → 模型对不同状态输出相同 → 没有学到状态区分")

print("\n" + "=" * 60)
print("诊断3: 中间层信号多样性")
print("=" * 60)

with torch.no_grad():
    state = env.reset()
    x = torch.FloatTensor(state).unsqueeze(0)
    
    # 手动逐层跟踪
    states_dict = net.init_states(1, x.device)
    
    # 跟踪每个时间步的脉冲模式
    spike_patterns = {
        "encoder": [],
        "decoder": [],
    }
    
    for t in range(8):
        enc_out, enc_spike, states_dict["encoder_membrane"], states_dict["encoder_refractory"], states_dict["encoder_adaptive_thresh"] = (
            net.encoder(x, states_dict["encoder_membrane"], states_dict["encoder_refractory"], states_dict["encoder_adaptive_thresh"])
        )
        spike_patterns["encoder"].append(enc_spike.squeeze().numpy().copy())
        
        layer_input = enc_out
        for i, proc in enumerate(net.processing_layers):
            layer_out, layer_spike, states_dict[f"proc_{i}_membrane"], states_dict[f"proc_{i}_refractory"], states_dict[f"proc_{i}_adaptive_thresh"] = (
                proc(layer_input, states_dict[f"proc_{i}_membrane"], states_dict[f"proc_{i}_refractory"], states_dict[f"proc_{i}_adaptive_thresh"])
            )
            layer_input = layer_out
        
        dec_out, dec_spike, states_dict["decoder_membrane"], states_dict["decoder_refractory"], states_dict["decoder_adaptive_thresh"] = (
            net.decoder(layer_input, states_dict["decoder_membrane"], states_dict["decoder_refractory"], states_dict["decoder_adaptive_thresh"])
        )
        spike_patterns["decoder"].append(dec_spike.squeeze().numpy().copy())
    
    # 分析脉冲模式
    enc_spikes = np.array(spike_patterns["encoder"])  # [8, 64]
    dec_spikes = np.array(spike_patterns["decoder"])  # [8, 16]
    
    print(f"Encoder: 每时间步激活率: {enc_spikes.mean(axis=1)}")
    print(f"Decoder: 每时间步激活率: {dec_spikes.mean(axis=1)}")
    print(f"Decoder: 每神经元激活率: {dec_spikes.mean(axis=0)}")
    print(f"Decoder: 累积脉冲: {dec_spikes.sum(axis=0)}")
    
    # 关键: decoder 的脉冲分布是否偏向某些神经元?
    dec_total = dec_spikes.sum(axis=0)
    if dec_total.sum() == 0:
        print("\n⚠️ Decoder 完全没有脉冲! 信号在到达 decoder 前消失了")
    else:
        print(f"\nDecoder脉冲集中度: 前4个={dec_total[:4]}, 后4个={dec_total[-4:]}")

print("\n" + "=" * 60)
print("诊断4: Readout层权重")
print("=" * 60)

readout = net.readout[0]
w = readout.weight.data
b = readout.bias.data
print(f"Readout权重: shape={w.shape}")
print(f"  权重均值: {w.mean():.6f}  std: {w.std():.6f}")
print(f"  偏置: {b}")
print(f"  每行(动作)权重绝对值均值: {w.abs().mean(dim=1)}")

print("\n" + "=" * 60)
print("诊断5: 完整游戏回放 (5步)")
print("=" * 60)

with torch.no_grad():
    state = env.reset()
    print(f"初始状态: 蛇头={env.snake[0]}, 食物={env.food}")
    for step in range(5):
        x = torch.FloatTensor(state).unsqueeze(0)
        q = net(x).squeeze()
        action = q.argmax().item()
        print(f"  Step {step}: Q={q.numpy().round(4)}, Action={action_names[action]}, 蛇头={env.snake[0]}, 食物={env.food}")
        state, reward, done, info = env.step(action)
        if done:
            print(f"  → 死亡! reward={reward}")
            break

print("\n" + "=" * 60)
print("诊断6: 梯度流检查 (训练模式)")
print("=" * 60)

net.train()
x = torch.FloatTensor(env.reset()).unsqueeze(0)
x.requires_grad_(False)
q = net(x)
loss = q.sum()
loss.backward()

print("各模块梯度范数:")
for name, param in net.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if grad_norm == 0:
            print(f"  ⚠️ {name}: grad_norm=0.000000 (死梯度!)")
        elif grad_norm < 1e-6:
            print(f"  ⚠️ {name}: grad_norm={grad_norm:.6f} (极小)")
        else:
            print(f"  ✅ {name}: grad_norm={grad_norm:.6f}")
    else:
        print(f"  ❌ {name}: 无梯度")
