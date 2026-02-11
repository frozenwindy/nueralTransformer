"""
BioNeuralNet - 快速测试配置 v3
v3 架构: 可微轴突 + 自适应树突分组 + 简化 LIF
"""

import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

NEURON_CONFIG = {
    "num_dendrites": 32,         # v3: 降低, 让每个树突有更大感受野
    "num_axon_branches": 8,      # v3: 用于 rewire 修剪的 top-K 数
    "membrane_decay": 0.5,
    "default_threshold": 0.3,
    "threshold_min": 0.01,
    "threshold_max": 2.0,
    "refractory_period": 1,      # v3: 保留接口但 Soma 不使用
    "signal_smoothing": 0.8,
    "threshold_increment": 0.05, # v3: 大幅降低, 减少过度抑制
    "threshold_decay": 0.95,     # v3: 更慢衰减, 更稳定
}

AXON_CONFIG = {
    "delay_min": 1,
    "delay_max": 3,
    "attenuation_min": 0.7,
    "attenuation_max": 1.0,
    "reconnect_interval": 500,   # v3: 降低修剪频率
    "prune_threshold": 0.01,
}

NETWORK_CONFIG = {
    "encoder_neurons": 64,
    "processing_layers": 2,
    "processing_neurons_per_layer": 128,
    "decoder_neurons": 32,       # v3: 增加到32, 给 readout 更多信息
    "num_actions": 4,
    "max_timesteps": 4,          # v3: 减少到4步, 可微轴突不需要长累积
}

SURROGATE_CONFIG = {
    "surrogate_slope": 25.0,
    "surrogate_type": "sigmoid",
}

GAME_CONFIG = {
    "grid_width": 10,
    "grid_height": 10,
    "input_dim": 124,
    "cell_size": 30,
    "fps": 10,
}

TRAIN_CONFIG = {
    "learning_rate": 1e-3,       # v3: 提高学习率, 可微路径允许更大步长
    "gamma": 0.99,
    "episodes": 2000,
    "batch_size": 64,            # v3: 增大 batch, 更稳定的梯度
    "replay_buffer_size": 50000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 800,        # v3: 更快衰减, 更早利用学到的策略
    "target_update": 10,
    "save_interval": 500,
    "log_interval": 50,
}
