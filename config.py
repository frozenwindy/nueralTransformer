"""
BioNeuralNet - 仿生神经网络配置文件
事件驱动并行计算架构
"""

import torch

# ============================================================
# 设备配置
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

# ============================================================
# 神经元结构参数
# ============================================================
NEURON_CONFIG = {
    # 树突数量（输入连接数）
    "num_dendrites": 64,
    # 轴突分支数（输出连接数）
    "num_axon_branches": 8,
    # 膜电位衰减系数 (每时间步衰减)
    "membrane_decay": 0.5,
    # 默认激活阈值
    "default_threshold": 0.3,
    # 阈值学习范围
    "threshold_min": 0.01,
    "threshold_max": 2.0,
    # 不应期
    "refractory_period": 1,
    # 信号连续变化的平滑系数
    "signal_smoothing": 0.8,
    # 自适应阈值: 每次脉冲后阈值增量
    "threshold_increment": 0.15,
    # 自适应阈值: 阈值恢复的衰减系数 (0~1, 越大恢复越慢)
    "threshold_decay": 0.9,
}

# ============================================================
# 轴突传导参数
# ============================================================
AXON_CONFIG = {
    # 传导延迟范围（时间步）
    "delay_min": 1,
    "delay_max": 5,
    # 信号衰减系数范围
    "attenuation_min": 0.7,
    "attenuation_max": 1.0,
    # 动态连接 - 连接更新周期（每多少训练步更新一次拓扑）
    "reconnect_interval": 100,
    # 连接修剪阈值（权重绝对值低于此值的连接被剪掉）
    "prune_threshold": 0.01,
}

# ============================================================
# 网络拓扑参数
# ============================================================
NETWORK_CONFIG = {
    # Encoder层 - 将游戏状态编码为脉冲模式
    "encoder_neurons": 128,
    # Processing层 - 中间处理层
    "processing_layers": 3,
    "processing_neurons_per_layer": 512,
    # Decoder层 - 汇聚为动作输出
    "decoder_neurons": 32,
    # 输出动作数（上下左右）
    "num_actions": 4,
    # 每次决策的最大传播时间步
    "max_timesteps": 16,
}

# ============================================================
# 替代梯度参数
# ============================================================
SURROGATE_CONFIG = {
    # 替代梯度函数的陡峭度
    "surrogate_slope": 25.0,
    # 替代梯度类型: "sigmoid", "atan", "triangle"
    "surrogate_type": "sigmoid",
}

# ============================================================
# 贪吃蛇游戏参数
# ============================================================
GAME_CONFIG = {
    "grid_width": 20,
    "grid_height": 20,
    # 输入维度 = grid_width * grid_height (方案A: 网格地图)
    "input_dim": 424,     # grid_width * grid_height + 24 features
    # 渲染
    "cell_size": 30,  # 像素
    "fps": 10,
}

# ============================================================
# 训练参数
# ============================================================
TRAIN_CONFIG = {
    "learning_rate": 1e-3,
    "gamma": 0.99,  # 折扣因子
    "episodes": 10000,
    "batch_size": 32,
    "replay_buffer_size": 10000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 5000,
    "target_update": 50,  # 每多少episode更新target网络
    "save_interval": 500,
    "log_interval": 50,
}
