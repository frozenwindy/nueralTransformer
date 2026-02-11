# BioNeuralNet — 仿生事件驱动神经网络

## 项目概述

**BioNeuralNet** 是一个基于 PyTorch 的仿生神经网络项目，核心理念是用 **脉冲神经网络 (Spiking Neural Network, SNN)** 的思路构建深度强化学习智能体，并以经典的 **贪吃蛇游戏** 作为训练与评估环境。

与传统的全连接/卷积神经网络不同，本项目从生物神经元的结构出发，将网络分解为：

| 生物概念 | 对应模块 | 作用 |
|---------|---------|------|
| 树突 (Dendrite) | `Dendrites` | 接收并整合上游信号 |
| 胞体 (Soma) | `Soma` | 膜电位累积 → 脉冲发放（LIF 模型） |
| 轴突 (Axon) | `Axon` | 将脉冲路由到下游神经元 |
| 突触可塑性 | 替代梯度 + 动态重连 | 学习与适应 |

训练采用 **Double DQN + 替代梯度法 (Surrogate Gradient)**，使不可微的脉冲发放过程可以端到端反向传播。

---

## 目录结构

```
nueralTransformer/
├── main.py              # 主入口：train / eval / demo / info 四个子命令
├── config.py            # 默认配置（20×20 地图，完整网络规模）
├── config_fast.py       # 快速测试配置（10×10 地图，缩小网络）
├── neuron.py            # 核心神经元模块 v3（Dendrites / Axon / Soma / NeuronLayer）
├── neuron_v1.py         # 早期版本的神经元模块 v1（保留供参考）
├── network.py           # 完整网络模型 BioNeuralNetwork
├── event_engine.py      # 事件驱动引擎：多时间步脉冲传播调度
├── snake_env.py         # 贪吃蛇游戏环境（Gym-like 接口）
├── trainer.py           # DQN 强化学习训练器
├── diagnose.py          # 信号传播诊断脚本
├── deep_diagnose.py     # 深度诊断脚本（Q值分布、梯度流等）
├── test_fast.py         # 快速架构验证测试脚本
├── requirements.txt     # 依赖列表
├── checkpoints/         # 默认配置训练的模型检查点
├── checkpoints_fast/    # 快速配置训练的模型检查点
└── doc/                 # 项目文档
```

---

## 核心架构

### 1. 神经元层 (`NeuronLayer`) — `neuron.py`

每一个 `NeuronLayer` 内部包含完整的生物仿真流水线：

```
输入信号 → [Dendrites 树突整合] → [SignalNorm 信号归一化] → [Soma LIF发放] → [Axon 轴突路由] → 输出
```

#### 1.1 树突 (Dendrites)
- **自适应分组**：当输入维度 < 树突数量时，自动调整有效树突数，保证每个树突至少看 4 个特征
- **突触权重**：每个神经元拥有独立的 `[num_dendrites, field_size]` 权重矩阵
- **非线性**：使用 GELU 激活函数（允许负值信号有梯度通过）
- **汇聚**：均值汇聚 + 可学习缩放

#### 1.2 胞体 (Soma)
- **LIF 模型**：膜电位 $V_{t+1} = \alpha \cdot V_t + I_{dendrite}$
- **自适应阈值**：每次脉冲后阈值临时升高，随时间衰减回基准值
- **替代梯度**：前向使用硬阈值，反向使用 sigmoid 导数的缩放版本
- **软重置**：发放后膜电位减去阈值（非归零），保留残余信息

#### 1.3 轴突 (Axon)
- **v3 设计**：使用可学习的 `nn.Linear` 权重矩阵（完全可微）
- **稀疏修剪 (rewire)**：定期保留每个源神经元 Top-K 最强连接，弱连接权重衰减
- **信号混合**：输出 = Axon(spike + leak)，其中 leak = sigmoid(α) × sigmoid(membrane)，提供连续梯度通道

### 2. 完整网络 (`BioNeuralNetwork`) — `network.py`

```
Encoder (128 neurons)
    ↓ axon → dendrite
Processing Layer 0 (512 neurons)
    ↓
Processing Layer 1 (512 neurons)
    ↓
Processing Layer 2 (512 neurons)
    ↓
Decoder (32 neurons)
    ↓
Readout (Linear → 4 actions)
```

- **Encoder**：将游戏状态（网格地图 + 特征向量）编码为脉冲模式
- **Processing Layers**：多层中间处理，提取时空特征
- **Decoder**：汇聚信号为低维表示
- **Readout**：线性层将累积脉冲映射为 4 个动作的 Q 值

### 3. 事件驱动引擎 (`EventEngine`) — `event_engine.py`

在 `max_timesteps`（默认 16）个离散时间步内模拟脉冲传播：

1. 每个时间步输入相同的游戏状态到 Encoder
2. 脉冲逐层传播：Encoder → Processing × N → Decoder
3. 累积所有时间步 Decoder 的轴突输出
4. 最终通过 Readout 层得到 Q 值

### 4. 替代梯度 (`SurrogateSpike`)

脉冲发放函数 $S = \Theta(V - V_{th})$ 不可微，通过自定义 `autograd.Function` 实现：

$$\frac{\partial S}{\partial V} \approx k \cdot \sigma(k(V - V_{th})) \cdot (1 - \sigma(k(V - V_{th})))$$

其中 $k$ 为陡峭度参数（默认 25.0），$\sigma$ 为 sigmoid 函数。

---

## 游戏环境 — `snake_env.py`

### 状态表示（混合模式）

| 组成部分 | 维度 | 说明 |
|---------|------|------|
| 网格地图 | width × height | 空地=0, 蛇身=1.0, 蛇头=0.8, 食物=0.5 |
| 危险信号 | 4 | 四方向是否会碰撞 |
| 食物方向 | 4 | 食物相对蛇头的 one-hot 方向 |
| 坐标特征 | 4 | 蛇头 & 食物的归一化坐标 |
| 障碍物距离 | 4 | 四方向到障碍物的归一化距离 |
| 当前方向 | 4 | one-hot 编码 |
| 其他特征 | 4 | 蛇长、曼哈顿距离、可用空间、存活步数 |

默认配置：20×20 地图 → 输入维度 424；快速配置：10×10 地图 → 输入维度 124。

### 奖励设计

| 事件 | 奖励 |
|------|------|
| 吃到食物 | +5.0 |
| 死亡（撞墙/自撞） | -1.0 |
| 每步存活 | -0.01 |
| 靠近食物 | +0.2 |
| 远离食物 | -0.2 |

### 动作空间

4 个离散动作：上 (0) / 下 (1) / 左 (2) / 右 (3)。不允许 180° 转向。

---

## 训练系统 — `trainer.py`

### 算法：Double DQN

1. **经验回放 (Experience Replay)**：存储 `(s, a, r, s', done)` 元组，随机采样打破时间相关性
2. **Epsilon-Greedy 探索**：$\epsilon$ 从 1.0 指数衰减到 0.05
3. **Double DQN**：
   - Policy Net 选择动作：$a^* = \arg\max_a Q_{policy}(s', a)$
   - Target Net 估算价值：$Q_{target} = r + \gamma \cdot Q_{target}(s', a^*)$
4. **损失函数**：Smooth L1 Loss (Huber Loss)
5. **梯度裁剪**：max_norm = 1.0
6. **动态重连**：每隔 `reconnect_interval` 步执行轴突权重修剪

### 训练参数（默认配置）

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 1e-3 | Adam 学习率 |
| gamma | 0.99 | 折扣因子 |
| episodes | 10000 | 训练轮次 |
| batch_size | 32 | 小批量大小 |
| replay_buffer_size | 10000 | 经验池容量 |
| epsilon_decay | 5000 | 探索率衰减步数 |
| target_update | 50 | Target 网络更新间隔 |

### 训练参数（快速配置）

| 参数 | 值 | 说明 |
|------|-----|------|
| learning_rate | 1e-3 | Adam 学习率 |
| episodes | 2000 | 训练轮次 |
| batch_size | 64 | 更大 batch 更稳定 |
| replay_buffer_size | 50000 | 更大经验池 |
| epsilon_decay | 800 | 更快探索衰减 |
| target_update | 10 | 更频繁的 Target 更新 |

---

## 配置系统

项目提供两套配置：

- **`config.py`**：完整配置，适合正式训练
  - 网格：20×20，输入维度 424
  - 网络：Encoder 128 → Processing 512×3 → Decoder 32
  - 最大时间步：16
- **`config_fast.py`**：快速配置，适合架构验证
  - 网格：10×10，输入维度 124
  - 网络：Encoder 64 → Processing 128×2 → Decoder 32
  - 最大时间步：4

通过 Python 的模块替换机制切换配置：
```python
import config_fast
sys.modules["config"] = config_fast
```

---

## 使用方法

### 依赖安装

```bash
pip install -r requirements.txt
```

依赖：
- `torch >= 2.0.0`
- `numpy >= 1.24.0`
- `pygame >= 2.5.0`（仅可视化需要）

### 命令行接口

```bash
# 训练模型
python main.py train [--episodes N] [--resume CHECKPOINT]

# 评估模型
python main.py eval [--checkpoint PATH] [--games N] [--render]

# 可视化演示
python main.py demo [--checkpoint PATH]

# 查看网络结构信息
python main.py info
```

### 快速测试

```bash
python test_fast.py
```

使用 10×10 小地图和缩减网络训练 2000 轮，验证架构收敛性。

### 诊断工具

```bash
# 信号传播诊断（逐层检查信号强度、树突结构、轴突覆盖率等）
python diagnose.py

# 深度诊断（Q值分布、动作多样性、梯度流、完整游戏回放分析）
python deep_diagnose.py
```

---

## 版本演进

### v1 (`neuron_v1.py`)
- 轴突使用 `scatter_add` + 随机 `target_indices` → **不可微的随机投射**
- 树突每个只看 ~2 个特征 → **过于稀疏**
- 含传导延迟缓冲区、信号衰减等完整生物模拟

### v3 (`neuron.py` — 当前版本)
- **树突**：自适应分组，保证足够的感受野；GELU 非线性
- **轴突**：用 `nn.Linear` 替代 `scatter_add`，完全可微；Top-K 稀疏修剪
- **胞体**：移除不应期，降低自适应阈值增量，减少过度抑制
- **信号混合**：spike + leak 提供连续梯度通道，解决梯度消失问题

---

## 关键技术亮点

1. **生物启发 + 可微分**：将仿生 SNN 概念与现代深度学习的自动微分结合
2. **替代梯度法**：通过自定义 `autograd.Function` 让脉冲函数可反向传播
3. **事件驱动多时间步**：模拟真实神经网络的时序累积特性
4. **动态拓扑**：轴突连接定期修剪与重建，类似突触可塑性
5. **双信号通路**：脉冲（离散）+ 膜电位泄漏（连续）混合输出，兼顾生物真实性与训练效率
6. **自适应阈值**：神经元发放后阈值升高再缓慢恢复，模拟生物神经元的适应性

---

## License

本项目使用 [LICENSE](../LICENSE) 中声明的许可协议。
