"""
BioNeuralNet - 强化学习训练器
使用 DQN + 替代梯度法训练仿生神经网络玩贪吃蛇

核心组件:
- Experience Replay Buffer: 经验回放
- Epsilon-Greedy: 探索策略
- Target Network: 稳定训练
- 动态重连: 定期修剪和重建轴突连接
"""

import os
import random
import math
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from config import TRAIN_CONFIG, GAME_CONFIG, DEVICE, DTYPE
from network import BioNeuralNetwork
from snake_env import SnakeGame


# 经验元组
Experience = namedtuple("Experience", ["state", "action", "reward", "next_state", "done"])


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=None):
        self.capacity = capacity or TRAIN_CONFIG["replay_buffer_size"]
        self.buffer = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor(np.array([e.state for e in batch])).to(DEVICE)
        actions = torch.LongTensor([e.action for e in batch]).to(DEVICE)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(DEVICE)
        next_states = torch.FloatTensor(np.array([e.next_state for e in batch])).to(DEVICE)
        dones = torch.FloatTensor([e.done for e in batch]).to(DEVICE)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class Trainer:
    """
    DQN 训练器

    训练流程:
    1. 智能体与环境交互，收集经验
    2. 从经验池采样 mini-batch
    3. 计算 TD target: r + γ * max_a Q_target(s', a)
    4. 反向传播更新网络 (替代梯度穿过脉冲函数)
    5. 定期更新 target 网络
    6. 定期执行轴突动态重连
    """

    def __init__(self, save_dir="checkpoints"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # 环境
        self.env = SnakeGame()

        # 策略网络和目标网络
        self.policy_net = BioNeuralNetwork(
            input_dim=GAME_CONFIG["input_dim"],
            num_actions=GAME_CONFIG.get("num_actions", 4),
        ).to(DEVICE)

        self.target_net = BioNeuralNetwork(
            input_dim=GAME_CONFIG["input_dim"],
            num_actions=GAME_CONFIG.get("num_actions", 4),
        ).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=TRAIN_CONFIG["learning_rate"],
        )

        # 经验回放
        self.replay_buffer = ReplayBuffer()

        # 训练统计
        self.episode_rewards = []
        self.episode_scores = []
        self.losses = []
        self.epsilon_history = []
        self.rewire_counts = []

        # 全局步计数
        self.global_step = 0

    def get_epsilon(self, episode):
        """计算当前 epsilon (探索率)"""
        eps = TRAIN_CONFIG["epsilon_end"] + (
            TRAIN_CONFIG["epsilon_start"] - TRAIN_CONFIG["epsilon_end"]
        ) * math.exp(-episode / TRAIN_CONFIG["epsilon_decay"])
        return eps

    def select_action(self, state, epsilon):
        """
        Epsilon-Greedy 动作选择

        Args:
            state: np.array 游戏状态
            epsilon: float 探索率

        Returns:
            action: int 选择的动作
        """
        if random.random() < epsilon:
            return random.randint(0, 3)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def optimize(self):
        """执行一步优化"""
        if len(self.replay_buffer) < TRAIN_CONFIG["batch_size"]:
            return None

        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            TRAIN_CONFIG["batch_size"]
        )

        # 计算当前 Q 值
        q_values = self.policy_net(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: 用 policy_net 选动作, target_net 估值
        with torch.no_grad():
            # Policy net 选出最优动作
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            # Target net 估算该动作的价值
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.gather(1, next_actions).squeeze(1)
            target_q = rewards + TRAIN_CONFIG["gamma"] * max_next_q * (1 - dones)

        # 损失 (Smooth L1 / Huber Loss)
        loss = nn.SmoothL1Loss()(q_values, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def train(self, num_episodes=None):
        """
        主训练循环

        Args:
            num_episodes: 训练的 episode 数量
        """
        num_episodes = num_episodes or TRAIN_CONFIG["episodes"]

        print("=" * 60)
        print("BioNeuralNet - 仿生神经网络训练")
        print("=" * 60)
        info = self.policy_net.get_network_info()
        print(f"神经元总数: {info['total_neurons']}")
        print(f"可训练参数: {info['trainable_params']:,}")
        print(f"树突/神经元: {info['dendrites_per_neuron']}")
        print(f"轴突分支/神经元: {info['axon_branches_per_neuron']}")
        print(f"最大传播时间步: {info['max_timesteps']}")
        print(f"设备: {DEVICE}")
        print("=" * 60)

        best_score = 0

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            epsilon = self.get_epsilon(episode)

            while not self.env.done:
                # 选择动作
                action = self.select_action(state, epsilon)

                # 执行动作
                next_state, reward, done, info = self.env.step(action)

                # 存入经验池
                self.replay_buffer.push(state, action, reward, next_state, done)

                # 每4步优化一次 (减少计算量, 让经验积累更多)
                if self.global_step % 4 == 0:
                    loss = self.optimize()
                    if loss is not None:
                        self.losses.append(loss)

                # 动态重连
                rewired = self.policy_net.maybe_rewire()
                if rewired > 0:
                    self.rewire_counts.append(rewired)

                state = next_state
                episode_reward += reward
                self.global_step += 1

            # Episode 结束
            self.episode_rewards.append(episode_reward)
            self.episode_scores.append(info["score"])
            self.epsilon_history.append(epsilon)

            # 更新 target 网络
            if episode % TRAIN_CONFIG["target_update"] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # 日志
            if episode % TRAIN_CONFIG["log_interval"] == 0:
                avg_reward = np.mean(self.episode_rewards[-TRAIN_CONFIG["log_interval"]:])
                avg_score = np.mean(self.episode_scores[-TRAIN_CONFIG["log_interval"]:])
                avg_loss = np.mean(self.losses[-100:]) if self.losses else 0
                max_recent = max(self.episode_scores[-TRAIN_CONFIG["log_interval"]:]) if self.episode_scores else 0
                print(
                    f"Episode {episode:5d} | "
                    f"Avg Reward: {avg_reward:7.2f} | "
                    f"Avg Score: {avg_score:5.2f} | "
                    f"Max Score: {max_recent:3.0f} | "
                    f"Loss: {avg_loss:.4f} | "
                    f"Eps: {epsilon:.3f} | "
                    f"Best: {best_score}"
                )

            # 保存最佳模型
            if info["score"] > best_score:
                best_score = info["score"]
                self.save_checkpoint("best.pth")

            # 定期保存
            if episode % TRAIN_CONFIG["save_interval"] == 0 and episode > 0:
                self.save_checkpoint(f"checkpoint_{episode}.pth")

        print("=" * 60)
        print(f"训练完成! 最高分: {best_score}")
        self.save_checkpoint("final.pth")

    def save_checkpoint(self, filename):
        """保存检查点"""
        path = os.path.join(self.save_dir, filename)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "episode_rewards": self.episode_rewards,
                "episode_scores": self.episode_scores,
                "global_step": self.global_step,
            },
            path,
        )

    def load_checkpoint(self, filename):
        """加载检查点"""
        path = os.path.join(self.save_dir, filename)
        checkpoint = torch.load(path, map_location=DEVICE)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.episode_rewards = checkpoint.get("episode_rewards", [])
        self.episode_scores = checkpoint.get("episode_scores", [])
        self.global_step = checkpoint.get("global_step", 0)
        print(f"已加载检查点: {path}")

    def evaluate(self, num_games=10, render=False):
        """
        评估模型表现

        Args:
            num_games: 评估游戏数
            render: 是否渲染

        Returns:
            avg_score: 平均分数
        """
        env = SnakeGame(render_mode="human" if render else None)
        scores = []

        self.policy_net.eval()
        with torch.no_grad():
            for game in range(num_games):
                state = env.reset()
                while not env.done:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                    q_values = self.policy_net(state_tensor)
                    action = q_values.argmax(dim=1).item()
                    state, _, _, info = env.step(action)

                    if render:
                        env.render()

                scores.append(info["score"])
                print(f"  Game {game + 1}: Score = {info['score']}")

        env.close()
        self.policy_net.train()

        avg_score = np.mean(scores)
        print(f"  平均分数: {avg_score:.2f}")
        return avg_score
