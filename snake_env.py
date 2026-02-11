"""
BioNeuralNet - 贪吃蛇游戏环境
方案A: 20x20 网格地图作为输入 (400维)
支持 Gym-like 接口用于强化学习训练
可选 Pygame 渲染用于可视化
"""

import random
import numpy as np
from enum import IntEnum
from config import GAME_CONFIG


class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


# 方向对应的坐标变化 (row, col)
DIR_DELTA = {
    Direction.UP: (-1, 0),
    Direction.DOWN: (1, 0),
    Direction.LEFT: (0, -1),
    Direction.RIGHT: (0, 1),
}

# 相反方向映射
OPPOSITE = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


class SnakeGame:
    """
    贪吃蛇游戏环境

    状态表示 (方案A - 网格地图):
        20x20 网格，每个格子的值:
        0.0 = 空地
        1.0 = 蛇身
        0.8 = 蛇头
        0.5 = 食物

    动作空间:
        0 = 上, 1 = 下, 2 = 左, 3 = 右

    奖励设计:
        +5.0   吃到食物
        -1.0   死亡（撞墙或自撞）
        -0.01  每步存活惩罚（鼓励高效觅食）
        +0.2   靠近食物
        -0.2   远离食物
    """

    def __init__(self, width=None, height=None, render_mode=None):
        self.width = width or GAME_CONFIG["grid_width"]
        self.height = height or GAME_CONFIG["grid_height"]
        self.render_mode = render_mode

        # Pygame 相关
        self._screen = None
        self._clock = None

        self.reset()

    def reset(self):
        """重置游戏，返回初始状态"""
        # 蛇初始位置: 中心，长度3，朝右
        center_r = self.height // 2
        center_c = self.width // 2
        self.snake = [
            (center_r, center_c),
            (center_r, center_c - 1),
            (center_r, center_c - 2),
        ]
        self.direction = Direction.RIGHT
        self.food = None
        self._place_food()
        self.score = 0
        self.steps = 0
        self.max_steps = self.width * self.height * 2  # 防止无限循环
        self.done = False

        return self._get_state()

    def step(self, action):
        """
        执行一步

        Args:
            action: int, 0=上 1=下 2=左 3=右

        Returns:
            state: np.array [height * width] 展平的网格状态
            reward: float 奖励
            done: bool 是否结束
            info: dict 额外信息
        """
        if self.done:
            return self._get_state(), 0.0, True, {"score": self.score}

        self.steps += 1
        action = Direction(action)

        # 不允许180度转向
        if action != OPPOSITE.get(self.direction, None):
            self.direction = action

        # 计算新蛇头位置
        head_r, head_c = self.snake[0]
        dr, dc = DIR_DELTA[self.direction]
        new_head = (head_r + dr, head_c + dc)

        # 检查碰撞
        reward = -0.01  # 存活惩罚

        # 撞墙
        if (
            new_head[0] < 0
            or new_head[0] >= self.height
            or new_head[1] < 0
            or new_head[1] >= self.width
        ):
            self.done = True
            return self._get_state(), -1.0, True, {"score": self.score, "death": "wall"}

        # 撞自己
        if new_head in self.snake:
            self.done = True
            return self._get_state(), -1.0, True, {"score": self.score, "death": "self"}

        # 移动蛇
        self.snake.insert(0, new_head)

        # 吃食物
        if new_head == self.food:
            self.score += 1
            reward = 5.0
            self._place_food()
        else:
            self.snake.pop()  # 没吃到食物，去掉尾巴

            # 距离奖励 (更强的引导信号)
            old_dist = abs(head_r - self.food[0]) + abs(head_c - self.food[1])
            new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])
            if new_dist < old_dist:
                reward += 0.2
            else:
                reward -= 0.2

        # 超时
        if self.steps >= self.max_steps:
            self.done = True
            return self._get_state(), -1.0, True, {"score": self.score, "death": "timeout"}

        return self._get_state(), reward, self.done, {"score": self.score}

    def _get_state(self):
        """
        获取当前游戏状态:
        混合表示 = 网格地图(100维) + 特征向量(24维) = 124维

        特征向量包含:
        - 4个方向的危险信号 (会撞墙或撞蛇身)
        - 食物相对方向 (4维 one-hot)
        - 蛇头归一化坐标 (2维)
        - 食物归一化坐标 (2维)
        - 4个方向的距离到障碍物 (归一化)
        - 当前方向 (4维 one-hot)
        - 蛇身长度 (归一化)
        - 距离食物的曼哈顿距离 (归一化)

        Returns:
            state: np.array [height * width + 24]
        """
        grid = np.zeros((self.height, self.width), dtype=np.float32)

        # 蛇身
        for r, c in self.snake[1:]:
            grid[r, c] = 1.0

        # 蛇头
        if self.snake:
            hr, hc = self.snake[0]
            if 0 <= hr < self.height and 0 <= hc < self.width:
                grid[hr, hc] = 0.8

        # 食物
        if self.food:
            grid[self.food[0], self.food[1]] = 0.5

        flat_grid = grid.flatten()

        # === 特征向量 ===
        features = np.zeros(24, dtype=np.float32)
        hr, hc = self.snake[0]

        # 1. 四个方向的危险信号 [0:4]
        for i, d in enumerate([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]):
            dr, dc = DIR_DELTA[d]
            nr, nc = hr + dr, hc + dc
            if nr < 0 or nr >= self.height or nc < 0 or nc >= self.width or (nr, nc) in self.snake:
                features[i] = 1.0

        # 2. 食物相对方向 [4:8]
        if self.food:
            fr, fc = self.food
            if fr < hr: features[4] = 1.0  # 食物在上
            if fr > hr: features[5] = 1.0  # 食物在下
            if fc < hc: features[6] = 1.0  # 食物在左
            if fc > hc: features[7] = 1.0  # 食物在右

        # 3. 蛇头归一化坐标 [8:10]
        features[8] = hr / max(1, self.height - 1)
        features[9] = hc / max(1, self.width - 1)

        # 4. 食物归一化坐标 [10:12]
        if self.food:
            features[10] = self.food[0] / max(1, self.height - 1)
            features[11] = self.food[1] / max(1, self.width - 1)

        # 5. 四个方向到障碍物的距离 (归一化) [12:16]
        for i, d in enumerate([Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]):
            dr, dc = DIR_DELTA[d]
            dist = 0
            nr, nc = hr + dr, hc + dc
            while 0 <= nr < self.height and 0 <= nc < self.width and (nr, nc) not in self.snake:
                dist += 1
                nr += dr
                nc += dc
            max_dim = max(self.height, self.width)
            features[12 + i] = dist / max_dim

        # 6. 当前方向 one-hot [16:20]
        features[16 + int(self.direction)] = 1.0

        # 7. 蛇身长度归一化 [20]
        features[20] = len(self.snake) / (self.width * self.height)

        # 8. 距离食物曼哈顿距离归一化 [21]
        if self.food:
            manhattan = abs(hr - self.food[0]) + abs(hc - self.food[1])
            features[21] = manhattan / (self.width + self.height)

        # 9. 可用空间比例 [22]
        features[22] = 1.0 - len(self.snake) / (self.width * self.height)

        # 10. 存活步数归一化 [23]
        features[23] = min(1.0, self.steps / self.max_steps)

        return np.concatenate([flat_grid, features])

    def _place_food(self):
        """在空位随机放置食物"""
        empty = []
        for r in range(self.height):
            for c in range(self.width):
                if (r, c) not in self.snake:
                    empty.append((r, c))
        if empty:
            self.food = random.choice(empty)
        else:
            # 蛇填满了整个地图 - 游戏胜利
            self.done = True
            self.food = None

    def render(self):
        """使用 Pygame 渲染游戏画面"""
        if self.render_mode != "human":
            return

        try:
            import pygame
        except ImportError:
            return

        cell = GAME_CONFIG["cell_size"]
        w = self.width * cell
        h = self.height * cell

        if self._screen is None:
            pygame.init()
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("BioNeuralNet Snake")
            self._clock = pygame.time.Clock()

        self._screen.fill((0, 0, 0))

        # 画网格线
        for x in range(0, w, cell):
            pygame.draw.line(self._screen, (40, 40, 40), (x, 0), (x, h))
        for y in range(0, h, cell):
            pygame.draw.line(self._screen, (40, 40, 40), (0, y), (w, y))

        # 画蛇身
        for i, (r, c) in enumerate(self.snake):
            color = (0, 200, 0) if i > 0 else (0, 255, 100)  # 蛇头更亮
            rect = pygame.Rect(c * cell + 1, r * cell + 1, cell - 2, cell - 2)
            pygame.draw.rect(self._screen, color, rect)

        # 画食物
        if self.food:
            fr, fc = self.food
            rect = pygame.Rect(fc * cell + 1, fr * cell + 1, cell - 2, cell - 2)
            pygame.draw.rect(self._screen, (255, 50, 50), rect)

        # 显示分数
        font = pygame.font.SysFont(None, 24)
        score_text = font.render(f"Score: {self.score}  Steps: {self.steps}", True, (255, 255, 255))
        self._screen.blit(score_text, (5, 5))

        pygame.display.flip()
        self._clock.tick(GAME_CONFIG["fps"])

        # 处理退出事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

    def close(self):
        """关闭渲染窗口"""
        if self._screen is not None:
            import pygame
            pygame.quit()
            self._screen = None
