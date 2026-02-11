"""
快速测试脚本 - 使用缩小的配置验证架构可行性和收敛趋势
"""
import sys
import time

# 用快速配置覆盖默认配置
import config_fast
sys.modules["config"] = config_fast

from trainer import Trainer


def main():
    print("=" * 60)
    print("快速架构验证测试")
    print("地图: 10x10, 输入维度: 124")
    print("=" * 60)

    trainer = Trainer(save_dir="checkpoints_fast")

    start = time.time()
    trainer.train(num_episodes=2000)
    elapsed = time.time() - start

    print(f"\n总耗时: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    print(f"平均每episode: {elapsed / 2000:.2f}s")

    # 评估
    print("\n--- 最终评估 ---")
    trainer.evaluate(num_games=5)


if __name__ == "__main__":
    main()
