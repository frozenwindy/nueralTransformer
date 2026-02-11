"""
BioNeuralNet - 主入口文件
支持训练、评估和可视化演示
"""

import argparse
import sys
import torch

from config import DEVICE


def main():
    parser = argparse.ArgumentParser(
        description="BioNeuralNet - 仿生事件驱动神经网络",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--episodes", type=int, default=None, help="训练 episode 数量 (默认: config中的值)"
    )
    train_parser.add_argument(
        "--resume", type=str, default=None, help="从检查点恢复训练"
    )

    # ---- eval ----
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pth", help="检查点路径"
    )
    eval_parser.add_argument(
        "--games", type=int, default=10, help="评估游戏数"
    )
    eval_parser.add_argument(
        "--render", action="store_true", help="是否渲染游戏画面"
    )

    # ---- demo ----
    demo_parser = subparsers.add_parser("demo", help="可视化演示")
    demo_parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pth", help="检查点路径"
    )

    # ---- info ----
    subparsers.add_parser("info", help="显示网络结构信息")

    args = parser.parse_args()

    if args.command == "train":
        from trainer import Trainer

        trainer = Trainer()
        if args.resume:
            trainer.load_checkpoint(args.resume)
        trainer.train(num_episodes=args.episodes)

    elif args.command == "eval":
        from trainer import Trainer

        trainer = Trainer()
        trainer.load_checkpoint(args.checkpoint)
        trainer.evaluate(num_games=args.games, render=args.render)

    elif args.command == "demo":
        from trainer import Trainer

        trainer = Trainer()
        trainer.load_checkpoint(args.checkpoint)
        trainer.evaluate(num_games=1, render=True)

    elif args.command == "info":
        from network import BioNeuralNetwork

        net = BioNeuralNetwork()
        info = net.get_network_info()
        print("=" * 50)
        print("BioNeuralNet 网络结构信息")
        print("=" * 50)
        for key, value in info.items():
            label = key.replace("_", " ").title()
            if isinstance(value, int) and value > 1000:
                print(f"  {label:30s}: {value:,}")
            else:
                print(f"  {label:30s}: {value}")
        print("=" * 50)
        print(f"  设备: {DEVICE}")
        print("=" * 50)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
