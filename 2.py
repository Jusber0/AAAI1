import numpy as np
import os
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch

def train(args):
    # 创建环境和算法执行器
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # 指定检查点目录
    checkpoint_dir = os.path.expanduser('/home/ubuntu/ACR/pointfootGym/logs/pointfoot_rough/Jun15_10-32-32_')
    checkpoint_filename = 'model_2000.pt'  # 假设从 model_2000.pt 文件恢复训练

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # 检查检查点是否存在
    if os.path.exists(checkpoint_path):
        # 加载检查点
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, weights_only=True)  # 使用 weights_only=True 以提高安全性
        print(f"Checkpoint loaded successfully")

        # 确保检查点包含 'ppo_state_dict' 和 'iteration'
        if 'ppo_state_dict' in checkpoint and 'iteration' in checkpoint:
            # 使用 ppo_state_dict 恢复模型
            ppo_runner.load(checkpoint_path)  # 调用自定义的 load 方法
            train_cfg.runner.start_iteration = checkpoint['iteration']  # 恢复训练迭代次数
            print(f"Resuming training from iteration {checkpoint['iteration']}")
        else:
            print("Error: 'ppo_state_dict' or 'iteration' not found in checkpoint.")
            return
    else:
        print(f"Checkpoint file {checkpoint_path} does not exist, starting from scratch.")

    # 继续训练
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

    # 保存新的检查点
    checkpoint = {
        'ppo_state_dict': ppo_runner.state_dict(),  # 保存模型的状态字典
        'iteration': train_cfg.runner.start_iteration  # 保存当前的训练迭代次数
    }
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    args = get_args()
    train(args)
