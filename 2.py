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
        checkpoint = torch.load(checkpoint_path, weights_only=True)  # 使用 weights_only=True
        print(f"Checkpoint loaded successfully")
        
        # 打印检查点内容以调试
        print("Checkpoint contents:", checkpoint)

        # 确保检查点包含 'model_state_dict' 和 'iter'
        if 'model_state_dict' in checkpoint and 'iter' in checkpoint:
            # 恢复模型状态
            ppo_runner.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])  # 恢复模型权重
            ppo_runner.alg.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 恢复优化器状态
            
            # 恢复训练迭代次数
            train_cfg.runner.start_iteration = checkpoint['iter']  # 恢复迭代次数
            print(f"Resuming training from iteration {checkpoint['iter']}")
        else:
            print("Error: 'model_state_dict' or 'iter' not found in checkpoint.")
            return
    else:
        print(f"Checkpoint file {checkpoint_path} does not exist, starting from scratch.")

    # 继续训练
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

    # 保存新的检查点
    checkpoint = {
        'model_state_dict': ppo_runner.state_dict(),  # 保存模型的状态字典
        'optimizer_state_dict': ppo_runner.alg.optimizer.state_dict(),  # 保存优化器状态字典
        'iter': train_cfg.runner.start_iteration  # 保存当前的训练迭代次数
    }
    checkpoint_save_path = os.path.join(checkpoint_dir, 'model_2000.pt')  # 继续使用相同的文件名
    torch.save(checkpoint, checkpoint_save_path)
    print(f"Checkpoint saved to {checkpoint_save_path}")

if __name__ == '__main__':
    args = get_args()
    train(args)
