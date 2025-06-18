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
    checkpoint_dir = args.checkpoint_dir  # 你可以传递检查点目录路径
    checkpoint_filename = 'model_2000.pt'  # 从 model_2000.pt 继续训练

    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
    
    if os.path.exists(checkpoint_path):
        # 加载检查点
        checkpoint = torch.load(checkpoint_path)
        ppo_runner.load_state_dict(checkpoint['ppo_state_dict'])
        train_cfg.runner.start_iteration = checkpoint['iteration']
        print(f"Resuming training from iteration {checkpoint['iteration']}")
    else:
        print("Checkpoint file does not exist, starting from scratch.")
    
    # 继续训练
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

    # 保存检查点
    # 每个训练迭代后，保存模型
    checkpoint = {
        'ppo_state_dict': ppo_runner.state_dict(),
        'iteration': train_cfg.runner.start_iteration
    }
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pt')
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    args = get_args()
    train(args)
