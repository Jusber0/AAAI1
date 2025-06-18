# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os
import torch
import argparse
import time
from legged_gym.envs import *
from legged_gym.utils import task_registry
from rsl_rl.algorithms import PPO
from torch.utils.tensorboard import SummaryWriter
from collections import deque

# Argument parser to receive command-line arguments
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True, help="Task name")
    parser.add_argument('--checkpoint_dir', type=str, required=True, help="Directory containing checkpoints")
    parser.add_argument('--checkpoint_filename', type=str, required=True, help="Checkpoint filename")
    return parser.parse_args()

def train(args):
    # Load the environment and algorithm runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # Construct the checkpoint path
    checkpoint_dir = args.checkpoint_dir
    checkpoint_filename = args.checkpoint_filename
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

    # Check if the checkpoint exists and load it
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        print(f"Checkpoint loaded successfully")

        # Load model and optimizer state from the checkpoint
        if 'model_state_dict' in checkpoint and 'iter' in checkpoint:
            ppo_runner.alg.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            ppo_runner.alg.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            train_cfg.runner.start_iteration = checkpoint['iter']
            print(f"Resuming training from iteration {checkpoint['iter']}")
        else:
            print("Error: 'model_state_dict' or 'iter' not found in checkpoint.")
            return
    else:
        print(f"Checkpoint file {checkpoint_path} does not exist, starting from scratch.")

    # Start the training process
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

    # Save the checkpoint after training
    checkpoint = {
        'model_state_dict': ppo_runner.alg.actor_critic.state_dict(),
        'optimizer_state_dict': ppo_runner.alg.optimizer.state_dict(),
        'iter': train_cfg.runner.start_iteration
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == '__main__':
    args = get_args()
    train(args)
