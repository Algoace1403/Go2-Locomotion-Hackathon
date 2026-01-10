"""
Level 2: Running Training Script
Train Go2 to run at ≥2.0 m/s
CRITICAL: This level is worth 40 points!
"""

import argparse
import os
import pickle
from datetime import datetime

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

# Add parent directory to path
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.run_env import RunEnv
from configs.level2_run import RunConfig


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--max_iterations', type=int, default=600)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--from_walk', type=str, default=None, help='Init from walking model')
    args = parser.parse_args()

    # Configuration
    config = RunConfig()

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', 'level2_run', timestamp)
    model_dir = os.path.join('models', 'level2_run', timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print("=" * 60)
    print("Level 2: Running Training (40 POINTS - CRITICAL!)")
    print("=" * 60)
    print(f"Target: ≥{config.min_vel} m/s (aiming for {config.target_vel} m/s)")
    print(f"Num envs: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Log dir: {log_dir}")
    print("=" * 60)

    # Create environment
    env = RunEnv(
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless
    )

    # Create actor-critic network
    actor_critic = ActorCritic(
        num_obs=config.num_observations,
        num_actions=config.num_joints,
        actor_hidden_dims=config.network['actor_hidden_dims'],
        critic_hidden_dims=config.network['critic_hidden_dims'],
        activation=config.network['activation'],
        init_noise_std=config.network['init_noise_std'],
    ).to(args.device)

    # Create PPO algorithm
    ppo = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=config.ppo['num_learning_epochs'],
        num_mini_batches=config.ppo['num_mini_batches'],
        clip_param=config.ppo['clip_param'],
        gamma=config.ppo['gamma'],
        lam=config.ppo['lam'],
        value_loss_coef=config.ppo['value_loss_coef'],
        entropy_coef=config.ppo['entropy_coef'],
        learning_rate=config.ppo['learning_rate'],
        max_grad_norm=config.ppo['max_grad_norm'],
        schedule=config.ppo['schedule'],
        desired_kl=config.ppo['desired_kl'],
        device=args.device,
    )

    # Create runner
    runner = OnPolicyRunner(
        env=env,
        algorithm=ppo,
        num_steps_per_env=config.num_steps_per_env,
        save_interval=config.save_interval,
        log_dir=log_dir,
        device=args.device,
    )

    # Resume or initialize from walking model
    if args.resume:
        print(f"Resuming from: {args.resume}")
        runner.load(args.resume)
    elif args.from_walk:
        print(f"Initializing from walking model: {args.from_walk}")
        # Load walking model weights as starting point
        checkpoint = torch.load(args.from_walk, map_location=args.device)
        actor_critic.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded walking model as initialization")

    # Train
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )

    # Save final model
    final_path = os.path.join(model_dir, 'final_model.pt')
    runner.save(final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")

    # Cleanup
    env.close()


if __name__ == '__main__':
    train()
