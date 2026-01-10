"""
Level 5: Dance Training Script (BONUS SHOWOFF!)
Train Go2 to dance with rhythm and style!
Perfect for presentation WOW factor!
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

from envs.dance_env import DanceEnv
from configs.level5_dance import DanceConfig


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--max_iterations', type=int, default=600)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--headless', action='store_true', default=True)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--bpm', type=int, default=120, help='Music BPM for dancing')
    args = parser.parse_args()

    # Configuration
    config = DanceConfig()

    # Create log directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join('logs', 'level5_dance', timestamp)
    model_dir = os.path.join('models', 'level5_dance', timestamp)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

    print("=" * 60)
    print("🕺 Level 5: DANCE Training (SHOWOFF MODE!) 💃")
    print("=" * 60)
    print(f"Music BPM: {args.bpm}")
    print(f"Dance frequency: {config.dance_frequency} Hz")
    print(f"Num envs: {args.num_envs}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Log dir: {log_dir}")
    print("=" * 60)
    print("\n🎵 Let's teach this robot to GROOVE! 🎵\n")

    # Create environment
    env = DanceEnv(
        num_envs=args.num_envs,
        device=args.device,
        headless=args.headless
    )

    # Set music BPM
    env.set_music_bpm(args.bpm)

    # Create actor-critic network (extended observations)
    actor_critic = ActorCritic(
        num_obs=config.num_observations,  # 48 for dance
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

    # Resume if checkpoint provided
    if args.resume:
        print(f"Resuming from: {args.resume}")
        runner.load(args.resume)

    # Train
    runner.learn(
        num_learning_iterations=args.max_iterations,
        init_at_random_ep_len=True,
    )

    # Save final model
    final_path = os.path.join(model_dir, 'final_model.pt')
    runner.save(final_path)
    print(f"\n🎉 Dance training complete! Final model saved to: {final_path}")
    print("🕺 Your robot is now ready to hit the dance floor! 💃")

    # Cleanup
    env.close()


if __name__ == '__main__':
    train()
