"""
Evaluation Script for Go2 Robot Models

Usage:
    python evaluate.py --checkpoint logs/go2_walk_xxx/model_xxx.pt
    python evaluate.py --checkpoint logs/go2_run_xxx/model_xxx.pt --episodes 10
"""

import argparse
import os
import pickle
import torch
import genesis as gs
from go2_env import Go2Env


def load_policy(checkpoint_path, num_obs, num_actions):
    """Load a trained policy from checkpoint"""
    from rsl_rl.modules import ActorCritic

    # Create policy network
    policy = ActorCritic(
        num_obs,
        num_obs,  # num_privileged_obs same as num_obs
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=gs.device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(gs.device)
    policy.eval()

    return policy


def evaluate(args):
    """Run evaluation on a trained model"""
    # Find checkpoint
    if os.path.isdir(args.checkpoint):
        # Find latest model in directory (sort by iteration number)
        models = [f for f in os.listdir(args.checkpoint) if f.startswith("model_") and f.endswith(".pt")]
        if not models:
            raise FileNotFoundError(f"No model files found in {args.checkpoint}")
        # Sort numerically by extracting the number from model_XXX.pt
        latest = sorted(models, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
        checkpoint_path = os.path.join(args.checkpoint, latest)
        cfg_path = os.path.join(args.checkpoint, "cfgs.pkl")
    else:
        checkpoint_path = args.checkpoint
        cfg_path = os.path.join(os.path.dirname(args.checkpoint), "cfgs.pkl")

    print(f"Loading checkpoint: {checkpoint_path}")

    # Load config
    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(f)

    # Initialize Genesis
    gs.init(backend=gs.gpu if not args.cpu else gs.cpu, precision="32", logging_level="warning")

    # Create environment with viewer
    print("Creating environment...")
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Load policy
    print("Loading policy...")
    policy = load_policy(checkpoint_path, obs_cfg["num_obs"], env_cfg["num_actions"])

    # Run evaluation
    print(f"\nRunning evaluation for {args.episodes} episodes...")
    print("Press Ctrl+C to stop\n")

    episode_rewards = []
    episode_lengths = []

    try:
        obs, _ = env.reset()

        for episode in range(args.episodes):
            episode_reward = 0
            episode_length = 0
            done = False

            while not done:
                # Get action from policy
                with torch.no_grad():
                    actions = policy.act_inference(obs)

                # Step environment
                obs, reward, reset, extras = env.step(actions)

                episode_reward += reward.item()
                episode_length += 1
                done = reset.item()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Length = {episode_length}")

            # Reset for next episode
            obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user")

    # Print summary
    if episode_rewards:
        print(f"\n{'='*40}")
        print("Evaluation Summary")
        print(f"{'='*40}")
        print(f"Episodes completed: {len(episode_rewards)}")
        print(f"Average reward: {sum(episode_rewards) / len(episode_rewards):.2f}")
        print(f"Average length: {sum(episode_lengths) / len(episode_lengths):.0f}")
        print(f"Max reward: {max(episode_rewards):.2f}")
        print(f"Min reward: {min(episode_rewards):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Go2 models")
    parser.add_argument("--checkpoint", "-c", type=str, required=True,
                        help="Path to checkpoint file or directory")
    parser.add_argument("--episodes", "-e", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
