"""
Separate Simulations Demo: Each behavior runs in its own simulation for 30 seconds.

- Walk: 30 seconds (separate simulation)
- Run: 30 seconds (separate simulation)
- Jump: 30 seconds (separate simulation)
- Spin: 30 seconds (separate simulation)

Usage:
    python demo_separate.py
    python demo_separate.py --duration 60  # Run each for 60 seconds
"""

import argparse
import os
import pickle
import torch
import time
import genesis as gs
from go2_env import Go2Env


def find_latest_model(logs_dir, mode):
    """Find the latest trained model for a given mode"""
    if not os.path.exists(logs_dir):
        return None, None

    mode_dirs = [d for d in os.listdir(logs_dir) if d.startswith(f"go2_{mode}_")]
    if not mode_dirs:
        return None, None

    latest_dir = sorted(mode_dirs)[-1]
    model_dir = os.path.join(logs_dir, latest_dir)

    models = [f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pt")]
    if not models:
        return None, None

    latest_model = sorted(models, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]

    return os.path.join(model_dir, latest_model), os.path.join(model_dir, "cfgs.pkl")


def load_policy(checkpoint_path, num_obs, num_actions):
    """Load a trained policy"""
    from rsl_rl.modules import ActorCritic

    policy = ActorCritic(
        num_obs,
        num_obs,
        num_actions,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
    )

    checkpoint = torch.load(checkpoint_path, map_location=gs.device, weights_only=False)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(gs.device)
    policy.eval()

    return policy


def run_single_mode(mode, model_path, cfg_path, duration, use_cpu):
    """Run a single mode in its own simulation"""

    print(f"\n{'='*60}")
    print(f"  {mode.upper()} SIMULATION - {duration} seconds")
    print(f"{'='*60}")
    print(f"Model: {model_path}\n")

    # Load config
    with open(cfg_path, "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(f)

    # Create fresh environment with viewer
    print("Creating simulation environment...")
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
    policy = load_policy(model_path, obs_cfg["num_obs"], env_cfg["num_actions"])

    # Set commands based on mode
    dt = 0.02
    total_steps = int(duration / dt)

    print(f"\nRunning {mode} for {duration} seconds ({total_steps} steps)...")
    print("Press Ctrl+C to skip to next mode\n")

    try:
        obs, _ = env.reset()

        # Set appropriate commands
        if mode == "walk":
            env.commands[:, 0] = 0.6   # Forward velocity (m/s)
            env.commands[:, 1] = 0.0   # Lateral
            env.commands[:, 2] = 0.0   # Angular
        elif mode == "run":
            env.commands[:, 0] = 2.0   # Fast forward (m/s)
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
        elif mode == "jump":
            env.commands[:, 0] = 0.2   # Slight forward
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
        elif mode == "spin":
            env.commands[:, 0] = 0.0   # No forward
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 2.5   # Spin (rad/s)

        start_time = time.time()
        step = 0
        resets = 0

        while step < total_steps:
            with torch.no_grad():
                actions = policy.act_inference(obs)

            obs, reward, reset, extras = env.step(actions)
            step += 1

            # Reset if fallen
            if reset.item():
                obs, _ = env.reset()
                # Restore commands after reset
                if mode == "walk":
                    env.commands[:, 0] = 0.6
                elif mode == "run":
                    env.commands[:, 0] = 2.0
                elif mode == "jump":
                    env.commands[:, 0] = 0.2
                elif mode == "spin":
                    env.commands[:, 2] = 2.5
                resets += 1

            # Progress update every 5 seconds
            if step % int(5 / dt) == 0:
                elapsed = time.time() - start_time
                remaining = duration - elapsed
                print(f"  [{mode.upper()}] {elapsed:.0f}s elapsed, {remaining:.0f}s remaining...")

        elapsed = time.time() - start_time
        print(f"\n{mode.upper()} completed in {elapsed:.1f}s (resets: {resets})")

    except KeyboardInterrupt:
        print(f"\n{mode.upper()} skipped by user")

    # Clean up the scene
    print("Closing simulation...")
    del env
    del policy


def main():
    parser = argparse.ArgumentParser(description="Separate Simulations Demo")
    parser.add_argument("--duration", "-d", type=int, default=30,
                        help="Duration for each mode in seconds (default: 30)")
    parser.add_argument("--modes", "-m", type=str, default="walk,run,jump,spin",
                        help="Comma-separated list of modes to run (default: walk,run,jump,spin)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    modes_to_run = [m.strip() for m in args.modes.split(",")]

    print("\n" + "=" * 60)
    print("  SEPARATE SIMULATIONS DEMO")
    print("=" * 60)
    print(f"\nModes: {', '.join(modes_to_run)}")
    print(f"Duration per mode: {args.duration} seconds")
    print(f"Total time: {len(modes_to_run) * args.duration} seconds")

    # Find all models first
    model_info = {}
    print("\nFinding trained models...")

    for mode in modes_to_run:
        model_path, cfg_path = find_latest_model("logs", mode)
        if model_path:
            model_info[mode] = (model_path, cfg_path)
            print(f"  {mode.upper()}: Found")
        else:
            print(f"  {mode.upper()}: NOT FOUND - will skip")

    available_modes = [m for m in modes_to_run if m in model_info]

    if not available_modes:
        print("\nNo trained models found!")
        print("Train models first with: python train.py --mode <mode>")
        return

    print(f"\nWill run: {', '.join(available_modes)}")
    input("\nPress Enter to start the demo...")

    # Initialize Genesis once
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    # Run each mode in its own simulation
    for mode in available_modes:
        model_path, cfg_path = model_info[mode]
        run_single_mode(mode, model_path, cfg_path, args.duration, args.cpu)

        if mode != available_modes[-1]:
            print("\n" + "-" * 40)
            print("Starting next simulation in 2 seconds...")
            print("-" * 40)
            time.sleep(2)

    print("\n" + "=" * 60)
    print("  ALL SIMULATIONS COMPLETE!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
