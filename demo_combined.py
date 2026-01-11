"""
Combined Demo: Walk -> Run -> Jump -> Spin
Each behavior runs for 7 seconds in sequence.

Usage:
    python demo_combined.py
    python demo_combined.py --walk_model logs/go2_walk_xxx --run_model logs/go2_run_xxx ...
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

    # Find latest model file
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


def run_demo(args):
    """Run the combined demo"""

    # Duration for each behavior (seconds)
    duration_per_mode = 7.0
    dt = 0.02  # 50Hz
    steps_per_mode = int(duration_per_mode / dt)

    # Find models
    modes = ["walk", "run", "jump", "spin"]
    model_paths = {}
    cfg_paths = {}

    print("=" * 60)
    print("COMBINED DEMO: Walk -> Run -> Jump -> Spin")
    print("=" * 60)
    print(f"\nEach behavior runs for {duration_per_mode} seconds\n")

    # Check for models
    for mode in modes:
        model_arg = getattr(args, f"{mode}_model", None)
        if model_arg and os.path.exists(model_arg):
            if os.path.isdir(model_arg):
                models = [f for f in os.listdir(model_arg) if f.startswith("model_") and f.endswith(".pt")]
                latest = sorted(models, key=lambda x: int(x.split("_")[1].split(".")[0]))[-1]
                model_paths[mode] = os.path.join(model_arg, latest)
                cfg_paths[mode] = os.path.join(model_arg, "cfgs.pkl")
            else:
                model_paths[mode] = model_arg
                cfg_paths[mode] = os.path.join(os.path.dirname(model_arg), "cfgs.pkl")
        else:
            # Auto-find latest model
            model_path, cfg_path = find_latest_model("logs", mode)
            if model_path:
                model_paths[mode] = model_path
                cfg_paths[mode] = cfg_path

    # Report found models
    for mode in modes:
        if mode in model_paths:
            print(f"  {mode.upper()}: {model_paths[mode]}")
        else:
            print(f"  {mode.upper()}: NOT FOUND - will skip")

    available_modes = [m for m in modes if m in model_paths]

    if not available_modes:
        print("\nNo trained models found! Please train at least one model first.")
        print("Run: python train.py --mode walk")
        return

    print(f"\nRunning demo with modes: {', '.join(available_modes)}")
    print("-" * 60)

    # Initialize Genesis
    gs.init(backend=gs.gpu if not args.cpu else gs.cpu, precision="32", logging_level="warning")

    # Load first config to create environment
    first_mode = available_modes[0]
    with open(cfg_paths[first_mode], "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg = pickle.load(f)

    # Create environment with viewer
    print("\nCreating environment...")
    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=True,
    )

    # Load all policies
    print("Loading policies...")
    policies = {}
    for mode in available_modes:
        print(f"  Loading {mode} policy...")
        policies[mode] = load_policy(model_paths[mode], obs_cfg["num_obs"], env_cfg["num_actions"])

    # Run demo
    print("\n" + "=" * 60)
    print("STARTING DEMO - Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    try:
        obs, _ = env.reset()

        while True:  # Loop the demo
            for mode in available_modes:
                print(f"\n>>> {mode.upper()} for {duration_per_mode} seconds...")

                policy = policies[mode]
                start_time = time.time()
                step = 0

                # Set appropriate commands based on mode
                if mode == "walk":
                    env.commands[:, 0] = 0.5   # Forward velocity
                    env.commands[:, 1] = 0.0   # Lateral
                    env.commands[:, 2] = 0.0   # Angular
                elif mode == "run":
                    env.commands[:, 0] = 2.0   # Fast forward
                    env.commands[:, 1] = 0.0
                    env.commands[:, 2] = 0.0
                elif mode == "jump":
                    env.commands[:, 0] = 0.3   # Slight forward
                    env.commands[:, 1] = 0.0
                    env.commands[:, 2] = 0.0
                elif mode == "spin":
                    env.commands[:, 0] = 0.0   # No forward
                    env.commands[:, 1] = 0.0
                    env.commands[:, 2] = 2.5   # Spin fast

                while step < steps_per_mode:
                    with torch.no_grad():
                        actions = policy.act_inference(obs)

                    obs, reward, reset, extras = env.step(actions)
                    step += 1

                    # Reset if fallen
                    if reset.item():
                        obs, _ = env.reset()

                elapsed = time.time() - start_time
                print(f"    Completed in {elapsed:.1f}s")

            print("\n" + "-" * 40)
            print("Demo cycle complete! Restarting...")
            print("-" * 40)
            obs, _ = env.reset()

    except KeyboardInterrupt:
        print("\n\nDemo stopped by user.")


def main():
    parser = argparse.ArgumentParser(description="Combined Demo: Walk, Run, Jump, Spin")
    parser.add_argument("--walk_model", type=str, help="Path to walk model directory")
    parser.add_argument("--run_model", type=str, help="Path to run model directory")
    parser.add_argument("--jump_model", type=str, help="Path to jump model directory")
    parser.add_argument("--spin_model", type=str, help="Path to spin model directory")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    args = parser.parse_args()

    run_demo(args)


if __name__ == "__main__":
    main()
