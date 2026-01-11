"""
Unified Training Script for Go2 Robot - Walk, Run, Jump

Usage:
    python train.py --mode walk       # Train walking (stable, 0.5 m/s)
    python train.py --mode run        # Train running (fast, 2.0+ m/s)
    python train.py --mode jump       # Train jumping
    python train.py --mode all        # Train all behaviors sequentially

    Add --viewer to see visualization
    Add --resume to continue from checkpoint
"""

import argparse
import os
import pickle
import shutil
from datetime import datetime

# Check for correct rsl-rl version
try:
    from importlib import metadata
    try:
        if metadata.version("rsl-rl"):
            raise ImportError("Please uninstall 'rsl-rl' and install 'rsl-rl-lib==2.2.4'")
    except metadata.PackageNotFoundError:
        pass
    try:
        version = metadata.version("rsl-rl-lib")
        if version != "2.2.4":
            print(f"Warning: rsl-rl-lib version {version} found, expected 2.2.4")
    except metadata.PackageNotFoundError:
        raise ImportError("Please install 'rsl-rl-lib==2.2.4': pip install rsl-rl-lib==2.2.4")
except Exception as e:
    print(f"Warning: {e}")

from rsl_rl.runners import OnPolicyRunner
import genesis as gs
from go2_env import Go2Env


# ==================== CONFIGURATION ====================

def get_base_env_cfg():
    """Base environment configuration shared across all modes"""
    return {
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        "kp": 20.0,
        "kd": 0.5,
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        "termination_if_roll_greater_than": 30,
        "termination_if_pitch_greater_than": 30,
        "termination_if_height_lower_than": 0.15,
    }


def get_base_obs_cfg():
    """Base observation configuration"""
    return {
        "num_obs": 45,
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "commands": 1.0,
        },
    }


def get_walk_cfg():
    """Configuration for walking mode"""
    env_cfg = get_base_env_cfg()
    env_cfg["mode"] = "walk"

    obs_cfg = get_base_obs_cfg()

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.30,
        "reward_scales": {
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -2.0,
            "ang_vel_xy": -0.05,
            "orientation": -1.0,
            "base_height": -30.0,
            "action_rate": -0.01,
            "similar_to_default": -0.1,
            "dof_acc": -2.5e-7,
            "survival": 0.5,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.3, 0.8],   # Walk speed
        "lin_vel_y_range": [-0.3, 0.3],  # Lateral movement
        "ang_vel_range": [-0.5, 0.5],    # Turning
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_run_cfg():
    """Configuration for running mode"""
    env_cfg = get_base_env_cfg()
    env_cfg["mode"] = "run"
    env_cfg["termination_if_roll_greater_than"] = 45  # More lenient for running
    env_cfg["termination_if_pitch_greater_than"] = 45

    obs_cfg = get_base_obs_cfg()

    reward_cfg = {
        "tracking_sigma": 0.5,  # Wider sigma for higher speeds
        "base_height_target": 0.28,  # Slightly lower for running stance
        "reward_scales": {
            "tracking_lin_vel": 2.0,
            "tracking_ang_vel": 0.3,
            "forward_vel": 1.0,        # Extra reward for forward speed
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.02,
            "orientation": -0.5,
            "base_height": -20.0,
            "action_rate": -0.005,
            "similar_to_default": -0.05,
            "energy": -0.0001,          # Encourage efficient running
            "survival": 1.0,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [1.0, 2.5],   # Run speed
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_jump_cfg():
    """Configuration for jumping mode"""
    env_cfg = get_base_env_cfg()
    env_cfg["mode"] = "jump"
    env_cfg["episode_length_s"] = 10.0  # Shorter episodes for jumping
    env_cfg["termination_if_roll_greater_than"] = 60
    env_cfg["termination_if_pitch_greater_than"] = 60
    env_cfg["termination_if_height_lower_than"] = 0.1

    obs_cfg = get_base_obs_cfg()

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.5,
        "jump_height_target": 0.6,
        "jump_height_threshold": 0.4,
        "reward_scales": {
            "jump_height": 5.0,
            "air_time": 2.0,
            "jump_upward_vel": 3.0,
            "land_stable": 2.0,
            "orientation": -0.5,
            "action_rate": -0.005,
            "similar_to_default": -0.02,
            "survival": 0.5,
            "feet_contact_forces": -0.01,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 0.5],   # Some forward motion
        "lin_vel_y_range": [0.0, 0.0],
        "ang_vel_range": [0.0, 0.0],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_spin_cfg():
    """Configuration for spinning mode"""
    env_cfg = get_base_env_cfg()
    env_cfg["mode"] = "spin"
    env_cfg["episode_length_s"] = 15.0
    env_cfg["termination_if_roll_greater_than"] = 45
    env_cfg["termination_if_pitch_greater_than"] = 45

    obs_cfg = get_base_obs_cfg()

    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.30,
        "reward_scales": {
            "spin_vel": 3.0,              # Reward spinning fast
            "spin_stability": 2.0,         # Stay upright
            "no_lateral_vel": -5.0,        # Don't move sideways
            "base_height": -30.0,          # Maintain height
            "orientation": -1.0,           # Stay level
            "action_rate": -0.01,
            "survival": 0.5,
        },
    }

    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.0, 0.0],    # No forward motion
        "lin_vel_y_range": [0.0, 0.0],    # No lateral motion
        "ang_vel_range": [2.0, 3.0],      # Spin command (rad/s)
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def get_train_cfg(exp_name, max_iterations, resume=False, resume_path=None):
    """Training configuration"""
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": resume,
            "resume_path": resume_path,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 50,
        "empirical_normalization": None,
        "seed": 1,
    }


# ==================== TRAINING ====================

def train_mode(mode, args):
    """Train a specific mode (walk, run, or jump)"""
    print(f"\n{'='*60}")
    print(f"Training Mode: {mode.upper()}")
    print(f"{'='*60}\n")

    # Get configuration for this mode
    if mode == "walk":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_walk_cfg()
    elif mode == "run":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_run_cfg()
    elif mode == "jump":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_jump_cfg()
    elif mode == "spin":
        env_cfg, obs_cfg, reward_cfg, command_cfg = get_spin_cfg()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # Setup experiment name and directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"go2_{mode}_{timestamp}"
    log_dir = f"logs/{exp_name}"

    # Handle resume
    resume_path = None
    if args.resume:
        # Find latest checkpoint for this mode
        mode_logs = [d for d in os.listdir("logs") if d.startswith(f"go2_{mode}_")]
        if mode_logs:
            latest = sorted(mode_logs)[-1]
            resume_path = f"logs/{latest}"
            print(f"Resuming from: {resume_path}")

    # Create log directory
    if not args.resume:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = resume_path

    # Save configurations
    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    # Get training config
    train_cfg = get_train_cfg(
        exp_name,
        args.max_iterations,
        resume=args.resume,
        resume_path=resume_path
    )

    # Create environment
    print(f"Creating environment with {args.num_envs} parallel instances...")
    env = Go2Env(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.viewer,
    )

    # Create runner and train
    print("Starting training...")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)

    print(f"\nTraining complete! Model saved to: {log_dir}")
    return log_dir


def main():
    parser = argparse.ArgumentParser(description="Train Go2 Robot - Walk, Run, Jump")
    parser.add_argument("--mode", type=str, default="walk",
                        choices=["walk", "run", "jump", "spin", "all"],
                        help="Training mode")
    parser.add_argument("--num_envs", "-n", type=int, default=4096,
                        help="Number of parallel environments")
    parser.add_argument("--max_iterations", "-i", type=int, default=300,
                        help="Maximum training iterations")
    parser.add_argument("--viewer", action="store_true",
                        help="Show visualization")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--cpu", action="store_true",
                        help="Use CPU instead of GPU")
    args = parser.parse_args()

    # Initialize Genesis
    backend = gs.cpu if args.cpu else gs.gpu
    print(f"Initializing Genesis with {backend} backend...")
    gs.init(backend=backend, precision="32", logging_level="warning")

    # Train
    if args.mode == "all":
        # Train all modes sequentially
        for mode in ["walk", "run", "jump", "spin"]:
            train_mode(mode, args)
    else:
        train_mode(args.mode, args)

    print("\nAll training complete!")


if __name__ == "__main__":
    main()
