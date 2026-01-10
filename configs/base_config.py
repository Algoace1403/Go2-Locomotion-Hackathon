"""
Base configuration for Go2 Locomotion Training
Contains shared hyperparameters across all levels
"""

import numpy as np


class BaseConfig:
    """Base configuration class with shared parameters"""

    # ============== Simulation ==============
    dt = 0.02                    # 50 Hz control frequency
    substeps = 2                 # Physics substeps per control step
    max_episode_length = 1000    # 20 seconds default

    # ============== Robot ==============
    num_joints = 12              # 4 legs × 3 joints

    # Default standing pose (radians)
    default_dof_pos = np.array([
        0.0,  0.8, -1.5,   # FL: hip, thigh, calf
        0.0,  0.8, -1.5,   # FR
        0.0,  1.0, -1.5,   # RL
        0.0,  1.0, -1.5,   # RR
    ])

    # Joint limits
    joint_limits = {
        'hip': (-0.8, 0.8),
        'thigh': (-1.0, 2.5),
        'calf': (-2.7, -0.9),
    }

    # PD Controller gains
    kp = 20.0                    # Proportional gain
    kd = 0.5                     # Derivative gain

    # ============== Actions ==============
    action_scale = 0.25          # Scale factor for actions
    clip_actions = 1.0           # Clip to [-1, 1]

    # ============== Observations ==============
    num_observations = 45        # Base observation dimension

    obs_scales = {
        'ang_vel': 0.25,
        'dof_pos': 1.0,
        'dof_vel': 0.05,
    }

    command_scales = {
        'lin_vel': 2.0,
        'ang_vel': 0.25,
    }

    # ============== Termination ==============
    termination = {
        'max_roll': 0.5,         # radians
        'max_pitch': 0.5,        # radians
        'min_height': 0.2,       # meters
    }

    # ============== PPO Training ==============
    ppo = {
        'clip_param': 0.2,
        'entropy_coef': 0.01,
        'value_loss_coef': 1.0,
        'max_grad_norm': 1.0,
        'learning_rate': 1e-3,
        'gamma': 0.99,
        'lam': 0.95,
        'num_learning_epochs': 5,
        'num_mini_batches': 4,
        'schedule': 'adaptive',
        'desired_kl': 0.01,
    }

    # ============== Network ==============
    network = {
        'actor_hidden_dims': [512, 256, 128],
        'critic_hidden_dims': [512, 256, 128],
        'activation': 'elu',
        'init_noise_std': 1.0,
    }

    # ============== Training ==============
    num_envs = 4096              # Parallel environments
    num_steps_per_env = 24       # Steps before update
    max_iterations = 500         # Training iterations
    save_interval = 100          # Save checkpoint every N iterations

    # ============== Logging ==============
    log_dir = 'logs'
    model_dir = 'models'
