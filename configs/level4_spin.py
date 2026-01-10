"""
Level 4: Spin in Place Configuration (BONUS)
Target: Pure rotation with high angular velocity, zero forward movement
Points: 10 (Bonus)
Metric: High ω with v=0
"""

from .base_config import BaseConfig
import numpy as np


class SpinConfig(BaseConfig):
    """Configuration for Level 4: Spin in Place"""

    # ============== Task Specific ==============
    level_name = "level4_spin"
    task_description = "Spin in place with high angular velocity, zero linear movement"

    # Spin parameters
    target_angular_vel = 2.0     # rad/s target
    max_linear_vel = 0.1         # Maximum allowed linear velocity
    max_drift = 0.2              # Maximum allowed position drift (meters)

    # Episode length
    max_episode_length = 1000    # 20s at 50Hz

    # Extended observations
    num_observations = 47        # Base 45 + start_position(2)

    # ============== Command Ranges ==============
    command_ranges = {
        'lin_vel_x': (0.0, 0.0),     # No forward movement
        'lin_vel_y': (0.0, 0.0),     # No lateral movement
        'ang_vel': (1.5, 2.5),       # High rotation
    }

    # ============== Reward Weights ==============
    reward_weights = {
        # Primary objectives
        'angular_velocity': 5.0,      # Spin fast
        'angular_tracking': 2.0,      # Track target ω

        # Critical penalties
        'zero_linear_velocity': -10.0,  # No forward/lateral movement
        'position_drift': -5.0,         # Stay in place

        # Stability
        'orientation_stability': 1.0,

        # Motion quality
        'symmetric_motion': 0.5,      # Legs move symmetrically

        # Smoothness (relaxed for dynamic motion)
        'action_smoothness': -0.005,

        # Survival
        'survival': 0.2,
    }

    # ============== Reward Parameters ==============
    reward_params = {
        'angular_sigma': 0.5,        # Tolerance for angular velocity
        'linear_penalty_scale': 2.0, # How harshly to penalize linear motion
    }

    # ============== Termination ==============
    termination = {
        'max_roll': 0.6,
        'max_pitch': 0.6,
        'min_height': 0.2,
        'max_drift': 0.5,            # Fail if drifted too far
    }

    # ============== Curriculum ==============
    curriculum = {
        'phase1': {
            'target_angular_vel': 0.5,
            'iterations': 100,
            'description': 'Slow rotation'
        },
        'phase2': {
            'target_angular_vel': 1.0,
            'iterations': 150,
            'description': 'Medium rotation'
        },
        'phase3': {
            'target_angular_vel': 1.5,
            'iterations': 150,
            'description': 'Fast rotation'
        },
        'phase4': {
            'target_angular_vel': 2.0,
            'iterations': 100,
            'description': 'Target rotation speed'
        },
    }

    # ============== Training ==============
    max_iterations = 500
    save_interval = 50
