"""
Level 1: Walking Configuration
Target: 0.5-1.0 m/s forward velocity, max distance in 60s
Points: 20
"""

from .base_config import BaseConfig
import numpy as np


class WalkConfig(BaseConfig):
    """Configuration for Level 1: Walking"""

    # ============== Task Specific ==============
    level_name = "level1_walk"
    task_description = "Stable walking at 0.5-1.0 m/s"

    # Target velocity range
    target_vel_min = 0.5         # m/s
    target_vel_max = 1.0         # m/s
    target_vel = 0.75            # Optimal target

    # Episode length (60 seconds for evaluation)
    max_episode_length = 3000    # 60s at 50Hz

    # ============== Command Ranges ==============
    command_ranges = {
        'lin_vel_x': (0.5, 1.0),   # Forward velocity
        'lin_vel_y': (-0.0, 0.0),  # No lateral movement
        'ang_vel': (-0.0, 0.0),    # No rotation
    }

    # ============== Reward Weights ==============
    reward_weights = {
        # Primary objectives
        'forward_velocity': 2.0,
        'velocity_tracking': 1.5,

        # Penalties
        'lateral_velocity': -0.5,
        'angular_velocity': -0.5,

        # Stability
        'orientation_stability': 1.0,
        'base_height': 0.5,

        # Smoothness & efficiency
        'action_smoothness': -0.01,
        'joint_torque': -0.0001,

        # Survival
        'survival': 0.2,
    }

    # ============== Reward Parameters ==============
    reward_params = {
        'target_height': 0.34,           # Standing height
        'tracking_sigma': 0.25,          # Gaussian width for velocity tracking
        'height_sigma': 0.1,             # Gaussian width for height
    }

    # ============== Termination ==============
    termination = {
        'max_roll': 0.5,                 # ~28 degrees
        'max_pitch': 0.5,
        'min_height': 0.2,
    }

    # ============== Curriculum ==============
    curriculum = {
        'phase1': {
            'target_vel': 0.3,
            'iterations': 100,
            'description': 'Learn to stand and take steps'
        },
        'phase2': {
            'target_vel': 0.5,
            'iterations': 150,
            'description': 'Increase to minimum speed'
        },
        'phase3': {
            'target_vel': 0.75,
            'iterations': 200,
            'description': 'Reach optimal walking speed'
        },
    }

    # ============== Training ==============
    max_iterations = 500
    save_interval = 50
