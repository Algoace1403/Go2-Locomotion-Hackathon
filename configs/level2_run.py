"""
Level 2: Running Configuration
Target: ≥2.0 m/s forward velocity
Points: 40 (CRITICAL - highest points)
"""

from .base_config import BaseConfig
import numpy as np


class RunConfig(BaseConfig):
    """Configuration for Level 2: Running"""

    # ============== Task Specific ==============
    level_name = "level2_run"
    task_description = "High-speed running at ≥2.0 m/s"

    # Target velocity
    target_vel = 2.5             # Aim higher than minimum
    min_vel = 2.0                # Hackathon requirement

    # Episode length
    max_episode_length = 1000    # 20s at 50Hz

    # ============== Command Ranges ==============
    command_ranges = {
        'lin_vel_x': (2.0, 3.0),   # High forward velocity
        'lin_vel_y': (-0.1, 0.1),  # Minimal lateral
        'ang_vel': (-0.1, 0.1),    # Minimal rotation
    }

    # ============== Reward Weights ==============
    reward_weights = {
        # Primary objectives (higher weights for speed)
        'forward_velocity': 3.0,
        'velocity_bonus': 5.0,       # Bonus for exceeding 2.0 m/s
        'velocity_tracking': 1.0,

        # Penalties (relaxed for running dynamics)
        'lateral_velocity': -0.3,
        'angular_velocity': -0.3,

        # Stability (relaxed)
        'orientation_stability': 0.5,
        'base_height': 0.3,

        # Gait quality
        'foot_clearance': 0.5,
        'gait_frequency': 0.5,

        # Smoothness (relaxed for dynamic motion)
        'action_smoothness': -0.005,

        # Survival (lower - we want risk-taking)
        'survival': 0.1,
    }

    # ============== Reward Parameters ==============
    reward_params = {
        'target_height': 0.32,           # Slightly lower for running
        'tracking_sigma': 0.5,           # Wider tolerance
        'min_vel_threshold': 2.0,        # Bonus threshold
    }

    # ============== Termination (More Lenient) ==============
    termination = {
        'max_roll': 0.7,                 # ~40 degrees
        'max_pitch': 0.7,
        'min_height': 0.15,              # Lower threshold
    }

    # ============== Curriculum ==============
    curriculum = {
        'phase1': {
            'target_vel': 1.0,
            'iterations': 100,
            'description': 'Fast walk / slow jog'
        },
        'phase2': {
            'target_vel': 1.5,
            'iterations': 150,
            'description': 'Transition to running'
        },
        'phase3': {
            'target_vel': 2.0,
            'iterations': 200,
            'description': 'Reach minimum requirement'
        },
        'phase4': {
            'target_vel': 2.5,
            'iterations': 150,
            'description': 'Push for higher speed'
        },
    }

    # ============== Training ==============
    max_iterations = 600         # More iterations for harder task
    save_interval = 50
