"""
Level 3: Circle Walking Configuration (BONUS)
Target: Walk in a perfect circle with R=2m
Points: 10 (Bonus)
Metric: Average RMSE for 10 rounds
"""

from .base_config import BaseConfig
import numpy as np


class CircleConfig(BaseConfig):
    """Configuration for Level 3: Circle Walking"""

    # ============== Task Specific ==============
    level_name = "level3_circle"
    task_description = "Walk in a circle of radius 2m"

    # Circle parameters
    circle_radius = 2.0          # meters
    circle_center = np.array([0.0, 0.0])
    target_speed = 0.5           # Walking speed on circle
    num_laps = 10                # Evaluation laps

    # Episode length (enough for 10 laps)
    # Circumference = 2π × 2m ≈ 12.56m
    # At 0.5 m/s, one lap ≈ 25s
    # 10 laps ≈ 250s = 12500 steps at 50Hz
    max_episode_length = 15000   # 300s buffer

    # Extended observations for circle tracking
    num_observations = 49        # Base 45 + position(2) + heading(1) + angular_error(1)

    # ============== Command Ranges ==============
    # For circle, commands are computed dynamically based on position
    command_ranges = {
        'lin_vel_x': (0.3, 0.7),
        'lin_vel_y': (-0.3, 0.3),
        'ang_vel': (0.1, 0.5),    # Constant turning rate for circle
    }

    # ============== Reward Weights ==============
    reward_weights = {
        # Circle tracking (primary)
        'radius_tracking': 3.0,
        'tangent_velocity': 2.0,
        'heading_alignment': 1.5,

        # Motion quality
        'angular_velocity': 1.0,
        'radial_velocity': -2.0,     # Penalize moving toward/away from center
        'forward_velocity': 1.0,

        # Stability
        'orientation_stability': 0.5,

        # Smoothness
        'action_smoothness': -0.01,

        # Survival
        'survival': 0.1,
    }

    # ============== Reward Parameters ==============
    reward_params = {
        'radius_sigma': 0.3,         # Tolerance for radius error
        'heading_sigma': 0.3,        # Tolerance for heading error
        'angular_vel_sigma': 0.2,    # Tolerance for yaw rate
    }

    # ============== Termination ==============
    termination = {
        'max_roll': 0.5,
        'max_pitch': 0.5,
        'min_height': 0.2,
        'max_radius_error': 1.5,     # Fail if too far from circle
    }

    # ============== Curriculum ==============
    curriculum = {
        'phase1': {
            'radius': 3.0,           # Start with larger circle (easier)
            'speed': 0.3,
            'iterations': 150,
            'description': 'Large circle, slow speed'
        },
        'phase2': {
            'radius': 2.5,
            'speed': 0.4,
            'iterations': 150,
            'description': 'Medium circle'
        },
        'phase3': {
            'radius': 2.0,
            'speed': 0.5,
            'iterations': 200,
            'description': 'Target circle'
        },
    }

    # ============== Training ==============
    max_iterations = 500
    save_interval = 50
