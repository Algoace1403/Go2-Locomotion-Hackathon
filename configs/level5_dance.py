"""
Level 5: Dance Configuration (BONUS SHOWOFF!)
Target: Rhythmic, coordinated movements with style
This is for presentation WOW factor!
"""

from .base_config import BaseConfig
import numpy as np


class DanceConfig(BaseConfig):
    """Configuration for Level 5: Dancing"""

    # ============== Task Specific ==============
    level_name = "level5_dance"
    task_description = "Rhythmic dancing with coordinated leg movements"

    # Dance parameters
    dance_frequency = 2.0        # Hz - beats per second
    dance_amplitude = 0.3        # Amplitude of movements
    beat_period = 0.5            # seconds per beat

    # Dance styles available
    dance_styles = ['groove', 'bounce', 'wave', 'shuffle', 'mixed']
    default_style = 'mixed'

    # Episode length
    max_episode_length = 1000    # 20s dance routine

    # Extended observations for rhythm tracking
    num_observations = 48        # Base 45 + beat_phase(1) + style_encoding(2)

    # ============== Command Ranges ==============
    command_ranges = {
        'lin_vel_x': (-0.2, 0.2),    # Slight forward/back
        'lin_vel_y': (-0.2, 0.2),    # Slight side-to-side
        'ang_vel': (-0.5, 0.5),      # Some rotation for style
    }

    # ============== Reward Weights ==============
    reward_weights = {
        # Rhythm and coordination
        'rhythm_sync': 3.0,           # Move to the beat!
        'leg_coordination': 2.5,      # Coordinated leg movements
        'symmetry': 2.0,              # Symmetric motions look good

        # Style points
        'height_variation': 1.5,      # Bouncing/grooving
        'body_roll': 1.0,             # Stylish body movement
        'pose_variety': 1.5,          # Don't be static

        # Stability (still need to stay up!)
        'orientation_stability': 1.0,
        'base_height': 0.5,

        # Smoothness
        'action_smoothness': -0.005,  # Fluid movements

        # Survival
        'survival': 0.2,
    }

    # ============== Reward Parameters ==============
    reward_params = {
        'target_height': 0.34,
        'height_variance_target': 0.05,  # Target bounce amplitude
        'rhythm_sigma': 0.1,             # Timing tolerance
    }

    # ============== Termination (Lenient for dancing) ==============
    termination = {
        'max_roll': 0.8,           # Allow more body movement
        'max_pitch': 0.8,
        'min_height': 0.15,        # Can crouch low
    }

    # ============== Dance Patterns ==============
    # Pre-defined movement patterns for curriculum
    patterns = {
        'groove': {
            'description': 'Side-to-side weight shifting',
            'frequency': 2.0,
            'amplitude': 0.2,
        },
        'bounce': {
            'description': 'Up-down bouncing motion',
            'frequency': 2.5,
            'amplitude': 0.15,
        },
        'wave': {
            'description': 'Wave motion through legs',
            'frequency': 1.5,
            'amplitude': 0.25,
        },
        'shuffle': {
            'description': 'Quick foot movements',
            'frequency': 3.0,
            'amplitude': 0.1,
        },
    }

    # ============== Curriculum ==============
    curriculum = {
        'phase1': {
            'style': 'bounce',
            'frequency': 1.5,
            'iterations': 150,
            'description': 'Learn basic bouncing'
        },
        'phase2': {
            'style': 'groove',
            'frequency': 2.0,
            'iterations': 150,
            'description': 'Add side-to-side groove'
        },
        'phase3': {
            'style': 'wave',
            'frequency': 2.0,
            'iterations': 150,
            'description': 'Learn leg waves'
        },
        'phase4': {
            'style': 'mixed',
            'frequency': 2.5,
            'iterations': 150,
            'description': 'Combine all moves!'
        },
    }

    # ============== Training ==============
    max_iterations = 600
    save_interval = 50
