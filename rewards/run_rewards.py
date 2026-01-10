"""
Level 2: Running Reward Functions
Target: ≥2.0 m/s forward velocity
Points: 40 (CRITICAL - highest points)
"""

import torch
from typing import Tuple, List, Callable
from .base_rewards import BaseRewards


class RunRewards(BaseRewards):
    """
    Reward functions for Level 2: Running
    Focus: High speed with relaxed stability constraints
    """

    def __init__(self, env):
        super().__init__(env)

        # Running-specific parameters
        self.target_vel = getattr(env.config, 'target_vel', 2.5)
        self.min_vel = getattr(env.config, 'min_vel', 2.0)
        self.target_height = getattr(
            env.config, 'reward_params', {}
        ).get('target_height', 0.32)

    def get_reward_functions(self) -> List[Callable]:
        """Return list of reward functions for running"""
        return [
            self.reward_forward_velocity,
            self.reward_velocity_bonus,
            self.reward_velocity_tracking,
            self.reward_lateral_penalty,
            self.reward_angular_penalty,
            self.reward_orientation,
            self.reward_height,
            self.reward_action_smoothness,
            self.reward_survival,
        ]

    # ============== Running-Specific Rewards ==============

    def reward_forward_velocity(self) -> Tuple[torch.Tensor, float]:
        """
        Primary reward: Forward velocity
        Higher weight than walking to encourage speed
        """
        vel_x = self.env._get_base_lin_vel()[:, 0]

        # No upper clipping - we want maximum speed
        reward = torch.clamp(vel_x, -0.5, 5.0)

        weight = self.env.config.reward_weights.get('forward_velocity', 3.0)
        return reward, weight

    def reward_velocity_bonus(self) -> Tuple[torch.Tensor, float]:
        """
        CRITICAL: Bonus reward for exceeding 2.0 m/s threshold
        This is the main driver for achieving the hackathon goal
        """
        vel_x = self.env._get_base_lin_vel()[:, 0]

        # Bonus only kicks in above threshold
        bonus = torch.where(
            vel_x >= self.min_vel,
            vel_x - self.min_vel,  # Linear bonus above threshold
            torch.zeros_like(vel_x)
        )

        weight = self.env.config.reward_weights.get('velocity_bonus', 5.0)
        return bonus, weight

    def reward_velocity_tracking(self) -> Tuple[torch.Tensor, float]:
        """
        Gaussian reward around target velocity (2.5 m/s)
        Wider sigma than walking for more tolerance
        """
        vel_x = self.env._get_base_lin_vel()[:, 0]
        vel_error = torch.abs(vel_x - self.target_vel)

        sigma = self.env.config.reward_params.get('tracking_sigma', 0.5)
        reward = torch.exp(-vel_error / sigma)

        weight = self.env.config.reward_weights.get('velocity_tracking', 1.0)
        return reward, weight

    def reward_lateral_penalty(self) -> Tuple[torch.Tensor, float]:
        """
        Penalize sideways movement - relaxed compared to walking
        Some drift is acceptable for high-speed running
        """
        vel_y = self.env._get_base_lin_vel()[:, 1]
        reward = -torch.abs(vel_y)

        # Lower weight than walking
        weight = self.env.config.reward_weights.get('lateral_velocity', 0.3)
        return reward, weight

    def reward_angular_penalty(self) -> Tuple[torch.Tensor, float]:
        """Penalize rotation - relaxed for running"""
        ang_vel_z = self.env._get_base_ang_vel()[:, 2]
        reward = -torch.abs(ang_vel_z)

        weight = self.env.config.reward_weights.get('angular_velocity', 0.3)
        return reward, weight

    def reward_orientation(self) -> Tuple[torch.Tensor, float]:
        """
        Keep robot upright - more lenient for running dynamics
        Running naturally involves more body motion
        """
        rpy = self.env._get_base_rpy()
        roll, pitch = rpy[:, 0], rpy[:, 1]

        # Reduced penalty weights
        reward = -0.5 * torch.abs(roll) - 0.5 * torch.abs(pitch)

        weight = self.env.config.reward_weights.get('orientation_stability', 0.5)
        return reward, weight

    def reward_height(self) -> Tuple[torch.Tensor, float]:
        """
        Maintain height - more lenient for running
        Only penalize if too low (about to fall)
        """
        height = self.env._get_base_position()[:, 2]

        # Only care if height drops too low
        reward = torch.where(
            height > 0.25,
            torch.ones_like(height),
            torch.zeros_like(height)
        )

        weight = self.env.config.reward_weights.get('base_height', 0.3)
        return reward, weight

    def reward_action_smoothness(self) -> Tuple[torch.Tensor, float]:
        """
        Smooth movements - relaxed for dynamic running
        Running requires more dynamic actions
        """
        action_diff = self.env.actions - self.env.prev_actions
        reward = -torch.sum(action_diff ** 2, dim=-1)

        # Lower penalty than walking
        weight = self.env.config.reward_weights.get('action_smoothness', 0.005)
        return reward, weight

    def reward_survival(self) -> Tuple[torch.Tensor, float]:
        """
        Survival bonus - lower than walking
        We want the policy to take risks for speed
        """
        reward = torch.ones(self.env.num_envs, device=self.device)

        weight = self.env.config.reward_weights.get('survival', 0.1)
        return reward, weight

    # ============== Compute Total Reward ==============

    def compute_total_reward(self) -> torch.Tensor:
        """Compute total reward as weighted sum of all components"""
        total_reward = torch.zeros(self.env.num_envs, device=self.device)

        for reward_fn in self.get_reward_functions():
            reward, weight = reward_fn()
            total_reward += reward * weight

        # Scale by dt
        total_reward *= self.env.dt

        return total_reward
