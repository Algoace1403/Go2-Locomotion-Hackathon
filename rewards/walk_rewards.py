"""
Level 1: Walking Reward Functions
Target: 0.5-1.0 m/s forward velocity, maximum distance in 60s
"""

import torch
from typing import Tuple, List, Callable
from .base_rewards import BaseRewards


class WalkRewards(BaseRewards):
    """
    Reward functions for Level 1: Walking
    Focus: Stable, controlled walking at moderate speed
    """

    def __init__(self, env):
        super().__init__(env)

        # Walking-specific parameters
        self.target_vel = getattr(env.config, 'target_vel', 0.75)
        self.target_height = getattr(
            env.config, 'reward_params', {}
        ).get('target_height', 0.34)

    def get_reward_functions(self) -> List[Callable]:
        """Return list of reward functions for walking"""
        return [
            self.reward_forward_velocity,
            self.reward_velocity_tracking,
            self.reward_lateral_penalty,
            self.reward_angular_penalty,
            self.reward_orientation,
            self.reward_height,
            self.reward_action_smoothness,
            self.reward_torque_penalty,
            self.reward_survival,
        ]

    # ============== Walking-Specific Rewards ==============

    def reward_forward_velocity(self) -> Tuple[torch.Tensor, float]:
        """
        Primary reward: Forward velocity
        Linear reward for moving forward
        """
        vel_x = self.env._get_base_lin_vel()[:, 0]

        # Clip to reasonable range to prevent exploitation
        reward = torch.clamp(vel_x, -0.5, 1.5)

        weight = self.env.config.reward_weights.get('forward_velocity', 2.0)
        return reward, weight

    def reward_velocity_tracking(self) -> Tuple[torch.Tensor, float]:
        """
        Reward for staying in target velocity range (0.5-1.0 m/s)
        Uses Gaussian centered at target_vel
        """
        vel_x = self.env._get_base_lin_vel()[:, 0]
        vel_error = torch.abs(vel_x - self.target_vel)

        sigma = self.env.config.reward_params.get('tracking_sigma', 0.25)
        reward = torch.exp(-vel_error / sigma)

        weight = self.env.config.reward_weights.get('velocity_tracking', 1.5)
        return reward, weight

    def reward_lateral_penalty(self) -> Tuple[torch.Tensor, float]:
        """Penalize sideways movement - walking should be straight"""
        vel_y = self.env._get_base_lin_vel()[:, 1]
        reward = -torch.abs(vel_y)

        weight = self.env.config.reward_weights.get('lateral_velocity', 0.5)
        return reward, weight

    def reward_angular_penalty(self) -> Tuple[torch.Tensor, float]:
        """Penalize rotation - walking should be in a straight line"""
        ang_vel_z = self.env._get_base_ang_vel()[:, 2]
        reward = -torch.abs(ang_vel_z)

        weight = self.env.config.reward_weights.get('angular_velocity', 0.5)
        return reward, weight

    def reward_orientation(self) -> Tuple[torch.Tensor, float]:
        """Keep robot upright - penalize roll and pitch"""
        rpy = self.env._get_base_rpy()
        roll, pitch = rpy[:, 0], rpy[:, 1]

        reward = -torch.abs(roll) - torch.abs(pitch)

        weight = self.env.config.reward_weights.get('orientation_stability', 1.0)
        return reward, weight

    def reward_height(self) -> Tuple[torch.Tensor, float]:
        """Maintain proper standing height"""
        height = self.env._get_base_position()[:, 2]
        height_error = torch.abs(height - self.target_height)

        sigma = self.env.config.reward_params.get('height_sigma', 0.1)
        reward = torch.exp(-height_error / sigma)

        weight = self.env.config.reward_weights.get('base_height', 0.5)
        return reward, weight

    def reward_action_smoothness(self) -> Tuple[torch.Tensor, float]:
        """Smooth movements - penalize jerky actions"""
        action_diff = self.env.actions - self.env.prev_actions
        reward = -torch.sum(action_diff ** 2, dim=-1)

        weight = self.env.config.reward_weights.get('action_smoothness', 0.01)
        return reward, weight

    def reward_torque_penalty(self) -> Tuple[torch.Tensor, float]:
        """Energy efficiency - penalize high torques"""
        torques = self.env._get_dof_torque()
        reward = -torch.sum(torques ** 2, dim=-1)

        weight = self.env.config.reward_weights.get('joint_torque', 0.0001)
        return reward, weight

    def reward_survival(self) -> Tuple[torch.Tensor, float]:
        """Bonus for staying alive (not falling)"""
        reward = torch.ones(self.env.num_envs, device=self.device)

        weight = self.env.config.reward_weights.get('survival', 0.2)
        return reward, weight

    # ============== Compute Total Reward ==============

    def compute_total_reward(self) -> torch.Tensor:
        """Compute total reward as weighted sum of all components"""
        total_reward = torch.zeros(self.env.num_envs, device=self.device)

        for reward_fn in self.get_reward_functions():
            reward, weight = reward_fn()
            total_reward += reward * weight

        # Scale by dt for proper temporal scaling
        total_reward *= self.env.dt

        return total_reward
