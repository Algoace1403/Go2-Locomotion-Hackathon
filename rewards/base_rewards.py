"""
Base Reward Functions
Common reward components used across all levels
"""

import torch
import numpy as np
from typing import Tuple


class BaseRewards:
    """
    Base class containing common reward functions.
    All reward functions return (reward_value, weight) tuple.
    """

    def __init__(self, env):
        """
        Initialize with reference to environment.

        Args:
            env: Go2Env instance
        """
        self.env = env
        self.device = env.device

    # ============== Velocity Rewards ==============

    def forward_velocity(self) -> Tuple[torch.Tensor, float]:
        """
        Reward for forward velocity.
        Higher velocity = higher reward.
        """
        vel_x = self.env._get_base_lin_vel()[:, 0]
        reward = vel_x
        weight = self.env.config.reward_weights.get('forward_velocity', 1.0)
        return reward, weight

    def velocity_tracking(self, target_vel: float, sigma: float = 0.25) -> Tuple[torch.Tensor, float]:
        """
        Reward for tracking target velocity using Gaussian.
        Maximized when velocity matches target.

        Args:
            target_vel: Target forward velocity (m/s)
            sigma: Gaussian width (tolerance)
        """
        vel_x = self.env._get_base_lin_vel()[:, 0]
        vel_error = torch.abs(vel_x - target_vel)
        reward = torch.exp(-vel_error / sigma)
        weight = self.env.config.reward_weights.get('velocity_tracking', 1.0)
        return reward, weight

    def lateral_velocity_penalty(self) -> Tuple[torch.Tensor, float]:
        """Penalize lateral (sideways) movement"""
        vel_y = self.env._get_base_lin_vel()[:, 1]
        reward = -torch.abs(vel_y)
        weight = self.env.config.reward_weights.get('lateral_velocity', 0.5)
        return reward, weight

    def angular_velocity_penalty(self) -> Tuple[torch.Tensor, float]:
        """Penalize unwanted rotation (yaw)"""
        ang_vel_z = self.env._get_base_ang_vel()[:, 2]
        reward = -torch.abs(ang_vel_z)
        weight = self.env.config.reward_weights.get('angular_velocity', 0.5)
        return reward, weight

    # ============== Stability Rewards ==============

    def orientation_stability(self) -> Tuple[torch.Tensor, float]:
        """Penalize roll and pitch (keep robot upright)"""
        rpy = self.env._get_base_rpy()
        roll, pitch = rpy[:, 0], rpy[:, 1]
        reward = -torch.abs(roll) - torch.abs(pitch)
        weight = self.env.config.reward_weights.get('orientation_stability', 1.0)
        return reward, weight

    def base_height(self, target_height: float = 0.34, sigma: float = 0.1) -> Tuple[torch.Tensor, float]:
        """
        Reward for maintaining target base height.

        Args:
            target_height: Target height (m)
            sigma: Gaussian width
        """
        height = self.env._get_base_position()[:, 2]
        height_error = torch.abs(height - target_height)
        reward = torch.exp(-height_error / sigma)
        weight = self.env.config.reward_weights.get('base_height', 0.5)
        return reward, weight

    def z_velocity_penalty(self) -> Tuple[torch.Tensor, float]:
        """Penalize vertical (bouncing) movement"""
        vel_z = self.env._get_base_lin_vel()[:, 2]
        reward = -torch.abs(vel_z)
        weight = self.env.config.reward_weights.get('z_velocity', 0.2)
        return reward, weight

    # ============== Action Quality Rewards ==============

    def action_smoothness(self) -> Tuple[torch.Tensor, float]:
        """Penalize abrupt action changes (action rate)"""
        action_diff = self.env.actions - self.env.prev_actions
        reward = -torch.sum(action_diff ** 2, dim=-1)
        weight = self.env.config.reward_weights.get('action_smoothness', 0.01)
        return reward, weight

    def joint_torque_penalty(self) -> Tuple[torch.Tensor, float]:
        """Penalize high joint torques (energy efficiency)"""
        torques = self.env._get_dof_torque()
        reward = -torch.sum(torques ** 2, dim=-1)
        weight = self.env.config.reward_weights.get('joint_torque', 0.0001)
        return reward, weight

    def default_pose_penalty(self) -> Tuple[torch.Tensor, float]:
        """Penalize deviation from default standing pose"""
        dof_pos = self.env._get_dof_pos()
        deviation = dof_pos - self.env.default_dof_pos
        reward = -torch.sum(deviation ** 2, dim=-1)
        weight = self.env.config.reward_weights.get('default_pose', 0.1)
        return reward, weight

    # ============== Survival Rewards ==============

    def survival_bonus(self) -> Tuple[torch.Tensor, float]:
        """Small bonus for staying alive"""
        reward = torch.ones(self.env.num_envs, device=self.device)
        weight = self.env.config.reward_weights.get('survival', 0.1)
        return reward, weight

    # ============== Gait Quality Rewards ==============

    def foot_clearance(self) -> Tuple[torch.Tensor, float]:
        """
        Reward for lifting feet (prevents shuffling).
        Requires foot contact/height information.
        """
        # Placeholder - requires foot position tracking
        reward = torch.zeros(self.env.num_envs, device=self.device)
        weight = self.env.config.reward_weights.get('foot_clearance', 0.0)
        return reward, weight

    def gait_symmetry(self) -> Tuple[torch.Tensor, float]:
        """
        Reward for symmetric leg motion.
        Left legs should mirror right legs for stable gaits.
        """
        dof_pos = self.env._get_dof_pos()

        # FL (0:3) + RL (6:9) vs FR (3:6) + RR (9:12)
        left = torch.cat([dof_pos[:, 0:3], dof_pos[:, 6:9]], dim=-1)
        right = torch.cat([dof_pos[:, 3:6], dof_pos[:, 9:12]], dim=-1)

        # Symmetric means left ≈ -right for hip joints
        # (simplified version)
        asymmetry = torch.sum((left + right) ** 2, dim=-1)
        reward = -asymmetry
        weight = self.env.config.reward_weights.get('gait_symmetry', 0.1)
        return reward, weight

    # ============== Command Following ==============

    def command_tracking_lin_vel(self) -> Tuple[torch.Tensor, float]:
        """Track commanded linear velocity"""
        cmd_vel = self.env.commands[:, :2]  # x, y velocity commands
        actual_vel = self.env._get_base_lin_vel()[:, :2]

        vel_error = torch.sum((cmd_vel - actual_vel) ** 2, dim=-1)
        reward = torch.exp(-vel_error / 0.25)
        weight = self.env.config.reward_weights.get('tracking_lin_vel', 1.0)
        return reward, weight

    def command_tracking_ang_vel(self) -> Tuple[torch.Tensor, float]:
        """Track commanded angular velocity"""
        cmd_ang = self.env.commands[:, 2]
        actual_ang = self.env._get_base_ang_vel()[:, 2]

        ang_error = (cmd_ang - actual_ang) ** 2
        reward = torch.exp(-ang_error / 0.25)
        weight = self.env.config.reward_weights.get('tracking_ang_vel', 0.5)
        return reward, weight
