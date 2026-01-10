"""
Level 3: Circle Walking Reward Functions (BONUS)
Target: Walk in a perfect circle with R=2m
Metric: Average RMSE for 10 rounds
"""

import torch
import numpy as np
from typing import Tuple, List, Callable
from .base_rewards import BaseRewards


class CircleRewards(BaseRewards):
    """
    Reward functions for Level 3: Circle Walking
    Focus: Precise path following while maintaining stable gait
    """

    def __init__(self, env):
        super().__init__(env)

        # Circle-specific parameters
        self.radius = getattr(env.config, 'circle_radius', 2.0)
        self.center = torch.tensor(
            getattr(env.config, 'circle_center', [0.0, 0.0]),
            device=self.device
        )
        self.target_speed = getattr(env.config, 'target_speed', 0.5)

        # Required angular velocity for circle
        # ω = v / r
        self.required_yaw_rate = self.target_speed / self.radius

    def get_reward_functions(self) -> List[Callable]:
        """Return list of reward functions for circle walking"""
        return [
            self.reward_radius_tracking,
            self.reward_tangent_velocity,
            self.reward_heading_alignment,
            self.reward_angular_velocity,
            self.reward_radial_velocity_penalty,
            self.reward_forward_velocity,
            self.reward_orientation,
            self.reward_action_smoothness,
            self.reward_survival,
        ]

    # ============== Circle-Specific Rewards ==============

    def reward_radius_tracking(self) -> Tuple[torch.Tensor, float]:
        """
        Primary reward: Stay on the circle (R=2m from center)
        Uses Gaussian for smooth reward gradient
        """
        pos = self.env._get_base_position()[:, :2]  # x, y
        dist_from_center = torch.norm(pos - self.center, dim=-1)

        radius_error = torch.abs(dist_from_center - self.radius)

        sigma = self.env.config.reward_params.get('radius_sigma', 0.3)
        reward = torch.exp(-radius_error / sigma)

        weight = self.env.config.reward_weights.get('radius_tracking', 3.0)
        return reward, weight

    def reward_tangent_velocity(self) -> Tuple[torch.Tensor, float]:
        """
        Reward for moving along the circle tangent
        Tangent direction is perpendicular to radius vector
        """
        pos = self.env._get_base_position()[:, :2]
        vel = self.env._get_base_lin_vel()[:, :2]

        # Angle from center to robot
        angle_to_center = torch.atan2(pos[:, 1], pos[:, 0])

        # Tangent is 90 degrees ahead (counter-clockwise)
        tangent_angle = angle_to_center + np.pi / 2

        # Tangent direction vector
        tangent_dir = torch.stack([
            torch.cos(tangent_angle),
            torch.sin(tangent_angle)
        ], dim=-1)

        # Velocity component along tangent
        tangent_vel = torch.sum(vel * tangent_dir, dim=-1)

        # Reward positive tangent velocity
        reward = tangent_vel

        weight = self.env.config.reward_weights.get('tangent_velocity', 2.0)
        return reward, weight

    def reward_heading_alignment(self) -> Tuple[torch.Tensor, float]:
        """
        Reward for facing tangent direction
        Robot should look where it's going on the circle
        """
        pos = self.env._get_base_position()[:, :2]
        rpy = self.env._get_base_rpy()
        heading = rpy[:, 2]  # yaw

        # Ideal heading is tangent to circle
        angle_to_center = torch.atan2(pos[:, 1], pos[:, 0])
        tangent_angle = angle_to_center + np.pi / 2

        # Heading error (handle wrap-around)
        heading_error = self._angle_diff(heading, tangent_angle)

        sigma = self.env.config.reward_params.get('heading_sigma', 0.3)
        reward = torch.exp(-torch.abs(heading_error) / sigma)

        weight = self.env.config.reward_weights.get('heading_alignment', 1.5)
        return reward, weight

    def reward_angular_velocity(self) -> Tuple[torch.Tensor, float]:
        """
        Reward for maintaining correct turning rate
        Required: ω = v / r ≈ 0.5 / 2.0 = 0.25 rad/s
        """
        ang_vel_z = self.env._get_base_ang_vel()[:, 2]

        yaw_error = torch.abs(ang_vel_z - self.required_yaw_rate)

        sigma = self.env.config.reward_params.get('angular_vel_sigma', 0.2)
        reward = torch.exp(-yaw_error / sigma)

        weight = self.env.config.reward_weights.get('angular_velocity', 1.0)
        return reward, weight

    def reward_radial_velocity_penalty(self) -> Tuple[torch.Tensor, float]:
        """
        Penalize movement toward or away from center
        We want to stay on the circle, not spiral in/out
        """
        pos = self.env._get_base_position()[:, :2]
        vel = self.env._get_base_lin_vel()[:, :2]

        # Radial direction (from center to robot)
        angle_to_center = torch.atan2(pos[:, 1], pos[:, 0])
        radial_dir = torch.stack([
            torch.cos(angle_to_center),
            torch.sin(angle_to_center)
        ], dim=-1)

        # Velocity component along radius
        radial_vel = torch.sum(vel * radial_dir, dim=-1)

        # Penalize radial velocity
        reward = -torch.abs(radial_vel)

        weight = self.env.config.reward_weights.get('radial_velocity', 2.0)
        return reward, weight

    def reward_forward_velocity(self) -> Tuple[torch.Tensor, float]:
        """Reward for maintaining forward motion on circle"""
        vel = self.env._get_base_lin_vel()[:, :2]
        speed = torch.norm(vel, dim=-1)

        # Reward up to target speed
        reward = torch.clamp(speed, 0, self.target_speed)

        weight = self.env.config.reward_weights.get('forward_velocity', 1.0)
        return reward, weight

    def reward_orientation(self) -> Tuple[torch.Tensor, float]:
        """Keep robot upright"""
        rpy = self.env._get_base_rpy()
        roll, pitch = rpy[:, 0], rpy[:, 1]

        reward = -torch.abs(roll) - torch.abs(pitch)

        weight = self.env.config.reward_weights.get('orientation_stability', 0.5)
        return reward, weight

    def reward_action_smoothness(self) -> Tuple[torch.Tensor, float]:
        """Smooth movements for stable circle walking"""
        action_diff = self.env.actions - self.env.prev_actions
        reward = -torch.sum(action_diff ** 2, dim=-1)

        weight = self.env.config.reward_weights.get('action_smoothness', 0.01)
        return reward, weight

    def reward_survival(self) -> Tuple[torch.Tensor, float]:
        """Survival bonus"""
        reward = torch.ones(self.env.num_envs, device=self.device)

        weight = self.env.config.reward_weights.get('survival', 0.1)
        return reward, weight

    # ============== Utility ==============

    def _angle_diff(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Compute smallest angle difference, handling wrap-around"""
        diff = a - b

        # Wrap to [-π, π]
        diff = torch.where(diff > np.pi, diff - 2 * np.pi, diff)
        diff = torch.where(diff < -np.pi, diff + 2 * np.pi, diff)

        return diff

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
