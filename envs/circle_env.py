"""
Level 3: Circle Walking Environment
"""

import torch
import numpy as np
from .base_go2_env import Go2Env
from configs.level3_circle import CircleConfig
from rewards.circle_rewards import CircleRewards


class CircleEnv(Go2Env):
    """Environment for Level 3: Circle Walking"""

    def __init__(self, num_envs: int = 4096, device: str = 'cuda', headless: bool = True):
        super().__init__(
            config=CircleConfig(),
            num_envs=num_envs,
            device=device,
            headless=headless
        )

        # Initialize circle rewards
        self.reward_module = CircleRewards(self)

        # Extended observation for circle tracking
        self.num_obs = self.config.num_observations  # 49

    def _reset_robot(self, env_ids: torch.Tensor):
        """Reset robot to starting position on circle"""
        # Start on the circle at radius R
        num_reset = len(env_ids)

        # Random angle on circle
        angles = torch.empty(num_reset, device=self.device).uniform_(0, 2 * np.pi)

        # Starting position on circle
        start_x = self.config.circle_radius * torch.cos(angles)
        start_y = self.config.circle_radius * torch.sin(angles)
        start_z = torch.full((num_reset,), 0.4, device=self.device)

        positions = torch.stack([start_x, start_y, start_z], dim=-1)

        # Starting orientation: facing tangent to circle
        # Tangent angle = angle + 90 degrees
        tangent_angles = angles + np.pi / 2

        # Convert yaw to quaternion
        quats = self._yaw_to_quat(tangent_angles)

        self.robot.set_pos(positions, zero_velocity=True, envs_idx=env_ids)
        self.robot.set_quat(quats, zero_velocity=True, envs_idx=env_ids)

        # Reset joints
        self.robot.set_dofs_position(
            self.default_dof_pos.expand(num_reset, -1),
            self.joint_names,
            zero_velocity=True,
            envs_idx=env_ids
        )

    def _yaw_to_quat(self, yaw: torch.Tensor) -> torch.Tensor:
        """Convert yaw angle to quaternion"""
        half_yaw = yaw / 2
        w = torch.cos(half_yaw)
        x = torch.zeros_like(yaw)
        y = torch.zeros_like(yaw)
        z = torch.sin(half_yaw)
        return torch.stack([w, x, y, z], dim=-1)

    def _compute_observations(self):
        """Compute extended observations for circle tracking"""
        # Standard observations
        super()._compute_observations()

        # Add position and heading for circle tracking
        pos = self._get_base_position()[:, :2]  # x, y
        rpy = self._get_base_rpy()
        heading = rpy[:, 2:3]  # yaw

        # Angular error from required turning rate
        ang_vel_z = self._get_base_ang_vel()[:, 2:3]
        required_rate = self.config.target_speed / self.config.circle_radius
        ang_error = ang_vel_z - required_rate

        # Extend observation buffer
        self.obs_buf = torch.cat([
            self.obs_buf,
            pos,           # 2D position
            heading,       # yaw
            ang_error,     # angular velocity error
        ], dim=-1)

    def _compute_rewards(self):
        """Compute circle-specific rewards"""
        self.reward_buf = self.reward_module.compute_total_reward()
