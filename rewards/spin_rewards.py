"""
Level 4: Spin in Place Reward Functions (BONUS)
Target: High angular velocity (ω), zero forward movement (v=0)
"""

import torch
from typing import Tuple, List, Callable
from .base_rewards import BaseRewards


class SpinRewards(BaseRewards):
    """
    Reward functions for Level 4: Spin in Place
    Focus: Maximum rotation with minimum translation
    """

    def __init__(self, env):
        super().__init__(env)

        # Spin-specific parameters
        self.target_angular_vel = getattr(env.config, 'target_angular_vel', 2.0)

        # Track starting position for drift penalty
        self.start_positions = None

    def reset_start_positions(self, env_ids: torch.Tensor = None):
        """Reset start positions for drift tracking"""
        pos = self.env._get_base_position()[:, :2]

        if env_ids is None:
            self.start_positions = pos.clone()
        else:
            if self.start_positions is None:
                self.start_positions = pos.clone()
            else:
                self.start_positions[env_ids] = pos[env_ids].clone()

    def get_reward_functions(self) -> List[Callable]:
        """Return list of reward functions for spinning"""
        return [
            self.reward_angular_velocity,
            self.reward_angular_tracking,
            self.reward_zero_linear_velocity,
            self.reward_position_drift,
            self.reward_orientation,
            self.reward_symmetric_motion,
            self.reward_action_smoothness,
            self.reward_survival,
        ]

    # ============== Spin-Specific Rewards ==============

    def reward_angular_velocity(self) -> Tuple[torch.Tensor, float]:
        """
        Primary reward: Angular velocity magnitude
        Higher rotation = higher reward (either direction)
        """
        ang_vel_z = self.env._get_base_ang_vel()[:, 2]

        # Reward absolute angular velocity
        reward = torch.abs(ang_vel_z)

        weight = self.env.config.reward_weights.get('angular_velocity', 5.0)
        return reward, weight

    def reward_angular_tracking(self) -> Tuple[torch.Tensor, float]:
        """
        Reward for hitting target angular velocity (2.0 rad/s)
        Uses Gaussian for smooth gradient
        """
        ang_vel_z = self.env._get_base_ang_vel()[:, 2]

        # Track magnitude, not direction
        ang_error = torch.abs(torch.abs(ang_vel_z) - self.target_angular_vel)

        sigma = self.env.config.reward_params.get('angular_sigma', 0.5)
        reward = torch.exp(-ang_error / sigma)

        weight = self.env.config.reward_weights.get('angular_tracking', 2.0)
        return reward, weight

    def reward_zero_linear_velocity(self) -> Tuple[torch.Tensor, float]:
        """
        CRITICAL: Penalize any forward or lateral movement
        This is the key constraint for "spin in place"
        """
        vel = self.env._get_base_lin_vel()[:, :2]  # x, y velocity
        linear_speed = torch.norm(vel, dim=-1)

        # Strong penalty for any linear motion
        reward = -linear_speed

        weight = self.env.config.reward_weights.get('zero_linear_velocity', 10.0)
        return reward, weight

    def reward_position_drift(self) -> Tuple[torch.Tensor, float]:
        """
        Penalize drifting from starting position
        Robot should spin in place, not spiral around
        """
        if self.start_positions is None:
            return torch.zeros(self.env.num_envs, device=self.device), 0.0

        pos = self.env._get_base_position()[:, :2]
        drift = torch.norm(pos - self.start_positions, dim=-1)

        # Penalize drift
        reward = -drift

        weight = self.env.config.reward_weights.get('position_drift', 5.0)
        return reward, weight

    def reward_orientation(self) -> Tuple[torch.Tensor, float]:
        """
        Keep robot upright during spin
        Some tolerance needed for dynamic spinning
        """
        rpy = self.env._get_base_rpy()
        roll, pitch = rpy[:, 0], rpy[:, 1]

        reward = -torch.abs(roll) - torch.abs(pitch)

        weight = self.env.config.reward_weights.get('orientation_stability', 1.0)
        return reward, weight

    def reward_symmetric_motion(self) -> Tuple[torch.Tensor, float]:
        """
        Reward symmetric leg motion for clean spin
        Left legs should mirror right legs for balanced rotation
        """
        dof_pos = self.env._get_dof_pos()

        # FL (0:3) + RL (6:9)
        left = torch.cat([dof_pos[:, 0:3], dof_pos[:, 6:9]], dim=-1)

        # FR (3:6) + RR (9:12)
        right = torch.cat([dof_pos[:, 3:6], dof_pos[:, 9:12]], dim=-1)

        # For spinning, we want opposite leg motions
        # Left and right should be mirror images
        asymmetry = torch.sum((left + right) ** 2, dim=-1)
        reward = -asymmetry

        weight = self.env.config.reward_weights.get('symmetric_motion', 0.5)
        return reward, weight

    def reward_action_smoothness(self) -> Tuple[torch.Tensor, float]:
        """
        Relaxed smoothness for dynamic spinning
        Spinning requires dynamic motions
        """
        action_diff = self.env.actions - self.env.prev_actions
        reward = -torch.sum(action_diff ** 2, dim=-1)

        weight = self.env.config.reward_weights.get('action_smoothness', 0.005)
        return reward, weight

    def reward_survival(self) -> Tuple[torch.Tensor, float]:
        """Survival bonus"""
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

        # Scale by dt
        total_reward *= self.env.dt

        return total_reward
