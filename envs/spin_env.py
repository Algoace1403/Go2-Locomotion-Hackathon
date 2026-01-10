"""
Level 4: Spin in Place Environment
"""

import torch
from .base_go2_env import Go2Env
from configs.level4_spin import SpinConfig
from rewards.spin_rewards import SpinRewards


class SpinEnv(Go2Env):
    """Environment for Level 4: Spin in Place"""

    def __init__(self, num_envs: int = 4096, device: str = 'cuda', headless: bool = True):
        super().__init__(
            config=SpinConfig(),
            num_envs=num_envs,
            device=device,
            headless=headless
        )

        # Initialize spin rewards
        self.reward_module = SpinRewards(self)

        # Extended observation
        self.num_obs = self.config.num_observations  # 47

    def reset(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Reset with start position tracking"""
        obs = super().reset(env_ids)

        # Track starting positions for drift penalty
        self.reward_module.reset_start_positions(env_ids)

        return obs

    def _sample_commands(self, env_ids: torch.Tensor):
        """No forward movement commands for spin"""
        num_reset = len(env_ids)

        # Zero linear velocity, high angular velocity
        self.commands[env_ids, 0] = 0.0  # No forward
        self.commands[env_ids, 1] = 0.0  # No lateral
        self.commands[env_ids, 2] = torch.empty(num_reset, device=self.device).uniform_(
            self.config.command_ranges['ang_vel'][0],
            self.config.command_ranges['ang_vel'][1]
        )

    def _compute_observations(self):
        """Compute extended observations for spin"""
        # Standard observations
        super()._compute_observations()

        # Add start position for drift tracking
        if self.reward_module.start_positions is not None:
            start_pos = self.reward_module.start_positions
        else:
            start_pos = torch.zeros((self.num_envs, 2), device=self.device)

        self.obs_buf = torch.cat([
            self.obs_buf,
            start_pos,  # 2D start position
        ], dim=-1)

    def _compute_rewards(self):
        """Compute spin-specific rewards"""
        self.reward_buf = self.reward_module.compute_total_reward()

    def _check_termination(self):
        """Check termination including drift"""
        super()._check_termination()

        # Additional termination for excessive drift
        if self.reward_module.start_positions is not None:
            pos = self._get_base_position()[:, :2]
            drift = torch.norm(pos - self.reward_module.start_positions, dim=-1)
            max_drift = self.config.termination.get('max_drift', 0.5)
            drift_term = drift > max_drift
            self.done_buf = self.done_buf | drift_term
