"""
Level 1: Walking Environment
"""

import torch
from .base_go2_env import Go2Env
from configs.level1_walk import WalkConfig
from rewards.walk_rewards import WalkRewards


class WalkEnv(Go2Env):
    """Environment for Level 1: Walking"""

    def __init__(self, num_envs: int = 4096, device: str = 'cuda', headless: bool = True):
        super().__init__(
            config=WalkConfig(),
            num_envs=num_envs,
            device=device,
            headless=headless
        )

        # Initialize walking rewards
        self.reward_module = WalkRewards(self)

    def _compute_rewards(self):
        """Compute walking-specific rewards"""
        self.reward_buf = self.reward_module.compute_total_reward()
