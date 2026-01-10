"""
Level 2: Running Environment
"""

import torch
from .base_go2_env import Go2Env
from configs.level2_run import RunConfig
from rewards.run_rewards import RunRewards


class RunEnv(Go2Env):
    """Environment for Level 2: Running"""

    def __init__(self, num_envs: int = 4096, device: str = 'cuda', headless: bool = True):
        super().__init__(
            config=RunConfig(),
            num_envs=num_envs,
            device=device,
            headless=headless
        )

        # Initialize running rewards
        self.reward_module = RunRewards(self)

    def _compute_rewards(self):
        """Compute running-specific rewards"""
        self.reward_buf = self.reward_module.compute_total_reward()
