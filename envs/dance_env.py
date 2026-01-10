"""
Level 5: Dance Environment
Make the Go2 dance with rhythm and style!
"""

import torch
import numpy as np
from .base_go2_env import Go2Env
from configs.level5_dance import DanceConfig
from rewards.dance_rewards import DanceRewards


class DanceEnv(Go2Env):
    """Environment for Level 5: Dancing"""

    def __init__(self, num_envs: int = 4096, device: str = 'cuda', headless: bool = True):
        super().__init__(
            config=DanceConfig(),
            num_envs=num_envs,
            device=device,
            headless=headless
        )

        # Initialize dance rewards
        self.reward_module = DanceRewards(self)

        # Extended observation for rhythm
        self.num_obs = self.config.num_observations  # 48

        # Beat tracking
        self.beat_phase = torch.zeros(num_envs, device=device)
        self.time_steps = torch.zeros(num_envs, device=device)

        # Music/rhythm parameters
        self.bpm = 120  # Beats per minute
        self.beat_period = 60.0 / self.bpm

    def reset(self, env_ids: torch.Tensor = None) -> torch.Tensor:
        """Reset with rhythm sync"""
        obs = super().reset(env_ids)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset rhythm tracking
        self.time_steps[env_ids] = 0
        self.beat_phase[env_ids] = 0

        # Reset reward module
        self.reward_module.reset()

        return obs

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample dance-appropriate commands (mostly stationary with flair)"""
        num_reset = len(env_ids)

        # Small random movements for variety
        self.commands[env_ids, 0] = torch.empty(num_reset, device=self.device).uniform_(-0.1, 0.1)
        self.commands[env_ids, 1] = torch.empty(num_reset, device=self.device).uniform_(-0.1, 0.1)
        self.commands[env_ids, 2] = torch.empty(num_reset, device=self.device).uniform_(-0.3, 0.3)

    def step(self, actions: torch.Tensor):
        """Step with rhythm update"""
        # Update time and beat phase
        self.time_steps += 1
        time = self.time_steps * self.dt
        self.beat_phase = (time / self.beat_period) % 1.0

        # Regular step
        return super().step(actions)

    def _compute_observations(self):
        """Compute extended observations with beat info"""
        # Standard observations
        super()._compute_observations()

        # Add beat phase (tells policy when the beat hits)
        beat_sin = torch.sin(self.beat_phase * 2 * np.pi).unsqueeze(-1)
        beat_cos = torch.cos(self.beat_phase * 2 * np.pi).unsqueeze(-1)

        # Style encoding (one-hot or continuous)
        style_encoding = torch.zeros((self.num_envs, 1), device=self.device)

        self.obs_buf = torch.cat([
            self.obs_buf,
            beat_sin,        # Beat phase (sin)
            beat_cos,        # Beat phase (cos)
            style_encoding,  # Dance style
        ], dim=-1)

    def _compute_rewards(self):
        """Compute dance-specific rewards"""
        self.reward_buf = self.reward_module.compute_total_reward()

    def _check_termination(self):
        """More lenient termination for dancing"""
        rpy = self._get_base_rpy()
        roll, pitch = rpy[:, 0], rpy[:, 1]
        height = self._get_base_position()[:, 2]

        # Lenient thresholds for dancing
        term_config = self.config.termination

        roll_term = torch.abs(roll) > term_config['max_roll']
        pitch_term = torch.abs(pitch) > term_config['max_pitch']
        height_term = height < term_config['min_height']
        time_term = self.episode_length_buf >= self.max_episode_length

        self.done_buf = roll_term | pitch_term | height_term | time_term

    def set_music_bpm(self, bpm: int):
        """Change the music tempo"""
        self.bpm = bpm
        self.beat_period = 60.0 / bpm
        print(f"Music BPM set to {bpm} (beat period: {self.beat_period:.3f}s)")
