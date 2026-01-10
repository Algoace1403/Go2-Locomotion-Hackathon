"""
Level 5: Dance Reward Functions (BONUS SHOWOFF!)
Make the robot groove, bounce, and dance with style!
"""

import torch
import numpy as np
from typing import Tuple, List, Callable
from .base_rewards import BaseRewards


class DanceRewards(BaseRewards):
    """
    Reward functions for Level 5: Dancing
    Focus: Rhythmic movements, coordination, and STYLE!
    """

    def __init__(self, env):
        super().__init__(env)

        # Dance parameters
        self.dance_frequency = getattr(env.config, 'dance_frequency', 2.0)
        self.beat_period = 1.0 / self.dance_frequency
        self.amplitude = getattr(env.config, 'dance_amplitude', 0.3)

        # Track rhythm
        self.time_step = 0
        self.beat_phase = 0.0

        # Movement history for variety tracking
        self.pose_history = []
        self.max_history = 50

    def update_beat(self):
        """Update the beat phase based on time"""
        self.time_step += 1
        time = self.time_step * self.env.dt
        self.beat_phase = (time * self.dance_frequency) % 1.0

    def reset(self):
        """Reset rhythm tracking"""
        self.time_step = 0
        self.beat_phase = 0.0
        self.pose_history = []

    def get_reward_functions(self) -> List[Callable]:
        """Return list of reward functions for dancing"""
        return [
            self.reward_rhythm_sync,
            self.reward_leg_coordination,
            self.reward_symmetry,
            self.reward_height_variation,
            self.reward_body_roll,
            self.reward_pose_variety,
            self.reward_orientation,
            self.reward_action_smoothness,
            self.reward_survival,
        ]

    # ============== Dance-Specific Rewards ==============

    def reward_rhythm_sync(self) -> Tuple[torch.Tensor, float]:
        """
        CORE REWARD: Synchronize movements with the beat!
        Movement velocity should peak at beat times.
        """
        # Update beat phase
        self.update_beat()

        # Get joint velocities
        dof_vel = self.env._get_dof_vel()
        movement_intensity = torch.mean(torch.abs(dof_vel), dim=-1)

        # Expected intensity based on beat phase
        # Peak at beat (phase=0) and off-beat (phase=0.5)
        beat_intensity = torch.abs(torch.sin(
            torch.tensor(self.beat_phase * 2 * np.pi, device=self.device)
        ))

        # Reward when movement matches beat
        sync_error = torch.abs(movement_intensity / (movement_intensity.max() + 1e-6) - beat_intensity)
        reward = torch.exp(-sync_error / 0.3)

        weight = self.env.config.reward_weights.get('rhythm_sync', 3.0)
        return reward, weight

    def reward_leg_coordination(self) -> Tuple[torch.Tensor, float]:
        """
        Reward coordinated leg movements.
        Diagonal legs should move together (trot-like rhythm).
        """
        dof_vel = self.env._get_dof_vel()

        # FL (0:3) and RR (9:12) should be similar (diagonal pair)
        fl_vel = dof_vel[:, 0:3]
        rr_vel = dof_vel[:, 9:12]
        diag1_sync = -torch.mean((fl_vel - rr_vel) ** 2, dim=-1)

        # FR (3:6) and RL (6:9) should be similar (other diagonal)
        fr_vel = dof_vel[:, 3:6]
        rl_vel = dof_vel[:, 6:9]
        diag2_sync = -torch.mean((fr_vel - rl_vel) ** 2, dim=-1)

        # Diagonals should be opposite phase
        diag_opposition = -torch.mean((fl_vel + fr_vel) ** 2, dim=-1)

        reward = diag1_sync + diag2_sync + 0.5 * diag_opposition

        weight = self.env.config.reward_weights.get('leg_coordination', 2.5)
        return reward, weight

    def reward_symmetry(self) -> Tuple[torch.Tensor, float]:
        """
        Reward symmetric movements (left-right).
        Dancing looks better when symmetric!
        """
        dof_pos = self.env._get_dof_pos()

        # Left side: FL (0:3) + RL (6:9)
        left = torch.cat([dof_pos[:, 0:3], dof_pos[:, 6:9]], dim=-1)

        # Right side: FR (3:6) + RR (9:12)
        right = torch.cat([dof_pos[:, 3:6], dof_pos[:, 9:12]], dim=-1)

        # Symmetric: left ≈ mirrored right (hip joints flip sign)
        # Simplified: reward similar magnitudes
        symmetry_error = torch.mean((torch.abs(left) - torch.abs(right)) ** 2, dim=-1)
        reward = torch.exp(-symmetry_error / 0.1)

        weight = self.env.config.reward_weights.get('symmetry', 2.0)
        return reward, weight

    def reward_height_variation(self) -> Tuple[torch.Tensor, float]:
        """
        Reward bouncing/grooving by varying body height with the beat.
        """
        height = self.env._get_base_position()[:, 2]

        # Target height should oscillate with beat
        target_height = 0.34 + 0.05 * torch.sin(
            torch.tensor(self.beat_phase * 2 * np.pi, device=self.device)
        )

        height_error = torch.abs(height - target_height)
        reward = torch.exp(-height_error / 0.05)

        weight = self.env.config.reward_weights.get('height_variation', 1.5)
        return reward, weight

    def reward_body_roll(self) -> Tuple[torch.Tensor, float]:
        """
        Reward stylish body roll/sway movements.
        Small controlled roll adds style!
        """
        rpy = self.env._get_base_rpy()
        roll = rpy[:, 0]

        # Target: small rhythmic roll
        target_roll = 0.1 * torch.sin(
            torch.tensor(self.beat_phase * 2 * np.pi, device=self.device)
        )

        roll_error = torch.abs(roll - target_roll)
        reward = torch.exp(-roll_error / 0.1)

        weight = self.env.config.reward_weights.get('body_roll', 1.0)
        return reward, weight

    def reward_pose_variety(self) -> Tuple[torch.Tensor, float]:
        """
        Reward variety in poses - don't be static!
        Track pose history and reward exploration.
        """
        dof_pos = self.env._get_dof_pos()

        # Store current pose (simplified: just first env)
        current_pose = dof_pos[0].detach().cpu().numpy()
        self.pose_history.append(current_pose)

        if len(self.pose_history) > self.max_history:
            self.pose_history.pop(0)

        # Reward if current pose is different from recent poses
        if len(self.pose_history) > 10:
            recent = np.array(self.pose_history[-10:])
            variance = np.mean(np.var(recent, axis=0))
            reward = torch.full((self.env.num_envs,), variance, device=self.device)
        else:
            reward = torch.zeros(self.env.num_envs, device=self.device)

        weight = self.env.config.reward_weights.get('pose_variety', 1.5)
        return reward, weight

    def reward_orientation(self) -> Tuple[torch.Tensor, float]:
        """
        Keep robot from falling, but allow stylish leans.
        More lenient than walking/running.
        """
        rpy = self.env._get_base_rpy()
        roll, pitch = rpy[:, 0], rpy[:, 1]

        # Only penalize extreme angles
        roll_penalty = torch.where(
            torch.abs(roll) > 0.5,
            torch.abs(roll) - 0.5,
            torch.zeros_like(roll)
        )
        pitch_penalty = torch.where(
            torch.abs(pitch) > 0.5,
            torch.abs(pitch) - 0.5,
            torch.zeros_like(pitch)
        )

        reward = -roll_penalty - pitch_penalty

        weight = self.env.config.reward_weights.get('orientation_stability', 1.0)
        return reward, weight

    def reward_action_smoothness(self) -> Tuple[torch.Tensor, float]:
        """
        Fluid dance movements - not jerky!
        """
        action_diff = self.env.actions - self.env.prev_actions
        reward = -torch.sum(action_diff ** 2, dim=-1)

        weight = self.env.config.reward_weights.get('action_smoothness', 0.005)
        return reward, weight

    def reward_survival(self) -> Tuple[torch.Tensor, float]:
        """Stay alive to keep dancing!"""
        reward = torch.ones(self.env.num_envs, device=self.device)

        weight = self.env.config.reward_weights.get('survival', 0.2)
        return reward, weight

    # ============== Special Dance Moves ==============

    def reward_moonwalk(self) -> Tuple[torch.Tensor, float]:
        """
        BONUS: Moonwalk detection!
        Moving backward while legs move forward.
        """
        vel_x = self.env._get_base_lin_vel()[:, 0]
        dof_vel = self.env._get_dof_vel()

        # Check if moving backward
        moving_back = vel_x < -0.1

        # Check if leg joints suggest forward motion
        leg_forward = torch.mean(dof_vel[:, [1, 4, 7, 10]], dim=-1) > 0  # thigh joints

        # Moonwalk = backward body + forward leg motion
        moonwalk = moving_back & leg_forward
        reward = moonwalk.float() * 5.0  # Big bonus!

        return reward, 1.0

    def reward_jump_attempt(self) -> Tuple[torch.Tensor, float]:
        """
        BONUS: Detect jump attempts!
        All feet leaving ground momentarily.
        """
        vel_z = self.env._get_base_lin_vel()[:, 2]
        height = self.env._get_base_position()[:, 2]

        # Jumping = upward velocity + above normal height
        jumping = (vel_z > 0.5) & (height > 0.4)
        reward = jumping.float() * 3.0

        return reward, 1.0

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
