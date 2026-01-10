"""
Unified Evaluator for All Levels
"""

import os
import numpy as np
import torch
from typing import Dict, Optional
from rsl_rl.modules import ActorCritic

from .metrics import MetricsCalculator


class Evaluator:
    """Evaluate trained policies for all levels"""

    def __init__(self, env, policy: ActorCritic, device: str = 'cuda'):
        """
        Initialize evaluator.

        Args:
            env: Environment instance
            policy: Trained actor-critic policy
            device: Device for inference
        """
        self.env = env
        self.policy = policy
        self.device = device
        self.metrics = MetricsCalculator()

        # Set policy to eval mode
        self.policy.eval()

    @torch.no_grad()
    def evaluate_level1(self, num_episodes: int = 10, episode_length: int = 3000) -> Dict:
        """
        Evaluate Level 1: Walking
        60 seconds, measure velocity and distance

        Args:
            num_episodes: Number of evaluation episodes
            episode_length: Steps per episode (3000 = 60s at 50Hz)

        Returns:
            Aggregated metrics
        """
        all_metrics = []

        for ep in range(num_episodes):
            obs = self.env.reset()
            velocities = []

            for step in range(episode_length):
                # Get action from policy
                action = self.policy.act_inference(obs)

                # Step environment
                obs, reward, done, info = self.env.step(action)

                # Record velocity
                vel_x = info['base_lin_vel'][0, 0].cpu().numpy()
                velocities.append(vel_x)

                if done.any():
                    break

            # Compute metrics for this episode
            velocities = np.array(velocities)
            ep_metrics = self.metrics.level1_metrics(velocities, self.env.dt)
            all_metrics.append(ep_metrics)

        # Aggregate metrics
        aggregated = self._aggregate_metrics(all_metrics)
        self.metrics.print_summary(1, aggregated)

        return aggregated

    @torch.no_grad()
    def evaluate_level2(self, num_episodes: int = 10, episode_length: int = 1000) -> Dict:
        """
        Evaluate Level 2: Running
        20 seconds, measure if ≥2.0 m/s achieved

        Args:
            num_episodes: Number of evaluation episodes
            episode_length: Steps per episode (1000 = 20s at 50Hz)

        Returns:
            Aggregated metrics
        """
        all_metrics = []

        for ep in range(num_episodes):
            obs = self.env.reset()
            velocities = []

            for step in range(episode_length):
                action = self.policy.act_inference(obs)
                obs, reward, done, info = self.env.step(action)

                vel_x = info['base_lin_vel'][0, 0].cpu().numpy()
                velocities.append(vel_x)

                if done.any():
                    break

            velocities = np.array(velocities)
            ep_metrics = self.metrics.level2_metrics(velocities, self.env.dt)
            all_metrics.append(ep_metrics)

        aggregated = self._aggregate_metrics(all_metrics)
        self.metrics.print_summary(2, aggregated)

        return aggregated

    @torch.no_grad()
    def evaluate_level3(self, num_laps: int = 10) -> Dict:
        """
        Evaluate Level 3: Circle Walking
        Complete 10 laps, measure RMSE

        Args:
            num_laps: Number of laps to complete

        Returns:
            Metrics including RMSE
        """
        obs = self.env.reset()
        positions = []

        # Track laps
        laps_completed = 0
        prev_angle = 0
        cumulative_angle = 0

        max_steps = 20000  # Safety limit

        for step in range(max_steps):
            action = self.policy.act_inference(obs)
            obs, reward, done, info = self.env.step(action)

            pos = info['base_position'][0, :2].cpu().numpy()
            positions.append(pos.copy())

            # Track lap completion
            angle = np.arctan2(pos[1], pos[0])
            angle_diff = angle - prev_angle

            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi

            cumulative_angle += angle_diff
            prev_angle = angle

            if abs(cumulative_angle) >= 2 * np.pi:
                laps_completed += 1
                cumulative_angle = 0

                if laps_completed >= num_laps:
                    break

            if done.any():
                break

        positions = np.array(positions)
        metrics = self.metrics.level3_metrics(
            positions,
            radius=self.env.config.circle_radius,
            center=np.array(self.env.config.circle_center)
        )
        metrics['laps_target'] = num_laps
        metrics['laps_completed'] = laps_completed

        self.metrics.print_summary(3, metrics)

        return metrics

    @torch.no_grad()
    def evaluate_level4(self, num_episodes: int = 10, episode_length: int = 1000) -> Dict:
        """
        Evaluate Level 4: Spin in Place
        20 seconds, measure angular velocity and drift

        Args:
            num_episodes: Number of evaluation episodes
            episode_length: Steps per episode

        Returns:
            Aggregated metrics
        """
        all_metrics = []

        for ep in range(num_episodes):
            obs = self.env.reset()

            angular_velocities = []
            linear_velocities = []
            positions = []
            start_position = None

            for step in range(episode_length):
                action = self.policy.act_inference(obs)
                obs, reward, done, info = self.env.step(action)

                pos = info['base_position'][0, :2].cpu().numpy()
                ang_vel = info['base_ang_vel'][0, 2].cpu().numpy()
                lin_vel = info['base_lin_vel'][0, :2].cpu().numpy()

                if start_position is None:
                    start_position = pos.copy()

                positions.append(pos.copy())
                angular_velocities.append(ang_vel)
                linear_velocities.append(lin_vel.copy())

                if done.any():
                    break

            ep_metrics = self.metrics.level4_metrics(
                np.array(angular_velocities),
                np.array(linear_velocities),
                np.array(positions),
                start_position
            )
            all_metrics.append(ep_metrics)

        aggregated = self._aggregate_metrics(all_metrics)
        self.metrics.print_summary(4, aggregated)

        return aggregated

    def _aggregate_metrics(self, metrics_list: list) -> Dict:
        """Aggregate metrics from multiple episodes"""
        if not metrics_list:
            return {}

        aggregated = {}
        keys = metrics_list[0].keys()

        for key in keys:
            values = [m[key] for m in metrics_list]

            if isinstance(values[0], bool):
                # For boolean: compute success rate
                aggregated[key] = np.mean(values)
                aggregated[f'{key}_rate'] = f"{np.mean(values)*100:.1f}%"
            else:
                # For numeric: compute mean and std
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)

        return aggregated

    def run_all_evaluations(self) -> Dict:
        """Run evaluation for all levels"""
        print("\n" + "=" * 60)
        print("RUNNING FULL EVALUATION SUITE")
        print("=" * 60)

        results = {}

        # Note: Each level requires its own environment and policy
        # This is a template - actual implementation needs level-specific setup

        return results


def load_policy(checkpoint_path: str, num_obs: int, num_actions: int,
                network_config: dict, device: str = 'cuda') -> ActorCritic:
    """
    Load trained policy from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        num_obs: Number of observations
        num_actions: Number of actions
        network_config: Network architecture config
        device: Device to load to

    Returns:
        Loaded policy
    """
    policy = ActorCritic(
        num_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=network_config['actor_hidden_dims'],
        critic_hidden_dims=network_config['critic_hidden_dims'],
        activation=network_config['activation'],
        init_noise_std=network_config['init_noise_std'],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.load_state_dict(checkpoint['model_state_dict'])

    return policy
