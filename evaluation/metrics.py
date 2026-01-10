"""
Metrics Calculation for All Levels
"""

import numpy as np
import torch
from typing import Dict, List


class MetricsCalculator:
    """Calculate hackathon-specific metrics for each level"""

    @staticmethod
    def level1_metrics(velocities: np.ndarray, dt: float = 0.02) -> Dict:
        """
        Level 1: Walking Metrics
        Target: 0.5-1.0 m/s, max distance in 60s

        Args:
            velocities: Forward velocities over episode (T,)
            dt: Timestep

        Returns:
            Dict with metrics
        """
        avg_velocity = np.mean(velocities)
        max_velocity = np.max(velocities)
        min_velocity = np.min(velocities)
        std_velocity = np.std(velocities)

        # Total distance
        distance = np.sum(velocities * dt)

        # Time in target range
        in_range = (velocities >= 0.5) & (velocities <= 1.0)
        time_in_range = np.mean(in_range)

        # Success: avg velocity in range
        success = 0.5 <= avg_velocity <= 1.0

        return {
            'avg_velocity': avg_velocity,
            'max_velocity': max_velocity,
            'min_velocity': min_velocity,
            'std_velocity': std_velocity,
            'distance': distance,
            'time_in_range': time_in_range,
            'success': success,
        }

    @staticmethod
    def level2_metrics(velocities: np.ndarray, dt: float = 0.02) -> Dict:
        """
        Level 2: Running Metrics
        Target: ≥2.0 m/s

        Args:
            velocities: Forward velocities over episode (T,)
            dt: Timestep

        Returns:
            Dict with metrics
        """
        avg_velocity = np.mean(velocities)
        peak_velocity = np.max(velocities)

        # Time above threshold
        above_threshold = velocities >= 2.0
        time_above_2ms = np.mean(above_threshold)

        # Success: avg velocity >= 2.0
        success = avg_velocity >= 2.0

        return {
            'avg_velocity': avg_velocity,
            'peak_velocity': peak_velocity,
            'time_above_2ms': time_above_2ms,
            'success': success,
        }

    @staticmethod
    def level3_metrics(
        positions: np.ndarray,
        radius: float = 2.0,
        center: np.ndarray = np.array([0, 0])
    ) -> Dict:
        """
        Level 3: Circle Walking Metrics
        Target: RMSE from ideal circle over 10 rounds

        Args:
            positions: XY positions over episode (T, 2)
            radius: Target circle radius
            center: Circle center

        Returns:
            Dict with metrics
        """
        # Distance from center for each point
        distances = np.linalg.norm(positions - center, axis=1)

        # Radius errors
        radius_errors = distances - radius
        rmse = np.sqrt(np.mean(radius_errors ** 2))
        mae = np.mean(np.abs(radius_errors))

        # Average radius
        avg_radius = np.mean(distances)

        # Count laps (full rotations)
        angles = np.arctan2(positions[:, 1], positions[:, 0])
        angle_diff = np.diff(angles)

        # Handle wrap-around
        angle_diff = np.where(angle_diff > np.pi, angle_diff - 2*np.pi, angle_diff)
        angle_diff = np.where(angle_diff < -np.pi, angle_diff + 2*np.pi, angle_diff)

        total_rotation = np.abs(np.sum(angle_diff))
        laps = total_rotation / (2 * np.pi)

        return {
            'rmse': rmse,
            'mae': mae,
            'avg_radius': avg_radius,
            'laps_completed': laps,
        }

    @staticmethod
    def level4_metrics(
        angular_velocities: np.ndarray,
        linear_velocities: np.ndarray,
        positions: np.ndarray,
        start_position: np.ndarray
    ) -> Dict:
        """
        Level 4: Spin Metrics
        Target: High ω, v=0

        Args:
            angular_velocities: Yaw rates (T,)
            linear_velocities: XY velocities (T, 2)
            positions: XY positions (T, 2)
            start_position: Starting position (2,)

        Returns:
            Dict with metrics
        """
        avg_omega = np.mean(np.abs(angular_velocities))
        peak_omega = np.max(np.abs(angular_velocities))

        linear_speeds = np.linalg.norm(linear_velocities, axis=1)
        avg_linear = np.mean(linear_speeds)

        # Position drift
        drifts = np.linalg.norm(positions - start_position, axis=1)
        max_drift = np.max(drifts)
        avg_drift = np.mean(drifts)

        # Spin score: high omega with low linear velocity
        linear_penalty = min(avg_linear * 2, 1.0)
        spin_score = avg_omega * (1 - linear_penalty)

        return {
            'avg_angular_velocity': avg_omega,
            'peak_angular_velocity': peak_omega,
            'avg_linear_velocity': avg_linear,
            'max_drift': max_drift,
            'avg_drift': avg_drift,
            'spin_score': spin_score,
        }

    @staticmethod
    def print_summary(level: int, metrics: Dict):
        """Print formatted metrics summary"""
        print(f"\n{'='*50}")
        print(f"Level {level} Evaluation Results")
        print('='*50)

        for key, value in metrics.items():
            if isinstance(value, bool):
                status = "PASS" if value else "FAIL"
                print(f"  {key}: {status}")
            elif isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

        print('='*50)
