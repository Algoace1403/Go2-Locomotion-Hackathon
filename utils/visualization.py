"""
Visualization Utilities
Plotting, video recording, and metrics visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List, Dict, Optional
import json


class TrainingVisualizer:
    """Visualize training progress and results"""

    def __init__(self, log_dir: str = 'logs'):
        self.log_dir = log_dir
        self.fig_dir = os.path.join(log_dir, 'figures')
        os.makedirs(self.fig_dir, exist_ok=True)

    def plot_reward_curve(self, rewards: List[float], title: str = "Training Rewards",
                          save_path: Optional[str] = None):
        """Plot reward curve over training iterations"""
        fig, ax = plt.subplots(figsize=(10, 6))

        iterations = range(len(rewards))
        ax.plot(iterations, rewards, 'b-', linewidth=2, label='Episode Reward')

        # Add moving average
        window = min(50, len(rewards) // 5)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window-1, len(rewards)), moving_avg, 'r-',
                   linewidth=2, label=f'Moving Avg ({window})')

        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    def plot_velocity_profile(self, velocities: np.ndarray, target_range: tuple = None,
                              title: str = "Velocity Profile"):
        """Plot velocity over time with target range"""
        fig, ax = plt.subplots(figsize=(12, 5))

        time = np.arange(len(velocities)) * 0.02  # Assuming 50Hz
        ax.plot(time, velocities, 'b-', linewidth=1.5, label='Velocity')

        # Add target range
        if target_range:
            ax.axhline(target_range[0], color='g', linestyle='--', label='Min Target')
            ax.axhline(target_range[1], color='r', linestyle='--', label='Max Target')
            ax.fill_between(time, target_range[0], target_range[1],
                           alpha=0.2, color='green', label='Target Range')

        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Velocity (m/s)', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_circle_trajectory(self, positions: np.ndarray, target_radius: float = 2.0,
                               center: np.ndarray = np.array([0, 0])):
        """Plot circle walking trajectory"""
        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot ideal circle
        circle = Circle(center, target_radius, fill=False, color='green',
                       linestyle='--', linewidth=2, label='Target Circle')
        ax.add_patch(circle)

        # Plot actual trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1.5,
               alpha=0.7, label='Actual Path')

        # Mark start and end
        ax.scatter(positions[0, 0], positions[0, 1], c='green', s=100,
                  marker='o', label='Start', zorder=5)
        ax.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100,
                  marker='x', label='End', zorder=5)

        # Mark center
        ax.scatter(center[0], center[1], c='black', s=50, marker='+', zorder=5)

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Circle Walking Trajectory', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_spin_analysis(self, angular_vels: np.ndarray, linear_vels: np.ndarray,
                           positions: np.ndarray, start_pos: np.ndarray):
        """Plot spin performance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        time = np.arange(len(angular_vels)) * 0.02

        # Angular velocity
        axes[0, 0].plot(time, angular_vels, 'b-', linewidth=1.5)
        axes[0, 0].axhline(2.0, color='g', linestyle='--', label='Target (2 rad/s)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Angular Velocity (rad/s)')
        axes[0, 0].set_title('Angular Velocity')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Linear velocity
        linear_speeds = np.linalg.norm(linear_vels, axis=1)
        axes[0, 1].plot(time, linear_speeds, 'r-', linewidth=1.5)
        axes[0, 1].axhline(0, color='g', linestyle='--', label='Target (0 m/s)')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Linear Speed (m/s)')
        axes[0, 1].set_title('Linear Velocity (should be ~0)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Position drift
        drifts = np.linalg.norm(positions - start_pos, axis=1)
        axes[1, 0].plot(time, drifts, 'm-', linewidth=1.5)
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Drift (m)')
        axes[1, 0].set_title('Position Drift from Start')
        axes[1, 0].grid(True, alpha=0.3)

        # XY trajectory
        axes[1, 1].plot(positions[:, 0], positions[:, 1], 'b-', linewidth=1, alpha=0.5)
        axes[1, 1].scatter(start_pos[0], start_pos[1], c='green', s=100,
                          marker='o', label='Start', zorder=5)
        axes[1, 1].scatter(positions[-1, 0], positions[-1, 1], c='red', s=100,
                          marker='x', label='End', zorder=5)
        axes[1, 1].set_xlabel('X (m)')
        axes[1, 1].set_ylabel('Y (m)')
        axes[1, 1].set_title('XY Position (should stay centered)')
        axes[1, 1].set_aspect('equal')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_dance_rhythm(self, joint_velocities: np.ndarray, beat_phases: np.ndarray,
                          bpm: int = 120):
        """Plot dance rhythm synchronization"""
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))

        time = np.arange(len(beat_phases)) * 0.02

        # Movement intensity vs beat
        movement_intensity = np.mean(np.abs(joint_velocities), axis=1)
        axes[0].plot(time, movement_intensity, 'b-', linewidth=1, alpha=0.7,
                    label='Movement Intensity')

        # Mark beats
        beat_period = 60.0 / bpm
        beat_times = np.arange(0, time[-1], beat_period)
        for bt in beat_times:
            axes[0].axvline(bt, color='r', alpha=0.3, linewidth=1)

        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Movement Intensity')
        axes[0].set_title(f'Dance Rhythm Sync (BPM: {bpm})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Beat phase
        axes[1].plot(time, beat_phases, 'g-', linewidth=1.5)
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Beat Phase (0-1)')
        axes[1].set_title('Beat Phase Over Time')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def create_metrics_dashboard(self, metrics: Dict, level: int):
        """Create a comprehensive metrics dashboard"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')

        # Title
        level_names = {
            1: "Walking",
            2: "Running",
            3: "Circle",
            4: "Spin",
            5: "Dance"
        }

        title = f"Level {level}: {level_names.get(level, 'Unknown')} - Results"
        ax.text(0.5, 0.95, title, fontsize=18, fontweight='bold',
               ha='center', transform=ax.transAxes)

        # Metrics table
        y_pos = 0.8
        for key, value in metrics.items():
            if isinstance(value, float):
                text = f"{key}: {value:.4f}"
            elif isinstance(value, bool):
                text = f"{key}: {'PASS' if value else 'FAIL'}"
            else:
                text = f"{key}: {value}"

            ax.text(0.1, y_pos, text, fontsize=12,
                   transform=ax.transAxes, family='monospace')
            y_pos -= 0.08

        plt.tight_layout()
        plt.show()

    def save_metrics_json(self, metrics: Dict, filename: str):
        """Save metrics to JSON file"""
        filepath = os.path.join(self.log_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"Metrics saved to: {filepath}")


class DemoRecorder:
    """Record demo videos of trained policies"""

    def __init__(self, output_dir: str = 'presentation/videos'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def record_episode(self, env, policy, num_steps: int = 500,
                       filename: str = 'demo.mp4'):
        """
        Record an episode as video.
        Note: Requires environment to support rendering.
        """
        import torch

        frames = []
        obs = env.reset()

        for step in range(num_steps):
            with torch.no_grad():
                action = policy.act_inference(obs)

            obs, reward, done, info = env.step(action)

            # Get frame if rendering is available
            if hasattr(env, 'render'):
                frame = env.render(mode='rgb_array')
                frames.append(frame)

            if done.any():
                break

        # Save video if frames were captured
        if frames:
            self._save_video(frames, os.path.join(self.output_dir, filename))
            print(f"Demo saved to: {os.path.join(self.output_dir, filename)}")
        else:
            print("No frames captured. Enable rendering in environment.")

    def _save_video(self, frames: List[np.ndarray], filepath: str, fps: int = 30):
        """Save frames as video using matplotlib animation or imageio"""
        try:
            import imageio
            imageio.mimsave(filepath, frames, fps=fps)
        except ImportError:
            print("Install imageio for video saving: pip install imageio[ffmpeg]")


def plot_level_comparison(all_metrics: Dict[int, Dict]):
    """Compare metrics across all levels"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    levels = list(all_metrics.keys())
    level_names = ['Walk', 'Run', 'Circle', 'Spin', 'Dance']

    # Success rates
    success_rates = [all_metrics.get(l, {}).get('success_rate', 0) for l in range(1, 6)]
    axes[0, 0].bar(level_names, success_rates, color=['green' if s > 0.8 else 'orange' if s > 0.5 else 'red' for s in success_rates])
    axes[0, 0].set_ylabel('Success Rate')
    axes[0, 0].set_title('Success Rate by Level')
    axes[0, 0].set_ylim(0, 1)

    # Velocities (for levels 1-2)
    velocities = [
        all_metrics.get(1, {}).get('avg_velocity_mean', 0),
        all_metrics.get(2, {}).get('avg_velocity_mean', 0),
        0, 0, 0
    ]
    targets = [0.75, 2.0, 0, 0, 0]
    x = np.arange(5)
    axes[0, 1].bar(x - 0.2, velocities[:5], 0.4, label='Achieved', color='blue')
    axes[0, 1].bar(x + 0.2, targets, 0.4, label='Target', color='green', alpha=0.5)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(level_names)
    axes[0, 1].set_ylabel('Velocity (m/s)')
    axes[0, 1].set_title('Velocity Achievement')
    axes[0, 1].legend()

    # Points achieved (mock data - replace with actual)
    points_possible = [20, 40, 10, 10, 0]  # Dance is bonus
    points_achieved = [
        20 if all_metrics.get(1, {}).get('success', False) else 0,
        40 if all_metrics.get(2, {}).get('success', False) else 0,
        10 if all_metrics.get(3, {}).get('rmse', float('inf')) < 0.5 else 0,
        10 if all_metrics.get(4, {}).get('spin_score', 0) > 1.0 else 0,
        0  # Bonus
    ]
    axes[0, 2].bar(level_names, points_possible, alpha=0.5, label='Possible', color='gray')
    axes[0, 2].bar(level_names, points_achieved, label='Achieved', color='green')
    axes[0, 2].set_ylabel('Points')
    axes[0, 2].set_title('Points Achievement')
    axes[0, 2].legend()

    # Hide unused subplots
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')

    # Total score
    total = sum(points_achieved)
    axes[1, 1].text(0.5, 0.5, f"Total Score: {total}/80 points\n(+20 presentation)",
                   fontsize=20, ha='center', va='center',
                   transform=axes[1, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.show()
