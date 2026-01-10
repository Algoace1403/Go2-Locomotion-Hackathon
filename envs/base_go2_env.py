"""
Base Go2 Environment for Locomotion Training
Based on Genesis physics simulator
"""

import numpy as np
import torch
import genesis as gs
from typing import Dict, Tuple, Optional

import sys
sys.path.append('..')
from configs.base_config import BaseConfig


class Go2Env:
    """
    Base environment for Unitree Go2 quadruped robot.
    Implements gym-style interface for RL training.
    """

    def __init__(
        self,
        config: BaseConfig = None,
        num_envs: int = 4096,
        device: str = 'cuda',
        headless: bool = True,
        **kwargs
    ):
        """
        Initialize the Go2 environment.

        Args:
            config: Configuration object with hyperparameters
            num_envs: Number of parallel environments
            device: Device to run simulation ('cuda' or 'cpu')
            headless: Whether to run without visualization
        """
        self.config = config if config else BaseConfig()
        self.num_envs = num_envs
        self.device = device
        self.headless = headless

        # Simulation parameters
        self.dt = self.config.dt
        self.max_episode_length = self.config.max_episode_length

        # Robot parameters
        self.num_joints = self.config.num_joints
        self.default_dof_pos = torch.tensor(
            self.config.default_dof_pos,
            dtype=torch.float32,
            device=self.device
        )

        # Action parameters
        self.action_scale = self.config.action_scale
        self.clip_actions = self.config.clip_actions

        # Observation parameters
        self.num_obs = self.config.num_observations
        self.obs_scales = self.config.obs_scales

        # PD control gains
        self.kp = self.config.kp
        self.kd = self.config.kd

        # State buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs),
            dtype=torch.float32,
            device=self.device
        )
        self.reward_buf = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device
        )
        self.done_buf = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device
        )
        self.episode_length_buf = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device
        )

        # Action buffers (for action rate penalty)
        self.actions = torch.zeros(
            (self.num_envs, self.num_joints),
            dtype=torch.float32,
            device=self.device
        )
        self.prev_actions = torch.zeros_like(self.actions)

        # Command buffer
        self.commands = torch.zeros(
            (self.num_envs, 3),
            dtype=torch.float32,
            device=self.device
        )

        # Initialize simulation
        self._init_simulation()

        # Reward functions (to be overridden by subclasses)
        self.reward_functions = []
        self.reward_weights = {}

    def _init_simulation(self):
        """Initialize Genesis simulation"""
        # Initialize Genesis
        gs.init(backend=gs.gpu if self.device == 'cuda' else gs.cpu)

        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=self.config.substeps,
            ),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
            ) if not self.headless else None,
            show_viewer=not self.headless,
        )

        # Add ground plane
        self.plane = self.scene.add_entity(gs.morphs.Plane())

        # Add Go2 robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file='urdf/go2/urdf/go2.urdf',
                pos=(0, 0, 0.4),
            ),
        )

        # Build scene
        self.scene.build(n_envs=self.num_envs)

        # Get joint indices
        self.joint_names = [
            'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
            'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
            'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
            'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
        ]

        # Set PD control gains
        self.robot.set_dofs_kp(
            [self.kp] * self.num_joints,
            self.joint_names
        )
        self.robot.set_dofs_kd(
            [self.kd] * self.num_joints,
            self.joint_names
        )

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reset specified environments.

        Args:
            env_ids: Indices of environments to reset. If None, reset all.

        Returns:
            observations: Initial observations
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        # Reset robot state
        self._reset_robot(env_ids)

        # Reset buffers
        self.episode_length_buf[env_ids] = 0
        self.prev_actions[env_ids] = 0
        self.actions[env_ids] = 0

        # Sample new commands
        self._sample_commands(env_ids)

        # Compute initial observations
        self._compute_observations()

        return self.obs_buf

    def _reset_robot(self, env_ids: torch.Tensor):
        """Reset robot to default pose"""
        # Reset position and orientation
        self.robot.set_pos(
            torch.tensor([0, 0, 0.4], device=self.device).expand(len(env_ids), -1),
            zero_velocity=True,
            envs_idx=env_ids
        )
        self.robot.set_quat(
            torch.tensor([1, 0, 0, 0], device=self.device).expand(len(env_ids), -1),
            zero_velocity=True,
            envs_idx=env_ids
        )

        # Reset joint positions
        self.robot.set_dofs_position(
            self.default_dof_pos.expand(len(env_ids), -1),
            self.joint_names,
            zero_velocity=True,
            envs_idx=env_ids
        )

    def _sample_commands(self, env_ids: torch.Tensor):
        """Sample velocity commands for environments"""
        # Default: sample from command ranges (overridden by subclasses)
        num_reset = len(env_ids)

        # For base class, use config command_ranges if available
        if hasattr(self.config, 'command_ranges'):
            ranges = self.config.command_ranges
            self.commands[env_ids, 0] = torch.empty(num_reset, device=self.device).uniform_(
                ranges['lin_vel_x'][0], ranges['lin_vel_x'][1]
            )
            self.commands[env_ids, 1] = torch.empty(num_reset, device=self.device).uniform_(
                ranges['lin_vel_y'][0], ranges['lin_vel_y'][1]
            )
            self.commands[env_ids, 2] = torch.empty(num_reset, device=self.device).uniform_(
                ranges['ang_vel'][0], ranges['ang_vel'][1]
            )
        else:
            # Default commands
            self.commands[env_ids, 0] = 0.5  # Forward velocity
            self.commands[env_ids, 1] = 0.0  # Lateral velocity
            self.commands[env_ids, 2] = 0.0  # Angular velocity

    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Execute one environment step.

        Args:
            actions: Joint position targets (num_envs, 12)

        Returns:
            observations: New observations
            rewards: Step rewards
            dones: Episode termination flags
            info: Additional information
        """
        # Store previous actions
        self.prev_actions = self.actions.clone()

        # Process actions
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions)

        # Scale actions and add to default positions
        target_positions = self.default_dof_pos + self.actions * self.action_scale

        # Apply actions (with 1-step latency for realism)
        self.robot.control_dofs_position(
            target_positions,
            self.joint_names
        )

        # Step simulation
        self.scene.step()

        # Update episode length
        self.episode_length_buf += 1

        # Compute observations
        self._compute_observations()

        # Compute rewards
        self._compute_rewards()

        # Check termination
        self._check_termination()

        # Handle resets
        reset_ids = self.done_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_ids) > 0:
            self.reset(reset_ids)

        # Prepare info dict
        info = {
            'episode_length': self.episode_length_buf.clone(),
            'base_lin_vel': self._get_base_lin_vel(),
            'base_ang_vel': self._get_base_ang_vel(),
            'base_position': self._get_base_position(),
        }

        return self.obs_buf, self.reward_buf, self.done_buf, info

    def _compute_observations(self):
        """Compute observations for all environments"""
        # Get robot state
        base_ang_vel = self._get_base_ang_vel()
        projected_gravity = self._get_projected_gravity()
        dof_pos = self._get_dof_pos()
        dof_vel = self._get_dof_vel()

        # Build observation vector
        self.obs_buf = torch.cat([
            base_ang_vel * self.obs_scales['ang_vel'],           # 3
            projected_gravity,                                     # 3
            self.commands * self.config.command_scales['lin_vel'], # 3
            (dof_pos - self.default_dof_pos) * self.obs_scales['dof_pos'],  # 12
            dof_vel * self.obs_scales['dof_vel'],                 # 12
            self.actions,                                          # 12
        ], dim=-1)

    def _compute_rewards(self):
        """Compute rewards - to be overridden by subclasses"""
        self.reward_buf = torch.zeros(self.num_envs, device=self.device)

        for reward_fn in self.reward_functions:
            reward, weight = reward_fn()
            self.reward_buf += reward * weight

    def _check_termination(self):
        """Check termination conditions"""
        # Get robot orientation
        rpy = self._get_base_rpy()
        roll, pitch = rpy[:, 0], rpy[:, 1]

        # Get base height
        height = self._get_base_position()[:, 2]

        # Termination conditions
        term_config = self.config.termination

        roll_term = torch.abs(roll) > term_config['max_roll']
        pitch_term = torch.abs(pitch) > term_config['max_pitch']
        height_term = height < term_config['min_height']
        time_term = self.episode_length_buf >= self.max_episode_length

        self.done_buf = roll_term | pitch_term | height_term | time_term

    # ============== State Getters ==============

    def _get_base_position(self) -> torch.Tensor:
        """Get base position (x, y, z)"""
        return self.robot.get_pos()

    def _get_base_quat(self) -> torch.Tensor:
        """Get base quaternion (w, x, y, z)"""
        return self.robot.get_quat()

    def _get_base_rpy(self) -> torch.Tensor:
        """Get base roll, pitch, yaw"""
        quat = self._get_base_quat()
        return self._quat_to_rpy(quat)

    def _get_base_lin_vel(self) -> torch.Tensor:
        """Get base linear velocity in body frame"""
        return self.robot.get_vel()

    def _get_base_ang_vel(self) -> torch.Tensor:
        """Get base angular velocity in body frame"""
        return self.robot.get_ang()

    def _get_projected_gravity(self) -> torch.Tensor:
        """Get gravity vector projected into body frame"""
        quat = self._get_base_quat()
        gravity = torch.tensor([0, 0, -1], device=self.device).expand(self.num_envs, -1)
        return self._quat_rotate_inverse(quat, gravity)

    def _get_dof_pos(self) -> torch.Tensor:
        """Get joint positions"""
        return self.robot.get_dofs_position(self.joint_names)

    def _get_dof_vel(self) -> torch.Tensor:
        """Get joint velocities"""
        return self.robot.get_dofs_velocity(self.joint_names)

    def _get_dof_torque(self) -> torch.Tensor:
        """Get joint torques"""
        return self.robot.get_dofs_force(self.joint_names)

    # ============== Utility Functions ==============

    @staticmethod
    def _quat_to_rpy(quat: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to roll, pitch, yaw"""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * torch.tensor(np.pi / 2, device=quat.device),
            torch.asin(sinp)
        )

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)

        return torch.stack([roll, pitch, yaw], dim=-1)

    @staticmethod
    def _quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        """Rotate vector by inverse of quaternion"""
        w, x, y, z = quat[:, 0:1], quat[:, 1:2], quat[:, 2:3], quat[:, 3:4]

        # Quaternion conjugate
        quat_conj = torch.cat([w, -x, -y, -z], dim=-1)

        # q * v * q^(-1)
        vec_quat = torch.cat([torch.zeros_like(w), vec], dim=-1)

        # Hamilton product
        result = Go2Env._quat_multiply(
            Go2Env._quat_multiply(quat_conj, vec_quat),
            quat
        )

        return result[:, 1:4]

    @staticmethod
    def _quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Hamilton product of two quaternions"""
        w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
        w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return torch.stack([w, x, y, z], dim=-1)

    def close(self):
        """Clean up resources"""
        if hasattr(self, 'scene'):
            self.scene.destroy()
