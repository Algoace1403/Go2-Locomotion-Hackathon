"""
Unified Go2 Environment for Walk, Run, and Jump
Clean implementation based on official Genesis locomotion example
"""

import math
import torch
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat


def gs_rand_float(lower, upper, shape, device):
    """Generate random floats in range [lower, upper]"""
    return (upper - lower) * torch.rand(shape, device=device, dtype=torch.float32) + lower


class Go2Env:
    """
    Unified Go2 environment supporting walk, run, and jump modes.

    Modes:
    - walk: Slow stable locomotion (0.3-0.8 m/s)
    - run: Fast locomotion (1.0-2.5 m/s)
    - jump: Jumping behavior with air time rewards
    """

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False, device=None):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]

        # Device setup
        self.device = device if device else gs.device

        # Simulation parameters
        self.simulate_action_latency = env_cfg.get("simulate_action_latency", True)
        self.dt = 0.02  # 50Hz control frequency
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        # Store configs
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"].copy()

        # Mode (walk, run, jump)
        self.mode = env_cfg.get("mode", "walk")

        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=2,
            ),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
            ),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=[0],
            ),
            show_viewer=show_viewer,
        )

        # Add ground plane
        self.scene.add_entity(
            gs.morphs.Plane(),
        )

        # Add robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            ),
        )

        # Build scene
        self.scene.build(n_envs=num_envs)

        # Get motor DOF indices
        self.motor_dofs = [self.robot.get_joint(name).dofs_idx_local[0] for name in env_cfg["joint_names"]]

        # Set PD gains
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motor_dofs)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motor_dofs)

        # Gravity vector
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)

        # Initial state tensors
        self.base_init_pos = torch.tensor(env_cfg["base_init_pos"], device=self.device, dtype=torch.float32)
        self.base_init_quat = torch.tensor(env_cfg["base_init_quat"], device=self.device, dtype=torch.float32)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)

        # Default joint positions
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]],
            device=self.device,
            dtype=torch.float32,
        )

        # Initialize all buffers
        self._init_buffers()

        # Prepare reward functions (scale by dt)
        self.reward_functions = {}
        self.episode_sums = {}
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            reward_fn = getattr(self, f"_reward_{name}", None)
            if reward_fn is not None:
                self.reward_functions[name] = reward_fn
                self.episode_sums[name] = torch.zeros(num_envs, device=self.device, dtype=torch.float32)

        # Extra info dict
        self.extras = {"observations": {}}

    def _init_buffers(self):
        """Initialize all state and observation buffers"""
        n = self.num_envs
        device = self.device

        # Robot state buffers
        self.base_pos = torch.zeros((n, 3), device=device, dtype=torch.float32)
        self.base_quat = torch.zeros((n, 4), device=device, dtype=torch.float32)
        self.base_lin_vel = torch.zeros((n, 3), device=device, dtype=torch.float32)
        self.base_ang_vel = torch.zeros((n, 3), device=device, dtype=torch.float32)
        self.base_euler = torch.zeros((n, 3), device=device, dtype=torch.float32)
        self.projected_gravity = torch.zeros((n, 3), device=device, dtype=torch.float32)

        # Joint state buffers
        self.dof_pos = torch.zeros((n, self.num_actions), device=device, dtype=torch.float32)
        self.dof_vel = torch.zeros((n, self.num_actions), device=device, dtype=torch.float32)
        self.last_dof_vel = torch.zeros((n, self.num_actions), device=device, dtype=torch.float32)

        # Action buffers
        self.actions = torch.zeros((n, self.num_actions), device=device, dtype=torch.float32)
        self.last_actions = torch.zeros((n, self.num_actions), device=device, dtype=torch.float32)

        # Command buffers
        self.commands = torch.zeros((n, self.num_commands), device=device, dtype=torch.float32)

        # Episode buffers
        self.episode_length_buf = torch.zeros(n, device=device, dtype=torch.int32)
        self.reset_buf = torch.ones(n, device=device, dtype=torch.bool)
        self.rew_buf = torch.zeros(n, device=device, dtype=torch.float32)
        self.obs_buf = torch.zeros((n, self.num_obs), device=device, dtype=torch.float32)

        # Foot contact tracking (for jump rewards)
        self.feet_air_time = torch.zeros(n, device=device, dtype=torch.float32)
        self.last_base_height = torch.zeros(n, device=device, dtype=torch.float32)

    def _resample_commands(self, env_ids):
        """Resample velocity commands for specified environments"""
        if env_ids is None or len(env_ids) == 0:
            return

        n = len(env_ids) if isinstance(env_ids, torch.Tensor) else self.num_envs

        # Sample new commands
        self.commands[env_ids, 0] = gs_rand_float(
            self.command_cfg["lin_vel_x_range"][0],
            self.command_cfg["lin_vel_x_range"][1],
            (n,), self.device
        )
        self.commands[env_ids, 1] = gs_rand_float(
            self.command_cfg["lin_vel_y_range"][0],
            self.command_cfg["lin_vel_y_range"][1],
            (n,), self.device
        )
        self.commands[env_ids, 2] = gs_rand_float(
            self.command_cfg["ang_vel_range"][0],
            self.command_cfg["ang_vel_range"][1],
            (n,), self.device
        )

    def step(self, actions):
        """Execute one environment step"""
        # Clip and store actions
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])

        # Apply action latency simulation
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions

        # Compute target positions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos

        # Apply PD control
        self.robot.control_dofs_position(target_dof_pos, self.motor_dofs)

        # Step simulation
        self.scene.step()

        # Update episode counter
        self.episode_length_buf += 1

        # Update robot state
        self._update_state()

        # Compute rewards
        self._compute_rewards()

        # Check terminations
        self._check_termination()

        # Resample commands periodically
        resample_ids = (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(resample_ids)

        # Reset terminated environments
        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_ids) > 0:
            self._reset_idx(reset_ids)

        # Update observations
        self._update_observation()

        # Store previous actions
        self.last_actions.copy_(self.actions)
        self.last_dof_vel.copy_(self.dof_vel)
        self.last_base_height.copy_(self.base_pos[:, 2])

        self.extras["observations"]["critic"] = self.obs_buf
        self.extras["time_outs"] = (self.episode_length_buf >= self.max_episode_length).float()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _update_state(self):
        """Update all robot state variables"""
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()

        inv_base_quat = inv_quat(self.base_quat)

        # Transform velocities to body frame
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)

        # Projected gravity
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)

        # Euler angles for termination check
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat),
            rpy=True, degrees=True
        )

        # Joint states
        self.dof_pos = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel = self.robot.get_dofs_velocity(self.motor_dofs)

        # Track air time for jumping
        is_in_air = self.base_pos[:, 2] > self.reward_cfg.get("jump_height_threshold", 0.35)
        self.feet_air_time = torch.where(is_in_air, self.feet_air_time + self.dt, torch.zeros_like(self.feet_air_time))

    def _compute_rewards(self):
        """Compute all rewards"""
        self.rew_buf.zero_()

        for name, reward_fn in self.reward_functions.items():
            rew = reward_fn() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

    def _check_termination(self):
        """Check episode termination conditions"""
        # Episode length exceeded
        self.reset_buf = self.episode_length_buf >= self.max_episode_length

        # Roll/pitch limits
        roll_limit = self.env_cfg.get("termination_if_roll_greater_than", 30)
        pitch_limit = self.env_cfg.get("termination_if_pitch_greater_than", 30)

        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > roll_limit
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > pitch_limit

        # Height too low (fallen)
        min_height = self.env_cfg.get("termination_if_height_lower_than", 0.15)
        self.reset_buf |= self.base_pos[:, 2] < min_height

    def _reset_idx(self, env_ids):
        """Reset specified environments"""
        if len(env_ids) == 0:
            return

        # Reset robot pose
        pos = self.base_init_pos.unsqueeze(0).expand(len(env_ids), -1)
        quat = self.base_init_quat.unsqueeze(0).expand(len(env_ids), -1)
        dof_pos = self.default_dof_pos.unsqueeze(0).expand(len(env_ids), -1)

        self.robot.set_pos(pos, zero_velocity=True, envs_idx=env_ids)
        self.robot.set_quat(quat, zero_velocity=True, envs_idx=env_ids)
        self.robot.set_dofs_position(dof_pos, self.motor_dofs, zero_velocity=True, envs_idx=env_ids)

        # Reset buffers
        self.base_pos[env_ids] = self.base_init_pos
        self.base_quat[env_ids] = self.base_init_quat
        self.base_lin_vel[env_ids] = 0
        self.base_ang_vel[env_ids] = 0
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0
        self.actions[env_ids] = 0
        self.last_actions[env_ids] = 0
        self.last_dof_vel[env_ids] = 0
        self.episode_length_buf[env_ids] = 0
        self.feet_air_time[env_ids] = 0
        self.last_base_height[env_ids] = self.base_init_pos[2]

        # Log episode rewards before reset
        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            if len(env_ids) > 0:
                self.extras["episode"][f"rew_{key}"] = value[env_ids].mean() / self.env_cfg["episode_length_s"]
            self.episode_sums[key][env_ids] = 0

        # Resample commands
        self._resample_commands(env_ids)

    def _update_observation(self):
        """Build observation vector"""
        self.obs_buf = torch.cat([
            self.base_ang_vel * self.obs_scales["ang_vel"],           # 3
            self.projected_gravity,                                     # 3
            self.commands * self.obs_scales.get("commands", 1.0),      # 3
            (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
            self.dof_vel * self.obs_scales["dof_vel"],                 # 12
            self.actions,                                               # 12
        ], dim=-1)

    def reset(self):
        """Reset all environments"""
        self.reset_buf.fill_(True)
        all_ids = torch.arange(self.num_envs, device=self.device)
        self._reset_idx(all_ids)
        self._update_observation()
        return self.obs_buf, None

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # ==================== REWARD FUNCTIONS ====================

    def _reward_tracking_lin_vel(self):
        """Reward tracking linear velocity commands (xy)"""
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        """Reward tracking angular velocity command (yaw)"""
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        """Penalize vertical velocity (bouncing)"""
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        """Penalize roll/pitch angular velocity"""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_orientation(self):
        """Penalize non-upright orientation"""
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """Penalize deviation from target base height"""
        target = self.reward_cfg.get("base_height_target", 0.3)
        return torch.square(self.base_pos[:, 2] - target)

    def _reward_action_rate(self):
        """Penalize action changes (jerkiness)"""
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)

    def _reward_action_smoothness(self):
        """Penalize second derivative of actions"""
        return torch.sum(torch.square(self.actions - 2 * self.last_actions), dim=1)

    def _reward_similar_to_default(self):
        """Penalize deviation from default joint pose"""
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_dof_vel(self):
        """Penalize high joint velocities"""
        return torch.sum(torch.square(self.dof_vel), dim=1)

    def _reward_dof_acc(self):
        """Penalize joint accelerations"""
        return torch.sum(torch.square(self.dof_vel - self.last_dof_vel), dim=1)

    def _reward_torques(self):
        """Penalize high torques (energy efficiency)"""
        # Approximate torques from PD error
        torques = self.env_cfg["kp"] * (self.actions * self.env_cfg["action_scale"]) - self.env_cfg["kd"] * self.dof_vel
        return torch.sum(torch.square(torques), dim=1)

    def _reward_energy(self):
        """Penalize energy consumption"""
        # power = torque * velocity
        torques = self.env_cfg["kp"] * (self.actions * self.env_cfg["action_scale"]) - self.env_cfg["kd"] * self.dof_vel
        power = torch.abs(torques * self.dof_vel)
        return torch.sum(power, dim=1)

    def _reward_forward_vel(self):
        """Reward forward velocity (for running)"""
        return self.base_lin_vel[:, 0]

    def _reward_survival(self):
        """Reward for staying alive"""
        return torch.ones(self.num_envs, device=self.device)

    # ==================== JUMP-SPECIFIC REWARDS ====================

    def _reward_jump_height(self):
        """Reward for jump height"""
        target = self.reward_cfg.get("jump_height_target", 0.5)
        height = self.base_pos[:, 2]
        return torch.clamp(height - 0.3, min=0) / target

    def _reward_air_time(self):
        """Reward for time spent in air during jump"""
        return self.feet_air_time

    def _reward_jump_upward_vel(self):
        """Reward upward velocity for jumping"""
        return torch.clamp(self.base_lin_vel[:, 2], min=0)

    def _reward_land_stable(self):
        """Reward for stable landing after jump"""
        is_landed = self.base_pos[:, 2] < 0.35
        was_jumping = self.feet_air_time > 0.1
        stable = torch.abs(self.base_euler[:, 0]) < 10
        stable &= torch.abs(self.base_euler[:, 1]) < 10
        return (is_landed & was_jumping & stable).float()

    def _reward_feet_contact_forces(self):
        """Penalize high foot contact forces"""
        # Approximate from base height change rate
        impact = torch.abs(self.base_pos[:, 2] - self.last_base_height) / self.dt
        return torch.clamp(impact - 1.0, min=0)

    # ==================== SPIN-SPECIFIC REWARDS ====================

    def _reward_spin_vel(self):
        """Reward for spinning (high yaw angular velocity)"""
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_spin_stability(self):
        """Reward for staying upright while spinning"""
        upright = 1.0 - torch.abs(self.projected_gravity[:, 2] + 1.0)
        return torch.clamp(upright, min=0)

    def _reward_no_lateral_vel(self):
        """Penalize lateral/forward movement while spinning"""
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
