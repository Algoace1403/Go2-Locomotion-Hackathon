# Unitree Go2 Locomotion Hackathon - Complete Architecture Document

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Control Theory Foundation](#2-control-theory-foundation)
3. [System Architecture](#3-system-architecture)
4. [Project Structure](#4-project-structure)
5. [Technical Stack](#5-technical-stack)
6. [Environment Design](#6-environment-design)
7. [Observation Space](#7-observation-space)
8. [Action Space](#8-action-space)
9. [Reward Engineering (All Levels)](#9-reward-engineering-all-levels)
10. [Training Pipeline](#10-training-pipeline)
11. [Evaluation & Metrics](#11-evaluation--metrics)
12. [Implementation Timeline](#12-implementation-timeline)
13. [Presentation Strategy](#13-presentation-strategy)

---

## 1. Executive Summary

### Hackathon Objective
Train a Unitree Go2 quadruped robot using Reinforcement Learning (RL) in the Genesis physics simulator to perform four locomotion behaviors.

### Scoring Breakdown
| Level | Task | Points | Priority |
|-------|------|--------|----------|
| Level 1 | Walking (0.5-1.0 m/s) | 20 | HIGH |
| Level 2 | Running (≥2.0 m/s) | 40 | CRITICAL |
| Level 3 | Circle Walking (R=2m) | 10 (Bonus) | MEDIUM |
| Level 4 | Spin in Place | 10 (Bonus) | MEDIUM |
| Presentation | Technical Explanation | 20 | HIGH |

**Total: 100 points (80 base + 20 bonus)**

### Core Philosophy
> "You are not the pilot; you are the coach."

We design **reward functions** that incentivize desired behaviors. The PPO algorithm learns motor commands to maximize cumulative reward.

---

## 2. Control Theory Foundation

The hackathon provides a DC Motor Position Control tutorial as foundational knowledge. Understanding this helps grasp how low-level motor control works before applying RL.

### 2.1 DC Motor Physical Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DC MOTOR SYSTEM MODEL                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    ELECTRICAL CIRCUIT                                │   │
│   │                                                                      │   │
│   │     V(t) ──┬──[R]──[L]──┬── Motor ──┐                               │   │
│   │            │            │           │                                │   │
│   │            │         Back-EMF       │                                │   │
│   │            │         (K·θ̇)         │                                │   │
│   │            └────────────────────────┘                                │   │
│   │                                                                      │   │
│   │   Kirchhoff's Voltage Law:                                          │   │
│   │   L(di/dt) + Ri = V - K·θ̇                                          │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    MECHANICAL SYSTEM                                 │   │
│   │                                                                      │   │
│   │            Torque (τ = K·i)                                         │   │
│   │                 │                                                    │   │
│   │                 ▼                                                    │   │
│   │   ┌─────────────────────────┐                                       │   │
│   │   │     Rotor (J, b)        │──► θ (position)                       │   │
│   │   │   Inertia + Friction    │                                       │   │
│   │   └─────────────────────────┘                                       │   │
│   │                                                                      │   │
│   │   Newton's 2nd Law:                                                 │   │
│   │   J·θ̈ + b·θ̇ = K·i                                                 │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Motor Parameters (Go2 Reference Values)

```python
# DC Motor Physical Parameters
MOTOR_PARAMS = {
    'J': 3.2284e-6,    # Moment of inertia (kg·m²)
    'b': 3.5077e-6,    # Viscous friction constant (N·m·s)
    'K': 0.0274,       # Motor constant (V/rad/s and N·m/A)
    'R': 4.0,          # Armature resistance (Ω)
    'L': 2.75e-6,      # Armature inductance (H)
}
```

### 2.3 Transfer Function

The open-loop transfer function from Voltage (V) to Position (θ):

```
                    K
Θ(s)/V(s) = ─────────────────────────────
            s·[(Js + b)(Ls + R) + K²]
```

**Expanded form:**
```
                           K
Θ(s)/V(s) = ─────────────────────────────────────────
            JL·s³ + (JR + bL)·s² + (bR + K²)·s
```

### 2.4 Control System Implementation

```python
import numpy as np
from scipy import signal

# Motor Parameters
J = 3.2284e-6   # Moment of inertia (kg·m²)
b = 3.5077e-6   # Viscous friction (N·m·s)
K = 0.0274      # Motor constant
R = 4.0         # Resistance (Ω)
L = 2.75e-6     # Inductance (H)

# Transfer Function: Θ(s)/V(s)
# Numerator: K
num = [K]

# Denominator: JL·s³ + (JR + bL)·s² + (bR + K²)·s + 0
den = [
    J * L,              # s³ coefficient
    (J * R) + (b * L),  # s² coefficient
    (b * R) + K**2,     # s¹ coefficient
    0                   # s⁰ coefficient (integrator)
]

# Create Transfer Function
sys_motor = signal.TransferFunction(num, den)
```

### 2.5 Proportional Control (P-Controller)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CLOSED-LOOP CONTROL SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Reference   ┌───────┐  Error   ┌────────────┐  Voltage  ┌─────────┐       │
│   Position ──►│   +   │────────►│ Controller │─────────►│  Motor  │──┬──►θ │
│   (θ_ref)     │   Σ   │         │   (Kp)     │    (V)    │  Plant  │  │     │
│               └───┬───┘         └────────────┘          └─────────┘  │     │
│                   │ -                                                 │     │
│                   │                                                   │     │
│                   └───────────────────────────────────────────────────┘     │
│                                    Feedback (θ)                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

Control Law: V = Kp · (θ_ref - θ)
```

```python
# Proportional Controller
Kp = 1.0  # Proportional gain

# Closed-Loop Transfer Function
# T(s) = Kp·G(s) / (1 + Kp·G(s))

# Open loop: Kp * Plant
num_ol = [Kp * K]
den_ol = den

# Closed loop (unity feedback)
num_cl = num_ol
den_cl = np.polyadd(den_ol, num_ol)

sys_closed_loop = signal.TransferFunction(num_cl, den_cl)
```

### 2.6 Effect of Controller Gain

| Kp Value | Response Speed | Overshoot | Steady-State Error |
|----------|----------------|-----------|-------------------|
| Low (1)  | Slow           | Low       | Higher            |
| Medium (10) | Moderate    | Moderate  | Lower             |
| High (20+) | Fast         | Higher    | Minimal           |

```python
# Comparing different gains
gains = [1, 10, 20, 50]

for Kp in gains:
    num_cl = [Kp * K]
    den_cl = np.polyadd(den, num_cl)
    sys_cl = signal.TransferFunction(num_cl, den_cl)

    t, y = signal.step(sys_cl, T=np.linspace(0, 0.5, 1000))
    # Plot response for each gain
```

### 2.7 Connection to Go2 Robot Control

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              FROM MOTOR CONTROL TO QUADRUPED LOCOMOTION                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   DC MOTOR (Single Joint)           GO2 ROBOT (12 Joints)                   │
│   ─────────────────────────         ───────────────────────                 │
│                                                                              │
│   • 1 DOF                           • 12 DOF (4 legs × 3 joints)            │
│   • Single transfer function        • Coupled multi-body dynamics           │
│   • PD control (Kp, Kd)            • PD control per joint                   │
│   • Position tracking              • Coordinated locomotion                 │
│                                                                              │
│   ┌─────────────┐                   ┌─────────────────────────────────────┐ │
│   │   V → θ     │                   │  Actions → Joint Positions → Gait   │ │
│   │   (direct)  │                   │  (RL Policy learns coordination)    │ │
│   └─────────────┘                   └─────────────────────────────────────┘ │
│                                                                              │
│   KEY INSIGHT:                                                              │
│   ─────────────                                                             │
│   In RL, we don't need to derive the transfer function.                     │
│   The neural network LEARNS the optimal control policy through              │
│   trial and error, using rewards as feedback instead of explicit            │
│   mathematical modeling.                                                    │
│                                                                              │
│   Traditional Control:  V = Kp·(θ_ref - θ) + Kd·(θ̇_ref - θ̇)              │
│   RL Control:           a = π_θ(observation)  [learned policy]             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.8 PD Control in Genesis/Go2

The Go2 robot uses PD (Proportional-Derivative) control at the joint level:

```python
# PD Controller for each joint
class JointPDController:
    def __init__(self, kp=20.0, kd=0.5):
        self.kp = kp  # Proportional gain
        self.kd = kd  # Derivative gain

    def compute_torque(self, target_pos, current_pos, current_vel):
        """
        Compute motor torque using PD control

        τ = Kp·(θ_target - θ_current) - Kd·θ̇_current
        """
        position_error = target_pos - current_pos
        torque = self.kp * position_error - self.kd * current_vel
        return torque

# Go2 uses these gains for all 12 joints
GO2_PD_GAINS = {
    'kp': 20.0,  # Position gain (stiffness)
    'kd': 0.5,   # Velocity gain (damping)
}
```

### 2.9 Why This Matters for the Hackathon

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONTROL THEORY → RL MAPPING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Classical Control Concept    │   RL Equivalent                            │
│   ────────────────────────────│──────────────────────────────────────────   │
│   Reference signal (θ_ref)     │   Velocity command (target vel)            │
│   Error signal (e = ref - y)   │   Part of observation/reward               │
│   Controller (PID)             │   Neural network policy π(s)               │
│   Plant (motor dynamics)       │   Genesis physics simulation               │
│   Feedback loop                │   Environment step → observation           │
│   Tuning gains (Kp, Kd)        │   Training (gradient descent on rewards)   │
│   Stability analysis           │   Episode success rate                     │
│   Step response                │   Evaluation rollouts                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Takeaways:**
1. **Low-level control is handled**: Genesis + PD controller manages joint-level torques
2. **RL learns high-level coordination**: The policy outputs desired joint positions
3. **Reward = feedback**: Instead of error signals, we use shaped rewards
4. **Training = tuning**: Instead of manual gain tuning, gradient descent optimizes the policy

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HIGH-LEVEL ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │              │    │                  │    │                          │   │
│  │   GENESIS    │◄──►│   ENVIRONMENT    │◄──►│     RL ALGORITHM         │   │
│  │   PHYSICS    │    │   (Go2Env)       │    │     (PPO)                │   │
│  │   ENGINE     │    │                  │    │                          │   │
│  │              │    │                  │    │                          │   │
│  └──────────────┘    └──────────────────┘    └──────────────────────────┘   │
│         │                    │                         │                     │
│         ▼                    ▼                         ▼                     │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│  │ • Collision  │    │ • Observations   │    │ • Actor Network          │   │
│  │ • Dynamics   │    │ • Rewards        │    │ • Critic Network         │   │
│  │ • Rendering  │    │ • Termination    │    │ • Experience Buffer      │   │
│  │ • GPU Accel  │    │ • Reset Logic    │    │ • Gradient Updates       │   │
│  └──────────────┘    └──────────────────┘    └──────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             TRAINING LOOP                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌─────────┐      ┌─────────┐      ┌─────────┐      ┌─────────┐           │
│    │  State  │─────►│ Policy  │─────►│ Action  │─────►│ Physics │           │
│    │   s_t   │      │  π(s)   │      │   a_t   │      │  Step   │           │
│    └─────────┘      └─────────┘      └─────────┘      └────┬────┘           │
│         ▲                                                   │                │
│         │                                                   ▼                │
│    ┌────┴────┐                                        ┌─────────┐           │
│    │  Next   │◄───────────────────────────────────────│ Reward  │           │
│    │ State   │                                        │  r_t    │           │
│    │ s_{t+1} │                                        └─────────┘           │
│    └─────────┘                                                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Project Structure

```
Robotbuild/
│
├── ARCHITECTURE.md              # This document
├── README.md                    # Quick start guide
│
├── configs/                     # Configuration files
│   ├── __init__.py
│   ├── base_config.py          # Shared hyperparameters
│   ├── level1_walk.py          # Walking-specific config
│   ├── level2_run.py           # Running-specific config
│   ├── level3_circle.py        # Circle walking config
│   └── level4_spin.py          # Spin in place config
│
├── envs/                        # Environment implementations
│   ├── __init__.py
│   ├── base_go2_env.py         # Base environment class
│   ├── walk_env.py             # Level 1: Walking
│   ├── run_env.py              # Level 2: Running
│   ├── circle_env.py           # Level 3: Circle
│   └── spin_env.py             # Level 4: Spin
│
├── rewards/                     # Reward function modules
│   ├── __init__.py
│   ├── base_rewards.py         # Common reward components
│   ├── walk_rewards.py         # Walking-specific rewards
│   ├── run_rewards.py          # Running-specific rewards
│   ├── circle_rewards.py       # Circle-specific rewards
│   └── spin_rewards.py         # Spin-specific rewards
│
├── training/                    # Training scripts
│   ├── __init__.py
│   ├── train_level1.py         # Train walking
│   ├── train_level2.py         # Train running
│   ├── train_level3.py         # Train circle
│   ├── train_level4.py         # Train spin
│   └── train_all.py            # Sequential training
│
├── evaluation/                  # Evaluation & metrics
│   ├── __init__.py
│   ├── evaluator.py            # Base evaluation class
│   ├── metrics.py              # Metric calculations
│   ├── eval_level1.py          # Evaluate walking
│   ├── eval_level2.py          # Evaluate running
│   ├── eval_level3.py          # Evaluate circle
│   └── eval_level4.py          # Evaluate spin
│
├── models/                      # Saved models
│   ├── level1_walk/
│   ├── level2_run/
│   ├── level3_circle/
│   └── level4_spin/
│
├── logs/                        # TensorBoard logs
│   ├── level1_walk/
│   ├── level2_run/
│   ├── level3_circle/
│   └── level4_spin/
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── math_utils.py           # Math helpers
│   ├── visualization.py        # Plotting & videos
│   └── logger.py               # Custom logging
│
├── scripts/                     # Helper scripts
│   ├── setup.sh                # Environment setup
│   ├── train_all.sh            # Train all levels
│   └── demo.sh                 # Run demos
│
└── presentation/                # Hackathon presentation
    ├── slides.md               # Presentation content
    ├── figures/                # Diagrams & plots
    └── videos/                 # Demo recordings
```

---

## 4. Technical Stack

### Core Dependencies

| Component | Library | Version | Purpose |
|-----------|---------|---------|---------|
| Physics Engine | Genesis | 0.3.11+ | Robot simulation |
| RL Algorithm | rsl-rl-lib | 2.2.4 | PPO implementation |
| Deep Learning | PyTorch | 2.0+ | Neural networks |
| Visualization | TensorBoard | 2.14+ | Training monitoring |
| Math/Science | NumPy | 1.24+ | Numerical operations |

### Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | NVIDIA GTX 1080 | NVIDIA RTX 3080+ |
| VRAM | 8 GB | 16 GB+ |
| RAM | 16 GB | 32 GB |
| CPU | 8 cores | 16+ cores |
| Storage | 20 GB | 50 GB SSD |

### Environment Setup

```bash
# Create conda environment
conda create -n genesis_env python=3.10 -y
conda activate genesis_env

# Install Genesis
pip install genesis-world

# Clone Genesis repository
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
cd Genesis/

# Install RL dependencies
pip install tensorboard rsl-rl-lib==2.2.4

# Additional utilities
pip install matplotlib seaborn pandas tqdm
```

---

## 5. Environment Design

### Go2 Robot Specifications

```
┌────────────────────────────────────────────────────────────────┐
│                    UNITREE GO2 SPECIFICATIONS                   │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Physical Properties:                                          │
│   ├── Mass: ~15 kg                                             │
│   ├── Dimensions: 0.7m x 0.3m x 0.4m (L x W x H)              │
│   └── Leg Configuration: 4 legs, 3 joints each                 │
│                                                                 │
│   Joint Structure (12 DOF total):                              │
│   ├── Front Left (FL):  hip, thigh, calf                      │
│   ├── Front Right (FR): hip, thigh, calf                      │
│   ├── Rear Left (RL):   hip, thigh, calf                      │
│   └── Rear Right (RR):  hip, thigh, calf                      │
│                                                                 │
│   Control Mode: Position control with PD gains                 │
│   ├── Kp (proportional): 20.0                                  │
│   └── Kd (derivative): 0.5                                     │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Joint Naming Convention

```python
JOINT_NAMES = [
    'FL_hip_joint',   'FL_thigh_joint',   'FL_calf_joint',   # Front Left
    'FR_hip_joint',   'FR_thigh_joint',   'FR_calf_joint',   # Front Right
    'RL_hip_joint',   'RL_thigh_joint',   'RL_calf_joint',   # Rear Left
    'RR_hip_joint',   'RR_thigh_joint',   'RR_calf_joint',   # Rear Right
]
```

### Default Joint Positions (Standing Pose)

```python
DEFAULT_DOF_POS = [
    0.0,  0.8, -1.5,   # FL: hip, thigh, calf (radians)
    0.0,  0.8, -1.5,   # FR
    0.0,  1.0, -1.5,   # RL
    0.0,  1.0, -1.5,   # RR
]
```

### Simulation Parameters

```python
SIM_CONFIG = {
    'dt': 0.02,              # 50 Hz control frequency
    'substeps': 2,           # Physics substeps per control step
    'gravity': [0, 0, -9.81],
    'ground_friction': 1.0,
    'max_episode_length': 1000,  # 20 seconds at 50 Hz
}
```

---

## 6. Observation Space

### Observation Vector (45 dimensions)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        OBSERVATION SPACE (45D)                               │
├────────────────────┬──────────┬─────────────────────────────────────────────┤
│ Component          │ Dims     │ Description                                  │
├────────────────────┼──────────┼─────────────────────────────────────────────┤
│ Base Angular Vel   │ 3        │ Roll, pitch, yaw rates (rad/s)              │
│ Projected Gravity  │ 3        │ Gravity vector in body frame                │
│ Velocity Commands  │ 3        │ Target vx, vy, yaw rate                     │
│ Joint Positions    │ 12       │ Current angles - default angles (rad)       │
│ Joint Velocities   │ 12       │ Angular velocities (rad/s)                  │
│ Previous Actions   │ 12       │ Last commanded positions                    │
├────────────────────┼──────────┼─────────────────────────────────────────────┤
│ TOTAL              │ 45       │                                             │
└────────────────────┴──────────┴─────────────────────────────────────────────┘
```

### Observation Scaling

```python
OBS_SCALES = {
    'ang_vel': 0.25,        # Scale down angular velocity
    'dof_pos': 1.0,         # Keep position as-is
    'dof_vel': 0.05,        # Scale down joint velocities
}

COMMAND_SCALES = {
    'lin_vel': 2.0,         # Linear velocity command
    'ang_vel': 0.25,        # Angular velocity command
}
```

### Extended Observations (For Advanced Tasks)

For Level 3 (Circle) and Level 4 (Spin), add:

```python
EXTENDED_OBS = {
    'base_position': 2,      # X, Y position (for circle tracking)
    'base_heading': 1,       # Yaw angle (for circle tangent)
    'angular_error': 1,      # Error from target angular velocity
}
# Total for extended: 45 + 4 = 49 dimensions
```

---

## 7. Action Space

### Action Vector (12 dimensions)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ACTION SPACE (12D)                                   │
├────────────────────┬──────────┬─────────────────────────────────────────────┤
│ Joint              │ Index    │ Range (radians)                             │
├────────────────────┼──────────┼─────────────────────────────────────────────┤
│ FL_hip             │ 0        │ [-0.8, 0.8]                                 │
│ FL_thigh           │ 1        │ [-1.0, 2.5]                                 │
│ FL_calf            │ 2        │ [-2.7, -0.9]                                │
│ FR_hip             │ 3        │ [-0.8, 0.8]                                 │
│ FR_thigh           │ 4        │ [-1.0, 2.5]                                 │
│ FR_calf            │ 5        │ [-2.7, -0.9]                                │
│ RL_hip             │ 6        │ [-0.8, 0.8]                                 │
│ RL_thigh           │ 7        │ [-1.0, 2.5]                                 │
│ RL_calf            │ 8        │ [-2.7, -0.9]                                │
│ RR_hip             │ 9        │ [-0.8, 0.8]                                 │
│ RR_thigh           │ 10       │ [-1.0, 2.5]                                 │
│ RR_calf            │ 11       │ [-2.7, -0.9]                                │
└────────────────────┴──────────┴─────────────────────────────────────────────┘
```

### Action Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ACTION PROCESSING PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐      │
│   │ Network  │───►│  Clip    │───►│  Scale   │───►│ Add to Default   │      │
│   │ Output   │    │ [-1, 1]  │    │  ×0.25   │    │ Positions        │      │
│   └──────────┘    └──────────┘    └──────────┘    └────────┬─────────┘      │
│                                                              │               │
│                                                              ▼               │
│                                                        ┌──────────┐         │
│                                                        │  Apply   │         │
│                                                        │ 1-step   │         │
│                                                        │ Latency  │         │
│                                                        └────┬─────┘         │
│                                                              │               │
│                                                              ▼               │
│                                                        ┌──────────┐         │
│                                                        │ PD       │         │
│                                                        │ Control  │         │
│                                                        └──────────┘         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Action Configuration

```python
ACTION_CONFIG = {
    'clip_actions': 1.0,     # Clip network output to [-1, 1]
    'action_scale': 0.25,    # Scale factor for actions
    'latency_steps': 1,      # Simulate real hardware delay
}
```

---

## 8. Reward Engineering (All Levels)

### 8.1 Level 1: Walking (20 points)

**Goal**: Stable walking at 0.5-1.0 m/s, maximum distance in 60 seconds.

```python
class WalkingRewards:
    """
    Reward configuration for Level 1: Walking
    Target: 0.5-1.0 m/s forward velocity
    """

    def __init__(self):
        self.target_vel_min = 0.5  # m/s
        self.target_vel_max = 1.0  # m/s
        self.target_vel = 0.75     # Optimal target

        # Reward weights
        self.weights = {
            'forward_velocity': 2.0,      # Primary objective
            'velocity_tracking': 1.5,     # Stay in range
            'lateral_velocity': -0.5,     # Penalize sideways
            'angular_velocity': -0.5,     # Penalize spinning
            'orientation_stability': 1.0, # Stay upright
            'base_height': 0.5,           # Maintain height
            'action_smoothness': -0.01,   # Smooth movements
            'joint_torque': -0.0001,      # Energy efficiency
            'survival': 0.2,              # Stay alive bonus
        }

    def compute(self, state, action, prev_action):
        reward = 0.0

        # 1. Forward velocity reward
        vel_x = state['base_lin_vel'][0]
        reward += self.weights['forward_velocity'] * vel_x

        # 2. Velocity tracking (Gaussian around target)
        vel_error = abs(vel_x - self.target_vel)
        tracking_reward = np.exp(-vel_error / 0.25)
        reward += self.weights['velocity_tracking'] * tracking_reward

        # 3. Penalize lateral movement
        vel_y = state['base_lin_vel'][1]
        reward += self.weights['lateral_velocity'] * abs(vel_y)

        # 4. Penalize spinning
        yaw_rate = state['base_ang_vel'][2]
        reward += self.weights['angular_velocity'] * abs(yaw_rate)

        # 5. Orientation stability (penalize roll and pitch)
        roll = state['orientation'][0]
        pitch = state['orientation'][1]
        reward += self.weights['orientation_stability'] * (
            -abs(roll) - abs(pitch)
        )

        # 6. Base height maintenance
        height = state['base_height']
        target_height = 0.34  # Default standing height
        height_error = abs(height - target_height)
        reward += self.weights['base_height'] * np.exp(-height_error / 0.1)

        # 7. Action smoothness
        action_diff = np.sum(np.square(action - prev_action))
        reward += self.weights['action_smoothness'] * action_diff

        # 8. Joint torque penalty (energy efficiency)
        torques = state['joint_torques']
        reward += self.weights['joint_torque'] * np.sum(np.square(torques))

        # 9. Survival bonus
        reward += self.weights['survival']

        return reward
```

**Termination Conditions**:
```python
WALK_TERMINATION = {
    'max_roll': 0.5,          # radians (~28 degrees)
    'max_pitch': 0.5,         # radians (~28 degrees)
    'min_height': 0.2,        # meters
    'max_episode_length': 3000,  # 60 seconds at 50 Hz
}
```

---

### 8.2 Level 2: Running (40 points) - CRITICAL

**Goal**: Achieve ≥2.0 m/s forward velocity.

```python
class RunningRewards:
    """
    Reward configuration for Level 2: Running
    Target: ≥2.0 m/s forward velocity
    """

    def __init__(self):
        self.target_vel = 2.5    # Aim higher than minimum
        self.min_vel = 2.0       # Hackathon requirement

        # Reward weights (tuned for speed)
        self.weights = {
            'forward_velocity': 3.0,      # Higher weight for speed
            'velocity_bonus': 5.0,        # Bonus for exceeding 2.0 m/s
            'lateral_velocity': -0.3,     # Reduced penalty (some drift ok)
            'angular_velocity': -0.3,     # Reduced penalty
            'orientation_stability': 0.5, # Relaxed stability
            'base_height': 0.3,           # Reduced importance
            'action_smoothness': -0.005,  # Less penalty (need dynamic moves)
            'foot_clearance': 0.5,        # Encourage leg lift
            'gait_frequency': 0.5,        # Encourage fast gait
            'survival': 0.1,              # Lower survival bonus
        }

    def compute(self, state, action, prev_action):
        reward = 0.0

        # 1. Forward velocity (primary)
        vel_x = state['base_lin_vel'][0]
        reward += self.weights['forward_velocity'] * vel_x

        # 2. Velocity bonus for exceeding 2.0 m/s
        if vel_x >= self.min_vel:
            reward += self.weights['velocity_bonus'] * (vel_x - self.min_vel)

        # 3. Velocity tracking (Gaussian around 2.5 m/s)
        vel_error = abs(vel_x - self.target_vel)
        tracking_reward = np.exp(-vel_error / 0.5)  # Wider tolerance
        reward += 1.0 * tracking_reward

        # 4. Penalize lateral movement (relaxed)
        vel_y = state['base_lin_vel'][1]
        reward += self.weights['lateral_velocity'] * abs(vel_y)

        # 5. Penalize spinning (relaxed)
        yaw_rate = state['base_ang_vel'][2]
        reward += self.weights['angular_velocity'] * abs(yaw_rate)

        # 6. Orientation stability (more lenient for running)
        roll = state['orientation'][0]
        pitch = state['orientation'][1]
        reward += self.weights['orientation_stability'] * (
            -0.5 * abs(roll) - 0.5 * abs(pitch)
        )

        # 7. Base height (relaxed for running dynamics)
        height = state['base_height']
        if height > 0.25:  # Only penalize if too low
            reward += self.weights['base_height']

        # 8. Action smoothness (relaxed)
        action_diff = np.sum(np.square(action - prev_action))
        reward += self.weights['action_smoothness'] * action_diff

        # 9. Foot clearance reward (encourage lifting feet)
        foot_heights = state['foot_heights']
        clearance = np.mean(np.maximum(foot_heights, 0))
        reward += self.weights['foot_clearance'] * clearance

        # 10. Gait frequency (encourage fast stepping)
        contact_changes = state['foot_contact_changes']
        reward += self.weights['gait_frequency'] * contact_changes

        # 11. Survival bonus (lower for running)
        reward += self.weights['survival']

        return reward
```

**Termination Conditions**:
```python
RUN_TERMINATION = {
    'max_roll': 0.7,          # More lenient (~40 degrees)
    'max_pitch': 0.7,         # More lenient
    'min_height': 0.15,       # Lower threshold
    'max_episode_length': 1000,  # 20 seconds
}
```

---

### 8.3 Level 3: Circle Walking (10 bonus points)

**Goal**: Walk in a perfect circle with R=2 meters.

```python
class CircleRewards:
    """
    Reward configuration for Level 3: Circle Walking
    Target: Walk in circle of radius 2m, minimize RMSE
    """

    def __init__(self):
        self.radius = 2.0        # Target radius in meters
        self.center = [0, 0]     # Circle center
        self.target_speed = 0.5  # Walking speed on circle

        # Reward weights
        self.weights = {
            'radius_tracking': 3.0,       # Stay on circle
            'tangent_velocity': 2.0,      # Move along tangent
            'heading_alignment': 1.5,     # Face correct direction
            'angular_velocity': 1.0,      # Maintain turning rate
            'radial_velocity': -2.0,      # Don't move toward/away center
            'forward_velocity': 1.0,      # Keep moving
            'orientation_stability': 0.5, # Stay upright
            'action_smoothness': -0.01,   # Smooth movements
            'survival': 0.1,
        }

    def compute(self, state, action, prev_action):
        reward = 0.0

        # Get robot position and velocity
        pos = state['base_position'][:2]  # [x, y]
        vel = state['base_lin_vel'][:2]   # [vx, vy]
        heading = state['base_heading']    # yaw angle

        # 1. Radius tracking (distance from ideal circle)
        dist_from_center = np.linalg.norm(pos - self.center)
        radius_error = abs(dist_from_center - self.radius)
        radius_reward = np.exp(-radius_error / 0.3)  # Gaussian
        reward += self.weights['radius_tracking'] * radius_reward

        # 2. Calculate tangent direction (perpendicular to radius)
        angle_to_center = np.arctan2(pos[1], pos[0])
        tangent_angle = angle_to_center + np.pi/2  # 90 degrees ahead
        tangent_dir = np.array([np.cos(tangent_angle), np.sin(tangent_angle)])

        # 3. Tangent velocity (component along tangent)
        tangent_vel = np.dot(vel, tangent_dir)
        reward += self.weights['tangent_velocity'] * tangent_vel

        # 4. Radial velocity (should be zero)
        radial_dir = np.array([np.cos(angle_to_center), np.sin(angle_to_center)])
        radial_vel = np.dot(vel, radial_dir)
        reward += self.weights['radial_velocity'] * abs(radial_vel)

        # 5. Heading alignment (robot should face tangent direction)
        heading_error = abs(self._angle_diff(heading, tangent_angle))
        heading_reward = np.exp(-heading_error / 0.3)
        reward += self.weights['heading_alignment'] * heading_reward

        # 6. Angular velocity for turning
        # Required yaw rate = speed / radius
        required_yaw_rate = self.target_speed / self.radius
        actual_yaw_rate = state['base_ang_vel'][2]
        yaw_error = abs(actual_yaw_rate - required_yaw_rate)
        reward += self.weights['angular_velocity'] * np.exp(-yaw_error / 0.2)

        # 7. Forward velocity
        speed = np.linalg.norm(vel)
        reward += self.weights['forward_velocity'] * min(speed, self.target_speed)

        # 8. Orientation stability
        roll = state['orientation'][0]
        pitch = state['orientation'][1]
        reward += self.weights['orientation_stability'] * (
            -abs(roll) - abs(pitch)
        )

        # 9. Action smoothness
        action_diff = np.sum(np.square(action - prev_action))
        reward += self.weights['action_smoothness'] * action_diff

        # 10. Survival
        reward += self.weights['survival']

        return reward

    def _angle_diff(self, a, b):
        """Compute smallest angle difference"""
        diff = a - b
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
```

**Evaluation Metric**:
```python
def compute_circle_rmse(trajectory, radius=2.0, center=[0, 0]):
    """
    Compute RMSE from ideal circle over 10 rounds.
    Lower RMSE = better score.
    """
    errors = []
    for pos in trajectory:
        dist = np.linalg.norm(pos - center)
        error = (dist - radius) ** 2
        errors.append(error)
    rmse = np.sqrt(np.mean(errors))
    return rmse
```

---

### 8.4 Level 4: Spin in Place (10 bonus points)

**Goal**: Pure rotation with high angular velocity, zero forward movement.

```python
class SpinRewards:
    """
    Reward configuration for Level 4: Spin in Place
    Target: High angular velocity (ω), zero linear velocity (v=0)
    """

    def __init__(self):
        self.target_angular_vel = 2.0  # rad/s target
        self.start_position = None      # Track drift

        # Reward weights
        self.weights = {
            'angular_velocity': 5.0,      # Primary: spin fast
            'zero_linear_velocity': -10.0, # Critical: no movement
            'position_drift': -5.0,        # Stay in place
            'angular_tracking': 2.0,       # Track target ω
            'orientation_stability': 1.0,  # Stay upright
            'symmetric_motion': 0.5,       # Legs move symmetrically
            'action_smoothness': -0.005,   # Allow dynamic motion
            'survival': 0.2,
        }

    def reset(self, initial_position):
        self.start_position = initial_position[:2].copy()

    def compute(self, state, action, prev_action):
        reward = 0.0

        pos = state['base_position'][:2]
        vel = state['base_lin_vel'][:2]
        yaw_rate = state['base_ang_vel'][2]

        # 1. Angular velocity reward (abs for either direction)
        reward += self.weights['angular_velocity'] * abs(yaw_rate)

        # 2. Angular velocity tracking (close to target)
        ang_error = abs(abs(yaw_rate) - self.target_angular_vel)
        reward += self.weights['angular_tracking'] * np.exp(-ang_error / 0.5)

        # 3. Zero linear velocity (CRITICAL)
        linear_speed = np.linalg.norm(vel)
        reward += self.weights['zero_linear_velocity'] * linear_speed

        # 4. Position drift penalty
        if self.start_position is not None:
            drift = np.linalg.norm(pos - self.start_position)
            reward += self.weights['position_drift'] * drift

        # 5. Orientation stability (upright)
        roll = state['orientation'][0]
        pitch = state['orientation'][1]
        reward += self.weights['orientation_stability'] * (
            -abs(roll) - abs(pitch)
        )

        # 6. Symmetric leg motion (for clean spin)
        # Left legs should mirror right legs
        dof_pos = state['dof_pos']
        left_legs = dof_pos[0:3] + dof_pos[6:9]   # FL + RL
        right_legs = dof_pos[3:6] + dof_pos[9:12] # FR + RR
        symmetry = -np.sum(np.square(np.array(left_legs) + np.array(right_legs)))
        reward += self.weights['symmetric_motion'] * symmetry

        # 7. Action smoothness (relaxed)
        action_diff = np.sum(np.square(action - prev_action))
        reward += self.weights['action_smoothness'] * action_diff

        # 8. Survival
        reward += self.weights['survival']

        return reward
```

**Evaluation Metric**:
```python
def compute_spin_score(angular_velocities, linear_velocities):
    """
    Score = Average angular velocity * (1 - linear velocity penalty)
    """
    avg_omega = np.mean(np.abs(angular_velocities))
    avg_linear = np.mean(np.linalg.norm(linear_velocities, axis=1))

    # Penalize any linear movement
    linear_penalty = min(avg_linear * 2, 1.0)  # Cap at 1.0

    score = avg_omega * (1 - linear_penalty)
    return score
```

---

## 9. Training Pipeline

### 9.1 PPO Algorithm Configuration

```python
PPO_CONFIG = {
    # Core PPO hyperparameters
    'clip_param': 0.2,
    'entropy_coef': 0.01,
    'value_loss_coef': 1.0,
    'max_grad_norm': 1.0,
    'learning_rate': 1e-3,
    'gamma': 0.99,
    'lam': 0.95,

    # Training schedule
    'num_learning_epochs': 5,
    'num_mini_batches': 4,
    'schedule': 'adaptive',
    'desired_kl': 0.01,

    # Rollout settings
    'num_steps_per_env': 24,
    'max_iterations': 500,
    'save_interval': 100,
}
```

### 9.2 Neural Network Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ACTOR-CRITIC NETWORK                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Input (Observations): 45 dimensions                                        │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                          ACTOR NETWORK                               │   │
│   │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────────────┐  │   │
│   │  │ Input   │──►│ Hidden  │──►│ Hidden  │──►│ Hidden              │  │   │
│   │  │ 45      │   │ 512     │   │ 256     │   │ 128                 │  │   │
│   │  │         │   │ ELU     │   │ ELU     │   │ ELU                 │  │   │
│   │  └─────────┘   └─────────┘   └─────────┘   └──────────┬──────────┘  │   │
│   │                                                        │             │   │
│   │                                              ┌─────────▼─────────┐   │   │
│   │                                              │ Output: 12        │   │   │
│   │                                              │ (Mean actions)    │   │   │
│   │                                              │ + log_std param   │   │   │
│   │                                              └───────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         CRITIC NETWORK                               │   │
│   │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────────────┐  │   │
│   │  │ Input   │──►│ Hidden  │──►│ Hidden  │──►│ Hidden              │  │   │
│   │  │ 45      │   │ 512     │   │ 256     │   │ 128                 │  │   │
│   │  │         │   │ ELU     │   │ ELU     │   │ ELU                 │  │   │
│   │  └─────────┘   └─────────┘   └─────────┘   └──────────┬──────────┘  │   │
│   │                                                        │             │   │
│   │                                              ┌─────────▼─────────┐   │   │
│   │                                              │ Output: 1         │   │   │
│   │                                              │ (State Value)     │   │   │
│   │                                              └───────────────────┘   │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

```python
NETWORK_CONFIG = {
    'actor_hidden_dims': [512, 256, 128],
    'critic_hidden_dims': [512, 256, 128],
    'activation': 'elu',
    'init_noise_std': 1.0,
}
```

### 9.3 Training Loop Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING LOOP FLOWCHART                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                              ┌───────────────┐                               │
│                              │    START      │                               │
│                              └───────┬───────┘                               │
│                                      │                                       │
│                              ┌───────▼───────┐                               │
│                              │  Initialize   │                               │
│                              │  Environment  │                               │
│                              │  & Networks   │                               │
│                              └───────┬───────┘                               │
│                                      │                                       │
│               ┌──────────────────────▼──────────────────────┐                │
│               │           MAIN TRAINING LOOP                 │                │
│               │         (for iteration in range(N))          │                │
│               └──────────────────────┬──────────────────────┘                │
│                                      │                                       │
│       ┌──────────────────────────────▼──────────────────────────────┐        │
│       │                    ROLLOUT PHASE                             │        │
│       │  ┌────────────────────────────────────────────────────────┐ │        │
│       │  │  for step in range(num_steps_per_env):                 │ │        │
│       │  │    1. Get observations from all parallel envs          │ │        │
│       │  │    2. Actor network predicts actions                   │ │        │
│       │  │    3. Execute actions in simulation                    │ │        │
│       │  │    4. Collect rewards, next_obs, dones                 │ │        │
│       │  │    5. Store transition in buffer                       │ │        │
│       │  └────────────────────────────────────────────────────────┘ │        │
│       └──────────────────────────────┬──────────────────────────────┘        │
│                                      │                                       │
│       ┌──────────────────────────────▼──────────────────────────────┐        │
│       │                    LEARNING PHASE                            │        │
│       │  ┌────────────────────────────────────────────────────────┐ │        │
│       │  │  1. Compute advantages (GAE)                           │ │        │
│       │  │  2. for epoch in range(num_learning_epochs):           │ │        │
│       │  │       - Shuffle and create mini-batches                │ │        │
│       │  │       - Compute policy loss (clipped)                  │ │        │
│       │  │       - Compute value loss                             │ │        │
│       │  │       - Compute entropy bonus                          │ │        │
│       │  │       - Backward pass & optimizer step                 │ │        │
│       │  └────────────────────────────────────────────────────────┘ │        │
│       └──────────────────────────────┬──────────────────────────────┘        │
│                                      │                                       │
│       ┌──────────────────────────────▼──────────────────────────────┐        │
│       │                    LOGGING & SAVING                          │        │
│       │  - Log metrics to TensorBoard                                │        │
│       │  - Save checkpoint if save_interval reached                  │        │
│       │  - Evaluate policy if eval_interval reached                  │        │
│       └──────────────────────────────┬──────────────────────────────┘        │
│                                      │                                       │
│                              ┌───────▼───────┐                               │
│                              │   Converged?  │───No───►(loop back)           │
│                              └───────┬───────┘                               │
│                                      │ Yes                                   │
│                              ┌───────▼───────┐                               │
│                              │     END       │                               │
│                              └───────────────┘                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 9.4 Curriculum Learning Strategy

```python
CURRICULUM = {
    'level1_walk': {
        'phase1': {
            'target_vel': 0.3,
            'iterations': 100,
            'description': 'Learn to stand and take steps'
        },
        'phase2': {
            'target_vel': 0.5,
            'iterations': 150,
            'description': 'Increase to minimum speed'
        },
        'phase3': {
            'target_vel': 0.75,
            'iterations': 200,
            'description': 'Reach optimal walking speed'
        },
    },
    'level2_run': {
        'phase1': {
            'target_vel': 1.0,
            'iterations': 100,
            'description': 'Fast walk / slow jog'
        },
        'phase2': {
            'target_vel': 1.5,
            'iterations': 150,
            'description': 'Transition to running'
        },
        'phase3': {
            'target_vel': 2.0,
            'iterations': 200,
            'description': 'Reach minimum requirement'
        },
        'phase4': {
            'target_vel': 2.5,
            'iterations': 150,
            'description': 'Push for higher speed'
        },
    },
}
```

---

## 10. Evaluation & Metrics

### 10.1 Metrics Dashboard

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           METRICS DASHBOARD                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ LEVEL 1: WALKING                                                     │    │
│  │ ├── Forward Velocity (m/s): ████████░░ 0.82 [target: 0.5-1.0]       │    │
│  │ ├── Distance in 60s (m):    ████████░░ 48.5                         │    │
│  │ ├── Stability Score:        █████████░ 0.91                         │    │
│  │ └── Episode Success Rate:   █████████░ 94%                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ LEVEL 2: RUNNING                                                     │    │
│  │ ├── Forward Velocity (m/s): ██████████ 2.34 [target: ≥2.0] ✓        │    │
│  │ ├── Peak Velocity (m/s):    ██████████ 2.67                         │    │
│  │ ├── Stability Score:        ████████░░ 0.78                         │    │
│  │ └── Episode Success Rate:   ████████░░ 82%                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ LEVEL 3: CIRCLE (Bonus)                                              │    │
│  │ ├── Circle RMSE (m):        ██░░░░░░░░ 0.23 [lower is better]       │    │
│  │ ├── Laps Completed:         ██████████ 10/10                        │    │
│  │ ├── Avg Radius Error (m):   ███░░░░░░░ 0.18                         │    │
│  │ └── Heading Error (rad):    ████░░░░░░ 0.12                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ LEVEL 4: SPIN (Bonus)                                                │    │
│  │ ├── Angular Velocity (rad/s): ████████░░ 1.84                       │    │
│  │ ├── Linear Velocity (m/s):    █░░░░░░░░░ 0.05 [target: 0]           │    │
│  │ ├── Position Drift (m):       █░░░░░░░░░ 0.08                       │    │
│  │ └── Spin Score:               ████████░░ 1.75                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 Evaluation Protocol

```python
class Evaluator:
    """Standardized evaluation for all levels"""

    def evaluate_level1(self, policy, num_episodes=10):
        """Evaluate walking performance"""
        metrics = {
            'avg_velocity': [],
            'distance_60s': [],
            'stability': [],
            'success_rate': 0,
        }

        for ep in range(num_episodes):
            obs = self.env.reset()
            total_distance = 0
            episode_velocities = []

            for step in range(3000):  # 60 seconds
                action = policy(obs)
                obs, reward, done, info = self.env.step(action)

                vel_x = info['base_lin_vel'][0]
                episode_velocities.append(vel_x)
                total_distance += vel_x * self.env.dt

                if done:
                    break

            avg_vel = np.mean(episode_velocities)
            metrics['avg_velocity'].append(avg_vel)
            metrics['distance_60s'].append(total_distance)

            # Success if in target range and didn't fall
            if 0.5 <= avg_vel <= 1.0 and not done:
                metrics['success_rate'] += 1

        metrics['success_rate'] /= num_episodes
        return metrics

    def evaluate_level2(self, policy, num_episodes=10):
        """Evaluate running performance"""
        metrics = {
            'avg_velocity': [],
            'peak_velocity': [],
            'time_above_2ms': [],
            'success_rate': 0,
        }

        for ep in range(num_episodes):
            obs = self.env.reset()
            episode_velocities = []

            for step in range(1000):  # 20 seconds
                action = policy(obs)
                obs, reward, done, info = self.env.step(action)

                vel_x = info['base_lin_vel'][0]
                episode_velocities.append(vel_x)

                if done:
                    break

            avg_vel = np.mean(episode_velocities)
            peak_vel = np.max(episode_velocities)
            time_above = np.mean(np.array(episode_velocities) >= 2.0)

            metrics['avg_velocity'].append(avg_vel)
            metrics['peak_velocity'].append(peak_vel)
            metrics['time_above_2ms'].append(time_above)

            if avg_vel >= 2.0:
                metrics['success_rate'] += 1

        metrics['success_rate'] /= num_episodes
        return metrics

    def evaluate_level3(self, policy, num_laps=10):
        """Evaluate circle walking - compute RMSE"""
        trajectory = []
        obs = self.env.reset()

        # Run until 10 laps completed
        laps = 0
        prev_angle = 0
        total_angle = 0

        while laps < num_laps:
            action = policy(obs)
            obs, reward, done, info = self.env.step(action)

            pos = info['base_position'][:2]
            trajectory.append(pos.copy())

            # Track laps
            angle = np.arctan2(pos[1], pos[0])
            angle_diff = angle - prev_angle
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            if angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            total_angle += angle_diff
            prev_angle = angle

            if abs(total_angle) >= 2 * np.pi:
                laps += 1
                total_angle = 0

            if done:
                break

        # Compute RMSE
        trajectory = np.array(trajectory)
        distances = np.linalg.norm(trajectory, axis=1)
        errors = (distances - 2.0) ** 2
        rmse = np.sqrt(np.mean(errors))

        return {
            'rmse': rmse,
            'laps_completed': laps,
            'avg_radius': np.mean(distances),
        }

    def evaluate_level4(self, policy, duration=20):
        """Evaluate spin in place"""
        obs = self.env.reset()
        start_pos = self.env.get_position()[:2]

        angular_velocities = []
        linear_velocities = []
        positions = []

        for step in range(int(duration / self.env.dt)):
            action = policy(obs)
            obs, reward, done, info = self.env.step(action)

            angular_velocities.append(info['base_ang_vel'][2])
            linear_velocities.append(info['base_lin_vel'][:2])
            positions.append(info['base_position'][:2])

            if done:
                break

        # Compute metrics
        avg_omega = np.mean(np.abs(angular_velocities))
        avg_linear = np.mean(np.linalg.norm(linear_velocities, axis=1))
        max_drift = np.max(np.linalg.norm(
            np.array(positions) - start_pos, axis=1
        ))

        # Score: high omega, low linear velocity
        score = avg_omega * max(0, 1 - avg_linear)

        return {
            'avg_angular_velocity': avg_omega,
            'avg_linear_velocity': avg_linear,
            'max_drift': max_drift,
            'spin_score': score,
        }
```

---

## 11. Implementation Timeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION TIMELINE                              │
│                            (1-3 Days)                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  DAY 1 (8-10 hours)                                                          │
│  ─────────────────                                                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ MORNING (4 hours)                                                    │    │
│  │ ├── [1 hr] Set up project structure                                 │    │
│  │ ├── [1 hr] Implement base environment (from Genesis example)        │    │
│  │ ├── [1 hr] Implement Level 1 reward function                        │    │
│  │ └── [1 hr] Start Level 1 training                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ AFTERNOON (4-6 hours)                                                │    │
│  │ ├── [1 hr] Monitor Level 1, adjust rewards if needed                │    │
│  │ ├── [1 hr] Implement Level 2 reward function                        │    │
│  │ ├── [2 hr] Train Level 2 (running requires more tuning)             │    │
│  │ └── [1-2 hr] Iterate on Level 2 reward weights                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  DAY 2 (6-8 hours)                                                           │
│  ─────────────────                                                           │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ MORNING (3-4 hours)                                                  │    │
│  │ ├── [1 hr] Finalize Level 2 (must hit ≥2.0 m/s)                     │    │
│  │ ├── [1 hr] Implement Level 3 (circle) reward function               │    │
│  │ └── [1-2 hr] Train Level 3                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ AFTERNOON (3-4 hours)                                                │    │
│  │ ├── [1 hr] Implement Level 4 (spin) reward function                 │    │
│  │ ├── [1-2 hr] Train Level 4                                          │    │
│  │ └── [1 hr] Run all evaluations, collect metrics                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  DAY 3 (Optional - 4 hours)                                                  │
│  ─────────────────────────                                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ POLISH & PRESENTATION                                                │    │
│  │ ├── [1 hr] Fine-tune any underperforming levels                     │    │
│  │ ├── [1 hr] Record demo videos for each level                        │    │
│  │ ├── [1 hr] Create presentation slides                               │    │
│  │ └── [1 hr] Practice presentation                                    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 12. Presentation Strategy (20 points)

### Slide Structure

```
Slide 1: Title
─────────────────
"Teaching Go2 to Walk, Run, and Dance"
Team Name | Hackathon Name | Date

Slide 2: Problem Statement
─────────────────
- Quadruped locomotion is hard
- Traditional control requires complex dynamics
- RL learns from experience, not equations

Slide 3: Our Approach
─────────────────
- Genesis physics simulator
- PPO algorithm with custom rewards
- Curriculum learning for progressive difficulty

Slide 4: Level 1 - Walking
─────────────────
[Video demo]
- Reward: Forward velocity + stability
- Result: 0.82 m/s, 48m in 60s
- Key insight: Balance speed vs stability

Slide 5: Level 2 - Running (HIGHLIGHT)
─────────────────
[Video demo]
- Reward: Speed bonus above 2.0 m/s
- Result: 2.34 m/s average
- Key insight: Relaxed stability constraints

Slide 6: Level 3 - Circle (Bonus)
─────────────────
[Video demo]
- Reward: Radius tracking + tangent velocity
- Result: RMSE = 0.23m
- Key insight: Heading alignment critical

Slide 7: Level 4 - Spin (Bonus)
─────────────────
[Video demo]
- Reward: Angular velocity - linear penalty
- Result: 1.84 rad/s, drift < 10cm
- Key insight: Symmetric leg motion

Slide 8: Technical Insights
─────────────────
- Reward shaping is the key skill
- Curriculum learning helps convergence
- Trade-offs: speed vs stability

Slide 9: Results Summary
─────────────────
| Level | Target | Achieved | Points |
|-------|--------|----------|--------|
| Walk  | 0.5-1.0| 0.82 m/s | 20/20  |
| Run   | ≥2.0   | 2.34 m/s | 40/40  |
| Circle| Low RMSE| 0.23m   | 10/10  |
| Spin  | High ω | 1.84 rad/s| 10/10 |

Slide 10: Future Work & Questions
─────────────────
- Multi-task single policy
- Sim-to-real transfer
- Questions?
```

### Presentation Tips

1. **Lead with demos** - Show videos early to grab attention
2. **Explain reward intuition** - Why each term matters
3. **Show failure cases** - What didn't work and why
4. **Quantify results** - Use exact numbers from evaluation
5. **Time management** - Practice to hit exact time limit

---

## Quick Reference: Key Files to Implement

| Priority | File | Purpose |
|----------|------|---------|
| 1 | `envs/base_go2_env.py` | Core environment |
| 2 | `rewards/walk_rewards.py` | Level 1 rewards |
| 3 | `training/train_level1.py` | Walking training |
| 4 | `rewards/run_rewards.py` | Level 2 rewards |
| 5 | `training/train_level2.py` | Running training |
| 6 | `rewards/circle_rewards.py` | Level 3 rewards |
| 7 | `rewards/spin_rewards.py` | Level 4 rewards |
| 8 | `evaluation/evaluator.py` | All metrics |

---

*Document Version: 1.0*
*Created for: Unitree Go2 Locomotion Hackathon*
