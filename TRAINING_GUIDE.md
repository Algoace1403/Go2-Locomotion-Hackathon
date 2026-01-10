# Complete Training Guide for Go2 Locomotion Hackathon

## Table of Contents
1. [Quick Start](#quick-start)
2. [Understanding the System](#understanding-the-system)
3. [Training Each Level](#training-each-level)
4. [Reward Tuning Guide](#reward-tuning-guide)
5. [Troubleshooting](#troubleshooting)
6. [Tips for Winning](#tips-for-winning)

---

## Quick Start

### Step 1: Environment Setup

```bash
# Create and activate conda environment
conda create -n genesis_env python=3.10 -y
conda activate genesis_env

# Install dependencies
pip install genesis-world
pip install tensorboard rsl-rl-lib==2.2.4
pip install matplotlib numpy torch imageio

# Clone Genesis for URDF files
git clone https://github.com/Genesis-Embodied-AI/Genesis.git
```

### Step 2: Navigate to Project

```bash
cd /Users/aks/Desktop/Robotbuild
```

### Step 3: Start Training (Priority Order)

```bash
# PRIORITY 1: Running (40 points!)
python training/train_level2.py --headless

# PRIORITY 2: Walking (20 points)
python training/train_level1.py --headless

# PRIORITY 3: Circle (10 bonus)
python training/train_level3.py --headless

# PRIORITY 4: Spin (10 bonus)
python training/train_level4.py --headless

# BONUS: Dance (presentation wow factor!)
python training/train_level5.py --headless --bpm 120
```

### Step 4: Monitor Training

```bash
# In a separate terminal
tensorboard --logdir=logs/
# Open http://localhost:6006 in browser
```

---

## Understanding the System

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    REINFORCEMENT LEARNING LOOP                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   1. OBSERVATION                                                 │
│      Robot sensors → 45-dimensional vector                       │
│      (joint angles, velocities, body orientation, etc.)         │
│                                                                  │
│   2. POLICY (Neural Network)                                    │
│      Observation → Action (12 joint positions)                  │
│      The network LEARNS what actions lead to high rewards       │
│                                                                  │
│   3. ENVIRONMENT                                                │
│      Actions → Physics simulation → New state                   │
│      Genesis simulator computes what happens                    │
│                                                                  │
│   4. REWARD                                                     │
│      New state → Reward signal                                  │
│      YOUR DESIGN tells the robot what's good/bad               │
│                                                                  │
│   5. UPDATE                                                     │
│      PPO algorithm updates network weights                      │
│      To get MORE reward in the future                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insight: You Are The Coach!

> "You are not the pilot; you are the coach."

You don't program HOW to walk. You design REWARDS that tell the robot:
- "Moving forward is good" (+reward)
- "Falling over is bad" (-reward)
- "Going fast is great!" (++reward)

The robot figures out the HOW through trial and error!

---

## Training Each Level

### Level 1: Walking (20 points)

**Goal**: Walk at 0.5-1.0 m/s for 60 seconds

**Key Rewards**:
```python
# These are the most important for walking:
'forward_velocity': 2.0,      # Move forward
'velocity_tracking': 1.5,     # Stay in speed range
'orientation_stability': 1.0, # Don't fall
'survival': 0.2,              # Stay alive
```

**Training Command**:
```bash
python training/train_level1.py --headless --max_iterations 500
```

**Expected Training Time**: ~30-60 minutes

**Success Indicators** (in TensorBoard):
- Reward steadily increasing
- Episode length approaching 3000 (60s)
- Mean velocity in 0.5-1.0 range

---

### Level 2: Running (40 points) - CRITICAL!

**Goal**: Run at ≥2.0 m/s

**Key Rewards**:
```python
# Speed is king for running:
'forward_velocity': 3.0,     # Primary objective
'velocity_bonus': 5.0,       # HUGE bonus above 2.0 m/s
'orientation_stability': 0.5, # Relaxed (running is dynamic)
```

**Training Command**:
```bash
python training/train_level2.py --headless --max_iterations 600
```

**Pro Tips**:
1. Start with walking model as initialization:
   ```bash
   python training/train_level2.py --from_walk models/level1_walk/*/final_model.pt
   ```
2. If stuck below 2.0 m/s, increase `velocity_bonus` weight
3. Running needs more iterations - be patient!

**Expected Training Time**: ~60-90 minutes

---

### Level 3: Circle (10 bonus points)

**Goal**: Walk in R=2m circle, minimize RMSE

**Key Rewards**:
```python
'radius_tracking': 3.0,      # Stay on the circle
'tangent_velocity': 2.0,     # Move along tangent
'heading_alignment': 1.5,    # Face the right way
'radial_velocity': -2.0,     # Don't drift in/out
```

**Training Command**:
```bash
python training/train_level3.py --headless --max_iterations 500
```

**Key Challenge**: Robot must coordinate turning while walking

---

### Level 4: Spin (10 bonus points)

**Goal**: High angular velocity, zero linear movement

**Key Rewards**:
```python
'angular_velocity': 5.0,       # Spin fast!
'zero_linear_velocity': -10.0, # CRITICAL: no movement
'position_drift': -5.0,        # Stay in place
```

**Training Command**:
```bash
python training/train_level4.py --headless --max_iterations 500
```

**Key Challenge**: Spin without drifting

---

### Level 5: Dance (Presentation Bonus!)

**Goal**: Rhythmic, coordinated movements

**Key Rewards**:
```python
'rhythm_sync': 3.0,          # Move to the beat!
'leg_coordination': 2.5,     # Coordinated legs
'height_variation': 1.5,     # Bouncing motion
'pose_variety': 1.5,         # Don't be static
```

**Training Command**:
```bash
python training/train_level5.py --headless --bpm 120
```

**This is for WOW factor in presentation!**

---

## Reward Tuning Guide

### General Principles

1. **Start Simple**: Begin with few rewards, add complexity
2. **Balance Carefully**: Too many penalties = robot does nothing
3. **Use Curriculum**: Start easy, gradually increase difficulty

### Common Patterns

**Want more speed?**
```python
# Increase these:
'forward_velocity': 3.0 → 5.0
'velocity_bonus': 5.0 → 8.0

# Decrease these:
'action_smoothness': -0.01 → -0.005
'orientation_stability': 1.0 → 0.5
```

**Robot keeps falling?**
```python
# Increase stability rewards:
'orientation_stability': 0.5 → 2.0
'survival': 0.1 → 0.5

# Reduce aggressive behaviors:
'forward_velocity': 3.0 → 2.0
```

**Movement too jerky?**
```python
# Increase smoothness penalty:
'action_smoothness': -0.005 → -0.02
```

### Reward Weight Cheat Sheet

| Behavior | Typical Weight |
|----------|---------------|
| Primary objective | 2.0 - 5.0 |
| Secondary objective | 1.0 - 2.0 |
| Stability | 0.5 - 1.5 |
| Smoothness (penalty) | -0.005 to -0.02 |
| Survival bonus | 0.1 - 0.3 |

---

## Troubleshooting

### Problem: Training Not Converging

**Symptoms**: Reward stays flat, not improving

**Solutions**:
1. Check learning rate (try 1e-4 instead of 1e-3)
2. Increase `num_envs` for more parallel experience
3. Check reward scale (rewards should be in -10 to +10 range)
4. Try curriculum learning (start easier)

### Problem: Robot Falls Immediately

**Symptoms**: Episode length stays at 0-50 steps

**Solutions**:
1. Increase `survival` reward
2. Increase `orientation_stability`
3. Start with standing still before walking
4. Check termination conditions (make more lenient)

### Problem: Running Too Slow

**Symptoms**: Velocity stuck below 2.0 m/s

**Solutions**:
1. Increase `velocity_bonus` significantly (try 10.0)
2. Reduce stability penalties
3. Initialize from walking model
4. Use curriculum (gradually increase target)

### Problem: Out of GPU Memory

**Solutions**:
1. Reduce `num_envs` (try 2048 or 1024)
2. Reduce network size in config
3. Use smaller batch size

---

## Tips for Winning

### Priority Strategy

1. **Focus on Level 2 first** (40 points!)
2. Get Level 1 working (20 points)
3. Attempt bonuses if time permits
4. Prepare killer presentation (20 points)

### Presentation Tips

1. **Show videos** of your best runs
2. **Explain your reward design** - why each term matters
3. **Show failure cases** - what you learned
4. **Quantify results** - exact velocities achieved
5. **End with dance demo** - WOW factor!

### Time Management

| Day | Focus |
|-----|-------|
| Day 1 Morning | Setup, start Level 2 training |
| Day 1 Afternoon | Tune Level 2, start Level 1 |
| Day 2 Morning | Finalize levels 1-2, try bonuses |
| Day 2 Afternoon | Evaluate, record demos, prepare slides |

### Checkpoints

Save frequently! Training can be resumed:
```bash
python training/train_level2.py --resume models/level2_run/*/checkpoint_*.pt
```

---

## File Reference

| File | Purpose |
|------|---------|
| `configs/level*_*.py` | Hyperparameters and reward weights |
| `rewards/*_rewards.py` | Reward function implementations |
| `envs/*_env.py` | Environment logic |
| `training/train_level*.py` | Training scripts |
| `evaluation/evaluator.py` | Evaluation tools |
| `utils/visualization.py` | Plotting and videos |

---

## Quick Commands Reference

```bash
# Training
python training/train_level1.py --headless
python training/train_level2.py --headless
python training/train_level3.py --headless
python training/train_level4.py --headless
python training/train_level5.py --headless --bpm 120

# Monitor
tensorboard --logdir=logs/

# Train all sequentially
python training/train_all.py
```

---

Good luck with your hackathon! 🤖🏃‍♂️💃
