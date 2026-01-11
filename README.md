# Go2 Robot Training - Walk, Run, Jump

Train a Unitree Go2 quadruped robot using Reinforcement Learning (PPO) in the Genesis physics simulator.

## Quick Start

### 1. Install Dependencies

```bash
# Create and activate conda environment
conda create -n genesis python=3.10 -y
conda activate genesis

# Install Genesis
pip install genesis-world

# Install RL library (IMPORTANT: must be this exact version)
pip install rsl-rl-lib==2.2.4

# Install other dependencies
pip install torch tensorboard
```

### 2. Train the Robot

```bash
cd /Users/aks/Desktop/Robotbuild

# Train walking (stable locomotion at 0.3-0.8 m/s)
python train.py --mode walk

# Train running (fast locomotion at 1.0-2.5 m/s)
python train.py --mode run

# Train jumping
python train.py --mode jump

# Train all modes sequentially
python train.py --mode all
```

#### Training Options

```bash
python train.py --mode walk \
    --num_envs 4096 \        # Number of parallel environments (default: 4096)
    --max_iterations 300 \   # Training iterations (default: 300)
    --viewer \               # Show visualization during training
    --resume                 # Resume from latest checkpoint
```

**For machines without GPU:**
```bash
python train.py --mode walk --cpu --num_envs 64
```

### 3. Monitor Training

```bash
tensorboard --logdir=logs/
```

Open http://localhost:6006 in your browser.

### 4. Evaluate Trained Models

```bash
# Evaluate a trained model with visualization
python evaluate.py --checkpoint logs/go2_walk_XXXXXX/

# Run more episodes
python evaluate.py --checkpoint logs/go2_run_XXXXXX/ --episodes 10
```

## Training Modes

| Mode | Speed Target | Description |
|------|--------------|-------------|
| walk | 0.3-0.8 m/s | Stable, energy-efficient walking |
| run  | 1.0-2.5 m/s | Fast running with dynamic gait |
| jump | N/A | Jumping with stable landing |

## Project Structure

```
Robotbuild/
├── train.py          # Main training script
├── evaluate.py       # Model evaluation script
├── go2_env.py        # Environment with all reward functions
├── urdf/             # Robot model files
│   ├── go2/          # Go2 robot URDF
│   └── plane/        # Ground plane URDF
├── logs/             # Training logs and checkpoints
└── models/           # Saved models
```

## Reward Functions

The environment includes comprehensive reward functions:

**Locomotion rewards:**
- `tracking_lin_vel` - Track commanded velocity
- `tracking_ang_vel` - Track commanded turning
- `forward_vel` - Reward forward motion (run mode)

**Stability penalties:**
- `lin_vel_z` - Penalize vertical bouncing
- `ang_vel_xy` - Penalize roll/pitch rotation
- `orientation` - Penalize tilting
- `base_height` - Maintain proper height

**Smoothness penalties:**
- `action_rate` - Penalize jerky movements
- `dof_acc` - Penalize joint acceleration
- `energy` - Penalize energy consumption

**Jump-specific:**
- `jump_height` - Reward height achieved
- `air_time` - Reward time in air
- `land_stable` - Reward stable landing

## Troubleshooting

**"No module named genesis"**
```bash
pip install genesis-world
```

**"Please install rsl-rl-lib==2.2.4"**
```bash
pip uninstall rsl-rl rsl_rl  # Remove any old versions
pip install rsl-rl-lib==2.2.4
```

**Training is very slow**
- Use GPU: remove `--cpu` flag
- Reduce environments: `--num_envs 1024`

**Robot falls immediately**
- This is normal at the start of training
- Wait for 50-100 iterations for learning to begin
- Check TensorBoard for reward curves

## Tips

1. **Start with walking** - It's the foundation for other behaviors
2. **Use GPU** - Training is 10-100x faster on GPU
3. **Monitor TensorBoard** - Watch for reward convergence
4. **300 iterations** is usually enough for basic walking
5. **Resume training** if you need to continue: `--resume`
