# Go2 Locomotion Hackathon Project

Train a Unitree Go2 quadruped robot using Reinforcement Learning in the Genesis physics simulator.

## Quick Start

### 1. Setup Environment

```bash
# Create conda environment
conda create -n genesis_env python=3.10 -y
conda activate genesis_env

# Install Genesis
pip install genesis-world

# Clone Genesis repository (for URDF files)
git clone https://github.com/Genesis-Embodied-AI/Genesis.git

# Install RL dependencies
pip install tensorboard rsl-rl-lib==2.2.4

# Install additional packages
pip install matplotlib numpy torch
```

### 2. Project Structure

```
Robotbuild/
├── configs/          # Configuration files for each level
├── envs/             # Environment implementations
├── rewards/          # Reward function modules
├── training/         # Training scripts
├── evaluation/       # Evaluation scripts
├── models/           # Saved model checkpoints
├── logs/             # TensorBoard logs
└── utils/            # Utility functions
```

### 3. Training

**Train individual levels:**

```bash
# Level 1: Walking (20 points)
python training/train_level1.py --headless

# Level 2: Running (40 points) - CRITICAL!
python training/train_level2.py --headless

# Level 3: Circle Walking (10 bonus points)
python training/train_level3.py --headless

# Level 4: Spin in Place (10 bonus points)
python training/train_level4.py --headless
```

**Train all levels sequentially:**

```bash
python training/train_all.py
```

### 4. Monitor Training

```bash
tensorboard --logdir=logs/
```

### 5. Evaluation

After training, evaluate your models:

```python
from evaluation.evaluator import Evaluator, load_policy
from envs.walk_env import WalkEnv
from configs.level1_walk import WalkConfig

# Load environment and policy
config = WalkConfig()
env = WalkEnv(num_envs=1, headless=False)
policy = load_policy('models/level1_walk/final_model.pt',
                     config.num_observations,
                     config.num_joints,
                     config.network)

# Run evaluation
evaluator = Evaluator(env, policy)
metrics = evaluator.evaluate_level1()
```

## Challenge Levels

| Level | Task | Target | Points |
|-------|------|--------|--------|
| 1 | Walking | 0.5-1.0 m/s | 20 |
| 2 | Running | ≥2.0 m/s | 40 |
| 3 | Circle | R=2m, low RMSE | 10 (bonus) |
| 4 | Spin | High ω, v=0 | 10 (bonus) |
| - | Presentation | - | 20 |

**Total: 100 points**

## Key Files

- `configs/base_config.py` - Shared hyperparameters
- `envs/base_go2_env.py` - Base environment class
- `rewards/walk_rewards.py` - Level 1 reward function
- `rewards/run_rewards.py` - Level 2 reward function
- `training/train_level2.py` - Critical training script (40 pts!)

## Tips

1. **Start with Level 2** - It's worth the most points (40!)
2. **Use curriculum learning** - Gradually increase difficulty
3. **Monitor TensorBoard** - Watch for reward convergence
4. **Tune reward weights** - Balance speed vs stability
5. **Save checkpoints** - Resume training if needed

## Architecture Documentation

See `ARCHITECTURE.md` for detailed technical documentation including:
- DC Motor control theory foundation
- Complete observation/action space definitions
- Reward function implementations for all levels
- Training pipeline architecture
- Evaluation protocols

## Hackathon Philosophy

> "You are not the pilot; you are the coach."

Design reward functions that teach the robot desired behaviors. The RL algorithm learns the actual motor commands.
