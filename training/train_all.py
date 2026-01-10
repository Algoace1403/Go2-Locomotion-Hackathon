"""
Train All Levels Sequentially
Runs training for all 4 levels in order of priority
"""

import subprocess
import sys
import os
from datetime import datetime


def run_training(level: int, **kwargs):
    """Run training for a specific level"""
    script_map = {
        1: 'train_level1.py',
        2: 'train_level2.py',
        3: 'train_level3.py',
        4: 'train_level4.py',
    }

    script = script_map.get(level)
    if not script:
        print(f"Invalid level: {level}")
        return False

    script_path = os.path.join(os.path.dirname(__file__), script)

    # Build command
    cmd = [sys.executable, script_path, '--headless']

    for key, value in kwargs.items():
        cmd.extend([f'--{key}', str(value)])

    print(f"\n{'='*60}")
    print(f"Starting Level {level} Training")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))

    return result.returncode == 0


def main():
    """Train all levels in priority order"""
    start_time = datetime.now()

    print("=" * 60)
    print("Go2 Locomotion Hackathon - Full Training Pipeline")
    print("=" * 60)
    print(f"Started at: {start_time}")
    print("\nTraining order (by points):")
    print("  1. Level 2: Running (40 points) - CRITICAL")
    print("  2. Level 1: Walking (20 points)")
    print("  3. Level 3: Circle  (10 bonus points)")
    print("  4. Level 4: Spin    (10 bonus points)")
    print("=" * 60)

    results = {}

    # Priority 1: Running (40 points)
    results['level2'] = run_training(2, max_iterations=600)

    # Priority 2: Walking (20 points)
    results['level1'] = run_training(1, max_iterations=500)

    # Priority 3: Circle (10 bonus)
    results['level3'] = run_training(3, max_iterations=500)

    # Priority 4: Spin (10 bonus)
    results['level4'] = run_training(4, max_iterations=500)

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total duration: {duration}")
    print("\nResults:")
    for level, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {level}: {status}")
    print("=" * 60)


if __name__ == '__main__':
    main()
