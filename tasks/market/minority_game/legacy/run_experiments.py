#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run all experiments for the Minority Game.

This script runs both experiments as specified in the requirements:
- Experiment 1: Time series for different M values (M=6, 8, 10)
- Experiment 2: Mixed population with M=1 to M=10
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    """Run a command and display its output."""
    print()
    print("=" * 80)
    print(description)
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode != 0:
        print(f"Error: Command failed with return code {result.returncode}")
        return False
    return True


def experiment_1():
    """
    Experiment 1: Time series for different M values.
    Shows that larger M leads to smaller fluctuations in A count.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: Time Series for Different M Values")
    print("=" * 80)
    print("Goal: Demonstrate that variance decreases as M increases")
    print("Parameters: N=1001, S=5, steps=20000, burn-in=2000, seed=42")
    print("Testing M = 6, 8, 10")
    print("=" * 80)
    
    N = 1001
    S = 5
    steps = 20000
    burn_in = 2000
    seed = 42
    M_values = [6, 8, 10]
    
    results = []
    
    for M in M_values:
        cmd = [
            sys.executable, "minority_game.py",
            "--N", str(N),
            "--M", str(M),
            "--S", str(S),
            "--steps", str(steps),
            "--burn-in", str(burn_in),
            "--seed", str(seed),
            "--plot"
        ]
        
        success = run_command(cmd, f"Running Experiment 1 with M={M}")
        results.append((M, success))
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 1 SUMMARY")
    print("=" * 80)
    for M, success in results:
        status = "✓ Success" if success else "✗ Failed"
        print(f"  M={M}: {status}")
    print("\nExpected result: Variance should decrease as M increases (M=6 > M=8 > M=10)")
    print("Check the generated PNG files for time series plots.")
    print("=" * 80)


def experiment_2():
    """
    Experiment 2: Mixed population with different memory lengths.
    Shows that win rate increases with M and plateaus around M≈6.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: Win Rate vs Memory Length (Mixed Population)")
    print("=" * 80)
    print("Goal: Show that agents with larger M have higher win rates")
    print("Parameters: N=1001, S=5, steps=20000, burn-in=2000, seed=123")
    print("Mixed M: 100 agents each for M=1..9, 101 agents for M=10")
    print("=" * 80)
    
    N = 1001
    S = 5
    steps = 20000
    burn_in = 2000
    seed = 123
    M_mix = "1:100,2:100,3:100,4:100,5:100,6:100,7:100,8:100,9:100,10:101"
    
    cmd = [
        sys.executable, "minority_game.py",
        "--N", str(N),
        "--S", str(S),
        "--steps", str(steps),
        "--burn-in", str(burn_in),
        "--seed", str(seed),
        "--M-mix", M_mix,
        "--dump-agent-stats",
        "--plot"
    ]
    
    success = run_command(cmd, "Running Experiment 2 (Mixed Population)")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT 2 SUMMARY")
    print("=" * 80)
    if success:
        print("✓ Experiment 2 completed successfully")
        print("\nGenerated files:")
        print("  - agent_stats.csv: Individual agent statistics")
        print("  - win_rate_by_M.png: Win rate vs memory length plot")
        print("  - minority_game_Mmixed_*.png: Time series plot")
        print("\nExpected result:")
        print("  - Win rate should increase with M")
        print("  - Plateau around M≈6 or higher")
    else:
        print("✗ Experiment 2 failed")
    print("=" * 80)


def main():
    """Run all experiments."""
    print("\n" + "╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "MINORITY GAME EXPERIMENTS" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"\nWorking directory: {script_dir}\n")
    
    # Run experiments
    try:
        experiment_1()
        experiment_2()
        
        print("\n" + "╔" + "=" * 78 + "╗")
        print("║" + " " * 25 + "ALL EXPERIMENTS COMPLETE" + " " * 29 + "║")
        print("╚" + "=" * 78 + "╝\n")
        
    except KeyboardInterrupt:
        print("\n\nExperiments interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError running experiments: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

