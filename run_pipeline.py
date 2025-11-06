"""
Main Pipeline Script: Expert → Diffusion → Prior-Guided RL

This script runs the complete pipeline:
1. M1: Generate expert trajectories from ODE model
2. M2: Replay to micro-level CTMP environment
3. M3: Train diffusion model and PPO variants

Usage:
    python run_pipeline.py --all          # Run complete pipeline
    python run_pipeline.py --m1           # Run M1 only
    python run_pipeline.py --m2           # Run M2 only
    python run_pipeline.py --m3           # Run M3 only
"""

import argparse
import sys
import subprocess
from pathlib import Path


def run_m1(args):
    """Run M1: Generate expert trajectories."""
    print("\n" + "="*70)
    print("M1: Generating Expert Trajectories from ODE Model")
    print("="*70 + "\n")

    script = Path(__file__).parent / 'expert' / 'export_expert_trajectories.py'

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=Path(__file__).parent
    )

    if result.returncode != 0:
        print("❌ M1 failed")
        return False

    print("\n✓ M1 Complete")
    return True


def run_m2(args):
    """Run M2: Replay to micro-level."""
    print("\n" + "="*70)
    print("M2: Replaying Macro Trajectories to Micro-Level CTMP")
    print("="*70 + "\n")

    script = Path(__file__).parent / 'expert' / 'replay_to_micro.py'

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=Path(__file__).parent
    )

    if result.returncode != 0:
        print("❌ M2 failed")
        return False

    print("\n✓ M2 Complete")
    return True


def run_m3(args):
    """Run M3: Train diffusion model and PPO variants."""
    print("\n" + "="*70)
    print("M3: Training Diffusion Model and PPO Variants")
    print("="*70 + "\n")

    # Train diffusion model
    print("\n--- Training Diffusion Model ---\n")
    diffusion_script = Path(__file__).parent / 'diffusion' / 'train.py'

    result = subprocess.run(
        [sys.executable, str(diffusion_script)],
        cwd=Path(__file__).parent
    )

    if result.returncode != 0:
        print("❌ Diffusion training failed")
        return False

    print("\n✓ Diffusion model trained")

    # Train PPO variants
    print("\n--- Training PPO Variants ---\n")
    ppo_script = Path(__file__).parent / 'rl' / 'train_ppo.py'

    result = subprocess.run(
        [sys.executable, str(ppo_script)],
        cwd=Path(__file__).parent
    )

    if result.returncode != 0:
        print("❌ PPO training failed")
        return False

    print("\n✓ M3 Complete")
    return True


def run_all(args):
    """Run complete pipeline."""
    print("\n" + "="*70)
    print("Running Complete Pipeline: Expert → Diffusion → Prior-Guided RL")
    print("="*70 + "\n")

    # M1
    if not run_m1(args):
        print("\n❌ Pipeline failed at M1")
        return False

    # M2
    if not run_m2(args):
        print("\n❌ Pipeline failed at M2")
        return False

    # M3
    if not run_m3(args):
        print("\n❌ Pipeline failed at M3")
        return False

    print("\n" + "="*70)
    print("✓ Complete Pipeline Finished Successfully!")
    print("="*70)
    print("\nResults saved to:")
    print("  - Expert trajectories: data/macro_expert/")
    print("  - Micro trajectories:  data/micro_replay/")
    print("  - Diffusion model:     logs/diffusion/")
    print("  - PPO results:         logs/ppo/")
    print("\nView comparison plot: logs/ppo/comparison_plot.png")
    print("="*70 + "\n")

    return True


def generate_report(args):
    """Generate final evaluation report."""
    print("\n" + "="*70)
    print("Generating Evaluation Report")
    print("="*70 + "\n")

    import pickle
    import numpy as np

    # Load results
    ppo_dir = Path(__file__).parent / 'logs' / 'ppo'
    results_file = ppo_dir / 'comparison_results.pkl'

    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Please run the complete pipeline first.")
        return False

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # Generate report
    report_lines = []
    report_lines.append("# Vaccine Allocation: Prior-Guided RL Results")
    report_lines.append("")
    report_lines.append("## Summary")
    report_lines.append("")

    for variant, metrics in results.items():
        final_rewards = metrics['episode_reward'][-20:]
        mean_reward = np.mean(final_rewards)
        std_reward = np.std(final_rewards)

        report_lines.append(f"### {variant.upper()}")
        report_lines.append(f"- Final Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
        report_lines.append(f"- Total Iterations: {len(metrics['iteration'])}")
        report_lines.append("")

    # Save report
    report_file = ppo_dir / 'README_results.md'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))

    print(f"✓ Report saved to {report_file}")
    print("\n" + '\n'.join(report_lines))

    return True


def main():
    parser = argparse.ArgumentParser(description='Run vaccine allocation pipeline')
    parser.add_argument('--all', action='store_true', help='Run complete pipeline')
    parser.add_argument('--m1', action='store_true', help='Run M1 only (expert trajectories)')
    parser.add_argument('--m2', action='store_true', help='Run M2 only (micro replay)')
    parser.add_argument('--m3', action='store_true', help='Run M3 only (diffusion + PPO)')
    parser.add_argument('--report', action='store_true', help='Generate evaluation report')
    parser.add_argument('--quick', action='store_true', help='Quick test run (small datasets)')

    args = parser.parse_args()

    # If no arguments, run all
    if not (args.all or args.m1 or args.m2 or args.m3 or args.report):
        args.all = True

    success = True

    if args.all:
        success = run_all(args)
    else:
        if args.m1:
            success = success and run_m1(args)
        if args.m2:
            success = success and run_m2(args)
        if args.m3:
            success = success and run_m3(args)

    if args.report or (args.all and success):
        generate_report(args)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
