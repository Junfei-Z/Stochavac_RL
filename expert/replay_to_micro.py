"""
Replay Macro Expert Trajectories to Micro-level CTMP Environment

This script takes macro-level expert trajectories and replays them in the
individual-level VaxEnv to generate micro-level (s, a, s', r) data.
"""

import sys
import os
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.vax_env import VaxEnv
from lifting.lifting import lift_macro_to_micro, create_micro_state_from_macro, compute_macro_quota


def replay_trajectory(
    macro_traj: dict,
    env: VaxEnv,
    lifting_rule: str = 'degree_risk',
    max_steps: int = 100,
    subsample_rate: int = 10
) -> dict:
    """
    Replay a macro trajectory in the micro environment.

    Args:
        macro_traj: Expert trajectory from simulate_episode_macro
        env: VaxEnv environment instance
        lifting_rule: Rule for lifting macro to micro allocations
        max_steps: Maximum number of steps to simulate
        subsample_rate: Subsample macro trajectory (ODE has finer timesteps)

    Returns:
        Dictionary of micro-level transitions
    """
    # Reset environment
    obs, info = env.reset(seed=macro_traj['meta']['seed'])

    transitions = []
    group_id = obs['group_id']

    # Macro trajectory has shape (T_macro, G)
    # We subsample it to match our micro timesteps
    T_macro = macro_traj['U'].shape[0]
    micro_steps = min(max_steps, T_macro // subsample_rate)

    for t in range(micro_steps):
        # Map micro time to macro time
        t_macro = min(t * subsample_rate, T_macro - 1)

        # Get macro quota at this time
        Ug = compute_macro_quota(macro_traj, t_macro)

        # Create micro state
        state = {
            'N': env.N,
            'group_id': group_id,
            'vaccinated': obs['vaccinated'],
            'supply_today': obs['supply_today'][0],
            'macro_quota': Ug,
        }

        # Lift to micro allocation
        action = lift_macro_to_micro(
            Ug=Ug,
            state=state,
            graph=env.graph,
            rule=lifting_rule
        )

        # Store transition
        transition = {
            's': obs.copy(),
            'a': action.copy(),
        }

        # Execute action
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        transition['s_next'] = next_obs.copy()
        transition['r'] = reward
        transition['done'] = terminated or truncated
        transition['info'] = next_info.copy()

        transitions.append(transition)

        obs = next_obs

        if terminated or truncated:
            break

    return {
        'transitions': transitions,
        'meta': macro_traj['meta'],
        'lifting_rule': lifting_rule,
        'n_steps': len(transitions),
    }


def main():
    """Replay expert trajectories and save micro-level data."""

    # Load macro expert trajectories
    macro_dir = Path(__file__).parent.parent / 'data' / 'macro_expert'
    macro_file = macro_dir / 'expert_trajectories.pkl'

    if not macro_file.exists():
        print(f"Error: {macro_file} not found. Run export_expert_trajectories.py first.")
        return

    print(f"Loading macro trajectories from {macro_file}...")
    with open(macro_file, 'rb') as f:
        macro_trajectories = pickle.load(f)

    print(f"Loaded {len(macro_trajectories)} macro trajectories")

    # Create output directory
    output_dir = Path(__file__).parent.parent / 'data' / 'micro_replay'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Replay subset of trajectories (500 out of 1000)
    n_replay = min(500, len(macro_trajectories))
    indices = np.random.choice(len(macro_trajectories), size=n_replay, replace=False)

    print(f"Replaying {n_replay} trajectories to micro-level...")

    micro_trajectories = []
    feasibility_stats = {
        'total_steps': 0,
        'feasible_steps': 0,
        'avg_supply_violation': 0,
    }

    for idx in tqdm(indices):
        macro_traj = macro_trajectories[idx]

        # Create environment with matching parameters
        env = VaxEnv(
            N=2000,
            G=3,
            f_high_risk=macro_traj['meta']['f_high_risk'],
            f_high_contact=macro_traj['meta']['f_high_contact'],
            max_steps=100,
            vax_supply_per_step=int(macro_traj['meta']['vax_rate'] * 2000),
            r_0=macro_traj['meta']['r_0'],
            seed=macro_traj['meta']['seed']
        )

        try:
            micro_traj = replay_trajectory(
                macro_traj=macro_traj,
                env=env,
                lifting_rule='degree_risk',
                max_steps=100,
                subsample_rate=10
            )

            micro_trajectories.append(micro_traj)

            # Compute feasibility statistics
            for trans in micro_traj['transitions']:
                feasibility_stats['total_steps'] += 1

                # Check if action respects supply
                action_sum = trans['a'].sum()
                supply = trans['s']['supply_today'][0]

                if action_sum <= supply * 1.01:  # Allow 1% tolerance
                    feasibility_stats['feasible_steps'] += 1
                else:
                    feasibility_stats['avg_supply_violation'] += (action_sum - supply)

        except Exception as e:
            print(f"Error replaying trajectory {idx}: {e}")
            continue

    # Save micro trajectories
    output_file = output_dir / 'micro_trajectories.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(micro_trajectories, f)

    print(f"✓ Saved {len(micro_trajectories)} micro trajectories to {output_file}")

    # Compute and save feasibility report
    if feasibility_stats['total_steps'] > 0:
        feasibility_rate = feasibility_stats['feasible_steps'] / feasibility_stats['total_steps']
        avg_violation = feasibility_stats['avg_supply_violation'] / max(feasibility_stats['total_steps'], 1)
    else:
        feasibility_rate = 0
        avg_violation = 0

    print("\n=== Feasibility Statistics ===")
    print(f"Total steps: {feasibility_stats['total_steps']}")
    print(f"Feasible steps: {feasibility_stats['feasible_steps']}")
    print(f"Feasibility rate: {feasibility_rate:.2%}")
    print(f"Avg supply violation: {avg_violation:.4f}")

    # Save statistics
    stats = {
        'n_trajectories': len(micro_trajectories),
        'feasibility_rate': feasibility_rate,
        'avg_supply_violation': avg_violation,
        'total_steps': feasibility_stats['total_steps'],
    }

    stats_file = output_dir / 'statistics.pkl'
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)

    if feasibility_rate >= 0.95:
        print(f"\n✓ M2 Complete: Feasibility rate {feasibility_rate:.2%} >= 95%")
    else:
        print(f"\n⚠ Warning: Feasibility rate {feasibility_rate:.2%} < 95%")

    return micro_trajectories, stats


if __name__ == '__main__':
    main()
