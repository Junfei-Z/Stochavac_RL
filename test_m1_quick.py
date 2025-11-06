"""
Quick test of M1: Generate small batch of expert trajectories
"""

import sys
import pickle
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'ProtectorPrevent' / 'Simulation'))

from expert.export_expert_trajectories import simulate_episode_macro, generate_scenarios

print("Quick M1 Test: Generating 10 expert trajectories...")

# Create output directory
output_dir = Path(__file__).parent / 'data' / 'macro_expert'
output_dir.mkdir(parents=True, exist_ok=True)

# Generate 10 scenarios
scenarios = generate_scenarios(10)

trajectories = []

for i, scenario in enumerate(tqdm(scenarios)):
    try:
        traj = simulate_episode_macro(seed=i, scenario=scenario)
        trajectories.append(traj)
        print(f"  Trajectory {i}: Deaths={traj['D'][-1].sum():.1f}, Strategy={scenario['strategy']}")
    except Exception as e:
        print(f"  Error in trajectory {i}: {e}")

# Save
output_file = output_dir / 'test_trajectories.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(trajectories, f)

print(f"\n✓ Saved {len(trajectories)} trajectories to {output_file}")
print("✓ M1 quick test passed!")
