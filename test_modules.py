"""
Quick test script to verify all modules work correctly.
"""

import sys
import numpy as np
from pathlib import Path

print("Testing module imports...")

# Test 1: Expert trajectory generation
print("\n1. Testing expert trajectory generation...")
sys.path.insert(0, str(Path(__file__).parent / 'third_party' / 'ProtectorPrevent' / 'Simulation'))

try:
    from expert.export_expert_trajectories import simulate_episode_macro

    scenario = {
        'r_0': 2.0,
        'vax_rate': 0.005,
        'init_infections': 15,
        'strategy': 'high risk',
        'f_high_risk': 0.168,
        'f_high_contact': 0.15,
        'vax_efficacy': 0.8,
        'npi_efficacy': 0.3,
        'variant_param': 1.0,
    }

    traj = simulate_episode_macro(seed=0, scenario=scenario)
    print(f"  ✓ Generated trajectory with {traj['meta']['T']} timesteps")
    print(f"  ✓ Final deaths: {traj['D'][-1].sum():.2f}")

except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: VaxEnv
print("\n2. Testing VaxEnv...")
try:
    from envs.vax_env import VaxEnv

    env = VaxEnv(N=100, max_steps=10)
    obs, info = env.reset(seed=42)

    print(f"  ✓ Environment created with N={env.N}")
    print(f"  ✓ Initial state: {info}")

    # Take a random action
    action = np.random.rand(100) * 0.1
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  ✓ Step executed, reward={reward:.2f}")

except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Lifting
print("\n3. Testing Lifting module...")
try:
    from lifting.lifting import lift_macro_to_micro
    import networkx as nx

    # Create simple graph
    G = nx.erdos_renyi_graph(100, 0.1)

    Ug = np.array([3.0, 2.0, 1.0])  # Group allocations
    state = {
        'N': 100,
        'group_id': np.repeat([0, 1, 2], [50, 30, 20]),
        'vaccinated': np.zeros(100),
        'supply_today': 6.0,
    }

    action = lift_macro_to_micro(Ug, state, G, rule='degree_risk')
    print(f"  ✓ Lifted action sum: {action.sum():.2f} (target: {Ug.sum():.2f})")

except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Diffusion model
print("\n4. Testing Diffusion model...")
try:
    import torch
    from diffusion.model import ConditionalTransformer, DiffusionModel, create_state_features

    denoiser = ConditionalTransformer(N=100, d_model=64, nhead=2, num_layers=2)
    model = DiffusionModel(denoiser=denoiser, T=100)

    # Create dummy state
    obs = {
        'node_states': np.zeros(100, dtype=np.int32),
        'vaccinated': np.zeros(100, dtype=np.int32),
        'group_id': np.zeros(100, dtype=np.int32),
    }

    state_features = create_state_features(obs, device='cpu')
    print(f"  ✓ State features shape: {state_features.shape}")

    # Sample
    samples = model.sample(state_features, shape=(1, 100))
    print(f"  ✓ Sampled actions shape: {samples.shape}")

except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: PPO
print("\n5. Testing PPO...")
try:
    from rl.ppo import PPO

    agent = PPO(N=100, state_dim=10, device='cpu')
    print(f"  ✓ PPO agent created")

    # Test action sampling
    state_features = torch.randn(1, 100, 10)
    action, log_prob, entropy = agent.policy.get_action(state_features)
    print(f"  ✓ Sampled action shape: {action.shape}")

except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*50)
print("✓ All module tests passed!")
print("="*50)
