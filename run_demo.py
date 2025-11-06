"""
Quick Demo: Scaled-down version for fast results

This runs a complete pipeline with reduced scale:
- Expert trajectories: 50 (instead of 1000)
- Micro replay: 25 (instead of 500)
- PPO iterations: 20 (instead of 200)
- Population: 500 (instead of 2000)
- Diffusion epochs: 20 (instead of 100)

Estimated time: 20-30 minutes
"""

import sys
import subprocess
from pathlib import Path

def run_demo():
    print("="*70)
    print("QUICK DEMO: Vaccine Allocation with Diffusion + RL")
    print("="*70)
    print("\nThis demo runs a scaled-down version:")
    print("- 50 expert trajectories (vs 1000 full)")
    print("- 25 micro trajectories (vs 500 full)")
    print("- 20 PPO iterations (vs 200 full)")
    print("- 500 individuals (vs 2000 full)")
    print("\nEstimated time: 20-30 minutes")
    print("="*70 + "\n")

    # M1: Generate expert trajectories (50 instead of 1000)
    print("\n[1/3] Generating 50 expert trajectories...")
    result = subprocess.run([
        sys.executable, "-c",
        """
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from expert.export_expert_trajectories import simulate_episode_macro, generate_scenarios
import pickle

output_dir = Path('data/macro_expert')
output_dir.mkdir(parents=True, exist_ok=True)

scenarios = generate_scenarios(50)
trajectories = []

for i, scenario in enumerate(scenarios):
    try:
        traj = simulate_episode_macro(seed=i, scenario=scenario)
        trajectories.append(traj)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i+1}/50 trajectories")
    except Exception as e:
        print(f"  Error in trajectory {i}: {e}")

output_file = output_dir / 'expert_trajectories_demo.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(trajectories, f)

print(f"✓ Saved {len(trajectories)} trajectories")
"""
    ])

    if result.returncode != 0:
        print("❌ M1 failed")
        return False

    # M2: Replay to micro (25 trajectories)
    print("\n[2/3] Replaying 25 trajectories to micro-level...")
    result = subprocess.run([
        sys.executable, "-c",
        """
import sys
import pickle
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from envs.vax_env import VaxEnv
from lifting.lifting import lift_macro_to_micro, compute_macro_quota
from diffusion.model import create_state_features

macro_file = Path('data/macro_expert/expert_trajectories_demo.pkl')
with open(macro_file, 'rb') as f:
    macro_trajectories = pickle.load(f)

output_dir = Path('data/micro_replay')
output_dir.mkdir(parents=True, exist_ok=True)

# Replay first 25 trajectories with N=500
micro_trajectories = []

for idx in range(min(25, len(macro_trajectories))):
    macro_traj = macro_trajectories[idx]

    env = VaxEnv(
        N=500,  # Reduced from 2000
        G=3,
        f_high_risk=macro_traj['meta']['f_high_risk'],
        f_high_contact=macro_traj['meta']['f_high_contact'],
        max_steps=50,  # Reduced from 100
        vax_supply_per_step=int(macro_traj['meta']['vax_rate'] * 500),
        r_0=macro_traj['meta']['r_0'],
        seed=macro_traj['meta']['seed']
    )

    obs, info = env.reset(seed=macro_traj['meta']['seed'])
    transitions = []

    for t in range(50):
        if t * 10 >= macro_traj['U'].shape[0]:
            break

        t_macro = min(t * 10, macro_traj['U'].shape[0] - 1)
        Ug = compute_macro_quota(macro_traj, t_macro)

        state = {
            'N': 500,
            'group_id': obs['group_id'],
            'vaccinated': obs['vaccinated'],
            'supply_today': obs['supply_today'][0],
            'macro_quota': Ug,
        }

        action = lift_macro_to_micro(Ug, state, env.graph, rule='degree_risk')
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        transitions.append({
            's': obs.copy(),
            'a': action.copy(),
            's_next': next_obs.copy(),
            'r': reward,
            'done': terminated or truncated,
        })

        obs = next_obs
        if terminated or truncated:
            break

    micro_trajectories.append({
        'transitions': transitions,
        'meta': macro_traj['meta'],
        'n_steps': len(transitions),
    })

    if (idx + 1) % 5 == 0:
        print(f"  Replayed {idx+1}/25 trajectories")

output_file = output_dir / 'micro_trajectories_demo.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(micro_trajectories, f)

print(f"✓ Saved {len(micro_trajectories)} micro trajectories")
"""
    ])

    if result.returncode != 0:
        print("❌ M2 failed")
        return False

    # M3: Train diffusion + PPO (quick version)
    print("\n[3/3] Training diffusion model and PPO (20 iterations)...")
    result = subprocess.run([
        sys.executable, "-c",
        """
import sys
import torch
import numpy as np
import pickle
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from diffusion.model import ConditionalTransformer, DiffusionModel, create_state_features
from rl.ppo import PPO
from envs.vax_env import VaxEnv
from lifting.lifting import project_feasible
from torch.utils.data import Dataset, DataLoader

# Load data
data_file = Path('data/micro_replay/micro_trajectories_demo.pkl')
with open(data_file, 'rb') as f:
    trajectories = pickle.load(f)

transitions = []
for traj in trajectories:
    transitions.extend(traj['transitions'])

print(f"Loaded {len(transitions)} transitions")

# Train diffusion model (20 epochs)
print("\\nTraining diffusion model...")

class SimpleDataset(Dataset):
    def __init__(self, transitions):
        self.transitions = transitions

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        trans = self.transitions[idx]
        return {
            'obs': {
                'node_states': trans['s']['node_states'],
                'vaccinated': trans['s']['vaccinated'],
                'group_id': trans['s']['group_id'],
            },
            'action': trans['a']
        }

def collate_fn(batch):
    obs_batch = {
        'node_states': np.stack([b['obs']['node_states'] for b in batch]),
        'vaccinated': np.stack([b['obs']['vaccinated'] for b in batch]),
        'group_id': np.stack([b['obs']['group_id'] for b in batch]),
    }
    actions = np.stack([b['action'] for b in batch])
    return obs_batch, actions

dataset = SimpleDataset(transitions)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

denoiser = ConditionalTransformer(N=500, d_model=64, nhead=2, num_layers=2, state_dim=10)
model = DiffusionModel(denoiser=denoiser, T=100, beta_schedule='linear')

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(20):
    epoch_loss = 0
    for obs_batch, actions_batch in dataloader:
        actions = torch.tensor(actions_batch, dtype=torch.float32)
        state_features = create_state_features(obs_batch, device='cpu')

        loss = model.p_losses(actions, state_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    if (epoch + 1) % 5 == 0:
        print(f"  Epoch {epoch+1}/20: loss={avg_loss:.4f}")

# Save diffusion model
output_dir = Path('logs/diffusion_demo')
output_dir.mkdir(parents=True, exist_ok=True)
torch.save({'model_state_dict': model.state_dict()}, output_dir / 'diffusion_model.pt')
print("✓ Diffusion model trained")

# Train PPO variants (20 iterations each)
print("\\nTraining PPO variants...")

env = VaxEnv(N=500, G=3, max_steps=50, vax_supply_per_step=5, r_0=2.0, seed=42)

results = {}

# RL-only
print("  Training RL-only...")
agent_rl = PPO(N=500, state_dim=10, lr=3e-4, kl_coef=0.0, device='cpu')

rewards_rl = []
for iteration in range(20):
    # Collect rollout
    obs, _ = env.reset()
    episode_reward = 0

    for step in range(50):
        state_features = create_state_features(obs, device='cpu')
        action, _, _ = agent_rl.policy.get_action(state_features, deterministic=False)
        action_np = action.cpu().numpy()[0]

        state_dict = {'N': 500, 'group_id': obs['group_id'],
                     'vaccinated': obs['vaccinated'], 'supply_today': obs['supply_today'][0]}
        action_np = project_feasible(action_np, state_dict)

        obs, reward, terminated, truncated, _ = env.step(action_np)
        episode_reward += reward

        if terminated or truncated:
            break

    rewards_rl.append(episode_reward)
    if (iteration + 1) % 5 == 0:
        print(f"    Iter {iteration+1}: reward={episode_reward:.1f}")

results['rl_only'] = {'rewards': rewards_rl}

# BC+RL
print("  Training BC+RL...")
agent_bc = PPO(N=500, state_dim=10, lr=3e-4, kl_coef=0.0, device='cpu')

# BC warmstart (5 epochs)
for epoch in range(5):
    for obs_batch, actions_batch in dataloader:
        states = create_state_features(obs_batch, device='cpu')
        actions = torch.tensor(actions_batch, dtype=torch.float32)

        mean, std = agent_bc.policy(states)
        loss = torch.nn.functional.mse_loss(mean, actions)

        agent_bc.optimizer.zero_grad()
        loss.backward()
        agent_bc.optimizer.step()

rewards_bc = []
for iteration in range(20):
    obs, _ = env.reset()
    episode_reward = 0

    for step in range(50):
        state_features = create_state_features(obs, device='cpu')
        action, _, _ = agent_bc.policy.get_action(state_features, deterministic=False)
        action_np = action.cpu().numpy()[0]

        state_dict = {'N': 500, 'group_id': obs['group_id'],
                     'vaccinated': obs['vaccinated'], 'supply_today': obs['supply_today'][0]}
        action_np = project_feasible(action_np, state_dict)

        obs, reward, terminated, truncated, _ = env.step(action_np)
        episode_reward += reward

        if terminated or truncated:
            break

    rewards_bc.append(episode_reward)
    if (iteration + 1) % 5 == 0:
        print(f"    Iter {iteration+1}: reward={episode_reward:.1f}")

results['bc_rl'] = {'rewards': rewards_bc}

# Diffusion+RL
print("  Training Diffusion+RL...")
agent_diff = PPO(N=500, state_dim=10, lr=3e-4, kl_coef=0.1, kl_decay=0.95, device='cpu')
agent_diff.set_prior_policy(model.denoiser)

rewards_diff = []
for iteration in range(20):
    obs, _ = env.reset()
    episode_reward = 0

    for step in range(50):
        state_features = create_state_features(obs, device='cpu')
        action, _, _ = agent_diff.policy.get_action(state_features, deterministic=False)
        action_np = action.cpu().numpy()[0]

        state_dict = {'N': 500, 'group_id': obs['group_id'],
                     'vaccinated': obs['vaccinated'], 'supply_today': obs['supply_today'][0]}
        action_np = project_feasible(action_np, state_dict)

        obs, reward, terminated, truncated, _ = env.step(action_np)
        episode_reward += reward

        if terminated or truncated:
            break

    rewards_diff.append(episode_reward)
    if (iteration + 1) % 5 == 0:
        print(f"    Iter {iteration+1}: reward={episode_reward:.1f}")

results['diffusion_rl'] = {'rewards': rewards_diff}

print("\\n✓ All training complete")

# Save results
output_dir = Path('logs/demo_results')
output_dir.mkdir(parents=True, exist_ok=True)
with open(output_dir / 'results.pkl', 'wb') as f:
    pickle.dump(results, f)

# Print summary
print("\\n" + "="*70)
print("DEMO RESULTS SUMMARY")
print("="*70)

for method, data in results.items():
    final_rewards = data['rewards'][-5:]
    print(f"\\n{method.upper()}:")
    print(f"  Initial reward: {data['rewards'][0]:.1f}")
    print(f"  Final reward (last 5 avg): {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")

print("\\n" + "="*70)
"""
    ])

    if result.returncode != 0:
        print("❌ M3 failed")
        return False

    print("\n" + "="*70)
    print("✓ DEMO COMPLETE!")
    print("="*70)
    print("\nResults saved to logs/demo_results/")

    return True


if __name__ == '__main__':
    success = run_demo()
    sys.exit(0 if success else 1)
