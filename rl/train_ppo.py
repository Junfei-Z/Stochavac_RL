"""
Train PPO Agents with Different Initialization Strategies

Three variants:
1. RL-only: PPO from scratch
2. BC+RL: Behavior cloning warm-start + PPO
3. Diffusion+RL: Diffusion prior + PPO with KL regularization
"""

import sys
import os
import torch
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from envs.vax_env import VaxEnv
from rl.ppo import PPO, PolicyNetwork
from diffusion.model import DiffusionModel, ConditionalTransformer, create_state_features
from lifting.lifting import project_feasible


def collect_rollout(
    env: VaxEnv,
    agent: PPO,
    max_steps: int = 100,
    device: str = 'cpu',
):
    """
    Collect a single rollout.

    Returns:
        Dictionary containing states, actions, rewards, etc.
    """
    states_list = []
    actions_list = []
    log_probs_list = []
    rewards_list = []
    values_list = []
    dones_list = []

    obs, info = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        # Convert obs to state features
        state_features = create_state_features(obs, device=device)

        # Get action from policy
        with torch.no_grad():
            action, log_prob, _ = agent.policy.get_action(state_features, deterministic=False)
            value = agent.value(state_features)

        action_np = action.cpu().numpy()[0]
        log_prob_np = log_prob.cpu().numpy()[0]
        value_np = value.cpu().numpy()[0]

        # Project action to feasible region
        state_dict = {
            'N': env.N,
            'group_id': obs['group_id'],
            'vaccinated': obs['vaccinated'],
            'supply_today': obs['supply_today'][0],
        }
        action_np = project_feasible(action_np, state_dict)

        # Execute action
        next_obs, reward, terminated, truncated, info = env.step(action_np)
        done = terminated or truncated

        # Store transition
        states_list.append(state_features.cpu())
        actions_list.append(torch.tensor(action_np, dtype=torch.float32))
        log_probs_list.append(torch.tensor(log_prob_np, dtype=torch.float32))
        rewards_list.append(reward)
        values_list.append(value_np)
        dones_list.append(float(done))

        obs = next_obs
        step += 1

    # Final value
    state_features = create_state_features(obs, device=device)
    with torch.no_grad():
        final_value = agent.value(state_features).cpu().numpy()[0]

    rollout = {
        'states': torch.cat(states_list, dim=0),  # (T, N, state_dim)
        'actions': torch.stack(actions_list),  # (T, N)
        'log_probs': torch.stack(log_probs_list),  # (T, N)
        'rewards': np.array(rewards_list, dtype=np.float32),  # (T,)
        'values': np.array(values_list, dtype=np.float32),  # (T,)
        'dones': np.array(dones_list, dtype=np.float32),  # (T,)
        'final_value': final_value,
    }

    return rollout


def train_ppo_variant(
    variant: str,
    env: VaxEnv,
    agent: PPO,
    n_iterations: int = 200,
    n_rollouts_per_iter: int = 4,
    max_steps_per_rollout: int = 100,
    output_dir: Path = None,
    device: str = 'cpu',
):
    """
    Train PPO agent with specified variant.

    Args:
        variant: 'rl_only', 'bc_rl', or 'diffusion_rl'
        env: Environment
        agent: PPO agent
        n_iterations: Number of training iterations
        n_rollouts_per_iter: Rollouts per iteration
        max_steps_per_rollout: Max steps per rollout
        output_dir: Output directory for logs
        device: Device

    Returns:
        Dictionary of training metrics
    """
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(output_dir / 'runs' / variant)
    else:
        writer = None

    metrics_history = {
        'iteration': [],
        'episode_reward': [],
        'episode_length': [],
        'policy_loss': [],
        'value_loss': [],
        'kl_div': [],
    }

    for iteration in tqdm(range(n_iterations), desc=f"Training {variant}"):
        # Collect rollouts
        all_states = []
        all_actions = []
        all_log_probs = []
        all_returns = []
        all_advantages = []

        total_reward = 0
        total_steps = 0

        for _ in range(n_rollouts_per_iter):
            rollout = collect_rollout(env, agent, max_steps_per_rollout, device)

            # Compute GAE
            values_with_final = np.concatenate([rollout['values'], [rollout['final_value']]])
            advantages, returns = agent.compute_gae(
                rewards=rollout['rewards'],
                values=values_with_final,
                dones=rollout['dones']
            )

            all_states.append(rollout['states'])
            all_actions.append(rollout['actions'])
            all_log_probs.append(rollout['log_probs'])
            all_returns.append(torch.tensor(returns, dtype=torch.float32))
            all_advantages.append(torch.tensor(advantages, dtype=torch.float32))

            total_reward += rollout['rewards'].sum()
            total_steps += len(rollout['rewards'])

        # Concatenate all rollouts
        states = torch.cat(all_states, dim=0).to(device)
        actions = torch.cat(all_actions, dim=0).to(device)
        log_probs = torch.cat(all_log_probs, dim=0).to(device)
        returns = torch.cat(all_returns, dim=0).to(device)
        advantages = torch.cat(all_advantages, dim=0).to(device)

        # Update agent
        update_metrics = agent.update(
            states=states,
            actions=actions,
            old_log_probs=log_probs,
            returns=returns,
            advantages=advantages,
            n_epochs=10,
            batch_size=64,
        )

        # Log metrics
        avg_reward = total_reward / n_rollouts_per_iter
        avg_length = total_steps / n_rollouts_per_iter

        metrics_history['iteration'].append(iteration)
        metrics_history['episode_reward'].append(avg_reward)
        metrics_history['episode_length'].append(avg_length)
        metrics_history['policy_loss'].append(update_metrics['policy_loss'])
        metrics_history['value_loss'].append(update_metrics['value_loss'])
        metrics_history['kl_div'].append(update_metrics['kl_div'])

        if writer:
            writer.add_scalar('train/episode_reward', avg_reward, iteration)
            writer.add_scalar('train/episode_length', avg_length, iteration)
            writer.add_scalar('train/policy_loss', update_metrics['policy_loss'], iteration)
            writer.add_scalar('train/value_loss', update_metrics['value_loss'], iteration)
            writer.add_scalar('train/kl_div', update_metrics['kl_div'], iteration)
            writer.add_scalar('train/kl_coef', update_metrics['kl_coef'], iteration)

        if (iteration + 1) % 20 == 0:
            print(f"{variant} - Iter {iteration+1}: reward={avg_reward:.2f}, length={avg_length:.1f}")

        # Save checkpoint
        if output_dir and (iteration + 1) % 50 == 0:
            checkpoint_path = output_dir / f'{variant}_checkpoint_{iteration+1}.pt'
            torch.save({
                'iteration': iteration,
                'policy_state_dict': agent.policy.state_dict(),
                'value_state_dict': agent.value.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
            }, checkpoint_path)

    if writer:
        writer.close()

    # Save final model
    if output_dir:
        final_path = output_dir / f'{variant}_final.pt'
        torch.save({
            'policy_state_dict': agent.policy.state_dict(),
            'value_state_dict': agent.value.state_dict(),
        }, final_path)

        # Save metrics
        metrics_path = output_dir / f'{variant}_metrics.pkl'
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics_history, f)

    return metrics_history


def behavior_cloning_warmstart(
    agent: PPO,
    data_file: Path,
    n_epochs: int = 20,
    batch_size: int = 32,
    device: str = 'cpu',
):
    """
    Pre-train policy via behavior cloning on expert data.

    Args:
        agent: PPO agent to warm-start
        data_file: Path to micro_trajectories.pkl
        n_epochs: Number of BC epochs
        batch_size: Batch size
        device: Device
    """
    print("Performing behavior cloning warm-start...")

    # Load expert data
    with open(data_file, 'rb') as f:
        trajectories = pickle.load(f)

    # Collect transitions
    transitions = []
    for traj in trajectories:
        transitions.extend(traj['transitions'])

    print(f"Loaded {len(transitions)} expert transitions")

    # BC training loop
    optimizer = torch.optim.Adam(agent.policy.parameters(), lr=1e-4)

    for epoch in range(n_epochs):
        # Shuffle transitions
        indices = np.random.permutation(len(transitions))
        epoch_loss = 0
        n_batches = 0

        for start in range(0, len(transitions), batch_size):
            end = min(start + batch_size, len(transitions))
            batch_indices = indices[start:end]

            # Get batch
            states_list = []
            actions_list = []

            for idx in batch_indices:
                trans = transitions[idx]
                obs = {
                    'node_states': trans['s']['node_states'][None, :],
                    'vaccinated': trans['s']['vaccinated'][None, :],
                    'group_id': trans['s']['group_id'][None, :],
                }
                state_features = create_state_features(obs, device=device)
                states_list.append(state_features)
                actions_list.append(torch.tensor(trans['a'], dtype=torch.float32, device=device))

            states = torch.cat(states_list, dim=0)
            actions = torch.stack(actions_list)

            # Forward pass
            mean, std = agent.policy(states)

            # MSE loss
            loss = F.mse_loss(mean, actions)

            # Update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if (epoch + 1) % 5 == 0:
            print(f"BC Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    print("✓ Behavior cloning complete")


def load_diffusion_prior(
    model_path: Path,
    device: str = 'cpu'
) -> PolicyNetwork:
    """
    Load diffusion model and wrap as policy network for prior.

    Args:
        model_path: Path to diffusion model checkpoint
        device: Device

    Returns:
        PolicyNetwork compatible interface
    """
    print(f"Loading diffusion model from {model_path}...")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Create denoiser
    denoiser = ConditionalTransformer(
        N=2000,
        d_model=128,
        nhead=4,
        num_layers=4,
        state_dim=10,
    ).to(device)

    # Create diffusion model
    diffusion_model = DiffusionModel(
        denoiser=denoiser,
        T=1000,
        beta_schedule='linear',
    ).to(device)

    diffusion_model.load_state_dict(checkpoint['model_state_dict'])
    diffusion_model.eval()

    # Wrap as policy network
    class DiffusionPolicyWrapper(nn.Module):
        def __init__(self, diffusion_model):
            super().__init__()
            self.diffusion_model = diffusion_model

        def forward(self, state):
            # Sample from diffusion model
            with torch.no_grad():
                B, N, _ = state.shape
                samples = self.diffusion_model.sample(
                    state=state,
                    shape=(B, N),
                    return_trajectory=False
                )

                # Estimate mean and std from samples
                # For simplicity, use the sample as mean and small fixed std
                mean = samples
                std = torch.ones_like(mean) * 0.1

            return mean, std

    prior_policy = DiffusionPolicyWrapper(diffusion_model)

    print("✓ Diffusion prior loaded")

    return prior_policy


def main():
    """Main training script."""

    # Paths
    data_dir = Path(__file__).parent.parent / 'data' / 'micro_replay'
    data_file = data_dir / 'micro_trajectories.pkl'
    diffusion_model_path = Path(__file__).parent.parent / 'logs' / 'diffusion' / 'diffusion_model_final.pt'
    output_dir = Path(__file__).parent.parent / 'logs' / 'ppo'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check if diffusion model exists
    use_diffusion = diffusion_model_path.exists()
    if not use_diffusion:
        print(f"Warning: Diffusion model not found at {diffusion_model_path}")
        print("Will skip Diffusion+RL variant")

    # Environment
    env = VaxEnv(
        N=2000,
        G=3,
        max_steps=100,
        vax_supply_per_step=10,
        r_0=2.0,
        seed=42
    )

    # Training parameters
    n_iterations = 200
    n_rollouts_per_iter = 4
    max_steps_per_rollout = 100

    results = {}

    # 1. RL-only
    print("\n" + "="*50)
    print("Training RL-only (PPO from scratch)")
    print("="*50)

    agent_rl = PPO(
        N=2000,
        state_dim=10,
        lr=3e-4,
        kl_coef=0.0,  # No KL regularization
        device=device
    )

    results['rl_only'] = train_ppo_variant(
        variant='rl_only',
        env=env,
        agent=agent_rl,
        n_iterations=n_iterations,
        n_rollouts_per_iter=n_rollouts_per_iter,
        max_steps_per_rollout=max_steps_per_rollout,
        output_dir=output_dir,
        device=device
    )

    # 2. BC+RL
    if data_file.exists():
        print("\n" + "="*50)
        print("Training BC+RL (Behavior Cloning + PPO)")
        print("="*50)

        agent_bc = PPO(
            N=2000,
            state_dim=10,
            lr=3e-4,
            kl_coef=0.0,
            device=device
        )

        # Warm-start with BC
        behavior_cloning_warmstart(
            agent=agent_bc,
            data_file=data_file,
            n_epochs=20,
            device=device
        )

        results['bc_rl'] = train_ppo_variant(
            variant='bc_rl',
            env=env,
            agent=agent_bc,
            n_iterations=n_iterations,
            n_rollouts_per_iter=n_rollouts_per_iter,
            max_steps_per_rollout=max_steps_per_rollout,
            output_dir=output_dir,
            device=device
        )

    # 3. Diffusion+RL
    if use_diffusion:
        print("\n" + "="*50)
        print("Training Diffusion+RL (Diffusion Prior + PPO with KL)")
        print("="*50)

        agent_diff = PPO(
            N=2000,
            state_dim=10,
            lr=3e-4,
            kl_coef=0.1,  # KL regularization
            kl_decay=0.995,
            device=device
        )

        # Load diffusion prior
        prior_policy = load_diffusion_prior(diffusion_model_path, device=device)
        agent_diff.set_prior_policy(prior_policy)

        results['diffusion_rl'] = train_ppo_variant(
            variant='diffusion_rl',
            env=env,
            agent=agent_diff,
            n_iterations=n_iterations,
            n_rollouts_per_iter=n_rollouts_per_iter,
            max_steps_per_rollout=max_steps_per_rollout,
            output_dir=output_dir,
            device=device
        )

    # Save comparison results
    comparison_path = output_dir / 'comparison_results.pkl'
    with open(comparison_path, 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ All experiments complete. Results saved to {output_dir}")

    # Plot comparison
    plot_comparison(results, output_dir)


def plot_comparison(results: dict, output_dir: Path):
    """Plot learning curves comparison."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for variant, metrics in results.items():
        iterations = metrics['iteration']
        rewards = metrics['episode_reward']

        # Smooth rewards
        window = 10
        if len(rewards) >= window:
            rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
            iterations_smooth = iterations[:len(rewards_smooth)]
        else:
            rewards_smooth = rewards
            iterations_smooth = iterations

        axes[0].plot(iterations_smooth, rewards_smooth, label=variant, linewidth=2)

    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Episode Reward')
    axes[0].set_title('Learning Curves Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot final performance comparison
    final_rewards = [results[v]['episode_reward'][-20:] for v in results.keys()]
    mean_final = [np.mean(r) for r in final_rewards]
    std_final = [np.std(r) for r in final_rewards]

    x = np.arange(len(results))
    axes[1].bar(x, mean_final, yerr=std_final, capsize=5, alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(results.keys())
    axes[1].set_ylabel('Final Episode Reward (last 20)')
    axes[1].set_title('Final Performance Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'comparison_plot.png', dpi=150)
    print(f"✓ Comparison plot saved to {output_dir / 'comparison_plot.png'}")


if __name__ == '__main__':
    import torch.nn.functional as F
    main()
