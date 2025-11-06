"""
Train Conditional Diffusion Model on Expert Trajectories
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from diffusion.model import ConditionalTransformer, DiffusionModel, create_state_features
from lifting.lifting import project_feasible


class MicroTrajectoryDataset(Dataset):
    """Dataset of micro-level transitions from expert replay."""

    def __init__(self, trajectories_file: Path):
        """
        Args:
            trajectories_file: Path to micro_trajectories.pkl
        """
        with open(trajectories_file, 'rb') as f:
            trajectories = pickle.load(f)

        # Flatten transitions from all trajectories
        self.transitions = []
        for traj in trajectories:
            self.transitions.extend(traj['transitions'])

        print(f"Loaded {len(self.transitions)} transitions from {len(trajectories)} trajectories")

    def __len__(self):
        return len(self.transitions)

    def __getitem__(self, idx):
        trans = self.transitions[idx]

        # Extract state and action
        state_dict = trans['s']
        action = trans['a']

        # Convert to tensors
        obs = {
            'node_states': state_dict['node_states'],
            'vaccinated': state_dict['vaccinated'],
            'group_id': state_dict['group_id'],
        }

        return {
            'obs': obs,
            'action': action,
        }


def collate_fn(batch):
    """Custom collate function for batching."""
    obs_batch = {
        'node_states': np.stack([b['obs']['node_states'] for b in batch]),
        'vaccinated': np.stack([b['obs']['vaccinated'] for b in batch]),
        'group_id': np.stack([b['obs']['group_id'] for b in batch]),
    }

    actions = np.stack([b['action'] for b in batch])

    return obs_batch, actions


def train_diffusion_model(
    data_file: Path,
    output_dir: Path,
    n_epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    log_interval: int = 100,
):
    """
    Train the conditional diffusion model.

    Args:
        data_file: Path to micro_trajectories.pkl
        output_dir: Directory to save checkpoints and logs
        n_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        device: Device to train on
        log_interval: Steps between logging
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'runs')

    # Load dataset
    print("Loading dataset...")
    dataset = MicroTrajectoryDataset(data_file)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Create model
    print("Creating model...")
    denoiser = ConditionalTransformer(
        N=2000,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        state_dim=10,
    ).to(device)

    model = DiffusionModel(
        denoiser=denoiser,
        T=1000,
        beta_schedule='linear',
        beta_start=1e-4,
        beta_end=0.02,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Training loop
    print(f"Training on {device}...")
    global_step = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for obs_batch, actions_batch in pbar:
            # Move to device
            actions = torch.tensor(actions_batch, dtype=torch.float32, device=device)
            state_features = create_state_features(obs_batch, device=device)

            # Forward pass
            loss = model.p_losses(actions, state_features)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Logging
            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix({'loss': loss.item()})

            if global_step % log_interval == 0:
                writer.add_scalar('train/loss', loss.item(), global_step)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], global_step)

        # Epoch summary
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")
        writer.add_scalar('train/epoch_loss', avg_loss, epoch)

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_model_path = output_dir / 'diffusion_model_final.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'N': 2000,
            'd_model': 128,
            'nhead': 4,
            'num_layers': 4,
            'T': 1000,
        }
    }, final_model_path)

    print(f"✓ Training complete. Final model saved to {final_model_path}")

    writer.close()

    return model


def evaluate_diffusion_model(
    model: DiffusionModel,
    data_file: Path,
    n_samples: int = 100,
    device: str = 'cpu',
):
    """
    Evaluate diffusion model on held-out data.

    Args:
        model: Trained diffusion model
        data_file: Path to evaluation data
        n_samples: Number of samples to evaluate
        device: Device to run on

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    # Load dataset
    dataset = MicroTrajectoryDataset(data_file)

    # Sample random transitions
    indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)

    mae_list = []
    supply_violations = []

    for idx in tqdm(indices, desc="Evaluating"):
        sample = dataset[idx]

        # Get state and ground truth action
        obs = {k: v[None, :] for k, v in sample['obs'].items()}  # Add batch dim
        action_gt = sample['action']

        # Create state features
        state_features = create_state_features(obs, device=device)

        # Generate action from diffusion model
        with torch.no_grad():
            action_pred = model.sample(
                state=state_features,
                shape=(1, len(action_gt)),
                return_trajectory=False
            )
            action_pred = action_pred.cpu().numpy()[0]

        # Compute MAE
        mae = np.abs(action_pred - action_gt).mean()
        mae_list.append(mae)

        # Check supply constraint
        supply = obs['supply_today'][0] if 'supply_today' in obs else action_gt.sum()
        supply_violation = max(0, action_pred.sum() - supply)
        supply_violations.append(supply_violation)

    metrics = {
        'mae_mean': np.mean(mae_list),
        'mae_std': np.std(mae_list),
        'supply_violation_mean': np.mean(supply_violations),
        'supply_violation_rate': np.mean([v > 0.01 for v in supply_violations]),
    }

    print("\n=== Evaluation Results ===")
    print(f"MAE: {metrics['mae_mean']:.4f} ± {metrics['mae_std']:.4f}")
    print(f"Supply violation: {metrics['supply_violation_mean']:.4f}")
    print(f"Supply violation rate: {metrics['supply_violation_rate']:.2%}")

    return metrics


def main():
    """Main training script."""

    # Paths
    data_dir = Path(__file__).parent.parent / 'data' / 'micro_replay'
    data_file = data_dir / 'micro_trajectories.pkl'

    if not data_file.exists():
        print(f"Error: {data_file} not found. Run replay_to_micro.py first.")
        return

    output_dir = Path(__file__).parent.parent / 'logs' / 'diffusion'

    # Train model
    model = train_diffusion_model(
        data_file=data_file,
        output_dir=output_dir,
        n_epochs=100,
        batch_size=32,
        lr=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # Evaluate model
    metrics = evaluate_diffusion_model(
        model=model,
        data_file=data_file,
        n_samples=100,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

    # Save metrics
    metrics_file = output_dir / 'metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

    if metrics['mae_mean'] < 0.15:
        print(f"\n✓ M3-Part1 Complete: MAE {metrics['mae_mean']:.4f} < 0.15")
    else:
        print(f"\n⚠ Warning: MAE {metrics['mae_mean']:.4f} >= 0.15")


if __name__ == '__main__':
    main()
