"""
Proximal Policy Optimization (PPO) with Prior-Guided KL Regularization

This implementation includes:
- Standard PPO with clipped objective
- KL divergence regularization to diffusion prior
- Advantage estimation with GAE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    """
    Policy network that outputs mean and std for continuous actions.
    Compatible with diffusion model architecture for initialization.
    """

    def __init__(
        self,
        N: int = 2000,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        state_dim: int = 10,
    ):
        super().__init__()

        self.N = N
        self.d_model = d_model

        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, N, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output heads
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),  # Actions in [0, 1]
        )

        self.log_std_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

        # Initialize log_std to small values
        self.log_std_head[-1].weight.data.mul_(0.01)
        self.log_std_head[-1].bias.data.fill_(-2.0)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            state: (B, N, state_dim) state features

        Returns:
            mean: (B, N) action means
            std: (B, N) action standard deviations
        """
        B, N, _ = state.shape

        # Embed state
        h = self.state_embed(state)  # (B, N, d_model)

        # Add positional encoding
        h = h + self.pos_embed[:, :N, :]

        # Transformer
        h = self.transformer(h)  # (B, N, d_model)

        # Output
        mean = self.mean_head(h).squeeze(-1)  # (B, N)
        log_std = self.log_std_head(h).squeeze(-1)  # (B, N)
        std = torch.exp(log_std).clamp(min=1e-3, max=1.0)

        return mean, std

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.

        Args:
            state: (B, N, state_dim) state features
            deterministic: If True, return mean action

        Returns:
            action: (B, N) sampled actions
            log_prob: (B, N) log probabilities
            entropy: (B,) entropy
        """
        mean, std = self.forward(state)

        if deterministic:
            action = mean
            log_prob = torch.zeros_like(action)
            entropy = torch.zeros(action.shape[0], device=action.device)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy().sum(dim=-1)  # (B,)

        # Clip to [0, 1]
        action = torch.clamp(action, 0, 1)

        return action, log_prob, entropy

    def evaluate_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probability and entropy of given action.

        Args:
            state: (B, N, state_dim) state features
            action: (B, N) actions

        Returns:
            log_prob: (B, N) log probabilities
            entropy: (B,) entropy
        """
        mean, std = self.forward(state)

        dist = Normal(mean, std)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy().sum(dim=-1)

        return log_prob, entropy


class ValueNetwork(nn.Module):
    """Value network for critic."""

    def __init__(
        self,
        N: int = 2000,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        state_dim: int = 10,
    ):
        super().__init__()

        self.N = N
        self.d_model = d_model

        # State embedding
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, N, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global pooling + value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, N, state_dim) state features

        Returns:
            value: (B,) state values
        """
        B, N, _ = state.shape

        # Embed state
        h = self.state_embed(state)  # (B, N, d_model)

        # Add positional encoding
        h = h + self.pos_embed[:, :N, :]

        # Transformer
        h = self.transformer(h)  # (B, N, d_model)

        # Global average pooling
        h_pooled = h.mean(dim=1)  # (B, d_model)

        # Value
        value = self.value_head(h_pooled).squeeze(-1)  # (B,)

        return value


class PPO:
    """
    PPO agent with prior-guided KL regularization.
    """

    def __init__(
        self,
        N: int = 2000,
        state_dim: int = 10,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        kl_coef: float = 0.1,
        kl_decay: float = 0.999,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
    ):
        self.N = N
        self.state_dim = state_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.kl_coef = kl_coef
        self.kl_coef_init = kl_coef
        self.kl_decay = kl_decay
        self.max_grad_norm = max_grad_norm
        self.device = device

        # Networks
        self.policy = PolicyNetwork(N=N, state_dim=state_dim).to(device)
        self.value = ValueNetwork(N=N, state_dim=state_dim).to(device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )

        # Prior policy (will be set externally, e.g., from diffusion model)
        self.prior_policy = None

    def set_prior_policy(self, prior_policy):
        """Set prior policy for KL regularization."""
        self.prior_policy = prior_policy
        self.prior_policy.eval()

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            rewards: (T,) rewards
            values: (T+1,) state values
            dones: (T,) done flags

        Returns:
            advantages: (T,) advantages
            returns: (T,) returns
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0

        for t in reversed(range(T)):
            if t == T - 1:
                next_value = values[t + 1]
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values[:-1]

        return advantages, returns

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        n_epochs: int = 10,
        batch_size: int = 64,
    ) -> Dict:
        """
        Update policy and value networks.

        Args:
            states: (T, N, state_dim) states
            actions: (T, N) actions
            old_log_probs: (T, N) old log probabilities
            returns: (T,) returns
            advantages: (T,) advantages
            n_epochs: Number of update epochs
            batch_size: Mini-batch size

        Returns:
            Dictionary of training metrics
        """
        T = states.shape[0]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {
            'policy_loss': 0,
            'value_loss': 0,
            'entropy': 0,
            'kl_div': 0,
            'total_loss': 0,
        }

        n_updates = 0

        for _ in range(n_epochs):
            # Shuffle indices
            indices = torch.randperm(T)

            for start in range(0, T, batch_size):
                end = min(start + batch_size, T)
                idx = indices[start:end]

                # Mini-batch
                state_batch = states[idx]
                action_batch = actions[idx]
                old_log_prob_batch = old_log_probs[idx]
                return_batch = returns[idx]
                advantage_batch = advantages[idx]

                # Evaluate current policy
                log_prob_batch, entropy_batch = self.policy.evaluate_action(state_batch, action_batch)

                # Policy ratio
                ratio = torch.exp(log_prob_batch - old_log_prob_batch)  # (B, N)

                # Clipped surrogate objective
                advantage_batch_expanded = advantage_batch[:, None]  # (B, 1)
                surr1 = ratio * advantage_batch_expanded
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage_batch_expanded
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_pred = self.value(state_batch)
                value_loss = F.mse_loss(value_pred, return_batch)

                # Entropy bonus
                entropy_loss = -entropy_batch.mean()

                # KL divergence to prior (if available)
                kl_loss = torch.tensor(0.0, device=self.device)
                if self.prior_policy is not None and self.kl_coef > 1e-6:
                    with torch.no_grad():
                        prior_mean, prior_std = self.prior_policy(state_batch)

                    current_mean, current_std = self.policy(state_batch)

                    # KL divergence between two Gaussians
                    kl_div = torch.log(prior_std / current_std) + \
                             (current_std ** 2 + (current_mean - prior_mean) ** 2) / (2 * prior_std ** 2) - 0.5
                    kl_loss = kl_div.mean()

                # Total loss
                total_loss = (
                    policy_loss +
                    self.vf_coef * value_loss +
                    self.ent_coef * entropy_loss +
                    self.kl_coef * kl_loss
                )

                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                # Metrics
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += -entropy_loss.item()
                metrics['kl_div'] += kl_loss.item()
                metrics['total_loss'] += total_loss.item()
                n_updates += 1

        # Decay KL coefficient
        self.kl_coef *= self.kl_decay

        # Average metrics
        for key in metrics:
            metrics[key] /= max(n_updates, 1)

        metrics['kl_coef'] = self.kl_coef

        return metrics
