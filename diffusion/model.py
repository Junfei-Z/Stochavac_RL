"""
Conditional Diffusion Model for Vaccine Allocation

Architecture: Transformer-based epsilon-prediction network with conditional inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding for time steps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time: (B,) tensor of diffusion time steps

        Returns:
            (B, dim) tensor of time embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ConditionalTransformer(nn.Module):
    """
    Transformer-based denoising network with conditional state inputs.
    """

    def __init__(
        self,
        N: int = 2000,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        state_dim: int = 10,  # Dimension of state features per individual
    ):
        super().__init__()

        self.N = N
        self.d_model = d_model

        # Time embedding
        self.time_embed = SinusoidalPositionEmbedding(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Action embedding (project noisy action to d_model)
        self.action_embed = nn.Linear(1, d_model)

        # State embedding (encode node features)
        self.state_embed = nn.Linear(state_dim, d_model)

        # Positional encoding for nodes
        self.pos_embed = nn.Parameter(torch.randn(1, N, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection (predict noise)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoise the action given current state.

        Args:
            x: (B, N) tensor of noisy actions
            t: (B,) tensor of diffusion time steps [0, T]
            state: (B, N, state_dim) tensor of node state features

        Returns:
            (B, N) tensor of predicted noise
        """
        B, N = x.shape

        # Time embedding
        t_emb = self.time_embed(t)  # (B, d_model)
        t_emb = self.time_mlp(t_emb)  # (B, d_model)

        # Action embedding
        x_emb = self.action_embed(x.unsqueeze(-1))  # (B, N, d_model)

        # State embedding
        s_emb = self.state_embed(state)  # (B, N, d_model)

        # Combine embeddings
        h = x_emb + s_emb + self.pos_embed[:, :N, :]  # (B, N, d_model)

        # Add time embedding (broadcast)
        h = h + t_emb[:, None, :]  # (B, N, d_model)

        # Transformer
        h = self.transformer(h)  # (B, N, d_model)

        # Output
        noise_pred = self.output_proj(h).squeeze(-1)  # (B, N)

        return noise_pred


class DiffusionModel(nn.Module):
    """
    Diffusion model wrapper with forward/reverse diffusion process.
    """

    def __init__(
        self,
        denoiser: nn.Module,
        T: int = 1000,
        beta_schedule: str = 'linear',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()

        self.denoiser = denoiser
        self.T = T

        # Beta schedule
        if beta_schedule == 'linear':
            self.beta = torch.linspace(beta_start, beta_end, T)
        elif beta_schedule == 'cosine':
            s = 0.008
            steps = T + 1
            t = torch.linspace(0, T, steps)
            alphas_cumprod = torch.cos(((t / T) + s) / (1 + s) * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.beta = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Precompute alpha values
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x0, (1 - alpha_bar_t) * I)

        Args:
            x0: (B, N) clean actions
            t: (B,) time steps
            noise: Optional (B, N) noise, sampled if None

        Returns:
            (B, N) noisy actions at time t
        """
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar_t = self.alpha_bar[t][:, None]  # (B, 1)

        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise

        return xt

    def p_losses(
        self,
        x0: torch.Tensor,
        state: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute denoising loss.

        Args:
            x0: (B, N) clean actions
            state: (B, N, state_dim) state features
            t: Optional (B,) time steps, sampled if None
            noise: Optional (B, N) noise, sampled if None

        Returns:
            Scalar loss
        """
        B, N = x0.shape

        # Sample time steps
        if t is None:
            t = torch.randint(0, self.T, (B,), device=x0.device).long()

        # Sample noise
        if noise is None:
            noise = torch.randn_like(x0)

        # Forward diffusion
        xt = self.q_sample(x0, t, noise)

        # Predict noise
        noise_pred = self.denoiser(xt, t, state)

        # MSE loss
        loss = F.mse_loss(noise_pred, noise)

        return loss

    @torch.no_grad()
    def p_sample(
        self,
        xt: torch.Tensor,
        t: int,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reverse diffusion single step: p(x_{t-1} | x_t)

        Args:
            xt: (B, N) noisy actions at time t
            t: Time step (scalar)
            state: (B, N, state_dim) state features

        Returns:
            (B, N) actions at time t-1
        """
        B, N = xt.shape
        device = xt.device

        # Predict noise
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = self.denoiser(xt, t_tensor, state)

        # Get alpha values
        alpha_t = self.alpha[t].to(device)
        alpha_bar_t = self.alpha_bar[t].to(device)

        # Compute mean
        if t > 0:
            alpha_bar_t_prev = self.alpha_bar[t - 1].to(device)
        else:
            alpha_bar_t_prev = torch.tensor(1.0, device=device)

        # DDPM formula
        x0_pred = (xt - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)

        # Posterior mean
        coef1 = torch.sqrt(alpha_bar_t_prev) * self.beta[t].to(device) / (1 - alpha_bar_t)
        coef2 = torch.sqrt(alpha_t) * (1 - alpha_bar_t_prev) / (1 - alpha_bar_t)
        mu = coef1 * x0_pred + coef2 * xt

        # Add noise (except at t=0)
        if t > 0:
            noise = torch.randn_like(xt)
            sigma_t = torch.sqrt(self.beta[t].to(device))
            return mu + sigma_t * noise
        else:
            return mu

    @torch.no_grad()
    def sample(
        self,
        state: torch.Tensor,
        shape: Tuple[int, int],
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples via reverse diffusion.

        Args:
            state: (B, N, state_dim) state features
            shape: (B, N) shape of actions to generate
            return_trajectory: If True, return full trajectory

        Returns:
            (B, N) generated actions (or list of actions if return_trajectory)
        """
        B, N = shape
        device = state.device

        # Start from pure noise
        xt = torch.randn(B, N, device=device)

        trajectory = [xt] if return_trajectory else None

        # Reverse diffusion
        for t in reversed(range(self.T)):
            xt = self.p_sample(xt, t, state)

            if return_trajectory:
                trajectory.append(xt)

        if return_trajectory:
            return trajectory
        else:
            return xt


def create_state_features(obs: dict, device: str = 'cpu') -> torch.Tensor:
    """
    Create state feature tensor from observation dictionary.

    Args:
        obs: Dictionary containing:
            - node_states: (B, N) or (N,)
            - vaccinated: (B, N) or (N,)
            - group_id: (B, N) or (N,)
            - time_step: (B, 1) or (1,)
            - supply_today: (B, 1) or (1,)

    Returns:
        (B, N, state_dim) tensor of state features
    """
    # Handle both batched and unbatched observations
    node_states = torch.tensor(obs['node_states'], dtype=torch.float32, device=device)
    vaccinated = torch.tensor(obs['vaccinated'], dtype=torch.float32, device=device)
    group_id = torch.tensor(obs['group_id'], dtype=torch.float32, device=device)

    if node_states.ndim == 1:
        # Unbatched, add batch dimension
        node_states = node_states.unsqueeze(0)
        vaccinated = vaccinated.unsqueeze(0)
        group_id = group_id.unsqueeze(0)

    B, N = node_states.shape

    # One-hot encode group_id (3 groups)
    group_onehot = F.one_hot(group_id.long(), num_classes=3).float()  # (B, N, 3)

    # One-hot encode node_states (6 states: S, E, I, R, V, D)
    state_onehot = F.one_hot(node_states.long(), num_classes=6).float()  # (B, N, 6)

    # Combine features
    features = torch.cat([
        state_onehot,  # 6 dims
        group_onehot,  # 3 dims
        vaccinated.unsqueeze(-1),  # 1 dim
    ], dim=-1)  # (B, N, 10)

    return features
