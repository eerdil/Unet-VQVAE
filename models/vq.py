"""
Vector quantizer with EMA codebook updates.
Shared codebook used across all U-Net levels.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizerEMA(nn.Module):
    """
    Vector quantizer with exponential moving average (EMA) codebook updates.
    More stable than straight-through gradient on the codebook itself.

    Args:
        vocab_size: number of codebook entries V
        d_embed: embedding dimension
        beta: commitment loss weight (default 0.25)
        decay: EMA decay rate (default 0.99)
        eps: small value to avoid division by zero in EMA
        restart_threshold: ema_cluster_size below this triggers dead-code restart.
            Set to 0 to disable. Only active after 500 warm-up steps.
    """
    def __init__(
        self,
        vocab_size: int = 4096,
        d_embed: int = 256,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
        restart_threshold: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.beta = beta
        self.decay = decay
        self.eps = eps
        self.restart_threshold = restart_threshold

        # Codebook embeddings - initialized randomly
        embed = torch.randn(vocab_size, d_embed)
        self.register_buffer('embedding', embed)
        self.register_buffer('ema_cluster_size', torch.zeros(vocab_size))
        self.register_buffer('ema_embed_sum', embed.clone())
        self.register_buffer('_step', torch.zeros(1, dtype=torch.long))

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h: projected encoder features (B, d_embed, H, W)
        Returns:
            z_q: quantized features with straight-through (B, d_embed, H, W)
            indices: codebook indices (B, H, W)
            commit_loss: commitment loss scalar
        """
        B, C, H, W = h.shape
        assert C == self.d_embed

        # Force float32 — distances overflow in fp16 with d_embed=256
        h = h.float()

        # Flatten spatial dims: (B*H*W, d_embed)
        h_flat = h.permute(0, 2, 3, 1).reshape(-1, C)

        # Find nearest codebook entry
        # d(h, e)^2 = ||h||^2 + ||e||^2 - 2 h.e
        dist = (
            h_flat.pow(2).sum(1, keepdim=True)
            + self.embedding.pow(2).sum(1)
            - 2 * h_flat @ self.embedding.t()
        )  # (B*H*W, V)
        indices_flat = dist.argmin(dim=1)  # (B*H*W,)
        indices = indices_flat.view(B, H, W)

        # Quantized embeddings
        z_q_flat = self.embedding[indices_flat]  # (B*H*W, d_embed)
        z_q = z_q_flat.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, d_embed, H, W)

        # EMA codebook update (only during training)
        if self.training:
            with torch.no_grad():
                # Use scatter_add to avoid O(N*V) one-hot allocation (V=4096, N can be 500K+)
                cluster_size = torch.zeros(self.vocab_size, device=h_flat.device, dtype=torch.float32)
                cluster_size.scatter_add_(0, indices_flat, torch.ones(indices_flat.shape[0], device=h_flat.device, dtype=torch.float32))

                embed_sum = torch.zeros(self.vocab_size, self.d_embed, device=h_flat.device, dtype=torch.float32)
                embed_sum.scatter_add_(0, indices_flat.unsqueeze(1).expand(-1, self.d_embed), h_flat)

                # Sync across DDP if needed
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(cluster_size)
                    torch.distributed.all_reduce(embed_sum)

                self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)
                self.ema_embed_sum.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # Normalize
                n = self.ema_cluster_size.sum()
                cluster_size_adj = (
                    (self.ema_cluster_size + self.eps)
                    / (n + self.vocab_size * self.eps) * n
                )
                self.embedding.copy_(self.ema_embed_sum / cluster_size_adj.unsqueeze(1))

                # Dead-code restart: reinitialize unused entries from current batch
                self._step += 1
                if self.restart_threshold > 0 and self._step.item() > 500:
                    self._restart_dead_codes(h_flat)

        # Commitment loss: encoder output tries to stay close to quantized
        commit_loss = F.mse_loss(h, z_q.detach())

        # Straight-through estimator: gradients flow back to encoder
        # Cast back to original dtype so downstream layers stay in AMP dtype
        orig_dtype = h.dtype  # float32 after our cast above; z_q_st kept in float32 is fine
        z_q_st = h + (z_q - h).detach()

        return z_q_st, indices, commit_loss

    def _restart_dead_codes(self, h_flat: torch.Tensor) -> None:
        """Replace dead codebook entries with random encoder outputs from the current batch."""
        dead = self.ema_cluster_size < self.restart_threshold  # (V,)
        n_dead = int(dead.sum().item())
        if n_dead == 0:
            return

        # Sample n_dead vectors from current batch; clamp to available batch size
        n_available = h_flat.shape[0]
        n_replace = min(n_dead, n_available)
        perm = torch.randperm(n_available, device=h_flat.device)[:n_replace]
        replacements = h_flat[perm].detach()  # (n_replace, d_embed)

        # Broadcast from rank 0 so all DDP workers use the same replacements
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.broadcast(replacements, src=0)

        dead_indices = dead.nonzero(as_tuple=False).squeeze(1)[:n_replace]
        self.embedding[dead_indices] = replacements
        # Reset EMA stats so the restarted codes start fresh but aren't immediately killed again
        self.ema_cluster_size[dead_indices] = self.restart_threshold
        self.ema_embed_sum[dead_indices] = replacements * self.restart_threshold

    def lookup(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Look up embeddings from indices.
        Args:
            indices: (B, H, W) or (B, L) integer tensor
        Returns:
            embeddings in d_embed dimension at last position
        """
        return self.embedding[indices]

    @torch.no_grad()
    def get_codebook_usage(self) -> float:
        """Returns fraction of codebook entries used (rough estimate)."""
        return (self.ema_cluster_size > 1.0).float().mean().item()
