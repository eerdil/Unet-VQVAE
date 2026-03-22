"""
Stage 1: Multi-Scale VQ U-Net Autoencoder.
Encoder produces features at 4 levels, each quantized with a shared codebook.
Decoder receives quantized skip connections.
"""
import torch
import torch.nn as nn
from typing import List, Tuple

from .unet import UNetEncoder, UNetDecoder
from .vq import VectorQuantizerEMA


class UNetVQVAE(nn.Module):
    """
    UNet-VQVAE: U-Net autoencoder with multi-scale vector quantization.

    K=4 encoder levels at resolutions [128, 64, 32, 16] (for 256x256 input).
    Shared codebook with per-level linear projection.
    Decoder reconstructs from quantized skip connections.
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_ch: int = 64,
        vocab_size: int = 4096,
        d_embed: int = 256,
        beta: float = 0.25,
        ema_decay: float = 0.99,
        restart_threshold: float = 0.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.beta = beta
        self.K = 4  # number of levels

        self.encoder = UNetEncoder(in_channels=in_channels, base_ch=base_ch)
        ch = self.encoder.channel_sizes  # [64, 128, 256, 512]

        # Per-level projection: C_k -> d_embed
        self.proj_layers = nn.ModuleList([
            nn.Linear(c, d_embed) for c in ch
        ])

        # Shared codebook
        self.quantizer = VectorQuantizerEMA(
            vocab_size=vocab_size, d_embed=d_embed, beta=beta, decay=ema_decay,
            restart_threshold=restart_threshold,
        )

        self.decoder = UNetDecoder(d_embed=d_embed, base_ch=base_ch, out_channels=in_channels)

    def encode(self, x: torch.Tensor, return_bottleneck: bool = False):
        """
        Encode image to multi-scale quantized features.
        Returns:
            z_skips: list of quantized features [z1..z4] with straight-through
            indices: list of codebook indices [r1..r4], each (B, H_k, W_k)
            commit_loss: summed commitment loss across levels
            h_K (optional): deepest projected features (B, d_embed, H_K, W_K) if return_bottleneck=True
        """
        feats = self.encoder(x)  # [f1, f2, f3, f4]

        z_skips = []
        indices = []
        total_commit = torch.tensor(0.0, device=x.device)
        h_K = None

        for k, fk in enumerate(feats):
            # Project to d_embed: (B, C, H, W) -> (B, H, W, C) -> Linear -> (B, H, W, d_embed)
            hk = self.proj_layers[k](fk.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (B, d_embed, H, W)

            if return_bottleneck and k == self.K - 1:
                h_K = hk  # save deepest projected features before quantization

            z_q_st, idx, commit = self.quantizer(hk)
            z_skips.append(z_q_st)
            indices.append(idx)
            total_commit = total_commit + commit

        if return_bottleneck:
            return z_skips, indices, total_commit / self.K, h_K
        return z_skips, indices, total_commit / self.K

    def decode(self, z_skips: List[torch.Tensor]) -> torch.Tensor:
        return self.decoder(z_skips)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full forward pass for Stage 1 training.
        Returns:
            recon: reconstructed image (B, 3, H, W)
            loss: total loss = L_recon + beta * L_commit
        """
        z_skips, indices, commit_loss = self.encode(x)
        recon = self.decode(z_skips)

        recon_loss = ((x - recon) ** 2).mean()
        loss = recon_loss + self.beta * commit_loss

        return recon, loss, recon_loss, commit_loss

    @torch.no_grad()
    def img_to_indices(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode image and return per-level codebook indices.
        Returns list [r1, r2, r3, r4] each of shape (B, H_k, W_k).
        Order: shallow (level 1) to deep (level 4).
        """
        feats = self.encoder(x)
        indices = []
        for k, fk in enumerate(feats):
            hk = self.proj_layers[k](fk.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            _z, idx, _c = self.quantizer(hk)
            indices.append(idx)
        return indices

    @torch.no_grad()
    def indices_to_image(self, indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Decode from per-level codebook indices to image.
        indices: [r1, r2, r3, r4] each (B, H_k, W_k)
        """
        z_skips = []
        for idx in indices:
            # (B, H, W) -> (B, H, W, d_embed) -> (B, d_embed, H, W)
            z = self.quantizer.lookup(idx).permute(0, 3, 1, 2)
            z_skips.append(z)
        return self.decode(z_skips).clamp(-1, 1)
