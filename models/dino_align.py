"""
DINO Alignment module for UNet-VAR Stage 1.
Aligns the deepest encoder level (bottleneck) with frozen DINOv3-S CLS token embeddings.

L_dino = || proj(pool(h_K)) - stopgrad(DINO_cls(im)) ||_2^2

where:
  h_K      : deepest projected features (B, d_embed, H_K, W_K) after P_K
  pool      : global average pooling -> (B, d_embed)
  proj      : learned linear (d_embed -> dino_dim) to match DINO's space
  DINO_cls  : CLS token from frozen DINOv3-S (B, 384)

Note: images fed to DINO must use ImageNet normalization (mean/std different from
the [-1,1] normalization used by the UNet). We handle that here.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

BMIC_DATA_PATH = "/usr/bmicnas02/data-biwi-01/fm_originalzoo/dinov3"

DINO_CKPT = {
    'dinov3_vits16':  os.path.join(BMIC_DATA_PATH, "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"),
    'dinov3_vitb16':  os.path.join(BMIC_DATA_PATH, "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"),
    'dinov3_vitl16':  os.path.join(BMIC_DATA_PATH, "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"),
}

DINO_DIM = {
    'dinov3_vits16': 384,
    'dinov3_vitb16': 768,
    'dinov3_vitl16': 1024,
}


class DINOAligner(nn.Module):
    """
    Wraps a frozen DINOv3 model and provides the DINO alignment loss.

    Args:
        d_embed: bottleneck embedding dimension from the UNet (default 256)
        model_name: which DINOv3 model to use (default 'dinov3_vits16')
    """
    def __init__(self, d_embed: int = 256, model_name: str = 'dinov3_vits16'):
        super().__init__()
        self.d_embed = d_embed
        self.model_name = model_name
        dino_dim = DINO_DIM[model_name]

        # Learned projection: maps pooled bottleneck features to DINO's CLS space
        self.proj = nn.Linear(d_embed, dino_dim)

        # Pixel mean/std for DINO's ImageNet normalization (images in [-1,1] need conversion)
        # Our images are in [-1, 1]; DINO expects [0,1] normalized with ImageNet stats
        self.register_buffer('dino_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('dino_std',  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # Load frozen DINO model
        dino_repo = os.path.join(BMIC_DATA_PATH, "dinov3")
        sys.path.insert(0, dino_repo)
        self.dino = torch.hub.load(
            dino_repo, model_name, source='local',
            weights=DINO_CKPT[model_name],
        )
        # Freeze all DINO parameters
        for p in self.dino.parameters():
            p.requires_grad_(False)
        self.dino.eval()

    def _preprocess_for_dino(self, x: torch.Tensor) -> torch.Tensor:
        """Convert from [-1,1] to DINO's ImageNet normalization."""
        x = (x + 1.0) / 2.0  # [-1,1] -> [0,1]
        x = (x - self.dino_mean) / self.dino_std
        return x

    @torch.no_grad()
    def get_dino_cls(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frozen CLS token from DINOv3. Returns (B, dino_dim)."""
        x_dino = self._preprocess_for_dino(x)
        # forward_features returns a dict with 'x_norm_clstoken'
        feats = self.dino.forward_features(x_dino)
        return feats['x_norm_clstoken']  # (B, dino_dim)

    def forward(self, h_K: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute DINO alignment loss.

        Args:
            h_K: projected bottleneck features (B, d_embed, H_K, W_K)
            x:   original input images in [-1, 1] (B, 3, H, W)
        Returns:
            dino_loss: scalar L2 alignment loss
        """
        # Pool bottleneck to (B, d_embed) and project to DINO's space
        pooled = h_K.mean(dim=(2, 3))        # (B, d_embed)
        pooled_proj = self.proj(pooled)       # (B, dino_dim)

        # Get frozen DINO CLS token (no grad)
        dino_cls = self.get_dino_cls(x.float())  # (B, dino_dim)

        return F.mse_loss(pooled_proj, dino_cls)
