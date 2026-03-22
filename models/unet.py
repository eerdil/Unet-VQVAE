"""
U-Net encoder and decoder with skip connections at each level.
K=4 levels, channel sizes [64, 128, 256, 512] from shallow to deep.
Input: 256x256 images -> encoder levels at 128, 64, 32, 16 spatial resolutions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConvBlock(nn.Module):
    """Conv 3x3 -> BN -> ReLU -> Conv 3x3 -> BN -> ReLU"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetEncoder(nn.Module):
    """
    4-level U-Net encoder.
    Input: (B, 3, 256, 256)
    Outputs feature maps at resolutions: 128, 64, 32, 16
    Channel sizes: [64, 128, 256, 512]
    Level indexing: level 1 = shallowest (128x128), level 4 = deepest (16x16)
    Returned as list [f1, f2, f3, f4] where f1 is shallowest.
    """
    def __init__(self, in_channels: int = 3, base_ch: int = 64):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]  # [64, 128, 256, 512]
        self.channel_sizes = ch

        # initial conv to bring 256x256 -> 128x128 features
        self.init_conv = ConvBlock(in_channels, ch[0])
        self.pool0 = nn.MaxPool2d(2)  # 256 -> 128 (this gives level 1)

        self.enc1 = ConvBlock(ch[0], ch[1])
        self.pool1 = nn.MaxPool2d(2)  # 128 -> 64

        self.enc2 = ConvBlock(ch[1], ch[2])
        self.pool2 = nn.MaxPool2d(2)  # 64 -> 32

        self.enc3 = ConvBlock(ch[2], ch[3])
        self.pool3 = nn.MaxPool2d(2)  # 32 -> 16 (bottleneck / level 4)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns [f1, f2, f3, f4] at resolutions [128, 64, 32, 16].
        f1 has ch[0] channels, f4 has ch[3] channels.
        """
        x0 = self.init_conv(x)   # (B, 64, 256, 256)
        f1 = self.pool0(x0)       # (B, 64, 128, 128)  -- level 1

        f2 = self.enc1(f1)        # (B, 128, 128, 128)
        f2 = self.pool1(f2)       # (B, 128, 64, 64)   -- level 2

        f3 = self.enc2(f2)        # (B, 256, 64, 64)
        f3 = self.pool2(f3)       # (B, 256, 32, 32)   -- level 3

        f4 = self.enc3(f3)        # (B, 512, 32, 32)
        f4 = self.pool3(f4)       # (B, 512, 16, 16)   -- level 4 (deepest)

        return [f1, f2, f3, f4]


class UNetDecoder(nn.Module):
    """
    4-level U-Net decoder receiving quantized skip connections.
    Reconstructs 256x256 image from skip connections at 4 levels.
    """
    def __init__(self, d_embed: int, base_ch: int = 64, out_channels: int = 3):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8]  # [64, 128, 256, 512]
        self.d_embed = d_embed

        # Project d_embed back to each level's channel count before decoding
        # skip connections come in as d_embed, need to be projected to ch[k]
        self.skip_proj4 = nn.Conv2d(d_embed, ch[3], 1)  # 16x16
        self.skip_proj3 = nn.Conv2d(d_embed, ch[2], 1)  # 32x32
        self.skip_proj2 = nn.Conv2d(d_embed, ch[1], 1)  # 64x64
        self.skip_proj1 = nn.Conv2d(d_embed, ch[0], 1)  # 128x128

        # Decoder blocks: upsample + conv (skip connection concatenated before conv)
        # Level 4 -> 3: (ch3 after up) + ch3 (skip) -> ch2
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch[3], ch[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[2]), nn.ReLU(inplace=True),
        )
        self.dec4 = ConvBlock(ch[2] + ch[2], ch[2])

        # Level 3 -> 2
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch[2], ch[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[1]), nn.ReLU(inplace=True),
        )
        self.dec3 = ConvBlock(ch[1] + ch[1], ch[1])

        # Level 2 -> 1
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch[1], ch[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]), nn.ReLU(inplace=True),
        )
        self.dec2 = ConvBlock(ch[0] + ch[0], ch[0])

        # Final upsample to 256x256
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch[0], ch[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(ch[0]), nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(ch[0], out_channels, 1)

    def forward(self, z_skips: List[torch.Tensor]) -> torch.Tensor:
        """
        z_skips: [z1, z2, z3, z4] where z_k has shape (B, d_embed, H_k, W_k)
        z1: (B, d_embed, 128, 128), z4: (B, d_embed, 16, 16)
        """
        z1, z2, z3, z4 = z_skips

        # Project skips to channel sizes
        s4 = self.skip_proj4(z4)  # (B, 512, 16, 16)
        s3 = self.skip_proj3(z3)  # (B, 256, 32, 32)
        s2 = self.skip_proj2(z2)  # (B, 128, 64, 64)
        s1 = self.skip_proj1(z1)  # (B, 64, 128, 128)

        x = self.up4(s4)                          # (B, 256, 32, 32)
        x = self.dec4(torch.cat([x, s3], dim=1))  # (B, 256, 32, 32)

        x = self.up3(x)                           # (B, 128, 64, 64)
        x = self.dec3(torch.cat([x, s2], dim=1))  # (B, 128, 64, 64)

        x = self.up2(x)                           # (B, 64, 128, 128)
        x = self.dec2(torch.cat([x, s1], dim=1))  # (B, 64, 128, 128)

        x = self.up1(x)                           # (B, 64, 256, 256)
        return self.final_conv(x)                 # (B, 3, 256, 256)
