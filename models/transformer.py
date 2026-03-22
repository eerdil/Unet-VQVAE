"""
Stage 2: VAR-style autoregressive transformer over U-Net multi-scale tokens.
Coarse-to-fine generation: level K (deepest, 16x16) first, level 1 (128x128) last.
Block-wise causal attention: tokens at level k attend to all coarser levels.
"""
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint


# ---- Attention helpers ----

class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(embed_dim))
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

        # kv caching for inference
        self.caching = False
        self.cached_k = None
        self.cached_v = None

    def kv_caching(self, enable: bool):
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    def forward(self, x: torch.Tensor, attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, C = x.shape
        zero_k_bias = torch.zeros(C, device=x.device, dtype=x.dtype)
        qkv = F.linear(x, self.qkv.weight,
                        torch.cat([self.q_bias, zero_k_bias, self.v_bias]))
        qkv = qkv.view(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each: (B, H, L, head_dim)

        if self.caching:
            if self.cached_k is None:
                self.cached_k = k
                self.cached_v = v
            else:
                k = self.cached_k = torch.cat([self.cached_k, k], dim=2)
                v = self.cached_v = torch.cat([self.cached_v, v], dim=2)

        # Use scaled_dot_product_attention if available (PyTorch >= 2.0)
        dropout_p = self.attn_drop if self.training else 0.0
        try:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias, dropout_p=dropout_p)
        except Exception:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attn_bias is not None:
                attn = attn + attn_bias
            attn = F.dropout(attn.softmax(dim=-1), p=dropout_p, training=self.training)
            out = attn @ v

        out = out.transpose(1, 2).reshape(B, L, C)
        return self.proj_drop(self.proj(out))


class FFN(nn.Module):
    def __init__(self, embed_dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(embed_dim * mlp_ratio)
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class AdaLNSelfAttnBlock(nn.Module):
    """Transformer block with AdaLN conditioning on class embedding."""
    def __init__(self, embed_dim: int, cond_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0,
                 drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(embed_dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.ffn = FFN(embed_dim, mlp_ratio=mlp_ratio, drop=drop)

        # AdaLN: predict scale1, shift1, gate1, scale2, shift2, gate2 from cond
        self.ada_lin = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * embed_dim))

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                attn_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # cond: (B, cond_dim)
        ada = self.ada_lin(cond).view(x.shape[0], 1, 6, x.shape[-1])
        scale1, shift1, gate1, scale2, shift2, gate2 = ada.unbind(2)

        # Attention
        x_norm = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * self.drop_path(self.attn(x_norm, attn_bias=attn_bias))

        # FFN
        x_norm = self.norm2(x) * (1 + scale2) + shift2
        x = x + gate2 * self.drop_path(self.ffn(x_norm))
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device)).div_(keep)
        return x * mask


# ---- 2D sinusoidal positional embedding ----

def make_2d_sincos_pos_embed(d_model: int, h: int, w: int) -> torch.Tensor:
    """Returns (h*w, d_model) sinusoidal 2D positional embeddings."""
    assert d_model % 4 == 0
    d = d_model // 4
    # row and col embeddings each have d_model//2 dims
    ys = torch.arange(h, dtype=torch.float32)
    xs = torch.arange(w, dtype=torch.float32)
    freq = torch.exp(-math.log(10000) * torch.arange(0, d, dtype=torch.float32) / d)
    ey = torch.outer(ys, freq)  # (h, d)
    ex = torch.outer(xs, freq)  # (w, d)
    # sin/cos for each
    ey = torch.cat([ey.sin(), ey.cos()], dim=-1)  # (h, d_model//2)
    ex = torch.cat([ex.sin(), ex.cos()], dim=-1)  # (w, d_model//2)
    # broadcast to (h, w, d_model)
    ey = ey.unsqueeze(1).expand(h, w, -1)
    ex = ex.unsqueeze(0).expand(h, w, -1)
    return torch.cat([ey, ex], dim=-1).reshape(h * w, d_model)


# ---- Main transformer ----

class UNetVAR(nn.Module):
    """
    VAR-style autoregressive transformer for UNet-VAR Stage 2.

    Generation order: coarse-to-fine (level K first, level 1 last).
    For K=4, U-Net level indexing (1=128x128, 4=16x16), AR order is [4,3,2,1].
    Sequence: [class_token], flatten(r_4), flatten(r_3), flatten(r_2), flatten(r_1)

    Args:
        vocab_size: codebook size V
        d_embed: codebook embedding dimension (from VQ)
        num_classes: number of ImageNet classes
        depth: transformer depth
        embed_dim: transformer model dimension d_model
        num_heads: attention heads
        level_sizes: spatial sizes (h,w) per level in AR order (coarse first)
                     e.g. [(16,16), (32,32), (64,64), (128,128)] for K=4
    """
    def __init__(
        self,
        vocab_size: int = 4096,
        d_embed: int = 256,
        num_classes: int = 1000,
        depth: int = 16,
        embed_dim: int = 1024,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        cond_drop_rate: float = 0.1,
        level_sizes: Optional[List[Tuple[int, int]]] = None,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if level_sizes is None:
            # AR order: deepest first
            level_sizes = [(16, 16), (32, 32), (64, 64), (128, 128)]

        self.vocab_size = vocab_size
        self.d_embed = d_embed
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.cond_drop_rate = cond_drop_rate
        self.K = len(level_sizes)
        self.level_sizes = level_sizes  # coarse to fine
        self.use_checkpoint = use_checkpoint

        # Token counts per level
        self.level_lens = [h * w for h, w in level_sizes]
        self.L = sum(self.level_lens)

        # begin/end positions for each level in the full sequence (after class token)
        self.begin_ends = []
        cur = 0
        for l in self.level_lens:
            self.begin_ends.append((cur, cur + l))
            cur += l

        # 1. Word embedding: token index -> d_model
        self.word_embed = nn.Linear(d_embed, embed_dim)

        # 2. Class embedding (+ unconditional token for CFG)
        self.class_emb = nn.Embedding(num_classes + 1, embed_dim)

        # 3. Level embedding: one per level
        self.level_emb = nn.Embedding(self.K, embed_dim)

        # 4. 2D sinusoidal positional embeddings per level (fixed, registered as buffer)
        pos_list = []
        for h, w in level_sizes:
            pos_list.append(make_2d_sincos_pos_embed(embed_dim, h, w))
        pos_all = torch.cat(pos_list, dim=0)  # (L, embed_dim)
        self.register_buffer('pos_embed', pos_all.unsqueeze(0))  # (1, L, embed_dim)

        # Level index for each token position (for level_emb lookup)
        lvl_idx = torch.cat([
            torch.full((l,), i, dtype=torch.long)
            for i, l in enumerate(self.level_lens)
        ])  # (L,)
        self.register_buffer('lvl_idx', lvl_idx.unsqueeze(0))  # (1, L)

        # 5. Block-wise causal attention mask
        # Tokens in level k can attend to all tokens in coarser levels (earlier in sequence),
        # but NOT to same or finer levels — except within the same level where it's fully masked
        # This is equivalent to: token i can attend to token j iff lvl[j] <= lvl[i]  (coarser first)
        # We use 0 for allowed, -inf for masked
        d = lvl_idx.view(self.L, 1)  # (L, 1) level index per token
        dT = d.t()                    # (1, L)
        # token i attends to token j if lvl[j] < lvl[i] (strictly coarser)
        # within same level: not attended (causal per level = attend to nothing in same level during training)
        # This matches VAR block-wise causal: d >= dT means same or coarser level
        mask = torch.where(d > dT, torch.zeros(1), torch.full((1,), float('-inf')))
        # Allow attending to strictly coarser: d > dT => 0 (visible), else -inf
        self.register_buffer('attn_bias', mask.unsqueeze(0).unsqueeze(0))  # (1, 1, L, L)

        # 6. Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            AdaLNSelfAttnBlock(
                embed_dim=embed_dim,
                cond_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        # 7. Output head
        self.head_norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        self.head_ada = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, 2 * embed_dim))
        self.head = nn.Linear(embed_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        std = (1 / self.embed_dim / 3) ** 0.5
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=std)
        # Scale output head down
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _get_logits(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.head_ada(cond).view(h.shape[0], 1, 2, self.embed_dim).unbind(2)
        h = self.head_norm(h) * (1 + scale) + shift
        return self.head(h)

    def forward(
        self,
        label_B: torch.Tensor,            # (B,) class labels
        token_embeds: torch.Tensor,        # (B, L, d_embed) teacher-forcing input (all levels)
    ) -> torch.Tensor:
        """
        Training forward pass with teacher forcing.
        token_embeds: continuous embeddings of ground truth tokens for all levels,
                      shape (B, L, d_embed).
        Returns logits (B, L, V).
        """
        B = label_B.shape[0]

        # Classifier-free guidance: randomly drop class conditioning
        drop_mask = torch.rand(B, device=label_B.device) < self.cond_drop_rate
        label_B = torch.where(drop_mask, torch.full_like(label_B, self.num_classes), label_B)
        cond = self.class_emb(label_B)  # (B, embed_dim)

        # Build input sequence: class embedding expands to first level, then word embeddings
        # Input at position t predicts token at position t (using coarser context)
        # For level k tokens: input = word_embed(z_{k's tokens}) + level_emb + pos_emb
        # First level tokens get class_emb as their "input" (shifted by 1 like GPT)
        # We implement: input_BLC[level k positions] = word_embed(token_embeds for level k)
        # The first level's input is the class token (sos)
        x = self.word_embed(token_embeds)  # (B, L, embed_dim)

        # Add positional and level embeddings
        x = x + self.pos_embed + self.level_emb(self.lvl_idx.expand(B, -1))

        # Prepend class token as context for the first level
        # Actually: in training we use teacher-forcing where level k's input comes from
        # embedding of levels K,...,k+1. The "sos" class token serves as input for level K.
        # We implement this by replacing level K's input with the class embedding.
        first_l = self.level_lens[0]
        sos = cond.unsqueeze(1).expand(B, first_l, -1)  # (B, first_l, embed_dim)
        # The tokens for the deepest level use class embedding as input
        # Remaining: teacher-forced from coarser levels
        # Reconstruct: x[:, :first_l] = sos (override), x[:, first_l:] stays
        x = torch.cat([sos + self.pos_embed[:, :first_l] + self.level_emb(self.lvl_idx[:, :first_l].expand(B, -1)),
                        x[:, first_l:]], dim=1)

        attn_bias = self.attn_bias  # (1, 1, L, L)
        dtype = x.dtype
        attn_bias = attn_bias.to(dtype)

        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = grad_checkpoint(blk, x, cond, attn_bias, use_reentrant=False)
            else:
                x = blk(x, cond=cond, attn_bias=attn_bias)

        return self._get_logits(x, cond)  # (B, L, V)

    @torch.no_grad()
    def generate(
        self,
        B: int,
        label_B: Optional[torch.Tensor],
        quantizer,           # VectorQuantizerEMA with .lookup() and .embedding
        cfg: float = 1.5,
        top_k: int = 2000,
        top_p: float = 0.95,
        temperature: float = 1.0,
        device: torch.device = None,
    ) -> List[torch.Tensor]:
        """
        Autoregressive generation, coarse to fine.
        Returns list of index tensors [r_K, ..., r_1] in AR order (coarse first).
        """
        if device is None:
            device = next(self.parameters()).device

        if label_B is None:
            label_B = torch.randint(0, self.num_classes, (B,), device=device)

        # Duplicate for CFG
        label_uncond = torch.full_like(label_B, self.num_classes)
        labels = torch.cat([label_B, label_uncond], dim=0)  # (2B,)
        cond = self.class_emb(labels)  # (2B, embed_dim)

        # Enable kv caching
        for blk in self.blocks:
            blk.attn.kv_caching(True)

        generated_indices = []
        seq_embeds = []  # accumulated token embeddings for the sequence so far

        for level_idx, (h, w) in enumerate(self.level_sizes):
            l = h * w
            first_l = self.level_lens[0]

            if level_idx == 0:
                # First level: input is class embedding
                pos = self.pos_embed[:, :first_l]  # (1, first_l, embed_dim)
                lvl = self.level_emb(self.lvl_idx[:, :first_l])
                x = cond.unsqueeze(1).expand(2 * B, first_l, -1) + pos + lvl
            else:
                # Subsequent levels: input is word_embed of previous level's quantized embeddings
                bg, ed = self.begin_ends[level_idx]
                pos = self.pos_embed[:, bg:ed]
                lvl = self.level_emb(self.lvl_idx[:, bg:ed])
                prev_embed = seq_embeds[-1]  # (B, prev_l, d_embed)
                prev_embed_2B = prev_embed.repeat(2, 1, 1)  # (2B,...)
                x = self.word_embed(prev_embed_2B) + pos + lvl

            # Forward through transformer blocks (kv-cached)
            for blk in self.blocks:
                x = blk(x, cond=cond, attn_bias=None)

            # Get logits for this level
            logits_2BV = self._get_logits(x, cond)  # (2B, l, V)
            # CFG
            logits_cond = logits_2BV[:B]
            logits_uncond = logits_2BV[B:]
            logits = logits_uncond + cfg * (logits_cond - logits_uncond)  # (B, l, V)

            # Sample
            logits = logits / temperature
            if top_k > 0:
                vals, _ = logits.topk(min(top_k, logits.shape[-1]), dim=-1)
                logits = logits.masked_fill(logits < vals[..., -1:], float('-inf'))
            if top_p > 0:
                probs = logits.softmax(dim=-1)
                sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                cumsum = sorted_probs.cumsum(dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0
                probs = probs.scatter_(-1, sorted_idx, sorted_probs)
                idx_Bl = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(B, l)
            else:
                idx_Bl = logits.argmax(dim=-1)

            generated_indices.append(idx_Bl.view(B, h, w))

            # Prepare embedding for next level input
            emb = quantizer.lookup(idx_Bl.view(B, h, w))  # (B, h, w, d_embed)
            seq_embeds.append(emb.view(B, l, self.d_embed))

        for blk in self.blocks:
            blk.attn.kv_caching(False)

        return generated_indices  # list of (B, H_k, W_k) in coarse-to-fine order
