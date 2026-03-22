"""
Stage 2 Training: VAR-style Autoregressive Transformer over UNet multi-scale tokens.
Requires a trained Stage 1 model. Freezes it and trains only the transformer.
Generation order: coarse-to-fine (level 4 = 16x16 first, level 1 = 128x128 last).
"""
import argparse
import os
import sys
import math
import time
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import wandb

sys.path.insert(0, str(Path(__file__).parent))
from models import UNetVQVAE, UNetVAR
from utils.data import build_imagenet_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='UNet-VAR Stage 2 Training')
    # Data
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=256)
    # Stage 1 model
    parser.add_argument('--stage1_ckpt', type=str, required=True, help='Path to Stage 1 checkpoint')
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--vocab_size', type=int, default=4096)
    parser.add_argument('--d_embed', type=int, default=256)
    # Transformer
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--cond_drop_rate', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=1000)
    # Training
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--label_smooth', type=float, default=0.1)
    # Logging / Saving
    parser.add_argument('--output_dir', type=str, default='./output/stage2')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--wandb_project', type=str, default='unetvqvae-stage2')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None)
    # DDP
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument('--num_workers', type=int, default=8)
    return parser.parse_args()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def setup_distributed(local_rank):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return True
    return False


def cosine_lr_with_warmup(optimizer, step, total_steps, warmup_steps, base_lr, min_lr=1e-6):
    if step < warmup_steps:
        lr = base_lr * (step + 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    return lr


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


@torch.no_grad()
def get_token_embeds_for_training(vae: UNetVQVAE, indices_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Given per-level indices in AR order (coarse first: [r4, r3, r2, r1]),
    build the teacher-forcing input embeddings for the transformer.

    For each level k in the sequence, the *input* to the transformer at that level's
    positions should be the embeddings of level k+1 (i.e. the next coarser level
    already generated). The deepest level uses class embedding (handled in transformer).

    Here we just return continuous embeddings for ALL levels in AR order:
    [embed(r4), embed(r3), embed(r2), embed(r1)] concatenated -> (B, L, d_embed).
    The transformer's forward() handles prepending the class token for level K.
    We strip the first level (handled inside transformer.forward) and return the rest
    as `token_embeds`.
    """
    embeds = []
    for idx in indices_list:
        B, H, W = idx.shape
        emb = vae.quantizer.lookup(idx)  # (B, H, W, d_embed)
        embeds.append(emb.reshape(B, H * W, -1))
    return torch.cat(embeds, dim=1)  # (B, L, d_embed)


def main():
    args = parse_args()
    ddp = setup_distributed(args.local_rank)
    device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.output_dir, exist_ok=True)

    if is_main_process():
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # ---- Load frozen Stage 1 model ----
    vae = UNetVQVAE(
        in_channels=3,
        base_ch=args.base_ch,
        vocab_size=args.vocab_size,
        d_embed=args.d_embed,
    ).to(device)
    ckpt = torch.load(args.stage1_ckpt, map_location='cpu')
    vae.load_state_dict(ckpt['model'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    if is_main_process():
        print(f'Loaded Stage 1 from {args.stage1_ckpt}')

    # ---- Data ----
    train_set, val_set = build_imagenet_dataset(args.data_path, args.image_size)
    train_sampler = DistributedSampler(train_set, shuffle=True) if ddp else None
    val_sampler = DistributedSampler(val_set, shuffle=False) if ddp else None

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, sampler=train_sampler,
        shuffle=(train_sampler is None), num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, sampler=val_sampler,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    # Level sizes in AR order: deepest (16x16) first, shallowest (128x128) last
    # For 256x256 input with K=4 UNet levels at [128, 64, 32, 16]
    level_sizes = [(16, 16), (32, 32), (64, 64), (128, 128)]

    # ---- Stage 2 Transformer ----
    transformer = UNetVAR(
        vocab_size=args.vocab_size,
        d_embed=args.d_embed,
        num_classes=args.num_classes,
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate,
        cond_drop_rate=args.cond_drop_rate,
        level_sizes=level_sizes,
    ).to(device)

    if ddp:
        transformer = DDP(transformer, device_ids=[args.local_rank], find_unused_parameters=False)
    transformer_without_ddp = transformer.module if ddp else transformer

    n_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    if is_main_process():
        print(f'Transformer params: {n_params / 1e6:.1f}M')

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        transformer.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    scaler = torch.amp.GradScaler("cuda")
    ce_loss = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

    # ---- Resume ----
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt2 = torch.load(args.resume, map_location='cpu')
        transformer_without_ddp.load_state_dict(ckpt2['model'])
        optimizer.load_state_dict(ckpt2['optimizer'])
        start_epoch = ckpt2.get('epoch', 0) + 1
        if is_main_process():
            print(f'Resumed transformer from epoch {start_epoch}')

    # ---- Training Loop ----
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_epochs * len(train_loader)
    global_step = start_epoch * len(train_loader)
    L = transformer_without_ddp.L

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        transformer.train()
        t0 = time.time()
        for step, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            lr = cosine_lr_with_warmup(optimizer, global_step, total_steps, warmup_steps, args.lr)

            # Get per-level indices from frozen VAE (in UNet level order: [r1, r2, r3, r4])
            with torch.no_grad():
                indices_unet_order = vae.img_to_indices(imgs)  # [r1(128x128), r2(64x64), r3(32x32), r4(16x16)]

            # Reverse to AR order: deepest first [r4, r3, r2, r1]
            indices_ar = list(reversed(indices_unet_order))

            # Build teacher-forcing token embeddings (B, L, d_embed) in AR order
            with torch.no_grad():
                token_embeds = get_token_embeds_for_training(vae, indices_ar)  # (B, L, d_embed)

            # Build ground-truth label sequence: (B, L) of codebook indices
            gt_indices = torch.cat([
                idx.reshape(labels.shape[0], -1) for idx in indices_ar
            ], dim=1)  # (B, L)

            with torch.amp.autocast("cuda"):
                logits_BLV = transformer(labels, token_embeds)  # (B, L, V)
                loss = ce_loss(logits_BLV.reshape(-1, args.vocab_size), gt_indices.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(transformer.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            global_step += 1

            if is_main_process() and global_step % args.log_every == 0:
                # Per-level accuracy
                with torch.no_grad():
                    preds = logits_BLV.argmax(dim=-1)  # (B, L)
                    acc_mean = (preds == gt_indices).float().mean().item() * 100
                    # Accuracy on finest level (last level_lens[-1] tokens)
                    last_l = transformer_without_ddp.level_lens[-1]
                    acc_fine = (preds[:, -last_l:] == gt_indices[:, -last_l:]).float().mean().item() * 100

                elapsed = time.time() - t0
                print(f'[Stage2] Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {acc_mean:.1f}% | '
                      f'AccFine: {acc_fine:.1f}% | LR: {lr:.2e} | '
                      f'GradNorm: {grad_norm:.3f} | Time: {elapsed:.1f}s')
                wandb.log({
                    'train/loss': loss.item(),
                    'train/acc_mean': acc_mean,
                    'train/acc_fine': acc_fine,
                    'train/lr': lr,
                    'train/grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                    'epoch': epoch,
                    'step': global_step,
                }, step=global_step)
                t0 = time.time()

        # ---- Validation ----
        if is_main_process():
            transformer.eval()
            val_loss = 0.0
            val_acc = 0.0
            val_n = 0
            with torch.no_grad():
                for imgs, labels in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    indices_unet_order = vae.img_to_indices(imgs)
                    indices_ar = list(reversed(indices_unet_order))
                    token_embeds = get_token_embeds_for_training(vae, indices_ar)
                    gt_indices = torch.cat([idx.reshape(labels.shape[0], -1) for idx in indices_ar], dim=1)
                    with torch.amp.autocast("cuda"):
                        logits_BLV = transformer_without_ddp(labels, token_embeds)
                        l = F.cross_entropy(logits_BLV.reshape(-1, args.vocab_size), gt_indices.reshape(-1))
                    val_loss += l.item() * imgs.shape[0]
                    val_acc += (logits_BLV.argmax(dim=-1) == gt_indices).float().mean().item() * imgs.shape[0]
                    val_n += imgs.shape[0]

            val_loss /= val_n
            val_acc /= val_n
            print(f'[Stage2 Val] Epoch {epoch} | Loss: {val_loss:.4f} | Acc: {val_acc*100:.1f}%')
            wandb.log({
                'val/loss': val_loss,
                'val/acc': val_acc * 100,
                'epoch': epoch,
            }, step=global_step)

        # ---- Save checkpoint ----
        if is_main_process() and (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f'stage2_ep{epoch:04d}.pth')
            save_checkpoint({
                'epoch': epoch,
                'model': transformer_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
            }, ckpt_path)
            print(f'Saved checkpoint: {ckpt_path}')

        if ddp:
            dist.barrier()

    if is_main_process():
        final_path = os.path.join(args.output_dir, 'stage2_final.pth')
        save_checkpoint({
            'epoch': args.epochs - 1,
            'model': transformer_without_ddp.state_dict(),
            'args': vars(args),
        }, final_path)
        print(f'Final checkpoint saved: {final_path}')
        wandb.finish()


if __name__ == '__main__':
    main()
