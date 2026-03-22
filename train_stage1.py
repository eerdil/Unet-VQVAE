"""
Stage 1 Training: Multi-Scale VQ U-Net Autoencoder
Trains the UNet encoder, decoder, projection layers, and VQ codebook (via EMA).
"""
import argparse
import os
import sys
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import wandb

# Allow imports from code/
sys.path.insert(0, str(Path(__file__).parent))
from models import UNetVQVAE
from utils.data import build_imagenet_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='UNet-VAR Stage 1 Training')
    # Data
    parser.add_argument('--data_path', type=str, required=True, help='Path to ImageNet root')
    parser.add_argument('--image_size', type=int, default=256)
    # Model
    parser.add_argument('--base_ch', type=int, default=64, help='Base channel size for UNet')
    parser.add_argument('--vocab_size', type=int, default=4096, help='Codebook size V')
    parser.add_argument('--d_embed', type=int, default=256, help='Codebook embedding dimension')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss weight')
    parser.add_argument('--ema_decay', type=float, default=0.99, help='EMA decay for codebook')
    parser.add_argument('--restart_threshold', type=float, default=0.0,
                        help='Dead-code restart: reinit codes with ema_cluster_size below this. 0=disabled.')
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32, help='Per-GPU batch size')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--warmup_steps', type=int, default=0, help='If > 0, overrides warmup_epochs')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    # DINO alignment (Extension 2)
    parser.add_argument('--dino_align', action='store_true', help='Add DINO bottleneck alignment loss')
    parser.add_argument('--dino_model', type=str, default='dinov3_vits16', help='Which DINOv3 model to use')
    parser.add_argument('--lambda_dino', type=float, default=0.1, help='Weight for DINO alignment loss')
    # Logging / Saving
    parser.add_argument('--output_dir', type=str, default='./output/stage1')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--log_every', type=int, default=50, help='Log every N iterations')
    parser.add_argument('--wandb_project', type=str, default='unetvqvae-stage1')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
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

    # ---- Model ----
    model = UNetVQVAE(
        in_channels=3,
        base_ch=args.base_ch,
        vocab_size=args.vocab_size,
        d_embed=args.d_embed,
        beta=args.beta,
        ema_decay=args.ema_decay,
        restart_threshold=args.restart_threshold,
    ).to(device)

    # ---- DINO Aligner (optional) ----
    dino_aligner = None
    if args.dino_align:
        from models.dino_align import DINOAligner
        dino_aligner = DINOAligner(d_embed=args.d_embed, model_name=args.dino_model).to(device)
        if is_main_process():
            print(f'DINO alignment enabled: {args.dino_model}, lambda={args.lambda_dino}')

    if ddp:
        model = DDP(model, device_ids=[args.local_rank], find_unused_parameters=False)
    model_without_ddp = model.module if ddp else model

    # ---- Optimizer: include dino_aligner.proj if active ----
    params = list(model.parameters())
    if dino_aligner is not None:
        params += list(dino_aligner.proj.parameters())
    optimizer = torch.optim.AdamW(
        params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95)
    )
    # Stage 1 trains in float32 — AMP fp16 overflows with deep U-Net random init

    # ---- Resume ----
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        if is_main_process():
            print(f'Resumed from epoch {start_epoch}')

    # ---- Training Loop ----
    total_steps = args.epochs * len(train_loader)
    warmup_steps = args.warmup_steps if args.warmup_steps > 0 else args.warmup_epochs * len(train_loader)
    global_step = start_epoch * len(train_loader)
    best_val_recon = float('inf')

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        model.train()
        t0 = time.time()
        for step, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)

            lr = cosine_lr_with_warmup(optimizer, global_step, total_steps, warmup_steps, args.lr)

            if dino_aligner is not None:
                # Need h_K for DINO alignment
                z_skips, indices, commit_loss, h_K = model_without_ddp.encode(imgs, return_bottleneck=True)
                recon = model_without_ddp.decode(z_skips)
                recon_loss = ((imgs - recon) ** 2).mean()
                dino_loss = dino_aligner(h_K, imgs)
                loss = recon_loss + args.beta * commit_loss + args.lambda_dino * dino_loss
            else:
                recon, loss, recon_loss, commit_loss = model(imgs)
                dino_loss = torch.tensor(0.0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()

            global_step += 1

            if is_main_process() and global_step % args.log_every == 0:
                usage = model_without_ddp.quantizer.get_codebook_usage()
                elapsed = time.time() - t0
                dino_str = f' | DINO: {dino_loss.item():.4f}' if args.dino_align else ''
                print(f'[Stage1] Epoch {epoch}/{args.epochs} | Step {step}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Recon: {recon_loss.item():.4f} | '
                      f'Commit: {commit_loss.item():.4f}{dino_str} | Codebook usage: {usage*100:.1f}% | '
                      f'LR: {lr:.2e} | GradNorm: {grad_norm:.3f} | Time: {elapsed:.1f}s')
                log_dict = {
                    'train/loss': loss.item(),
                    'train/recon_loss': recon_loss.item(),
                    'train/commit_loss': commit_loss.item(),
                    'train/codebook_usage': usage,
                    'train/lr': lr,
                    'train/grad_norm': grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm,
                    'epoch': epoch,
                    'step': global_step,
                }
                if args.dino_align:
                    log_dict['train/dino_loss'] = dino_loss.item()
                wandb.log(log_dict, step=global_step)
                t0 = time.time()

        # ---- Validation ----
        if is_main_process():
            model.eval()
            val_recon_loss = 0.0
            val_commit_loss = 0.0
            val_n = 0
            log_imgs = None  # save one batch for image logging
            with torch.no_grad():
                for imgs, _ in val_loader:
                    imgs = imgs.to(device, non_blocking=True)
                    recon, loss, recon_loss, commit_loss = model_without_ddp(imgs)
                    val_recon_loss += recon_loss.item() * imgs.shape[0]
                    val_commit_loss += commit_loss.item() * imgs.shape[0]
                    val_n += imgs.shape[0]
                    if log_imgs is None:
                        log_imgs = (imgs[:8].cpu(), recon[:8].cpu())  # save first 8 for logging

            val_recon_loss /= val_n
            val_commit_loss /= val_n
            print(f'[Stage1 Val] Epoch {epoch} | Recon: {val_recon_loss:.4f} | Commit: {val_commit_loss:.4f}')

            # Log reconstructions to wandb: interleave original and reconstructed
            if log_imgs is not None:
                orig, rec = log_imgs
                # Denormalize from [-1,1] to [0,1]
                orig = (orig.clamp(-1, 1) + 1) / 2
                rec = (rec.clamp(-1, 1) + 1) / 2
                paired = torch.stack([orig, rec], dim=1).flatten(0, 1)  # (16, 3, H, W)
                wandb.log({
                    'val/reconstructions': [wandb.Image(paired[i]) for i in range(len(paired))],
                    'val/recon_loss': val_recon_loss,
                    'val/commit_loss': val_commit_loss,
                    'epoch': epoch,
                }, step=global_step)
            else:
                wandb.log({
                    'val/recon_loss': val_recon_loss,
                    'val/commit_loss': val_commit_loss,
                    'epoch': epoch,
                }, step=global_step)

        # ---- Save checkpoint ----
        if is_main_process():
            # Periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.output_dir, f'stage1_ep{epoch:04d}.pth')
                save_checkpoint({
                    'epoch': epoch,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args),
                }, ckpt_path)
                print(f'Saved checkpoint: {ckpt_path}')
            # Best model checkpoint (based on val recon loss)
            if val_recon_loss < best_val_recon:
                best_val_recon = val_recon_loss
                best_path = os.path.join(args.output_dir, 'stage1_best.pth')
                save_checkpoint({
                    'epoch': epoch,
                    'val_recon_loss': val_recon_loss,
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': vars(args),
                }, best_path)
                print(f'New best val recon {val_recon_loss:.4f} — saved {best_path}')

        if ddp:
            dist.barrier()

    # Save final model
    if is_main_process():
        final_path = os.path.join(args.output_dir, 'stage1_final.pth')
        save_checkpoint({
            'epoch': args.epochs - 1,
            'model': model_without_ddp.state_dict(),
            'args': vars(args),
        }, final_path)
        print(f'Final checkpoint saved: {final_path}')
        wandb.finish()


if __name__ == '__main__':
    main()
