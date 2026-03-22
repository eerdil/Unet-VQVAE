"""
Visualize VQ-VAE reconstructions from a checkpoint and log to wandb.
Usage:
    python visualize_recons.py --checkpoint <path_to_ckpt.pth> [--n_images 16] [--wandb_run_name <name>]
"""
import argparse
import sys
from pathlib import Path

import torch
import wandb

sys.path.insert(0, str(Path(__file__).parent))
from models import UNetVQVAE
from utils.data import build_imagenet_dataset
from torch.utils.data import DataLoader, Subset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_path', type=str,
                        default='/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/ILSVRC2012_imagenet/imagenet')
    parser.add_argument('--n_images', type=int, default=16, help='Number of images to visualize')
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--wandb_project', type=str, default='unetvqvae-stage1')
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ckpt_args = ckpt.get('args', {})
    epoch = ckpt.get('epoch', '?')
    val_recon = ckpt.get('val_recon_loss', float('nan'))
    print(f'Loaded checkpoint: epoch={epoch}, val_recon={val_recon:.4f}')

    # Build model from checkpoint args
    model = UNetVQVAE(
        in_channels=3,
        base_ch=ckpt_args.get('base_ch', 64),
        vocab_size=ckpt_args.get('vocab_size', 4096),
        d_embed=ckpt_args.get('d_embed', 256),
        beta=ckpt_args.get('beta', 0.25),
        ema_decay=ckpt_args.get('ema_decay', 0.99),
    ).to(device)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()

    # Load a small subset of val images
    _, val_set = build_imagenet_dataset(args.data_path, args.image_size)
    subset = Subset(val_set, list(range(args.n_images)))
    loader = DataLoader(subset, batch_size=args.n_images, shuffle=False, num_workers=args.num_workers)
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)

    # Run forward pass
    with torch.no_grad():
        recon, _, recon_loss, commit_loss = model(imgs)
        usage = model.quantizer.get_codebook_usage()

    print(f'Recon loss: {recon_loss.item():.4f} | Commit loss: {commit_loss.item():.4f} | Codebook usage: {usage*100:.1f}%')

    # Denormalize from [-1, 1] to [0, 1]
    imgs_vis = (imgs.clamp(-1, 1) + 1) / 2
    recon_vis = (recon.clamp(-1, 1) + 1) / 2

    # Build interleaved pairs: orig, recon, orig, recon, ...
    paired = torch.stack([imgs_vis.cpu(), recon_vis.cpu()], dim=1).flatten(0, 1)  # (2*N, 3, H, W)

    # Log to wandb
    run_name = args.wandb_run_name or f'recon_viz_ep{epoch}'
    wandb.init(project=args.wandb_project, name=run_name, config={
        'checkpoint': args.checkpoint,
        'epoch': epoch,
        'val_recon_loss': val_recon,
        'recon_loss_on_viz': recon_loss.item(),
        'codebook_usage': usage,
        **ckpt_args,
    })

    wandb.log({
        'reconstructions': [wandb.Image(paired[i], caption='orig' if i % 2 == 0 else 'recon')
                            for i in range(len(paired))],
        'recon_loss': recon_loss.item(),
        'commit_loss': commit_loss.item(),
        'codebook_usage': usage,
        'epoch': epoch,
    })
    wandb.finish()
    print('Done. Images logged to wandb.')


if __name__ == '__main__':
    main()
