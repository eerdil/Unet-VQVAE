#!/bin/bash
#SBATCH --job-name=unetvqvae_stage1_dino
#SBATCH --output=/usr/bmicnas02/data-biwi-01/erdile_data/projects/GenerativeUnet/logs/stage1_dino_%j.out
#SBATCH --error=/usr/bmicnas02/data-biwi-01/erdile_data/projects/GenerativeUnet/logs/stage1_dino_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=48:00:00

mkdir -p /usr/bmicnas02/data-biwi-01/erdile_data/projects/GenerativeUnet/logs

export TMPDIR=/usr/benderstor01/data-biwi-01/erdile_data/data/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR

source /usr/benderstor01/data-biwi-01/erdile_data/data/software/miniconda3/bin/activate
conda activate GenUnet

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29502   # different port from baseline job
export WORLD_SIZE=$SLURM_NTASKS

DATA_PATH="/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/ILSVRC2012_imagenet/imagenet"
OUTPUT_DIR="/usr/bmicnas02/data-biwi-01/erdile_data/projects/GenerativeUnet/outputs/stage1_dino"
mkdir -p $OUTPUT_DIR

cd /usr/bmicnas02/data-biwi-01/erdile_data/projects/GenerativeUnet/code

torchrun \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_stage1.py \
    --data_path $DATA_PATH \
    --image_size 256 \
    --base_ch 64 \
    --vocab_size 4096 \
    --d_embed 256 \
    --beta 0.25 \
    --ema_decay 0.99 \
    --epochs 200 \
    --batch_size 32 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --warmup_epochs 5 \
    --warmup_steps 1000 \
    --grad_clip 1.0 \
    --dino_align \
    --dino_model dinov3_vits16 \
    --lambda_dino 0.1 \
    --output_dir $OUTPUT_DIR \
    --save_every 10 \
    --log_every 50 \
    --num_workers 8 \
    --wandb_project unetvqvae-stage1 \
    --wandb_run_name stage1_dino_vits16
