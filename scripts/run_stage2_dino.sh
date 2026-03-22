#!/bin/bash
#SBATCH --job-name=stage2_dino
#SBATCH --output=/usr/bmicnas02/data-biwi-01/erdile_data/projects/GenerativeUnet/logs/stage2_dino_%j.out
#SBATCH --error=/usr/bmicnas02/data-biwi-01/erdile_data/projects/GenerativeUnet/logs/stage2_dino_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=48:00:00

export TMPDIR=/usr/benderstor01/data-biwi-01/erdile_data/data/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /usr/benderstor01/data-biwi-01/erdile_data/data/software/miniconda3/bin/activate
conda activate GenUnet

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29502
export WORLD_SIZE=$SLURM_NTASKS

DATA_PATH="/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/ILSVRC2012_imagenet/imagenet"
STAGE1_CKPT="/usr/bmicnas02/data-biwi-01/erdile_data/projects/GenerativeUnet/outputs/stage1_dino/stage1_best.pth"
OUTPUT_DIR="/usr/bmicnas02/data-biwi-01/erdile_data/projects/GenerativeUnet/outputs/stage2_dino"

cd /usr/bmicnas02/data-biwi-01/erdile_data/projects/GenerativeUnet/code

torchrun \
    --nproc_per_node=2 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_stage2.py \
    --data_path $DATA_PATH \
    --image_size 256 \
    --stage1_ckpt $STAGE1_CKPT \
    --base_ch 64 \
    --vocab_size 4096 \
    --d_embed 256 \
    --depth 16 \
    --embed_dim 1024 \
    --num_heads 16 \
    --mlp_ratio 4.0 \
    --drop_path_rate 0.1 \
    --cond_drop_rate 0.1 \
    --num_classes 1000 \
    --epochs 300 \
    --batch_size 8 \
    --grad_checkpoint \
    --lr 1e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 10 \
    --grad_clip 1.0 \
    --label_smooth 0.1 \
    --output_dir $OUTPUT_DIR \
    --save_every 10 \
    --log_every 50 \
    --num_workers 8 \
    --wandb_project unetvqvae-stage2 \
    --wandb_run_name stage2_dino
