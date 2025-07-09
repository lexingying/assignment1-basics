#!/bin/bash
#SBATCH --job-name=training_together_owt
#SBATCH --output=training_together_owt.out
#SBATCH --error=training_together_owt.err
#SBATCH --time=1:30:00
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH -c 8

pwd
uv run training_together.py \
    --dataset 'owt' \
    --train_data_path "results/owt/train.bin" \
    --val_data_path "results/owt/val.bin" \
    --vocab_size 32000 \
    --d_model 1024 \
    --num_heads 16 \
    --num_layers 12 \
    --d_ff 4096 \
    --context_length 512 \
    --theta 10000.0 \
    --batch_size 32 \
    --max_iters 10000 \
    --alpha_max 2e-3 \
    --alpha_min 1e-4 \
    --t_warmup 1500 \
    --weight_decay 0.02 \
    --beta1 0.9 \
    --beta2 0.99 \
    --grad_clip 1.0 \
    --compile \
    --eval_interval 1000 \
    --save_interval 10000 \
    --use_wandb \
    --wandb_project "cs336_hw1" \
    --wandb_run_name "owt" \
    --checkpoint_dir "results/owt/checkpoints" \
    --device "cuda" \