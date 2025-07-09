#!/bin/bash
#SBATCH --job-name=training_together_tinystories_ablation
#SBATCH --output=training_together_tinystories_ablation.out
#SBATCH --error=training_together_tinystories_ablation.err
#SBATCH --time=1:00:00
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH --mem=100G
#SBATCH -c 8

pwd
uv run training_together.py \
    --dataset 'tinystories' \
    --train_data_path "results/tinystories/train.bin" \
    --val_data_path "results/tinystories/val.bin" \
    --vocab_size 10000 \
    --d_model 512 \
    --num_heads 16 \
    --num_layers 6 \
    --d_ff 1344 \
    --context_length 256 \
    --theta 10000.0 \
    --batch_size 32 \
    --max_iters 10000 \
    --alpha_max 1e-3 \
    --alpha_ratio 0.05 \
    --t_warmup 2500 \
    --weight_decay 0.01 \
    --beta1 0.9 \
    --beta2 0.99 \
    --grad_clip 1.0 \
    --compile \
    --eval_interval 1000 \
    --save_interval 10000 \
    --use_wandb \
    --wandb_project "cs336_hw1" \
    --wandb_run_name "tinystories_withoutRMS" \
    --checkpoint_dir "results/tinystories/checkpoints" \
    --device "cuda" \