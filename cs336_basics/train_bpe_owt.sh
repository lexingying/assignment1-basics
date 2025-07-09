#!/bin/bash
#SBATCH --job-name=train_bpe_owt
#SBATCH --output=train_bpe_owt.out
#SBATCH --error=train_bpe_owt.err
#SBATCH --time=4:00:00
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH --mem=100G
#SBATCH -c 8


uv run run_train_bpe.py \
    --input /data/a1-basics/owt_train.txt \
    --vocab-size 32000 \
    --output-dir results/owt \
    --special-tokens "<|endoftext|>" \
    --progress-bar \
    --num-workers 8

