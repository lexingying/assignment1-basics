#!/bin/bash
#SBATCH --job-name=train_bpe_tinystories
#SBATCH --output=train_bpe_tinystories.out
#SBATCH --error=train_bpe_tinystories.err
#SBATCH --time=00:20:00
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH --gpus=1
#SBATCH --mem=50G
#SBATCH -c 8


uv run run_train_bpe.py \
    --input ../data/TinyStoriesV2-GPT4-train.txt \
    --vocab-size 10000 \
    --output-dir results/tinystories \
    --special-tokens "<|endoftext|>" \
#    --num-processes 8
