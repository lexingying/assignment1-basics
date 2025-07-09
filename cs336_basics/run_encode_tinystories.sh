#!/bin/bash
#SBATCH --job-name=encode_tinystories
#SBATCH --output=encode_tinystories.out
#SBATCH --error=encode_tinystories.err
#SBATCH --time=2:00:00
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH --mem=100G
#SBATCH -c 8

pwd
uv run encode.py \
    --datasets tinystories