#!/bin/bash
#SBATCH --job-name=encode_owt
#SBATCH --output=encode_owt.out
#SBATCH --error=encode_owt.err
#SBATCH --time=2:00:00
#SBATCH --partition=a1-batch-cpu
#SBATCH --qos=a1-batch-cpu-qos
#SBATCH --mem=100G
#SBATCH -c 8

pwd
uv run encode.py \
    --datasets owt