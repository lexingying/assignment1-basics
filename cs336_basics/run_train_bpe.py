#!/usr/bin/env python3

import argparse
import json
import os
import psutil
import time
import cProfile

from train_bpe import *

def save_tokenizer(vocab, merges, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Converting
    serializable_vocab = {k: list(v) for k, v in vocab.items()}
    
    serializable_merges = []
    for (first, second) in merges:
        serializable_merges.append([list(first), list(second)])

    # Saving
    with open(os.path.join(output_dir, "vocab.json"), "w") as f:
        json.dump(serializable_vocab, f)

    with open(os.path.join(output_dir, "merges.json"), "w") as f:
        json.dump(serializable_merges, f)
    
    print(f"Tokenizer saved to {output_dir}")

def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / (10 ** 9) 

def main():
    parser = argparse.ArgumentParser(description="Train a BPE tokenizer on the target dataset")
    parser.add_argument("--input", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--vocab-size", type=int, default=8000, help="Target vocabulary size")
    parser.add_argument("--output-dir", type=str, default="tokenizer", help="Directory to save the tokenizer files")
    parser.add_argument("--special-tokens", nargs="+", default=["<|endoftext|>", "<|startoftext|>", "<|pad|>"], 
                        help="Special tokens to include in the vocabulary")
    parser.add_argument("--progress-bar", action="store_true", help="Show progress bar during training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of processes to use for training")
    
    args = parser.parse_args()
    
    print(f"Training BPE tokenizer on {args.input} with target vocabulary size {args.vocab_size}")
    print(f"Special tokens: {args.special_tokens}")
    print(f"progress bar: {args.progress_bar}")
    print(f"Number of processes: {args.num_workers}")

    start_memory = get_memory_usage()
    print(f"Initial memory usage: {start_memory:.2f} GB")
    
    # Start the timer
    start_time = time.time()

    # pr = cProfile.Profile()
    # pr.enable()
    vocab, merges = train_bpe(args.input, args.vocab_size, args.special_tokens, args.progress_bar, args.num_workers)
    # pr.disable()
    # pr.print_stats(sort='time')
    end_time = time.time()
    training_time = end_time - start_time

    end_memory = get_memory_usage()
    memory_used = end_memory - start_memory

    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Peak memory usage: {end_memory:.2f} GB")
    print(f"Memory increase during training: {memory_used:.2f} GB")
    
    save_tokenizer(vocab, merges, args.output_dir)

if __name__ == "__main__":
    main()