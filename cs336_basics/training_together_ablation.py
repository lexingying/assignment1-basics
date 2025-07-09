import argparse
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import os
import math
import wandb
from pathlib import Path
from tqdm.auto import tqdm
from typing import Iterable


from cs336_basics.optimizer import *
from cs336_basics.model import *
from cs336_basics.checkpointing import *
from cs336_basics.data_loading import *


def data_generator(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    while True:
        yield load_data_batch(dataset, batch_size, context_length, device)


def train_transformer(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = args.device if torch.cuda.is_available() and args.device.startswith('cuda') else 'cpu'
    print(f"device: {device}")
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    time_str = time.strftime('%Y-%m-%d_%H-%M-%S')
    
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode='r')


    vocab_size = args.vocab_size
    
    print(f"Vocabulary size: {vocab_size}")
    
    model = TransformerLM_without_RMS(
        vocab_size=vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta
    ).to(device)
    
    # compilation
    if hasattr(torch, 'compile') and args.compile:
        print("Compiling...")
        model = torch.compile(model)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {n_params/1e6:.2f}M")
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.alpha_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    
    # Resume from checkpoint (optional)
    start_iter = 0
    if args.resume_from_checkpoint and args.checkpoint_path:
        if os.path.exists(args.checkpoint_path):
            start_iter = load_checkpoint(args.checkpoint_path, model, optimizer)
            print(f"Resumed from iteration {start_iter}")
        else:
            print(f"No checkpoint at {args.checkpoint_path}")
    
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"transformer-lm-{time_str}",
            config=vars(args)
        )
    
    
    data_gen = data_generator(train_data, args.batch_size, args.context_length, device)
    
    print("training...")
    best_val_loss = float('inf')
    
    try:
        pbar = tqdm(range(start_iter, args.max_iters))
        for iter_num in pbar:
            # Update learning rate according to schedule
            current_lr = get_lr_cosine_schedule(
                iter_num, 
                args.alpha_max, 
                args.alpha_ratio * args.alpha_max,
                args.t_warmup, 
                args.max_iters
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
            
            # get next batch
            X, Y = next(data_gen)
            
            X = torch.clamp(X, 0, vocab_size - 1)
            Y = torch.clamp(Y, 0, vocab_size - 1)
            
            optimizer.zero_grad()
            logits = model(X)
            
            loss = cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            
            loss.backward()
            
            if args.grad_clip > 0:
                clip_gradients(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.1e}"
            })
            
            # Evaluation
            if iter_num % args.eval_interval == 0:
                model.eval()
                with torch.no_grad():
                    val_x, val_y = load_data_batch(val_data, args.batch_size, args.context_length, device)
                    val_x = torch.clamp(val_x, 0, vocab_size - 1)
                    val_y = torch.clamp(val_y, 0, vocab_size - 1)
                    
                    val_logits = model(val_x)
                    val_loss = cross_entropy(val_logits.view(-1, val_logits.size(-1)), val_y.view(-1))
                    
                    # perplexity
                    train_ppl = math.exp(loss.item())
                    val_ppl = math.exp(val_loss.item())
                    
                    print(f"\nStep {iter_num}: train_loss:{loss.item():.4f}, val_loss:{val_loss.item():.4f}, "
                          f"train_ppl:{train_ppl:.2f}, val_ppl:{val_ppl:.2f}, lr:{current_lr:.1e}")
                    
                    if args.use_wandb:
                        wandb.log({
                            'iter': iter_num,
                            'train/loss': loss.item(),
                            'val/loss': val_loss.item(),
                            'train/perplexity': train_ppl,
                            'val/perplexity': val_ppl,
                            'learning_rate': current_lr
                        }, step=iter_num)
                    
                    # the best model
                    if val_loss.item() < best_val_loss:
                        best_val_loss = val_loss.item()
                        best_model_path = checkpoint_dir / f"best_model.pt"
                        save_checkpoint(model, optimizer, iter_num, best_model_path)
                        print(f"New best model: {best_model_path}")
                
                model.train()
            
            # Save checkpoints
            if iter_num > 0 and iter_num % args.save_interval == 0:
                checkpoint_path = checkpoint_dir / f"{args.wandb_run_name or 'model'}_{time_str}_{iter_num//1000}k.pt"
                save_checkpoint(model, optimizer, iter_num, checkpoint_path)
                print(f"\nCheckpoint saved: {checkpoint_path}")
    
    except KeyboardInterrupt:
        print("\nTraining interrupted")
    
    # Final checkpoint
    final_checkpoint_path = checkpoint_dir / f"{args.wandb_run_name or 'model'}_{time_str}_final.pt"
    save_checkpoint(model, optimizer, iter_num, final_checkpoint_path)
    print(f"Final checkpoint: {final_checkpoint_path}")
    
    if args.use_wandb:
        wandb.finish()
    
    return model, optimizer

def main():
    parser = argparse.ArgumentParser(description='Train a Transformer LM')
    
    # Data param
    parser.add_argument('--dataset', type=str, default='tinystories', help='Dataset name (folder in data/)')
    parser.add_argument('--vocab_size', type=int, default=None, help='Vocabulary size (default: detect from dataset)')
    parser.add_argument('--train_data_path', type=str, default=None, help='Path to training data')
    parser.add_argument('--val_data_path', type=str, default=None, help='Path to validation data')
    
    # Model param
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--d_ff', type=int, default=1344, help='Feed-forward dimension')
    parser.add_argument('--context_length', type=int, default=256, help='Context length for training')
    parser.add_argument('--theta', type=float, default=10000.0, help='Base for rotary positional embedding')
    
    # Training param
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_iters', type=int, default=10000, help='Maximum training iterations')
    parser.add_argument('--alpha_max', type=float, default=3e-4, help='Maximum learning rate')
    parser.add_argument('--alpha_ratio', type=float, default=0.1, help='Minimum learning rate ratio')
    parser.add_argument('--t_warmup', type=float, default=0.05, help='Warmup iterations')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay for AdamW')

    parser.add_argument('--beta1', type=float, default=0.9, help='AdamW beta1')
    parser.add_argument('--beta2', type=float, default=0.95, help='AdamW beta2')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping (0 to disable)')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile() for faster training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    # Other param
    parser.add_argument('--eval_interval', type=int, default=500, help='Evaluate every N iterations')
    parser.add_argument('--save_interval', type=int, default=5000, help='Save checkpoint every N iterations')
    parser.add_argument('--use_wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='transformer-lm', help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name')
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Directory for checkpoints')
    parser.add_argument('--resume_from_checkpoint', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to checkpoint for resuming')
    
    parser.add_argument('--device', type=str, default='cuda', help='Device for training (cuda or cpu)')
    
    args = parser.parse_args()
    train_transformer(args)

if __name__ == '__main__':
    main()