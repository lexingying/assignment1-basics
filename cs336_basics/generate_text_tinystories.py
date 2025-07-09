import torch
from cs336_basics.decoding import *
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
import torch.nn as nn


datasets = {
    'tinystories': {
        'train': 'data/TinyStoriesV2-GPT4-train.txt',
        'val': 'data/TinyStoriesV2-GPT4-valid.txt',
        'vocab_filepath': 'cs336_basics/results/tinystories/vocab.json',
        'merges_filepath': 'cs336_basics/results/tinystories/merges.json',
        'special_tokens': ['<|endoftext|>'],
        'output_dir': 'cs336_basics/results/tinystories'
    }
}

def load_model_from_checkpoint(checkpoint_path, vocab_size, context_length, d_model, num_heads, d_ff, num_layers, theta, device="cuda" if torch.cuda.is_available() else "cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    transformer_lm = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        theta=theta,
    )
    weights = checkpoint['model_state_dict']
    transformer_lm.token_embedding.weight = nn.Parameter(weights['_orig_mod.token_embedding.weight'])

    transformer_lm.ln_final.weight = nn.Parameter(weights['_orig_mod.ln_final.weight'])
    transformer_lm.lm_head.weight = nn.Parameter(weights['_orig_mod.lm_head.weight'])
    
    for layer_idx in range(num_layers):
        prefix = f'_orig_mod.transformer_blocks.{layer_idx}.'
        
        # Attention weights
        transformer_lm.transformer_blocks[layer_idx].attention.W_q.weight = nn.Parameter(weights[f'{prefix}attention.W_q.weight'])
        transformer_lm.transformer_blocks[layer_idx].attention.W_k.weight = nn.Parameter(weights[f'{prefix}attention.W_k.weight'])
        transformer_lm.transformer_blocks[layer_idx].attention.W_v.weight = nn.Parameter(weights[f'{prefix}attention.W_v.weight'])
        transformer_lm.transformer_blocks[layer_idx].attention.W_o.weight = nn.Parameter(weights[f'{prefix}attention.W_o.weight'])
        
        # RoPE caches
        if hasattr(transformer_lm.transformer_blocks[layer_idx].attention.rope, 'cos_cache'):
            transformer_lm.transformer_blocks[layer_idx].attention.rope.cos_cache = weights[f'{prefix}attention.rope.cos_cache']
        if hasattr(transformer_lm.transformer_blocks[layer_idx].attention.rope, 'sin_cache'):
            transformer_lm.transformer_blocks[layer_idx].attention.rope.sin_cache = weights[f'{prefix}attention.rope.sin_cache']
        
        # Layer normalization weights
        transformer_lm.transformer_blocks[layer_idx].attention_norm.weight = nn.Parameter(weights[f'{prefix}attention_norm.weight'])
        transformer_lm.transformer_blocks[layer_idx].ff_norm.weight = nn.Parameter(weights[f'{prefix}ff_norm.weight'])
        
        # Feedforward weights
        transformer_lm.transformer_blocks[layer_idx].ffn.w1.weight = nn.Parameter(weights[f'{prefix}ffn.w1.weight'])
        transformer_lm.transformer_blocks[layer_idx].ffn.w2.weight = nn.Parameter(weights[f'{prefix}ffn.w2.weight'])
        transformer_lm.transformer_blocks[layer_idx].ffn.w3.weight = nn.Parameter(weights[f'{prefix}ffn.w3.weight'])
    transformer_lm.to(device)
    transformer_lm.eval()
    
    return transformer_lm

checkpoint_path = "cs336_basics/results/tinystories/checkpoints/best_model.pt"
vocab_size = 10000
context_length = 256
d_model = 512
num_heads = 16
d_ff = 1344
num_layers = 6
theta = 10000.0
model = load_model_from_checkpoint(checkpoint_path, vocab_size, context_length, d_model, num_heads, d_ff, num_layers, theta)



tokenizer = Tokenizer.from_files(
    vocab_filepath=datasets['tinystories']['vocab_filepath'],
    merges_filepath=datasets['tinystories']['merges_filepath'],
    special_tokens=datasets['tinystories']['special_tokens']
)

# generate text
prompt = "A little boy finished his homework."
generated_text = generate_text(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=300,
    temperature=0.7,
    top_p=0.9,
)

print(generated_text)