import torch
from torch import nn
from einops import einsum
import math
from typing import Optional
    

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **kwargs))
        std = math.sqrt(2.0 / (in_features + out_features))
        a = -3.0 * std
        b =  3.0 * std
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=a, b=b)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return einsum(x, self.weight, '... d_in, d_out d_in -> ... d_out')

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), **kwargs))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize weights using truncated normal distribution
        nn.init.trunc_normal_(self.weight)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        kwargs = {'device': device, 'dtype': dtype}

        self.weight = nn.Parameter(torch.ones(d_model, **kwargs))

        self.eps = eps
        self.d_model = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        x_normalized = x / rms

        result = x_normalized * self.weight
        
        return result.to(in_dtype)
    


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        
        self.d_model = d_model
        self.d_ff = d_ff
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        silu_out = w1x * torch.sigmoid(w1x) 
        w3_out = self.w3(x)
        gated = silu_out * w3_out
        
        return self.w2(gated)
    

class silu(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        
        self.d_model = d_model
        self.d_ff = d_ff
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        silu_out = w1x * torch.sigmoid(w1x) 

        
        return self.w2(silu_out)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        
        # Build sinusoidal embeddings directly on the correct device
        freqs = 1.0 / (theta ** (torch.arange(0, d_k, 2).float() / d_k))
        t = torch.arange(max_seq_len, device=device).float().unsqueeze(1)
        freqs = freqs.unsqueeze(0)
        angles = t * freqs
        
        # Pre-compute cos and sin values on the device
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        
        # Register buffers
        self.register_buffer("cos_cache", cos_cache)
        self.register_buffer("sin_cache", sin_cache)
        
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Ensure inputs are on the correct device
        
        # Ensure positions are within bounds
        assert torch.max(token_positions) < self.max_seq_len, f"Position {torch.max(token_positions)} exceeds max_seq_len {self.max_seq_len}"
        
        # Shape adjustments for broadcasting
        if len(x.shape) == 4 and token_positions.dim() == 2:
            token_positions = token_positions.unsqueeze(1)
        
        # Get cos and sin values on the same device as the input
        cos = self.cos_cache[token_positions]
        sin = self.sin_cache[token_positions]
        
        # Apply rotation
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_even * cos - x_odd * sin
        x_out[..., 1::2] = x_even * sin + x_odd * cos
        
        return x_out
    

def softmax(x, dim):
    max_x = x.max(dim=dim, keepdim=True)[0]
    
    x_stable = x - max_x
    
    exp_x = torch.exp(x_stable)
    
    sum_exp_x = exp_x.sum(dim=dim, keepdim=True)
    
    softmax_x = exp_x / sum_exp_x
    
    return softmax_x



def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    
    scores = torch.matmul(Q, K.transpose(-2, -1)) # (..., seq_len_q, seq_len_k)
    
    scores = scores / (d_k ** 0.5)
    
    if mask is not None:
        scores = scores.masked_fill(mask == False, float('-inf'))
    
    attention_weights = softmax(scores, dim=-1)

    output = torch.matmul(attention_weights, V) # (..., seq_len_q, d_v)
    
    return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super(MultiHeadSelfAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 
        
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        
        self.rope = RotaryPositionalEmbedding(
            theta=theta, 
            d_k=self.d_k,  
            max_seq_len=max_seq_len
        )
        
    def forward(self, x, token_positions=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        causal_mask = torch.tril(torch.ones(seq_length, seq_length)).to(x.device)
        
        q = self.W_q(x)  # (batch_size, seq_length, d_model)
        k = self.W_k(x)  
        v = self.W_v(x)  
        
        # d_model -> (num_heads, d_k)
        q = q.view(batch_size, seq_length, self.num_heads, self.d_k)
        k = k.view(batch_size, seq_length, self.num_heads, self.d_k)
        v = v.view(batch_size, seq_length, self.num_heads, self.d_k)
        
        q = q.transpose(1, 2) # (batch_size, num_heads, seq_length, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        if token_positions is not None:
            positions = token_positions
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        
        attention_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        
        attention_output = attention_output.transpose(1, 2) # (batch_size, seq_length, num_heads, d_k)
        
        # Concatenate heads
        concat_attention = attention_output.contiguous().view(batch_size, seq_length, self.d_model)
        
        output = self.W_o(concat_attention)
        
        return output
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0):
        super(TransformerBlock, self).__init__()
        
        # RMSNorm
        self.attention_norm = RMSNorm(d_model)
        self.ff_norm = RMSNorm(d_model)
        
        # Multihead attention
        self.attention = MultiHeadSelfAttention(d_model, num_heads, theta)
        
        # Feedforward
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        
    def forward(self, x, token_positions: Optional[torch.Tensor] = None):
        x = x + self.attention(self.attention_norm(x), token_positions)
        
        x = x + self.ffn(self.ff_norm(x))
        
        return x
    


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, 
                 num_heads: int, d_ff: int, num_layers: int, theta: float = 10000.0):
        super(TransformerLM, self).__init__()
        
        # Token embedding
        self.token_embedding = Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, theta)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.ln_final = RMSNorm(d_model)
        
        # Output projection
        self.lm_head = Linear(d_model, vocab_size)
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff

    
    def forward(self, tokens, token_positions=None):
        batch_size, seq_length = tokens.shape
        if(seq_length > self.context_length):
            tokens = tokens[:, -self.context_length:]
            seq_length = self.context_length
        assert seq_length <= self.context_length, f"Input sequence length ({seq_length}) exceeds model's maximum context length ({self.context_length})"
        if token_positions is None:
            token_positions = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)
        # Embed tokens
        x = self.token_embedding(tokens)  # [batch_size, seq_length, d_model]
        
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        
        x = self.ln_final(x)

        logits = self.lm_head(x)  # [batch_size, seq_length, vocab_size]
        
        return logits
    

class TransformerBlock_without_RMS(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0):
        super(TransformerBlock_without_RMS, self).__init__()
        
        # Multihead attention
        self.attention = MultiHeadSelfAttention(d_model, num_heads, theta)
        
        # Feedforward
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        
    def forward(self, x, token_positions: Optional[torch.Tensor] = None):
        x = x + self.attention(x, token_positions)
        
        x = x + self.ffn(x)
        
        return x
    

class TransformerLM_without_RMS(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, 
                 num_heads: int, d_ff: int, num_layers: int, theta: float = 10000.0):
        super(TransformerLM_without_RMS, self).__init__()
        
        # Token embedding
        self.token_embedding = Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock_without_RMS(d_model, num_heads, d_ff, theta)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.ln_final = RMSNorm(d_model)
        
        # Output projection
        self.lm_head = Linear(d_model, vocab_size)
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff

    
    def forward(self, tokens, token_positions=None):
        batch_size, seq_length = tokens.shape
        assert seq_length <= self.context_length, f"Input sequence length ({seq_length}) exceeds model's maximum context length ({self.context_length})"
        if token_positions is None:
            token_positions = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)
        # Embed tokens
        x = self.token_embedding(tokens)  # [batch_size, seq_length, d_model]
        
        for block in self.transformer_blocks:
            x = block(x, token_positions)

        logits = self.lm_head(x)  # [batch_size, seq_length, vocab_size]
        
        return logits
    

class TransformerBlock_postnorm(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0):
        super(TransformerBlock_postnorm, self).__init__()
        
        # RMSNorm
        self.attention_norm = RMSNorm(d_model)
        self.ff_norm = RMSNorm(d_model)
        
        # Multihead attention
        self.attention = MultiHeadSelfAttention(d_model, num_heads, theta)
        
        # Feedforward
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        
    def forward(self, x, token_positions: Optional[torch.Tensor] = None):
        x = self.attention_norm(x + self.attention(x, token_positions))
        
        x = self.ff_norm(x + self.ffn(x))
        
        return x

class TransformerLM_postnorm(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, 
                 num_heads: int, d_ff: int, num_layers: int, theta: float = 10000.0):
        super(TransformerLM_postnorm, self).__init__()
        
        # Token embedding
        self.token_embedding = Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock_postnorm(d_model, num_heads, d_ff, theta)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.ln_final = RMSNorm(d_model)
        
        # Output projection
        self.lm_head = Linear(d_model, vocab_size)
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff

    
    def forward(self, tokens, token_positions=None):
        batch_size, seq_length = tokens.shape
        assert seq_length <= self.context_length, f"Input sequence length ({seq_length}) exceeds model's maximum context length ({self.context_length})"
        if token_positions is None:
            token_positions = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)
        # Embed tokens
        x = self.token_embedding(tokens)  # [batch_size, seq_length, d_model]
        
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        
        x = self.ln_final(x)

        logits = self.lm_head(x)  # [batch_size, seq_length, vocab_size]
        
        return logits
    
class MultiHeadSelfAttention_NOPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = 1024, theta: float = 10000.0):
        super(MultiHeadSelfAttention_NOPE, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads 
        
        self.W_q = Linear(d_model, d_model)
        self.W_k = Linear(d_model, d_model)
        self.W_v = Linear(d_model, d_model)
        self.W_o = Linear(d_model, d_model)
        
        self.rope = RotaryPositionalEmbedding(
            theta=theta, 
            d_k=self.d_k,  
            max_seq_len=max_seq_len
        )
        
    def forward(self, x, token_positions=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        causal_mask = torch.tril(torch.ones(seq_length, seq_length)).to(x.device)
        
        q = self.W_q(x)  # (batch_size, seq_length, d_model)
        k = self.W_k(x)  
        v = self.W_v(x)  
        
        # d_model -> (num_heads, d_k)
        q = q.view(batch_size, seq_length, self.num_heads, self.d_k)
        k = k.view(batch_size, seq_length, self.num_heads, self.d_k)
        v = v.view(batch_size, seq_length, self.num_heads, self.d_k)
        
        q = q.transpose(1, 2) # (batch_size, num_heads, seq_length, d_k)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attention_output = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        
        attention_output = attention_output.transpose(1, 2) # (batch_size, seq_length, num_heads, d_k)
        
        # Concatenate heads
        concat_attention = attention_output.contiguous().view(batch_size, seq_length, self.d_model)
        
        output = self.W_o(concat_attention)
        
        return output

class TransformerBlock_NOPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0):
        super(TransformerBlock_NOPE, self).__init__()
        
        # RMSNorm
        self.attention_norm = RMSNorm(d_model)
        self.ff_norm = RMSNorm(d_model)
        
        # Multihead attention
        self.attention = MultiHeadSelfAttention_NOPE(d_model, num_heads, theta)
        
        # Feedforward
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
        
    def forward(self, x, token_positions: Optional[torch.Tensor] = None):
        x = x + self.attention(self.attention_norm(x), token_positions)
        
        x = x + self.ffn(self.ff_norm(x))
        
        return x
    


class TransformerLM_NOPE(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, 
                 num_heads: int, d_ff: int, num_layers: int, theta: float = 10000.0):
        super(TransformerLM_NOPE, self).__init__()
        
        # Token embedding
        self.token_embedding = Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock_NOPE(d_model, num_heads, d_ff, theta)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.ln_final = RMSNorm(d_model)
        
        # Output projection
        self.lm_head = Linear(d_model, vocab_size)
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff

    
    def forward(self, tokens, token_positions=None):
        batch_size, seq_length = tokens.shape
        assert seq_length <= self.context_length, f"Input sequence length ({seq_length}) exceeds model's maximum context length ({self.context_length})"
        if token_positions is None:
            token_positions = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)
        # Embed tokens
        x = self.token_embedding(tokens)  # [batch_size, seq_length, d_model]
        
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        
        x = self.ln_final(x)

        logits = self.lm_head(x)  # [batch_size, seq_length, vocab_size]
        
        return logits
    

class PositionwiseFeedForward_SiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        
        self.d_model = d_model
        self.d_ff = d_ff
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1x = self.w1(x)
        silu_out = w1x * torch.sigmoid(w1x) 
        
        return self.w2(silu_out)
    
class TransformerBlock_SiLU(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float = 10000.0):
        super(TransformerBlock_SiLU, self).__init__()
        
        # RMSNorm
        self.attention_norm = RMSNorm(d_model)
        self.ff_norm = RMSNorm(d_model)
        
        # Multihead attention
        self.attention = MultiHeadSelfAttention(d_model, num_heads, theta)
        
        # Feedforward
        self.ffn = PositionwiseFeedForward_SiLU(d_model, d_ff)
        
    def forward(self, x, token_positions: Optional[torch.Tensor] = None):
        x = x + self.attention(self.attention_norm(x), token_positions)
        
        x = x + self.ffn(self.ff_norm(x))
        
        return x
    


class TransformerLM_SiLU(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, 
                 num_heads: int, d_ff: int, num_layers: int, theta: float = 10000.0):
        super(TransformerLM_SiLU, self).__init__()
        
        # Token embedding
        self.token_embedding = Embedding(vocab_size, d_model)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock_SiLU(d_model, num_heads, d_ff, theta)
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.ln_final = RMSNorm(d_model)
        
        # Output projection
        self.lm_head = Linear(d_model, vocab_size)
        
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.d_ff = d_ff

    
    def forward(self, tokens, token_positions=None):
        batch_size, seq_length = tokens.shape
        assert seq_length <= self.context_length, f"Input sequence length ({seq_length}) exceeds model's maximum context length ({self.context_length})"
        if token_positions is None:
            token_positions = torch.arange(seq_length).unsqueeze(0).expand(batch_size, seq_length)
        # Embed tokens
        x = self.token_embedding(tokens)  # [batch_size, seq_length, d_model]
        
        for block in self.transformer_blocks:
            x = block(x, token_positions)
        
        x = self.ln_final(x)

        logits = self.lm_head(x)  # [batch_size, seq_length, vocab_size]
        
        return logits