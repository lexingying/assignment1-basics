import torch

from cs336_basics.model import softmax

def decode(
    model,
    prompt_tokens,
    max_tokens=100,
    temperature=1.0,
    top_p=1.0,
    end_token_id=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.eval()
    if isinstance(prompt_tokens, list):
        prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long)
    
    prompt_tokens = prompt_tokens.to(device)
    
    if len(prompt_tokens.shape) > 1:
        prompt_tokens = prompt_tokens.squeeze()
    
    generated = prompt_tokens.clone()
    
    with torch.no_grad():
        for _ in range(max_tokens):
            input_tokens = generated.unsqueeze(0); print(generated.shape, input_tokens.shape)
            logits = model(input_tokens)
            next_token_logits = logits[0, -1, :]
            
            # temperature scaling
            if temperature > 0:
                next_token_logits = next_token_logits / temperature
            else:
                # greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=0)
                
                if end_token_id is not None and next_token.item() == end_token_id:
                    break
                    
                continue
            
            # final prob
            probs = softmax(next_token_logits, dim=-1)
            
            # top-p sampling
            if top_p < 1.0:
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # indices where cumulative probability exceeds top_p
                sorted_indices_to_remove = cumulative_probs > top_p

                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_keep = torch.zeros_like(probs, dtype=torch.bool)
                indices_to_keep[sorted_indices] = ~sorted_indices_to_remove
                
                # filter the probabilities
                probs = probs.masked_fill(~indices_to_keep, 0.0)
                
                probs = probs / probs.sum()
            
            # sample from the filtered probabilities
            next_token = torch.multinomial(probs, num_samples=1)
            
            # concatenate the generated token
            generated = torch.cat([generated, next_token], dim=0)
            
            if end_token_id is not None and next_token.item() == end_token_id:
                break
    
    return generated


def generate_text(
    model,
    tokenizer,
    prompt,
    max_tokens=100,
    temperature=1.0,
    top_p=1.0,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    prompt_tokens = tokenizer.encode(prompt)
    
    # end token id
    end_token_id = tokenizer.encode("<|endoftext|>")[0] if hasattr(tokenizer, "encode") else None
    
    # generate!
    generated_tokens = decode(
        model,
        prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        end_token_id=end_token_id,
        device=device,
    )
    generated_text = tokenizer.decode(generated_tokens.tolist())
    
    return generated_text