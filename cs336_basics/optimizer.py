import torch
import math
from torch.optim import Optimizer

def cross_entropy(logits, targets):
    batch_dims = logits.shape[:-1]  # excluding the vocab_size dimension
    vocab_size = logits.shape[-1]
    
    # do some reshape
    batch_size = torch.prod(torch.tensor(batch_dims)) if batch_dims else torch.tensor(1)
    flat_logits = logits.reshape(-1, vocab_size)
    flat_targets = targets.reshape(-1)
    
    max_logits, _ = flat_logits.max(dim=1, keepdim=True)
    logits_stable = flat_logits - max_logits
    
    sum_exp_logits = torch.exp(logits_stable).sum(dim=1, keepdim=True)
    log_sum_exp = max_logits + torch.log(sum_exp_logits)
    
    batch_indices = torch.arange(batch_size, device=logits.device)
    target_logits = flat_logits[batch_indices, flat_targets]
    
    cross_entropy_values = -target_logits + log_sum_exp.squeeze(1)
    
    if batch_dims:
        cross_entropy_values = cross_entropy_values.reshape(batch_dims)

    return cross_entropy_values.mean()



class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):   
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                
                lr = group['lr']
                beta1, beta2 = group['betas']
                eps = group['eps']
                weight_decay = group['weight_decay']
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data) 
                    state['v'] = torch.zeros_like(p.data) 
                
                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']
                
                # Update biased moment estimates
                m.mul_(beta1).add_((1 - beta1) * grad)    
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                adjusted_lr = lr * math.sqrt(bias_correction2) / bias_correction1

                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-adjusted_lr)
                
                # weight decay
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)
                
        return loss
    


def get_lr_cosine_schedule(current_iter, alpha_max, alpha_min, T_warmup, T_cosine):
    if current_iter < T_warmup:
        return (current_iter / T_warmup) * alpha_max

    elif current_iter <= T_cosine:
        cosine_factor = 0.5 * (1 + math.cos(math.pi * (current_iter - T_warmup) / (T_cosine - T_warmup)))
        return alpha_min + cosine_factor * (alpha_max - alpha_min)
    
    else:
        return alpha_min
    



def clip_gradients(parameters, max_norm, eps=1e-6):
    parameters_with_grad = [p for p in parameters if p.grad is not None]
    
    if len(parameters_with_grad) == 0:
        return 0.0
    
    total_norm = torch.norm(
        torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters_with_grad]), 2
    )
    
    clip_coef = max_norm / (total_norm + eps)
    
    if clip_coef < 1.0:
        for p in parameters_with_grad:
            p.grad.detach().mul_(clip_coef)
    
    return total_norm.item()