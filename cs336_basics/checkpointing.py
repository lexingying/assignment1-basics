import torch
from typing import Union, IO, BinaryIO
import os


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer, 
                   iteration: int, 
                   out: Union[str, os.PathLike, BinaryIO, IO[bytes]]) -> None:
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    # Save the checkpoint
    torch.save(checkpoint, out)


def load_checkpoint(src: Union[str, os.PathLike, BinaryIO, IO[bytes]], 
                   model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer) -> int:
    # Load the checkpoint
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']