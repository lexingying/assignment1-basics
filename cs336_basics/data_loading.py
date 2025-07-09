import torch
import numpy as np
from numpy import typing as npt


def load_data_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    dataset_length = len(dataset)
    sample_range = dataset_length - context_length
    sample_idcs = np.random.randint(0, sample_range, batch_size)
    samples = np.stack([dataset[idx:idx + context_length] for idx in sample_idcs])
    samples_offset = np.stack([dataset[idx + 1:idx + context_length + 1] for idx in sample_idcs])
    
    return torch.tensor(samples, dtype=torch.long, device=device), torch.tensor(samples_offset, dtype=torch.long, device=device)