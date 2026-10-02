"""Pick the compute device: NVIDIA GPU (0), Apple Silicon GPU ("mps"), or "cpu"."""

import torch


def best_device():
    if torch.cuda.is_available():
        return 0
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
