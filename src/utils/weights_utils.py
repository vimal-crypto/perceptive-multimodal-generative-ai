import torch
import torch.nn as nn
import os
import random
import numpy as np


def weights_init(module: nn.Module):
    """
    Initialize weights for Conv and BatchNorm layers using zero-mean Gaussians.
    To be called via model.apply(weights_init).

    Conv weights: N(0, 0.02)
    BatchNorm weights: N(1, 0.02), biases set to 0.

    Args:
        module: A PyTorch nn.Module.
    """
    classname = module.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv2d') == -1:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 and classname.find('BatchNorm2d') == -1:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0)


class AverageValueMeter:
    """
    Running average meter for tracking loss values during training.
    Tracks current value, running sum, count, and mean.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val: float, n: int = 1):
        """
        Update with a new value.

        Args:
            val: New value to add.
            n: Batch size or count this value represents.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return f"AverageValueMeter(avg={self.avg:.4f}, count={int(self.count)})"


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: nn.Module, optimizer, epoch: int, loss: float, path: str):
    """
    Save a training checkpoint.

    Args:
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        epoch: Current epoch number.
        loss: Current loss value.
        path: File path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f"[INFO] Checkpoint saved: {path}")


def load_checkpoint(model: nn.Module, path: str, optimizer=None, device: str = None):
    """
    Load a training checkpoint into a model.

    Args:
        model: PyTorch model to load into.
        path: Path to checkpoint file.
        optimizer: Optional optimizer to restore state.
        device: Device to map tensors to.

    Returns:
        Tuple of (model, optimizer, epoch, loss).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    print(f"[INFO] Checkpoint loaded: {path} (epoch {ckpt.get('epoch', '?')})")
    return model, optimizer, ckpt.get('epoch', 0), ckpt.get('loss', 0.0)
