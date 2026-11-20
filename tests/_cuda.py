"""G-track tests need a CUDA GPU (default target: RTX 5090, sm_120) and nvcc; elsewhere they are skipped."""
import pytest

try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:          # CPU-only checkout without torch
    torch = None
    HAS_CUDA = False

requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="G drills need an NVIDIA GPU + torch with CUDA")


def cc():
    return torch.cuda.get_device_capability() if HAS_CUDA else (0, 0)
