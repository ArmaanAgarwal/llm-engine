"""
Loader for the fused RMSNorm CUDA kernel (csrc/rmsnorm.cu).

    from engine import kernels
    kernels.build()          # compiles with nvcc via torch.utils.cpp_extension (~1 min, cached)
    kernels.USE_FUSED = True # RMSNorm.forward now routes to the kernel on CUDA tensors
"""

import os
import torch

USE_FUSED = False
_ext = None


def build(verbose=False):
    global _ext
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _ext = load(
        name="fused_rmsnorm",
        sources=[os.path.join(here, "csrc", "rmsnorm.cu")],
        extra_cuda_cflags=["-O3"],
        verbose=verbose,
    )
    return _ext


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if _ext is None:
        build()
    shape = x.shape
    y = _ext.forward(x.contiguous().view(-1, shape[-1]), weight, eps)
    return y.view(shape)
