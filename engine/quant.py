"""
INT8 weight-only quantization.

Each nn.Linear weight W [out, in] becomes:
    scale = max(|W|, dim=1) / 127          [out, 1]  (one scale per output row)
    W_q   = round(W / scale)               [out, in] int8

Forward dequantizes on the fly:  y = x @ (W_q * scale).T + b

Why weight-only: decode is bandwidth-bound, so halving weight bytes is a direct
speedup. Activations stay in fp16, so the only error is rounding in the weights.
Why per-row: a single scale per tensor is dominated by the largest weight and
rounds everything else coarsely; per-row scales keep each output's precision.
"""

import torch
import torch.nn as nn


class QuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        W = linear.weight.data.float()
        scale = W.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
        self.register_buffer("w_q", torch.round(W / scale).to(torch.int8))
        self.register_buffer("scale", scale.to(linear.weight.dtype))
        self.bias = linear.bias
        self.out_features, self.in_features = W.shape

    def forward(self, x):
        W = self.w_q.to(x.dtype) * self.scale.to(x.dtype)
        return nn.functional.linear(x, W, self.bias)


def quantize_model(model, skip=("embed_tokens", "lm_head")):
    """Replace every nn.Linear in the 24 blocks with QuantLinear. Embedding/unembed stay as-is."""
    n = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and not any(s in full for s in skip):
                setattr(module, child_name, QuantLinear(child))
                n += 1
    return n


def weight_bytes(model):
    total = 0
    for p in model.parameters():
        total += p.numel() * p.element_size()
    for b in model.buffers():
        total += b.numel() * b.element_size()
    return total
