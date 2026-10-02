"""
Qwen2.5-style transformer, written from scratch for inference.

Day 1 version: no KV cache, no batching tricks. Every forward call
recomputes everything. Slow on purpose. Correctness first.

The architecture (Llama/Qwen2 family), per layer:

    x ──► RMSNorm ──► Attention ──► + ──► RMSNorm ──► MLP ──► + ──► next layer
    │                              ▲ │                        ▲
    └──────────────────────────────┘ └────────────────────────┘
              residual                        residual

Shapes for Qwen2.5-0.5B:
    hidden (d_model)   = 896
    layers             = 24
    query heads        = 14   (head_dim = 896 / 14 = 64)
    key/value heads    = 2    (grouped-query attention: 7 query heads share each KV head)
    MLP intermediate   = 4864
    vocab              = 151,936
    rope theta         = 1,000,000
    rms eps            = 1e-6
    tie_word_embeddings = True  (the output projection reuses the embedding matrix)
"""

from dataclasses import dataclass
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 151936
    hidden: int = 896
    n_layers: int = 24
    n_heads: int = 14
    n_kv_heads: int = 2
    intermediate: int = 4864
    rope_theta: float = 1_000_000.0
    rms_eps: float = 1e-6
    tie_embeddings: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden // self.n_heads


# ---------------------------------------------------------------------------
# 1. RMSNorm
#
# LayerNorm subtracts the mean and divides by std. RMSNorm skips the mean:
#     y = x / sqrt(mean(x^2) + eps) * weight
# Cheaper, and it's what every Llama-family model uses.
# Note: we compute in float32 even if x is float16. Summing 896 squares in
# fp16 overflows. This matters on Day 2 when we switch to half precision.
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(you): implement.  ~3 lines.
        #   1. xf = x.float()
        #   2. rms = sqrt(mean(xf**2, last dim, keepdim) + eps)
        #   3. return (xf / rms).to(x.dtype) * weight
        raise NotImplementedError("RMSNorm.forward")


# ---------------------------------------------------------------------------
# 2. Rotary position embedding (RoPE)
#
# The model has no idea what order tokens are in unless we tell it.
# RoPE tells it by ROTATING each (q, k) vector by an angle that depends on
# the token's position. Pairs of dimensions (0,32), (1,33), ... each get
# their own rotation frequency, so position shows up as a pattern of
# rotations across the vector.
#
# Why rotate instead of add? Because the dot product q·k then depends only
# on the DIFFERENCE in positions, which is what attention needs.
#
# Implementation: precompute cos/sin tables of shape [max_len, head_dim],
# then apply:  x_rot = x * cos + rotate_half(x) * sin
# where rotate_half swaps the two halves and negates one:
#     [x1, x2] -> [-x2, x1]
# This is the HF convention. Keep it; it has to match HF's weights.
# ---------------------------------------------------------------------------
def rope_tables(head_dim: int, max_len: int, theta: float, device, dtype):
    # inv_freq[i] = 1 / theta^(2i / head_dim)   for i in [0, head_dim/2)
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)                  # [max_len, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)             # [max_len, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x:   [batch, heads, seq, head_dim]
    # cos: [seq, head_dim]  -> broadcast over batch and heads
    return x * cos[None, None] + rotate_half(x) * sin[None, None]


# ---------------------------------------------------------------------------
# 3. Attention (grouped-query)
#
# Each token builds a query q, a key k, a value v. Token i attends to every
# token j <= i (causal), weighted by softmax(q_i · k_j / sqrt(d)).
#
# Grouped-query: 14 query heads but only 2 KV heads. Query heads 0-6 share
# KV head 0; heads 7-13 share KV head 1. We expand K and V with
# repeat_interleave so the shapes line up. The point of GQA is that the
# KV cache (Day 2) is 7x smaller than it would be with 14 KV heads.
#
# Qwen2 quirk: q, k, v projections HAVE bias; o_proj does NOT.
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden, cfg.n_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden, cfg.n_kv_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden, cfg.n_kv_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.hidden, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        # Project, then split the last dim into (heads, head_dim), then put heads before seq.
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)     # [B, H, T, D]
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)  # [B, KV, T, D]
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # GQA: expand KV heads to match query heads.
        rep = self.n_heads // self.n_kv_heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        # Causal attention. This one call is softmax(q k^T / sqrt(d)) v with a
        # lower-triangular mask. We'll write our own kernel for RMSNorm, not this.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, T, D]

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


# ---------------------------------------------------------------------------
# 4. MLP (SwiGLU)
#
#     y = down( silu(gate(x)) * up(x) )
#
# Two parallel projections up to 4864 dims, one gated by SiLU, multiplied,
# projected back down to 896. This is where most of the parameters live
# and most of the "knowledge" is stored.
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.up_proj = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate, cfg.hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO(you): implement.  1 line.
        #   return down_proj( silu(gate_proj(x)) * up_proj(x) )
        raise NotImplementedError("MLP.forward")


# ---------------------------------------------------------------------------
# 5. One transformer block = pre-norm attention + pre-norm MLP, each with a
#    residual connection.
# ---------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden, cfg.rms_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden, cfg.rms_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# 6. The model
# ---------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden, cfg.rms_eps)
        if cfg.tie_embeddings:
            self.lm_head = None  # reuse embed_tokens.weight
        else:
            self.lm_head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        self._rope_cache = None

    def _rope(self, T: int, device, dtype):
        if self._rope_cache is None or self._rope_cache[0].shape[0] < T or self._rope_cache[0].device != device:
            self._rope_cache = rope_tables(self.cfg.head_dim, max(T, 2048), self.cfg.rope_theta, device, dtype)
        cos, sin = self._rope_cache
        return cos[:T], sin[:T]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: [B, T] -> logits: [B, T, vocab]"""
        x = self.embed_tokens(input_ids)
        cos, sin = self._rope(x.shape[1], x.device, x.dtype)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.norm(x)
        if self.lm_head is None:
            return x @ self.embed_tokens.weight.T
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy decode. Day 1: recompute the whole sequence every step (no cache)."""
        ids = input_ids
        for _ in range(max_new_tokens):
            logits = self(ids)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
        return ids


# ---------------------------------------------------------------------------
# 7. Loading Hugging Face weights
#
# HF names look like:  model.layers.3.self_attn.q_proj.weight
# Ours look like:               layers.3.self_attn.q_proj.weight
# So: strip the "model." prefix. Everything else was named to match.
# ---------------------------------------------------------------------------
def load_from_hf_state_dict(model: Model, sd: dict) -> None:
    ours = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k == "lm_head.weight" and model.lm_head is None:
            continue  # tied; we use embed_tokens.weight
        ours[k] = v
    missing, unexpected = model.load_state_dict(ours, strict=False)
    missing = [m for m in missing]  # list for printing
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert not missing, f"missing keys: {missing[:5]}"


def from_pretrained(repo: str = "Qwen/Qwen2.5-0.5B-Instruct", device="cuda", dtype=torch.float32) -> Model:
    """Download HF weights and load them into our Model."""
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    import json

    cfg_path = hf_hub_download(repo, "config.json")
    hc = json.load(open(cfg_path))
    cfg = Config(
        vocab_size=hc["vocab_size"],
        hidden=hc["hidden_size"],
        n_layers=hc["num_hidden_layers"],
        n_heads=hc["num_attention_heads"],
        n_kv_heads=hc["num_key_value_heads"],
        intermediate=hc["intermediate_size"],
        rope_theta=hc["rope_theta"],
        rms_eps=hc["rms_norm_eps"],
        tie_embeddings=hc.get("tie_word_embeddings", True),
    )
    model = Model(cfg)
    sd = load_file(hf_hub_download(repo, "model.safetensors"))
    load_from_hf_state_dict(model, sd)
    return model.to(device=device, dtype=dtype).eval()
