"""
Qwen2.5-style transformer for inference, written from scratch.

Pipeline:
    tokens → embedding → [Block × 24] → RMSNorm → scores over vocab

Each Block:
    x = x + Attention(RMSNorm(x))      # tokens exchange information
    x = x + MLP(RMSNorm(x))            # each token applies stored knowledge

Supports:
    - full recompute (no cache)        model(ids)
    - KV cache: prefill + decode       model.prefill(ids, cache); model.decode(tok, cache)
    - batching with left-padding       any batch size; pass attention_mask
    - optional fused RMSNorm kernel    set engine.kernels.USE_FUSED = True after building csrc/
"""

from dataclasses import dataclass
import json
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


# ----------------------------------------------------------------------------
# KV cache: preallocated buffers, written in place at position t.
# Shape per layer: [batch, kv_heads, max_len, head_dim]
# ----------------------------------------------------------------------------
class KVCache:
    def __init__(self, cfg: Config, batch: int, max_len: int, device, dtype):
        shape = (cfg.n_layers, batch, cfg.n_kv_heads, max_len, cfg.head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.pos = 0            # number of positions filled so far
        self.max_len = max_len

    def update(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """k, v: [batch, kv_heads, T, head_dim] for positions [pos, pos+T). Returns full k, v up to pos+T."""
        T = k.shape[2]
        self.k[layer, :, :, self.pos:self.pos + T] = k
        self.v[layer, :, :, self.pos:self.pos + T] = v
        return self.k[layer, :, :, :self.pos + T], self.v[layer, :, :, :self.pos + T]


# ----------------------------------------------------------------------------
# RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight.  Computed in fp32.
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from . import kernels
        if kernels.USE_FUSED and x.is_cuda:
            return kernels.rmsnorm(x, self.weight, self.eps)
        xf = x.float()
        rms = torch.sqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf / rms).to(x.dtype) * self.weight


# ----------------------------------------------------------------------------
# RoPE: rotate pairs (i, i+32) of each 64-dim head vector by position*rate_i.
# ----------------------------------------------------------------------------
def rope_tables(head_dim: int, max_len: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(max_len, device=device).float()
    freqs = torch.outer(pos, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: [B, H, T, D]; cos/sin: [T, D] or [B, T, D]
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[:, None], sin[:, None]
    return x * cos + rotate_half(x) * sin


# ----------------------------------------------------------------------------
# Attention with grouped-query heads. q/k/v have bias, o has none (Qwen2).
# ----------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.head_dim = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden, cfg.n_heads * cfg.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden, cfg.n_kv_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden, cfg.n_kv_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.hidden, bias=False)

    def forward(self, x, cos, sin, cache=None, layer=0, attn_mask=None):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer, k, v)        # [B, KV, pos+T, D]

        rep = self.n_heads // self.n_kv_heads
        k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)

        if attn_mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=(cache is None or T > 1))
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        out = out.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(out)


# ----------------------------------------------------------------------------
# MLP (SwiGLU): down( silu(gate(x)) * up(x) )
# ----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.up_proj = nn.Linear(cfg.hidden, cfg.intermediate, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate, cfg.hidden, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden, cfg.rms_eps)
        self.self_attn = Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden, cfg.rms_eps)
        self.mlp = MLP(cfg)

    def forward(self, x, cos, sin, cache=None, layer=0, attn_mask=None):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cache, layer, attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.hidden, cfg.rms_eps)
        self.lm_head = None if cfg.tie_embeddings else nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)
        self._rope = None

    def rope(self, device, dtype, max_len=4096):
        if self._rope is None or self._rope[0].device != device or self._rope[0].dtype != dtype or self._rope[0].shape[0] < max_len:
            self._rope = rope_tables(self.cfg.head_dim, max_len, self.cfg.rope_theta, device, dtype)
        return self._rope

    def unembed(self, x):
        return x @ self.embed_tokens.weight.T if self.lm_head is None else self.lm_head(x)

    def _run(self, ids, positions, cache=None, attn_mask=None):
        """ids: [B, T]; positions: [B, T] or [T]."""
        x = self.embed_tokens(ids)
        cos_all, sin_all = self.rope(x.device, x.dtype)
        cos, sin = cos_all[positions], sin_all[positions]
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cache, i, attn_mask)
        return self.unembed(self.norm(x))

    # --- No cache: full recompute. Used for correctness testing. ---
    def forward(self, ids):
        T = ids.shape[1]
        positions = torch.arange(T, device=ids.device)
        return self._run(ids, positions)

    # --- With cache ---
    @torch.no_grad()
    def prefill(self, ids, cache, attention_mask=None):
        """ids: [B, T] left-padded. attention_mask: [B, T] with 1 for real tokens. Returns logits [B, T, V]."""
        B, T = ids.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=ids.device, dtype=torch.long)
        # positions: real tokens count from 0; pads get 0
        positions = (attention_mask.cumsum(-1) - 1).clamp(min=0)
        # mask: causal AND key is a real token
        causal = torch.tril(torch.ones(T, T, device=ids.device, dtype=torch.bool))
        key_ok = attention_mask.bool()[:, None, None, :]          # [B,1,1,T]
        mask = causal[None, None] & key_ok                        # [B,1,T,T]
        logits = self._run(ids, positions, cache, mask)
        cache.pos = T
        cache.attention_mask = attention_mask
        return logits

    @torch.no_grad()
    def decode(self, tok, cache):
        """tok: [B, 1]. Returns logits [B, 1, V]."""
        B = tok.shape[0]
        am = torch.cat([cache.attention_mask, torch.ones(B, 1, device=tok.device, dtype=torch.long)], dim=1)
        positions = (am.sum(-1, keepdim=True) - 1)               # [B,1]
        mask = am.bool()[:, None, None, :]                        # [B,1,1,pos+1]
        logits = self._run(tok, positions, cache, mask)
        cache.pos += 1
        cache.attention_mask = am
        return logits

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, attention_mask=None, use_cache=True, eos_id=None):
        """Greedy decode. ids: [B, T] left-padded. Returns [B, T+max_new_tokens]."""
        B, T = ids.shape
        if not use_cache:
            out = ids
            for _ in range(max_new_tokens):
                nxt = self(out)[:, -1].argmax(-1, keepdim=True)
                out = torch.cat([out, nxt], dim=1)
            return out

        p = next(self.parameters())
        cache = KVCache(self.cfg, B, T + max_new_tokens, p.device, p.dtype)
        logits = self.prefill(ids, cache, attention_mask)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out = [nxt]
        done = torch.zeros(B, dtype=torch.bool, device=ids.device)
        for _ in range(max_new_tokens - 1):
            if eos_id is not None:
                done |= nxt[:, 0] == eos_id
                if done.all():
                    break
            logits = self.decode(nxt, cache)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt)
        return torch.cat([ids] + out, dim=1)


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_from_hf_state_dict(model: Model, sd: dict) -> None:
    ours = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[len("model."):]
        if k == "lm_head.weight" and model.lm_head is None:
            continue
        ours[k] = v
    missing, unexpected = model.load_state_dict(ours, strict=False)
    assert not unexpected, f"unexpected keys: {list(unexpected)[:5]}"
    assert not missing, f"missing keys: {list(missing)[:5]}"


def config_from_hf(hc: dict) -> Config:
    return Config(
        vocab_size=hc["vocab_size"], hidden=hc["hidden_size"], n_layers=hc["num_hidden_layers"],
        n_heads=hc["num_attention_heads"], n_kv_heads=hc["num_key_value_heads"],
        intermediate=hc["intermediate_size"],
        rope_theta=hc.get("rope_theta") or hc.get("rope_parameters", {}).get("rope_theta", 1_000_000.0),
        rms_eps=hc["rms_norm_eps"], tie_embeddings=hc.get("tie_word_embeddings", True),
    )


def from_pretrained(repo="Qwen/Qwen2.5-0.5B-Instruct", device="cuda", dtype=torch.float32) -> Model:
    from safetensors.torch import load_file
    from huggingface_hub import hf_hub_download
    cfg = config_from_hf(json.load(open(hf_hub_download(repo, "config.json"))))
    model = Model(cfg)
    load_from_hf_state_dict(model, load_file(hf_hub_download(repo, "model.safetensors")))
    return model.to(device=device, dtype=dtype).eval()
