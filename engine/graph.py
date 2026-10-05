"""
CUDA-graph decode.

At batch 1 a decode step is ~300 small kernel launches; the GPU finishes each in
microseconds and then waits for Python to launch the next. A CUDA graph records the
whole step once and replays it as a single launch, removing that overhead entirely.

Requirements: every tensor the step touches must have a fixed address and shape.
So we preallocate the token, position, and mask buffers once, write the cache via
index_copy_ at a device-side position, and attend over the full cache with a float
mask that hides the unfilled slots. Between replays we update the static buffers
in place (new token, open one more mask slot, pos += 1) and replay.

    dec = GraphDecoder(model, batch=1, max_len=256)
    logits = dec.prefill(ids, attention_mask)      # eager prefill fills the cache
    for _ in range(n): tok = dec.step(tok)         # each step is one graph replay
"""

import torch
from .model import KVCache


class GraphDecoder:
    def __init__(self, model, batch: int, max_len: int):
        self.m = model
        p = next(model.parameters())
        self.device, self.dtype = p.device, p.dtype
        self.B, self.L = batch, max_len
        self.cache = KVCache(model.cfg, batch, max_len, self.device, self.dtype)
        # static buffers
        self.tok = torch.zeros(batch, 1, dtype=torch.long, device=self.device)
        self.pos_t = torch.zeros(1, dtype=torch.long, device=self.device)
        self.mask_f = torch.full((batch, 1, 1, max_len), float("-inf"), device=self.device, dtype=self.dtype)
        self.logits = None
        self.graph = None

    @torch.no_grad()
    def prefill(self, ids, attention_mask=None):
        B, T = ids.shape
        assert B == self.B and T < self.L
        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long, device=self.device)
        logits = self.m.prefill(ids, self.cache, attention_mask)
        # open mask slots for the real tokens, keep pads at -inf
        self.mask_f.fill_(float("-inf"))
        self.mask_f[:, 0, 0, :T] = torch.where(attention_mask.bool(), 0.0, float("-inf")).to(self.dtype)
        self.pos_t.fill_(T)
        return logits

    def _step_eager(self):
        return self.m.decode_static(self.tok, self.cache, self.mask_f, self.pos_t)

    @torch.no_grad()
    def capture(self):
        """Warm up on a side stream, then capture one decode step into a graph."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                self._step_eager()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.logits = self._step_eager()

    @torch.no_grad()
    def step(self, tok):
        """tok: [B,1] long. Returns logits [B,1,V]. Advances the cache by one position."""
        self.tok.copy_(tok)
        self.mask_f[:, 0, 0].scatter_(1, self.pos_t.expand(self.B, 1), 0.0)   # open this slot
        if self.graph is None:
            out = self._step_eager()
        else:
            self.graph.replay()
            out = self.logits
        self.pos_t += 1
        return out

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, attention_mask=None):
        logits = self.prefill(ids, attention_mask)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        out = [nxt]
        if self.graph is None and self.device.type == "cuda":
            # capture on the first real step so the warmup writes land at the right position
            saved_pos = self.pos_t.clone(); saved_mask = self.mask_f.clone()
            saved_k = self.cache.k.clone(); saved_v = self.cache.v.clone()
            self.tok.copy_(nxt)
            self.mask_f[:, 0, 0].scatter_(1, self.pos_t.expand(self.B, 1), 0.0)
            self.capture()
            self.pos_t.copy_(saved_pos); self.mask_f.copy_(saved_mask)
            self.cache.k.copy_(saved_k); self.cache.v.copy_(saved_v)
            del saved_k, saved_v
        for _ in range(max_new_tokens - 1):
            logits = self.step(nxt)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            out.append(nxt)
        return torch.cat([ids] + out, dim=1)
