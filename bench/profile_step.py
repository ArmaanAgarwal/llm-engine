import sys, torch
from engine.model import from_pretrained, KVCache
from engine import kernels
fused = sys.argv[1] == 'fused'
m = from_pretrained(device='cuda', dtype=torch.float16)
if fused: kernels.build(); kernels.USE_FUSED = True
ids = torch.randint(0, 1000, (1, 16), device='cuda')
cache = KVCache(m.cfg, 1, 64, 'cuda', torch.float16)
m.prefill(ids, cache)
tok = ids[:, -1:]
for _ in range(5): tok = m.decode(tok, cache)[:, -1].argmax(-1, keepdim=True)   # warmup
torch.cuda.synchronize()
torch.cuda.nvtx.range_push('decode_steps')
for _ in range(20): tok = m.decode(tok, cache)[:, -1].argmax(-1, keepdim=True)
torch.cuda.synchronize(); torch.cuda.nvtx.range_pop()
print('done', 'fused' if fused else 'pytorch')
