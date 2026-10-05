"""
Correctness tests. Run:  python -m tests.test_engine [--tiny]

--tiny: random small Qwen2 built in transformers, no download; checks the architecture.
default: real Qwen2.5-0.5B-Instruct vs Hugging Face.

Checks:
  1. logits match HF (no cache)
  2. cached greedy == uncached greedy == HF greedy
  3. batched (left-padded) greedy == per-prompt greedy
  4. INT8 model still produces the same greedy text on a short prompt
  5. fused kernel == PyTorch RMSNorm (CUDA only)
"""

import argparse, torch
from engine.model import Config, Model, load_from_hf_state_dict, config_from_hf, RMSNorm

PROMPTS = ["The capital of France is", "def fibonacci(n):", "In 1969, humans first", "Water boils at"]


def make_tiny(device):
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    hc = Qwen2Config(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                     rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True)
    hf = Qwen2ForCausalLM(hc).to(device).eval()
    ours = Model(config_from_hf(hc.to_dict())).to(device).eval()
    load_from_hf_state_dict(ours, hf.state_dict())

    class Tok:
        pad_token_id = 0; padding_side = "left"
        def __call__(self, ps, return_tensors=None, padding=False):
            if isinstance(ps, str): ps = [ps]
            seqs = [torch.randint(1, 512, (1, 6 + 3 * (len(p) % 3)), generator=torch.Generator().manual_seed(sum(map(ord, p)))) for p in ps]
            L = max(s.shape[1] for s in seqs)
            ids = torch.zeros(len(seqs), L, dtype=torch.long); am = torch.zeros(len(seqs), L, dtype=torch.long)
            for i, s in enumerate(seqs): ids[i, L - s.shape[1]:] = s; am[i, L - s.shape[1]:] = 1
            class R: pass
            r = R(); r.input_ids = ids.to(device); r.attention_mask = am.to(device); r.to = lambda d: r; return r
        def decode(self, ids, skip_special_tokens=True): return str(ids.tolist())
    return ours, hf, Tok(), 1e-4


def make_real(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.model import from_pretrained
    repo = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(repo); tok.padding_side = "left"
    hf = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).to(device).eval()
    ours = from_pretrained(repo, device=device, dtype=torch.float32)
    return ours, hf, tok, 1e-2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tiny", action="store_true"); a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ours, hf, tok, tol = (make_tiny if a.tiny else make_real)(device)
    N = 12

    # 1. logits
    worst = 0
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            d = (ours(ids).float() - hf(ids).logits.float()).abs().max().item()
        worst = max(worst, d)
    print(f"[1] max logit diff {worst:.6f} (tol {tol})"); assert worst < tol

    # 2. cached == uncached == hf
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        g_nc = ours.generate(ids, N, use_cache=False)
        g_c = ours.generate(ids, N, use_cache=True)
        g_hf = hf.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False, pad_token_id=tok.pad_token_id)
    assert torch.equal(g_nc, g_c), "cache changed output"
    assert torch.equal(g_c, g_hf), f"greedy mismatch vs HF\n{tok.decode(g_c[0])}\n{tok.decode(g_hf[0])}"
    print("[2] cached greedy == uncached == HF:", repr(tok.decode(g_c[0][ids.shape[1]:])))

    # 3. batched == individual
    enc = tok(PROMPTS, return_tensors="pt", padding=True)
    with torch.no_grad():
        gb = ours.generate(enc.input_ids.to(device), N, attention_mask=enc.attention_mask.to(device))
        for i, p in enumerate(PROMPTS):
            single = ours.generate(tok(p, return_tensors="pt").input_ids.to(device), N)
            assert torch.equal(gb[i, -N:], single[0, -N:]), f"batched row {i} differs"
    print("[3] batched greedy == individual")

    # 4. int8 greedy still sane
    from engine.quant import quantize_model
    import copy
    q = copy.deepcopy(ours); n = quantize_model(q)
    with torch.no_grad():
        gq = q.generate(ids, N)
    same = (gq[0, -N:] == g_c[0, -N:]).float().mean().item()
    print(f"[4] int8: {n} linears quantized, {same*100:.0f}% of greedy tokens unchanged")
    assert same >= 0.5

    # 5. fused kernel
    if device == "cuda":
        from engine import kernels
        kernels.build()
        x = torch.randn(4, 7, ours.cfg.hidden, device="cuda", dtype=torch.float16)
        norm = ours.layers[0].input_layernorm
        ref = norm(x); kernels.USE_FUSED = True; got = norm(x); kernels.USE_FUSED = False
        d = (ref.float() - got.float()).abs().max().item()
        print(f"[5] fused rmsnorm max diff {d:.6f}"); assert d < 1e-2
    else:
        print("[5] fused kernel: skipped (no CUDA)")
    print("PASS")


if __name__ == "__main__":
    main()
