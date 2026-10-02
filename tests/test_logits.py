"""
The Day 1 gate. Nothing moves on until this passes.

Two checks:
  1. Logits: our model vs Hugging Face on the same prompts. Max abs diff < tol.
  2. Greedy decode: same 30 tokens, exactly.

Run on Colab:   python -m tests.test_logits
Run offline (architecture only, random weights, tiny config):
                python -m tests.test_logits --tiny
"""

import argparse
import torch
from engine.model import Config, Model, load_from_hf_state_dict

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In 1969, humans first",
    "The three primary colors are",
    "Water boils at",
]


def compare(ours, theirs, tok, device, tol, n_new=30):
    ours.eval(); theirs.eval()
    worst = 0.0
    for p in PROMPTS:
        ids = tok(p, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            a = ours(ids).float()
            b = theirs(ids).logits.float()
        d = (a - b).abs().max().item()
        worst = max(worst, d)
        print(f"{d:.5f}  {p!r}")
    print(f"max abs logit diff: {worst:.5f}  (tol {tol})")
    assert worst < tol, "logits do not match"

    # Greedy decode check
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        a = ours.generate(ids, n_new)
        b = theirs.generate(ids, max_new_tokens=n_new, do_sample=False)
    assert torch.equal(a, b), f"greedy mismatch:\n ours:   {tok.decode(a[0])}\n theirs: {tok.decode(b[0])}"
    print("greedy decode matches:", repr(tok.decode(a[0][ids.shape[1]:])))
    print("PASS")


def main_real(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from engine.model import from_pretrained
    repo = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(repo)
    theirs = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float32).to(device)
    ours = from_pretrained(repo, device=device, dtype=torch.float32)
    compare(ours, theirs, tok, device, tol=1e-2)


def main_tiny(device):
    """No download. Build a tiny random Qwen2 in transformers, copy weights into ours, compare."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    hc = Qwen2Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        rope_theta=10000.0, rms_norm_eps=1e-6, tie_word_embeddings=True,
    )
    theirs = Qwen2ForCausalLM(hc).to(device).eval()
    cfg = Config(vocab_size=512, hidden=64, n_layers=2, n_heads=4, n_kv_heads=2,
                 intermediate=128, rope_theta=10000.0, rms_eps=1e-6, tie_embeddings=True)
    ours = Model(cfg).to(device).eval()
    load_from_hf_state_dict(ours, theirs.state_dict())

    class FakeTok:
        def __call__(self, p, return_tensors=None):
            g = torch.Generator().manual_seed(abs(hash(p)) % (2**31))
            ids = torch.randint(0, 512, (1, 12), generator=g)
            class R: pass
            r = R(); r.input_ids = ids; return r
        def decode(self, ids): return str(ids.tolist())

    compare(ours, theirs, FakeTok(), device, tol=1e-4, n_new=10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    (main_tiny if args.tiny else main_real)(device)
