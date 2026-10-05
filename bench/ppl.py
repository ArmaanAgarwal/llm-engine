"""
Perplexity on WikiText-2 for fp16 vs INT8.

Perplexity = exp(mean negative log-likelihood of the true next token).
Lower is better; it measures how "surprised" the model is by real text.
Same 512-token windows for both precisions, so the only difference is quantization.

Usage:  python -m bench.ppl [--windows 40]
Output: bench/ppl.csv
"""

import argparse, csv, math, os
import torch

REPO = "Qwen/Qwen2.5-0.5B-Instruct"


@torch.no_grad()
def perplexity(model, ids, window=512, n_windows=40):
    nlls = []
    for i in range(n_windows):
        s = i * window
        chunk = ids[:, s:s + window + 1]
        if chunk.shape[1] < window + 1: break
        x, y = chunk[:, :-1], chunk[:, 1:]
        logits = model(x).float()
        nll = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1), reduction="mean")
        nlls.append(nll.item())
    return math.exp(sum(nlls) / len(nlls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=40)
    ap.add_argument("--out", default="bench/ppl.csv")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from engine.model import from_pretrained
    from engine.quant import quantize_model, weight_bytes

    tok = AutoTokenizer.from_pretrained(REPO)
    text = "\n\n".join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids.cuda()

    rows = []
    m = from_pretrained(device="cuda", dtype=torch.float16)
    p16 = perplexity(m, ids, n_windows=a.windows); b16 = weight_bytes(m)
    print(f"fp16  ppl={p16:.3f}  weights={b16/1e6:.0f} MB"); rows.append({"precision": "fp16", "perplexity": round(p16, 3), "weight_mb": round(b16/1e6)})

    quantize_model(m)
    p8 = perplexity(m, ids, n_windows=a.windows); b8 = weight_bytes(m)
    print(f"int8  ppl={p8:.3f}  weights={b8/1e6:.0f} MB  (delta {p8-p16:+.3f})"); rows.append({"precision": "int8", "perplexity": round(p8, 3), "weight_mb": round(b8/1e6)})

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["precision", "perplexity", "weight_mb"]); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
