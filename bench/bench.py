"""
Benchmark: tokens/sec for each engine configuration at batch sizes 1, 8, 32.

Configurations:
    hf_fp16        Hugging Face generate(), fp16                 (baseline)
    ours_nocache   our model, fp32, full recompute every step    (Day 1)
    ours_cache     our model, fp16, KV cache                     (Day 2)
    ours_int8      + INT8 weight-only quantization               (Day 3)
    ours_fused     + fused RMSNorm CUDA kernel                   (Day 3)
    vllm           vLLM, fp16                                    (if installed)

Method: same prompts every run, 1 untimed warmup, then median of N timed runs of
NEW_TOKENS decode steps with cuda.synchronize() around the timer. Prefill is
timed separately and excluded from the decode tok/s.

Usage:  python -m bench.bench [--runs 5] [--new-tokens 128] [--skip hf,vllm]
Output: bench/results.csv
"""

import argparse, csv, os, statistics, sys, time
import torch

REPO = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "The history of the Roman Empire begins with",
    "def quicksort(arr):",
    "In machine learning, gradient descent is",
    "The three laws of thermodynamics state that",
    "Once upon a time in a small village,",
    "The main differences between TCP and UDP are",
    "To bake sourdough bread, first you need",
    "Photosynthesis is the process by which",
] * 4  # 32 prompts


def timed(fn):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize(); return out, time.perf_counter() - t0


def bench_ours(model, tok, batch, new_tokens, runs):
    from engine.model import KVCache
    enc = tok(PROMPTS[:batch], return_tensors="pt", padding=True).to("cuda")
    ids, am = enc.input_ids, enc.attention_mask
    p = next(model.parameters())
    decode_times, prefill_times = [], []
    for r in range(runs + 1):
        cache = KVCache(model.cfg, batch, ids.shape[1] + new_tokens, p.device, p.dtype)
        _, tp = timed(lambda: model.prefill(ids, cache, am))
        nxt = torch.zeros(batch, 1, dtype=torch.long, device="cuda")
        def run():
            n = nxt
            for _ in range(new_tokens):
                n = model.decode(n, cache)[:, -1].argmax(-1, keepdim=True)
        _, td = timed(run)
        if r > 0:
            prefill_times.append(tp); decode_times.append(td)
    td = statistics.median(decode_times)
    return batch * new_tokens / td, statistics.median(prefill_times)


def bench_ours_graph(model, tok, batch, new_tokens, runs):
    from engine.graph import GraphDecoder
    enc = tok(PROMPTS[:batch], return_tensors="pt", padding=True).to("cuda")
    ids, am = enc.input_ids, enc.attention_mask
    dec = GraphDecoder(model, batch, ids.shape[1] + new_tokens + 8)
    # capture once
    dec.prefill(ids, am); nxt = torch.zeros(batch, 1, dtype=torch.long, device="cuda"); dec.tok.copy_(nxt)
    dec.mask_f[:, 0, 0].scatter_(1, dec.pos_t.expand(batch, 1), 0.0); dec.capture()
    times = []
    for r in range(runs + 1):
        dec.prefill(ids, am)
        def run():
            n = nxt
            for _ in range(new_tokens):
                n = dec.step(n)[:, -1].argmax(-1, keepdim=True)
        _, td = timed(run)
        if r > 0: times.append(td)
    return batch * new_tokens / statistics.median(times), float("nan")


def bench_ours_nocache(model, tok, batch, new_tokens, runs):
    enc = tok(PROMPTS[:batch], return_tensors="pt", padding=True).to("cuda")
    ids = enc.input_ids
    times = []
    nt = min(new_tokens, 32)   # quadratic; keep it short
    for r in range(runs + 1):
        _, t = timed(lambda: model.generate(ids, nt, use_cache=False))
        if r > 0: times.append(t)
    return batch * nt / statistics.median(times), float("nan")


def bench_hf(tok, batch, new_tokens, runs):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(REPO, torch_dtype=torch.float16).cuda().eval()
    enc = tok(PROMPTS[:batch], return_tensors="pt", padding=True).to("cuda")
    times = []
    for r in range(runs + 1):
        _, t = timed(lambda: m.generate(**enc, max_new_tokens=new_tokens, min_new_tokens=new_tokens, do_sample=False, repetition_penalty=1.0, temperature=None, top_p=None, top_k=None, pad_token_id=tok.pad_token_id))
        if r > 0: times.append(t)
    del m; torch.cuda.empty_cache()
    return batch * new_tokens / statistics.median(times), float("nan")


def bench_vllm(batch, new_tokens, runs):
    from vllm import LLM, SamplingParams
    llm = LLM(REPO, dtype="float16", gpu_memory_utilization=0.6, enforce_eager=False)
    sp = SamplingParams(temperature=0, max_tokens=new_tokens, min_tokens=new_tokens)
    times = []
    for r in range(runs + 1):
        t0 = time.perf_counter(); llm.generate(PROMPTS[:batch], sp, use_tqdm=False); t = time.perf_counter() - t0
        if r > 0: times.append(t)
    return batch * new_tokens / statistics.median(times), float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--skip", default="")
    ap.add_argument("--out", default="bench/results.csv")
    a = ap.parse_args()
    skip = set(a.skip.split(",")) if a.skip else set()
    batches = [int(b) for b in a.batches.split(",")]

    from transformers import AutoTokenizer
    from engine.model import from_pretrained
    from engine.quant import quantize_model
    from engine import kernels
    tok = AutoTokenizer.from_pretrained(REPO); tok.padding_side = "left"
    rows = []

    def record(name, batch, tps, prefill):
        rows.append({"config": name, "batch": batch, "tok_per_s": round(tps, 1), "prefill_s": round(prefill, 4) if prefill == prefill else ""})
        print(f"{name:14s} batch={batch:3d}  {tps:9.1f} tok/s", flush=True)

    if "hf" not in skip:
        for b in batches: record("hf_fp16", b, *bench_hf(tok, b, a.new_tokens, a.runs))

    if "nocache" not in skip:
        m = from_pretrained(device="cuda", dtype=torch.float32)
        for b in batches: record("ours_nocache", b, *bench_ours_nocache(m, tok, b, a.new_tokens, a.runs))
        del m; torch.cuda.empty_cache()

    m = from_pretrained(device="cuda", dtype=torch.float16)
    for b in batches: record("ours_cache", b, *bench_ours(m, tok, b, a.new_tokens, a.runs))

    if "graph" not in skip:
        try:
            for b in batches: record("ours_graph", b, *bench_ours_graph(m, tok, b, a.new_tokens, a.runs))
        except Exception as e:
            print("cuda graph skipped:", e)

    if "int8" not in skip:
        quantize_model(m)
        for b in batches: record("ours_int8", b, *bench_ours(m, tok, b, a.new_tokens, a.runs))

    if "fused" not in skip:
        try:
            kernels.build(); kernels.USE_FUSED = True
            for b in batches: record("ours_fused", b, *bench_ours(m, tok, b, a.new_tokens, a.runs))
        except Exception as e:
            print("fused kernel skipped:", e)
        finally:
            kernels.USE_FUSED = False
    del m; torch.cuda.empty_cache()

    if "vllm" not in skip:
        try:
            for b in batches: record("vllm", b, *bench_vllm(b, a.new_tokens, a.runs))
        except Exception as e:
            print("vllm skipped:", e)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "batch", "tok_per_s", "prefill_s"]); w.writeheader(); w.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
