# llm-engine

A from-scratch inference engine for Qwen2.5-0.5B-Instruct, built to measure where decode time goes and how far each optimization moves it.

## The problem

Generating one token requires reading every weight in the model from GPU memory (~1 GB in fp16 for this model). On a T4 (320 GB/s) that is ~3 ms per token from bandwidth alone, while the arithmetic takes ~15 µs. Decode is **memory-bandwidth-bound**: the GPU's compute units idle >95% of the time waiting on weights. Every optimization here either reads fewer bytes per token or produces more tokens per read.

## What's here

| Step | What | Where |
|---|---|---|
| Forward pass | Embedding, RMSNorm, RoPE, grouped-query attention, SwiGLU MLP, 24 layers, tied unembedding — written by hand, verified against Hugging Face to <1e-2 logit error | `engine/model.py` |
| KV cache | Preallocated per-layer K/V buffers written in place; `prefill` + `decode` split. Turns generation from quadratic to linear | `engine/model.py` (`KVCache`) |
| Batching | Left-padded batches with attention masks and per-row positions; one weight read serves N sequences | `engine/model.py` (`prefill`/`decode`) |
| INT8 | Weight-only, per-output-row scale, dequantized on the fly. Halves weight bytes | `engine/quant.py` |
| Fused RMSNorm | One CUDA kernel (warp-shuffle reduction) replacing three PyTorch ops: one HBM read + one write instead of three | `csrc/rmsnorm.cu`, `engine/kernels.py` |
| Benchmarks | Decode tok/s per configuration × batch size, median of N runs | `bench/bench.py` |
| Perplexity | WikiText-2, fp16 vs INT8, same windows | `bench/ppl.py` |
| Tests | Logits vs HF, cached == uncached == HF greedy, batched == single, INT8 sanity, kernel vs PyTorch | `tests/test_engine.py` |

## Results (T4, Qwen2.5-0.5B-Instruct, 128 decode tokens)

<!-- filled from bench/results.csv -->
![throughput](bench/throughput.png)

| Configuration | batch 1 | batch 8 | batch 32 |
|---|---|---|---|
| Hugging Face fp16 | | | |
| Ours, KV cache fp16 | | | |
| Ours + INT8 | | | |
| Ours + fused RMSNorm | | | |
| vLLM | | | |

Perplexity (WikiText-2): fp16 **X.XX**, INT8 **X.XX** (Δ +0.XX) with weight size 988 MB → 5XX MB.

## What vLLM still does that this doesn't

CUDA graphs (the whole decode step captured once and replayed with no Python launch overhead), paged KV memory (no wasted cache space when batch rows have different lengths), and fully fused attention kernels. Those account for most of the remaining gap.

## Run

```
pip install torch transformers safetensors datasets matplotlib
python -m tests.test_engine --tiny      # architecture check, no download
python -m tests.test_engine             # vs Qwen2.5-0.5B-Instruct
python -m bench.bench                   # throughput → bench/results.csv
python -m bench.ppl                     # perplexity → bench/ppl.csv
python -m bench.plot                    # chart + resume numbers
```

Or open `run_all.ipynb` in Colab with a T4 runtime and Run all.
