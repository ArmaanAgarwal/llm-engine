"""Chart bench/results.csv → bench/throughput.png, and print the resume numbers."""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rows = list(csv.DictReader(open("bench/results.csv")))
configs = []
for r in rows:
    if r["config"] not in configs: configs.append(r["config"])
batches = sorted({int(r["batch"]) for r in rows})
labels = {"hf_fp16": "Hugging Face (fp16)", "ours_nocache": "Ours: no cache (fp32)", "ours_cache": "Ours: KV cache (fp16)",
          "ours_int8": "Ours: + INT8", "ours_fused": "Ours: + fused RMSNorm", "vllm": "vLLM"}

fig, ax = plt.subplots(figsize=(8, 4.5))
w = 0.8 / len(configs)
for i, c in enumerate(configs):
    ys = [next((float(r["tok_per_s"]) for r in rows if r["config"] == c and int(r["batch"]) == b), 0) for b in batches]
    ax.bar([j + i * w for j in range(len(batches))], ys, w, label=labels.get(c, c))
ax.set_xticks([j + 0.4 - w / 2 for j in range(len(batches))]); ax.set_xticklabels([f"batch {b}" for b in batches])
ax.set_ylabel("decode tokens / sec (aggregate)"); ax.set_yscale("log")
ax.set_title("Qwen2.5-0.5B decode throughput on T4"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("bench/throughput.png", dpi=150)
print("wrote bench/throughput.png")

def get(c, b):
    return next((float(r["tok_per_s"]) for r in rows if r["config"] == c and int(r["batch"]) == b), None)

print("\n=== Resume numbers ===")
for b in batches:
    hf, cache, int8, fused = get("hf_fp16", b), get("ours_cache", b), get("ours_int8", b), get("ours_fused", b)
    best = fused or int8 or cache
    line = f"batch {b:2d}: ours {best:.0f} tok/s"
    if hf: line += f" vs HF {hf:.0f} ({best/hf:.1f}x)"
    if cache and fused: line += f"; fused kernel {100*(1 - cache/fused):+.0f}% throughput vs cache-only"
    print(line)
