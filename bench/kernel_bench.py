"""
Microbenchmark the fused RMSNorm kernel against PyTorch's three-op version.

Reports per-call latency, speedup, and achieved memory bandwidth (GB/s) vs. the
T4's ~320 GB/s peak. Shapes: decode at batch 1 / 32, and a prefill-sized block.

Usage:  python -m bench.kernel_bench
Output: bench/kernel.csv
"""

import csv, os, time, torch
from engine.model import RMSNorm
from engine import kernels

PEAK_GBS = 320.0  # T4 HBM2


def time_fn(fn, iters=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    kernels.build()
    dim = 896
    norm = RMSNorm(dim, 1e-6).cuda().half()
    rows = []
    for name, shape in [("decode b=1", (1, 1, dim)), ("decode b=32", (32, 1, dim)), ("prefill 32x128", (32, 128, dim))]:
        x = torch.randn(*shape, device="cuda", dtype=torch.float16)
        kernels.USE_FUSED = False; t_py = time_fn(lambda: norm(x))
        kernels.USE_FUSED = True;  t_fu = time_fn(lambda: norm(x))
        # correctness
        kernels.USE_FUSED = False; ref = norm(x); kernels.USE_FUSED = True; got = norm(x); kernels.USE_FUSED = False
        err = (ref.float() - got.float()).abs().max().item()
        bytes_moved = 2 * x.numel() * 2  # read x + write y, fp16
        gbs_py, gbs_fu = bytes_moved / t_py / 1e9, bytes_moved / t_fu / 1e9
        rows.append({"shape": name, "pytorch_us": round(t_py * 1e6, 1), "fused_us": round(t_fu * 1e6, 1),
                     "speedup": round(t_py / t_fu, 2), "fused_gbs": round(gbs_fu, 1), "pct_peak": round(100 * gbs_fu / PEAK_GBS, 1), "max_err": err})
        print(f"{name:16s} pytorch {t_py*1e6:7.1f} us   fused {t_fu*1e6:7.1f} us   {t_py/t_fu:4.2f}x   "
              f"fused {gbs_fu:6.1f} GB/s ({100*gbs_fu/PEAK_GBS:4.1f}% of peak)   err {err:.1e}")
    os.makedirs("bench", exist_ok=True)
    with open("bench/kernel.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
