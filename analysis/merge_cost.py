#!/usr/bin/env python3
"""Wall-clock time and peak memory of the merge operators.

Measures GAM, TIES and averaging as a function of the number of deltas n and of
the LoRA rank r, and checks that the fast (factored) GAM equals the dense one.

  * GAM (dense):    forms dW_i = B_i A_i (d x k) and runs a dense SVD
  * GAM (factored): two thin QRs and an r x r SVD per adapter, never forms d x k
  * TIES (on factors, PEFT): add_weighted_adapter(combination_type="ties"),
    applied to lora_A and lora_B separately (the baseline used in the paper)
  * TIES (on dense dW): element-wise TIES on the full updates

Usage
    python analysis/merge_cost.py --n-list 2 3 4 --r-list 8 16 32 64 --repeat 3
    # whole-model timing with the real module shapes of an adapter
    python analysis/merge_cost.py --real-adapter ckpt/delta_a --threads 16
    # GPU timing
    python analysis/merge_cost.py --device cuda

Random matrices are used for timing, so the numbers measure the operators
themselves. Merging is a one-time offline cost, and the merged adapter has the
same rank and inference cost as a single delta.
"""

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pda.gam import average_merge, gam_merge, gam_merge_dense  # noqa: E402

# TF32 matmul (default on Ampere+) has ~1e-3 relative precision and would make
# the equivalence check fail spuriously. Merging is offline, so use full precision.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")


def read_real_shapes(adapter_path):
    """[(module, d, k, r), ...] taken from the lora_A / lora_B shapes of an adapter."""
    from safetensors.torch import load_file
    st = os.path.join(adapter_path, "adapter_model.safetensors")
    bn = os.path.join(adapter_path, "adapter_model.bin")
    sd = load_file(st) if os.path.exists(st) else torch.load(bn, map_location="cpu")
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        cfg = json.load(f)
    out = []
    for key in sd:
        if "lora_A" not in key or key.replace("lora_A", "lora_B") not in sd:
            continue
        A, B = sd[key], sd[key.replace("lora_A", "lora_B")]
        out.append((key, B.shape[0], A.shape[1], A.shape[0]))
    return out, cfg.get("r")


# --------------------------------------------------------------------------- operators

def avg_dense(Bs, As, r):
    m = torch.stack([B @ A for B, A in zip(Bs, As)]).mean(0)
    U, S, Vt = torch.linalg.svd(m, full_matrices=False)
    return (U[:, :r] * S[:r]) @ Vt[:r, :]


def avg_factored(Bs, As, r):
    return average_merge(Bs, As, r)


def _ties(tensors, density):
    pruned = []
    for t in tensors:
        flat = t.reshape(-1)
        kk = max(int(density * flat.numel()), 1)
        idx = torch.topk(flat.abs(), k=kk, largest=True).indices
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        pruned.append((flat * mask).reshape(t.shape))
    st = torch.stack(pruned, 0)
    sg = st.sign()
    maj = torch.where(sg.sum(0) >= 0, 1.0, -1.0)
    keep = (sg == maj) & (st != 0)
    return (st * keep).sum(0) / keep.sum(0).clamp(min=1.0)


def ties_on_factors(Bs, As, r, density=0.5):
    return _ties(Bs, density), _ties(As, density)


def ties_dense(Bs, As, r, density=0.5):
    return _ties([B @ A for B, A in zip(Bs, As)], density)


def gam_dense(Bs, As, r):
    return gam_merge_dense([B @ A for B, A in zip(Bs, As)], r)


def gam_factored(Bs, As, r):
    return gam_merge(Bs, As, r)


OPS = {
    "Averaging (dense)": avg_dense,
    "Averaging (factored)": avg_factored,
    "TIES (on factors, PEFT)": ties_on_factors,
    "TIES (on dense dW)": ties_dense,
    "GAM (dense)": gam_dense,
    "GAM (factored)": gam_factored,
}


# --------------------------------------------------------------------------- timing

def _rss_mb():
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        return None


def bench(fn, Bs, As, r, repeat, device):
    """(median seconds, peak additional memory in MB). CPU memory is the RSS increase."""
    fn(Bs, As, r)  # warm-up
    times = []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
    else:
        import gc
        gc.collect()
        base = _rss_mb()
        peak_rss = base or 0.0
    for _ in range(repeat):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn(Bs, As, r)
        if device == "cuda":
            torch.cuda.synchronize()
        else:
            cur = _rss_mb()
            if cur is not None:
                peak_rss = max(peak_rss, cur)
        times.append(time.perf_counter() - t0)
        del out
    if device == "cuda":
        peak = (torch.cuda.max_memory_allocated() - base) / 1e6
    else:
        peak = (peak_rss - base) if base is not None else float("nan")
    times.sort()
    return times[len(times) // 2], peak


def analytic_peak_mb(name, d, k, n, r, bytes_per=4):
    """Size of the largest intermediate tensors, computed analytically."""
    dk = d * k * bytes_per / 1e6
    fac = (d * r + r * k) * bytes_per / 1e6
    if "factored" in name or "on factors" in name:
        return n * fac
    if name == "TIES (on dense dW)":
        return 2 * n * dk
    if name == "Averaging (dense)":
        return (n + 1) * dk
    return n * dk + dk


def _rand(n, d, k, r, dev):
    return ([torch.randn(d, r, device=dev) * 0.02 for _ in range(n)],
            [torch.randn(r, k, device=dev) * 0.02 for _ in range(n)])


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--shape", nargs=2, type=int, default=[4096, 12288],
                   help="shape of one weight matrix (default: Qwen3-8B gate/up projection)")
    p.add_argument("--n-list", nargs="+", type=int, default=[2, 3, 4])
    p.add_argument("--r-list", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--n-matrices", type=int, default=252,
                   help="number of adapted matrices for the rough whole-model estimate")
    p.add_argument("--real-adapter", default=None,
                   help="adapter directory; time every module at its real shape")
    p.add_argument("--threads", type=int, default=None,
                   help="fix the number of CPU threads (recommended on shared nodes)")
    args = p.parse_args()

    if args.threads:
        torch.set_num_threads(args.threads)
    d, k = args.shape
    dev = args.device
    torch.manual_seed(0)
    print(f"device={dev}  matrix {d}x{k}  median of {args.repeat} runs  "
          f"threads={torch.get_num_threads()}  cpu={platform.processor() or platform.machine()}"
          + (f"  gpu={torch.cuda.get_device_name(0)}" if dev == "cuda" else ""))

    # ---- equivalence of the dense and factored implementations
    r0 = max(args.r_list)
    Bs, As = _rand(2, d, k, r0, dev)
    g1 = gam_dense(Bs, As, r0)
    L2, R2 = gam_factored(Bs, As, r0)
    a1 = avg_dense(Bs, As, r0)
    aL, aR = avg_factored(Bs, As, r0)
    print(f"[check] GAM dense vs factored, rel. diff = {((g1 - L2 @ R2).norm() / g1.norm()).item():.2e}")
    print(f"[check] AVG dense vs factored, rel. diff = {((a1 - aL @ aR).norm() / a1.norm()).item():.2e}\n")

    # ---- A. scaling with n (r = 64)
    print("=" * 78)
    print("A. Time vs number of deltas n (r = 64)")
    print("=" * 78)
    print(f"{'operator':<28}" + "".join(f"{f'n={n}':>14}" for n in args.n_list))
    rows_n = {}
    for name, fn in OPS.items():
        row = {}
        for n in args.n_list:
            Bs, As = _rand(n, d, k, 64, dev)
            row[n] = bench(fn, Bs, As, 64, args.repeat, dev)
        rows_n[name] = row
        print(f"{name:<28}" + "".join(f"{row[n][0] * 1000:>12.1f}ms" for n in args.n_list))

    print("\nPeak memory (measured / analytic largest intermediates)")
    print(f"{'operator':<28}" + "".join(f"{f'n={n}':>18}" for n in args.n_list))
    for name, row in rows_n.items():
        cells = []
        for n in args.n_list:
            meas = row[n][1]
            ms = f"{meas:.0f}" if meas == meas else "n/a"
            cells.append(f"{ms} / {analytic_peak_mb(name, d, k, n, 64):.0f}MB")
        print(f"{name:<28}" + "".join(f"{c:>18}" for c in cells))

    # ---- B. scaling with r (n = 4)
    print("\n" + "=" * 78)
    print("B. Time vs rank r (n = 4)")
    print("=" * 78)
    print(f"{'operator':<28}" + "".join(f"{f'r={r}':>14}" for r in args.r_list))
    for name, fn in OPS.items():
        cells = []
        for r in args.r_list:
            Bs, As = _rand(4, d, k, r, dev)
            cells.append(f"{bench(fn, Bs, As, r, args.repeat, dev)[0] * 1000:>12.1f}ms")
        print(f"{name:<28}" + "".join(cells))

    # ---- C. whole model
    print("\n" + "=" * 78)
    n4 = max(args.n_list)
    if args.real_adapter:
        shapes, real_r = read_real_shapes(args.real_adapter)
        rr = real_r or 64
        print(f"C. Whole-model merge time, every module at its real shape "
              f"({len(shapes)} matrices, r={rr}, n={n4})")
        print("=" * 78)
        for (d_, k_), c in sorted(Counter((d_, k_) for _, d_, k_, _ in shapes).items(),
                                  key=lambda x: -x[1]):
            print(f"   {d_:>6} x {k_:<6} {c:>4} matrices")
        totals = {name: 0.0 for name in OPS if "dense" not in name}
        for i, (_, d_, k_, _) in enumerate(shapes):
            Bs, As = _rand(n4, d_, k_, rr, dev)
            for name in totals:
                totals[name] += bench(OPS[name], Bs, As, rr, 1, dev)[0]
            if (i + 1) % 40 == 0:
                print(f"   {i + 1}/{len(shapes)}")
        print()
        for name, tot in totals.items():
            print(f"   {name:<30}{tot:>10.2f} s")
        print("   Dense operators are not timed per module (tens of minutes per model);")
        print("   multiply the per-matrix times above by the number of matrices for an estimate.")
    else:
        print(f"C. Rough whole-model estimate: per-matrix time x {args.n_matrices} (overestimates)")
        print("=" * 78)
        for name, row in rows_n.items():
            print(f"{name:<28}{row[n4][0] * args.n_matrices:>10.1f} s")
        print("   Use --real-adapter for per-module timing at real shapes.")


if __name__ == "__main__":
    main()
