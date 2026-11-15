#!/usr/bin/env python3
"""Check whether the adapters share their LoRA initialization, and which side of
the alignment carries GAM.

Three quick checks (CPU, a few minutes):

  0. the training seed recorded in each adapter directory, if any;
  1. cosine similarity of the lora_A matrices across adapters. LoRA initializes
     B = 0, so with a shared seed, a small learning rate and few steps the A
     matrices barely move. Near-identical A matrices mean that the overlap of the
     right singular subspaces reflects the shared initialization;
  2. basis-energy retention on the left (U) and right (V) side, read from the
     JSON written by analysis/geometry_audit.py.

Usage
    python analysis/check_init.py \
        --adapters ckpt/delta_a ckpt/delta_b ckpt/delta_c ckpt/delta_d \
        --audit-json results/geometry/geometry_audit.json
"""

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pda.lora_io import load_adapters  # noqa: E402

SEED_FILES = ["training_args.yaml", "llamaboard_config.yaml", "trainer_state.json",
              "all_results.json", "train_results.json", "adapter_config.json"]


def report_seeds(paths):
    print("=" * 68)
    print("0. Training seeds")
    print("=" * 68)
    seeds = []
    for path in paths:
        name = os.path.basename(path.rstrip("/"))
        seed, src = None, None
        for fn in SEED_FILES:
            fp = os.path.join(path, fn)
            if not os.path.exists(fp):
                continue
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    low = line.lower()
                    if "seed" in low and "seedling" not in low:
                        seed, src = line.strip(), fn
                        break
            if seed:
                break
        if seed:
            seeds.append(seed)
            print(f"  {name:24s} {seed}   (from {src})")
        else:
            print(f"  {name:24s} no training-argument file found")
    if not seeds:
        print("  No seed found. The LLaMA-Factory default seed is 42.")
    elif len(set(seeds)) == 1 and len(seeds) > 1:
        print("  All adapters were trained with the same seed.")
    else:
        print("  Seeds differ across adapters.")
    print()


def check_init_similarity(paths, n_layers):
    print("=" * 68)
    print("1. Similarity of lora_A across adapters")
    print("=" * 68)
    adapters, _ = load_adapters(paths, verbose=False)
    names = list(adapters[0])[:n_layers]
    cos = []
    for ln in names:
        A = [a[ln][1] for a in adapters]
        for i in range(1, len(A)):
            cos.append((A[0].flatten() @ A[i].flatten() / (A[0].norm() * A[i].norm())).item())
    mean_cos, min_cos = statistics.mean(cos), min(cos)
    print(f"  cos(A_0, A_i) over {len(names)} matrices: mean {mean_cos:.4f}, min {min_cos:.4f}")
    if mean_cos > 0.95:
        print("  The A matrices are nearly identical: they keep the shared initialization,")
        print("  so overlap of the right singular subspaces comes from initialization.")
    elif mean_cos > 0.5:
        print("  The A matrices are partially similar.")
    else:
        print("  The A matrices differ substantially: right-subspace overlap is learned.")
    print()


def check_energy_retention(audit_json):
    print("=" * 68)
    print("2. Basis-energy retention by side")
    print("=" * 68)
    with open(audit_json) as f:
        rows = json.load(f)
    eu = statistics.mean(r["energy_retention_U"] for r in rows)
    ev = statistics.mean(r["energy_retention_V"] for r in rows)
    print(f"  left  (U): {eu:.4f}")
    print(f"  right (V): {ev:.4f}")
    print("  (1.0 = alignment does not change the averaged basis; larger values mean")
    print("   alignment prevents more cancellation when averaging)")
    if ev > eu * 1.3:
        print("  Alignment acts mainly on the right side.")
    elif eu > ev * 1.3:
        print("  Alignment acts mainly on the left side.")
    else:
        print("  Both sides contribute similarly.")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapters", nargs="+", required=True)
    p.add_argument("--audit-json", default=None,
                   help="geometry_audit.json from analysis/geometry_audit.py")
    p.add_argument("--n-layers", type=int, default=10, help="number of matrices to sample")
    args = p.parse_args()

    report_seeds(args.adapters)
    check_init_similarity(args.adapters, args.n_layers)
    if args.audit_json:
        check_energy_retention(args.audit_json)


if __name__ == "__main__":
    main()
