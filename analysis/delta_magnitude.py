#!/usr/bin/env python3
"""Relative size of a LoRA update with respect to the base weights, ||dW|| / ||W||.

Reads the base-model safetensors shards lazily (the full model is never loaded)
and computes ||B A||_F / ||W||_F for every adapted matrix. Works for a single
preference delta or for a merged adapter written by scripts/merge_adapters.py.

Usage
    python analysis/delta_magnitude.py --base /path/to/Qwen3-8B \
        --adapter ckpt/delta_a --apply-scaling
"""

import argparse
import glob
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pda.lora_io import load_adapter, lora_scaling  # noqa: E402


def iter_base_tensors(base_dir):
    from safetensors import safe_open
    files = sorted(glob.glob(os.path.join(base_dir, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors shards in {base_dir}")
    for f in files:
        with safe_open(f, framework="pt", device="cpu") as fh:
            for key in fh.keys():
                yield key, fh


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="local base model directory (safetensors shards)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--apply-scaling", action="store_true",
                   help="multiply dW by the LoRA scaling, as PEFT does at inference")
    args = p.parse_args()

    factors, cfg = load_adapter(args.adapter, dtype=torch.float32)
    # PEFT module names look like base_model.model.model.layers.0.self_attn.q_proj,
    # base parameter names like model.layers.0.self_attn.q_proj.weight
    lora = {}
    for name, ba in factors.items():
        stem = name[len("base_model.model."):] if name.startswith("base_model.model.") else name
        lora[stem] = ba
    sc = lora_scaling(cfg) if args.apply_scaling else 1.0
    print(f"{len(lora)} adapted matrices, LoRA scaling {lora_scaling(cfg)} "
          f"({'applied' if args.apply_scaling else 'not applied'})\n")

    rows = []
    for key, fh in iter_base_tensors(args.base):
        stem = key[:-len(".weight")] if key.endswith(".weight") else key
        if stem not in lora:
            continue
        W = fh.get_tensor(key).float()
        B, A = lora[stem]
        dW = sc * (B @ A)
        if dW.shape != W.shape:
            print(f"  skip {stem}: shape {tuple(dW.shape)} vs {tuple(W.shape)}")
            continue
        nw, nd = W.norm().item(), dW.norm().item()
        rows.append((stem, nd, nw, nd / max(nw, 1e-30)))
        if len(rows) % 40 == 0:
            print(f"  {len(rows)} matrices")

    if not rows:
        print("no matrices matched: check that the base model and adapter use the same module names")
        return

    ratios = torch.tensor([r[3] for r in rows])
    tot_d = sum(r[1] ** 2 for r in rows) ** 0.5
    tot_w = sum(r[2] ** 2 for r in rows) ** 0.5
    print("\n" + "=" * 66)
    print(f"Results over {len(rows)} matrices")
    print("=" * 66)
    print(f"  per-matrix ||dW||/||W||   median {ratios.median():.5f}   mean {ratios.mean():.5f}")
    print(f"                            min {ratios.min():.5f}   max {ratios.max():.5f}")
    print(f"  global ||dW||/||W|| (all matrices concatenated) = {tot_d / tot_w:.5f} "
          f"({100 * tot_d / tot_w:.3f}%)")
    print("\n  by module type (median):")
    by_type = defaultdict(list)
    for stem, _, _, ratio in rows:
        by_type[stem.split(".")[-1]].append(ratio)
    for t in sorted(by_type, key=lambda x: -sum(by_type[x]) / len(by_type[x])):
        v = torch.tensor(by_type[t])
        print(f"    {t:<12} {v.median():.5f}  ({len(v)} matrices)")


if __name__ == "__main__":
    main()
