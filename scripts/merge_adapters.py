#!/usr/bin/env python3
"""Merge several preference-delta LoRA adapters into one LoRA adapter.

Methods
  gam        Geometric Alignment Merging, fast factored implementation (default)
  gam_dense  the same operator computed on dense d x k updates (reference, slow)
  average    average of the updates, truncated to rank r

The first adapter in --adapters is the Procrustes reference unless --ref-idx is
given. In the paper the reference is the single delta with the best validation
performance.

Example
  python scripts/merge_adapters.py \
      --adapters ckpt/delta_searchr1 ckpt/delta_qwen3 ckpt/delta_acesearcher ckpt/delta_llama \
      --method gam \
      --output merged/pda_gam_4delta

  # optional: also write a standalone model with the merged update folded in
  python scripts/merge_adapters.py ... --export-full-model --base-model Qwen/Qwen3-8B
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pda import (average_merge, dense_to_factors, gam_merge, gam_merge_dense,  # noqa: E402
                 load_adapters, save_adapter)

METHODS = ("gam", "gam_dense", "average")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapters", nargs="+", required=True,
                   help="PEFT LoRA adapter directories (all with the same r and lora_alpha)")
    p.add_argument("--output", required=True, help="output directory for the merged adapter")
    p.add_argument("--method", default="gam", choices=METHODS)
    p.add_argument("--rank", type=int, default=None,
                   help="rank of the merged adapter (default: the adapter rank)")
    p.add_argument("--ref-idx", type=int, default=0,
                   help="index of the Procrustes reference adapter (default: 0)")
    p.add_argument("--procrustes", default="orthogonal", choices=("orthogonal", "rotation"),
                   help="'orthogonal' solves over O(r) as in the paper (default). "
                        "'rotation' restricts to SO(r) and reproduces the earliest code.")
    p.add_argument("--target-modules", nargs="*", default=None,
                   help="only merge modules whose name contains one of these strings")
    p.add_argument("--device", default="cpu", help="device for the merge computation")
    p.add_argument("--compute-dtype", default="float64", choices=("float64", "float32"))
    p.add_argument("--export-full-model", action="store_true",
                   help="also save base model + merged update as a standalone model")
    p.add_argument("--base-model", default=None,
                   help="base model path or HF id (required with --export-full-model)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.export_full_model and not args.base_model:
        sys.exit("--export-full-model requires --base-model")
    if not 0 <= args.ref_idx < len(args.adapters):
        sys.exit("--ref-idx out of range")

    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = getattr(torch, args.compute_dtype)

    print(f"[1/3] loading {len(args.adapters)} adapters")
    adapters, cfgs = load_adapters(args.adapters, dtype=torch.float32,
                                   target_modules=args.target_modules)
    r = args.rank or cfgs[0]["r"]
    names = list(adapters[0])

    print(f"[2/3] merging {len(names)} matrices with method={args.method} "
          f"(rank={r}, ref_idx={args.ref_idx}, procrustes={args.procrustes})")
    t0 = time.time()
    merged = {}
    for i, name in enumerate(names):
        Bs = [a[name][0].to(args.device, dtype) for a in adapters]
        As = [a[name][1].to(args.device, dtype) for a in adapters]
        if args.method == "gam":
            B, A = gam_merge(Bs, As, r, args.ref_idx, args.procrustes)
        elif args.method == "gam_dense":
            dW = gam_merge_dense([b @ a for b, a in zip(Bs, As)], r, args.ref_idx,
                                 args.procrustes)
            B, A = dense_to_factors(dW, r)
        else:
            B, A = average_merge(Bs, As, r)
        merged[name] = (B.cpu(), A.cpu())
        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{len(names)}")
    print(f"      done in {time.time() - t0:.1f}s")

    print(f"[3/3] saving merged adapter to {args.output}")
    info = {
        "method": args.method,
        "adapters": args.adapters,
        "ref_idx": args.ref_idx,
        "procrustes": args.procrustes,
        "rank": r,
        "n_matrices": len(names),
    }
    save_adapter(merged, args.output, args.adapters[0], extra_info=info)

    if args.export_full_model:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        out = os.path.join(args.output, "full_model")
        print(f"      folding the merged adapter into {args.base_model} -> {out}")
        model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, args.output).merge_and_unload()
        model.save_pretrained(out, safe_serialization=True)
        AutoTokenizer.from_pretrained(args.base_model).save_pretrained(out)
    print("done")


if __name__ == "__main__":
    main()
