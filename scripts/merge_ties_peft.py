#!/usr/bin/env python3
"""
merge_ties_peft.py: TIES-Merging baseline for LoRA adapters.

Uses PEFT `add_weighted_adapter(combination_type="ties")`, which applies TIES
to the lora_A and lora_B factors. This is the TIES baseline reported in the paper.

Usage:
    python scripts/merge_ties_peft.py \
        --base_model <path_or_hf_id> \
        --loras <lora1> <lora2> [<lora3> ...] \
        --output <output_path> \
        [--weights 2.0 1.0 1.0] \
        [--density 0.5] \
        [--majority_sign_method frequency] \
        [--combination_type ties] \
        [--dtype float16] \
        [--device auto]

Example:
    python scripts/merge_ties_peft.py \
        --base_model Qwen/Qwen3-8B \
        --loras ckpt/delta_a ckpt/delta_b ckpt/delta_c \
        --density 0.5 \
        --output merged/ties
"""

import argparse
import sys
import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge multiple LoRA adapters using TIES (or DARE+TIES) via PEFT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        help=(
            "Path or HuggingFace model ID for the base model. "
            "If not provided, it will be inferred from the first LoRA's PeftConfig."
        ),
    )
    parser.add_argument(
        "--loras",
        type=str,
        nargs="+",
        required=True,
        help="Local paths (or HF IDs) to two or more LoRA adapter directories.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory to save the merged model and tokenizer.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Per-adapter weights (importance scores). "
            "Must match the number of --loras if provided. "
            "Defaults to 1.0 for each adapter."
        ),
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.5,
        help=(
            "Fraction of delta weights to KEEP during the Trim step (0.0–1.0). "
            "Lower = more aggressive pruning. Default: 0.5"
        ),
    )
    parser.add_argument(
        "--majority_sign_method",
        type=str,
        choices=["total", "frequency"],
        default="frequency",
        help=(
            "'frequency' uses sign-vote (recommended); "
            "'total' uses magnitude-weighted sign. Default: frequency"
        ),
    )
    parser.add_argument(
        "--combination_type",
        type=str,
        choices=["ties", "dare_ties", "dare_linear", "linear"],
        default="ties",
        help="Merging strategy. 'dare_ties' applies DARE pruning before TIES. Default: ties",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="torch dtype for loading the base model. Default: float16",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="device_map for base model loading. Default: auto",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base model in 4-bit (requires bitsandbytes). Cannot save merged model in quantized form.",
    )
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load base model in 8-bit (requires bitsandbytes). Cannot save merged model in quantized form.",
    )
    parser.add_argument(
        "--merge_and_unload",
        action="store_true",
        default=True,
        help=(
            "After TIES merging, fuse the final merged adapter into the base model weights "
            "and save as a standalone model. Default: True"
        ),
    )
    parser.add_argument(
        "--save_adapter_only",
        action="store_true",
        help=(
            "Save only the merged LoRA adapter weights instead of the full fused model. "
            "Overrides --merge_and_unload."
        ),
    )

    return parser.parse_args()


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def validate_args(args):
    if len(args.loras) < 2:
        print("ERROR: At least two --loras are required for merging.", file=sys.stderr)
        sys.exit(1)

    if args.weights is not None and len(args.weights) != len(args.loras):
        print(
            f"ERROR: --weights has {len(args.weights)} values but --loras has {len(args.loras)}.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not 0.0 < args.density <= 1.0:
        print("ERROR: --density must be in (0.0, 1.0].", file=sys.stderr)
        sys.exit(1)

    for path in args.loras:
        if not os.path.exists(path) and "/" not in path:
            print(f"WARNING: LoRA path '{path}' does not exist locally.", file=sys.stderr)

    if args.load_in_4bit and args.load_in_8bit:
        print("ERROR: Cannot use both --load_in_4bit and --load_in_8bit.", file=sys.stderr)
        sys.exit(1)


def resolve_base_model(args):
    """If base_model not given, infer from the first LoRA's PeftConfig."""
    if args.base_model:
        return args.base_model
    print(f"  No --base_model provided. Inferring from first LoRA: {args.loras[0]}")
    config = PeftConfig.from_pretrained(args.loras[0])
    base = config.base_model_name_or_path
    print(f"  Inferred base model: {base}")
    return base


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()
    validate_args(args)

    weights = args.weights if args.weights is not None else [1.0] * len(args.loras)
    torch_dtype = DTYPE_MAP[args.dtype]

    print("\n" + "=" * 60)
    print("  TIES LoRA Merger")
    print("=" * 60)
    print(f"  LoRAs          : {args.loras}")
    print(f"  Weights        : {weights}")
    print(f"  Density        : {args.density}")
    print(f"  Combination    : {args.combination_type}")
    print(f"  Sign method    : {args.majority_sign_method}")
    print(f"  Output         : {args.output}")
    print(f"  dtype          : {args.dtype}")
    print("=" * 60 + "\n")

    # ── 1. Resolve base model ──────────────────────────────────────
    base_model_id = resolve_base_model(args)

    # ── 2. Load base model ─────────────────────────────────────────
    print(f"[1/5] Loading base model: {base_model_id}")
    load_kwargs = dict(
        torch_dtype=torch_dtype,
        device_map=args.device,
    )
    if args.load_in_4bit:
        load_kwargs["load_in_4bit"] = True
    elif args.load_in_8bit:
        load_kwargs["load_in_8bit"] = True

    model = AutoModelForCausalLM.from_pretrained(base_model_id, **load_kwargs)
    model.eval()
    print("  Base model loaded.\n")

    # ── 3. Load tokenizer ──────────────────────────────────────────
    print(f"[2/5] Loading tokenizer from: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    print("  Tokenizer loaded.\n")

    # ── 4. Load LoRA adapters ──────────────────────────────────────
    print(f"[3/5] Loading {len(args.loras)} LoRA adapters...")
    adapter_names = []
    for i, lora_path in enumerate(args.loras):
        name = f"adapter_{i}"
        adapter_names.append(name)
        if i == 0:
            model = PeftModel.from_pretrained(model, lora_path, adapter_name=name)
        else:
            model.load_adapter(lora_path, adapter_name=name)
        print(f"  Loaded '{name}' from: {lora_path}")
    print()

    # ── 5. Merge with TIES ─────────────────────────────────────────
    merged_name = "ties_merged"
    print(f"[4/5] Merging adapters with combination_type='{args.combination_type}'...")

    merge_kwargs = dict(
        combination_type=args.combination_type,
        density=args.density,
    )
    # majority_sign_method only applies to ties-based methods
    if "ties" in args.combination_type:
        merge_kwargs["majority_sign_method"] = args.majority_sign_method

    model.add_weighted_adapter(
        adapter_names,
        weights,
        merged_name,
        **merge_kwargs,
    )
    model.set_adapter(merged_name)
    print(f"  Merged adapter '{merged_name}' created and set as active.\n")

    # ── 6. Save ────────────────────────────────────────────────────
    os.makedirs(args.output, exist_ok=True)
    print(f"[5/5] Saving to: {args.output}")

    if args.save_adapter_only:
        # Save just the merged LoRA adapter
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print("  Saved merged LoRA adapter (adapter weights only).")
    else:
        # Fuse the merged adapter into base model weights → standalone model
        print("  Fusing adapter into base model weights (merge_and_unload)...")
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        print("  Saved full fused model (ready for inference without PEFT).")

    print(f"\nDone. Merged model saved to: {args.output}\n")


if __name__ == "__main__":
    main()