#!/usr/bin/env python3
"""Build standalone models for the GAM component ablation.

Each method adds one ingredient to the previous one:

    avg         (1/n) sum_i B_i A_i                       rank up to n*r, no truncation
    avg_r       rank-r truncation of avg                  + low-rank truncation
    gam_id      GAM with every Procrustes matrix = I      + factor-wise averaging
    gam         full GAM                                  + Procrustes alignment

Two norm-matched controls separate direction from magnitude:

    gam_id_nm   gam_id rescaled per matrix to the Frobenius norm of gam
    avg_r_nm    avg_r rescaled per matrix to the Frobenius norm of gam   (optional)

    gam vs gam_id_nm     same magnitude, different direction -> effect of alignment
    gam_id_nm vs gam_id  same direction, different magnitude -> effect of magnitude

The outputs are full models (base weights + scaling * merged update), so they can
be evaluated with vLLM or transformers without loading any adapter.

Example
    python scripts/build_ablation_models.py \
        --base Qwen/Qwen3-8B \
        --adapters ckpt/delta_a ckpt/delta_b ckpt/delta_c ckpt/delta_d \
        --out merged/ablation \
        --methods avg avg_r gam_id gam_id_nm gam

    # check name mapping and magnitudes without writing anything
    python scripts/build_ablation_models.py --base ... --adapters ... --out ... --dry-run

Disk: about 16 GB per 8B model.  Memory: about 20 GB (the base model is reloaded
for each method).  No GPU needed.
"""

import argparse
import json
import os
import shutil
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pda.gam import average_merge, gam_merge, thin_svd_from_factors  # noqa: E402
from pda.lora_io import load_adapters, lora_scaling, to_model_param_name  # noqa: E402

DTYPE = torch.float64
METHODS = ["avg", "avg_r", "gam_id", "gam_id_nm", "gam", "avg_r_nm"]
DEFAULT_METHODS = ["avg", "avg_r", "gam_id", "gam_id_nm", "gam"]


def lowrank_fro(L, R):
    """||L @ R||_F without forming L @ R."""
    return torch.sqrt(torch.clamp(torch.einsum("ij,ji->", L.T @ L, R @ R.T), min=0.0))


def build_all_variants(B_list, A_list, r, ref_idx, procrustes_mode):
    """Return {method: (L, R)} with dW = L @ R for every ablation variant."""
    n = len(B_list)
    out = {"avg": (torch.cat(B_list, dim=1) / n, torch.cat(A_list, dim=0))}
    out["avg_r"] = average_merge(B_list, A_list, r)

    Us, Ss, Vs = [], [], []
    for B, A in zip(B_list, A_list):
        u, s, v = thin_svd_from_factors(B, A)
        Us.append(u[:, :r]); Ss.append(torch.diag(s[:r])); Vs.append(v[:, :r])
    out["gam_id"] = (torch.stack(Us).mean(0) @ torch.stack(Ss).mean(0),
                     torch.stack(Vs).mean(0).T)
    out["gam"] = gam_merge(B_list, A_list, r, ref_idx, procrustes_mode)

    gam_norm = lowrank_fro(*out["gam"])
    for src, dst in (("gam_id", "gam_id_nm"), ("avg_r", "avg_r_nm")):
        L, R = out[src]
        n0 = lowrank_fro(L, R)
        ratio = gam_norm / n0 if n0 > 0 else torch.tensor(1.0, dtype=DTYPE)
        out[dst] = (L * ratio, R)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, help="base model directory or HF id")
    p.add_argument("--adapters", nargs="+", required=True)
    p.add_argument("--out", required=True, help="output root; one subdirectory per method")
    p.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=METHODS)
    p.add_argument("--ref-idx", type=int, default=0,
                   help="Procrustes reference adapter; use the same order as for merging")
    p.add_argument("--procrustes", default="orthogonal", choices=("orthogonal", "rotation"))
    p.add_argument("--dry-run", action="store_true",
                   help="check name mapping and magnitudes without writing files")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[1/3] loading adapters")
    adapters, cfgs = load_adapters(args.adapters, dtype=DTYPE)
    r = cfgs[0]["r"]
    scaling = lora_scaling(cfgs[0])
    names = list(adapters[0])
    print(f"      LoRA scaling = {scaling} (applied when adding the update to the base)")

    print("[2/3] computing merged updates")
    deltas = {m: {} for m in args.methods}
    for i, ln in enumerate(names):
        variants = build_all_variants([a[ln][0] for a in adapters], [a[ln][1] for a in adapters],
                                      r, args.ref_idx, args.procrustes)
        for m in args.methods:
            L, R = variants[m]
            deltas[m][ln] = (L.float(), R.float())
        if (i + 1) % 50 == 0:
            print(f"      {i + 1}/{len(names)}")

    print("      mean ||dW||_F over the first 5 matrices:")
    for m in args.methods:
        vals = [(L @ R).norm().item() for L, R in (deltas[m][ln] for ln in names[:5])]
        print(f"        {m:10s} {sum(vals) / len(vals):.6f}")

    print("[3/3] folding updates into the base model")
    for m in args.methods:
        print(f"  --- {m}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
        sd = model.state_dict()
        keys = set(sd.keys())

        applied, missing = 0, []
        for ln in names:
            pname = to_model_param_name(ln, keys)
            if pname is None:
                missing.append(ln)
                continue
            L, R = deltas[m][ln]
            dW = (L @ R) * scaling
            if sd[pname].shape != dW.shape:
                sys.exit(f"shape mismatch at {pname}: {tuple(sd[pname].shape)} vs {tuple(dW.shape)}")
            sd[pname] = (sd[pname].float() + dW).to(torch.bfloat16)
            applied += 1
        print(f"      applied {applied}/{len(names)} matrices")
        if missing:
            print(f"      {len(missing)} adapter modules have no matching base parameter, e.g.")
            for x in missing[:5]:
                print(f"        {x}")
            sys.exit("name mapping failed")

        if args.dry_run:
            del model, sd
            continue

        outdir = os.path.join(args.out, m)
        os.makedirs(outdir, exist_ok=True)
        model.load_state_dict(sd)
        model.save_pretrained(outdir, safe_serialization=True)
        AutoTokenizer.from_pretrained(args.base, trust_remote_code=True).save_pretrained(outdir)
        gen_cfg = os.path.join(args.base, "generation_config.json")
        if os.path.exists(gen_cfg):
            shutil.copy(gen_cfg, os.path.join(outdir, "generation_config.json"))
        with open(os.path.join(outdir, "merge_info.json"), "w") as f:
            json.dump({"method": m, "base_model": args.base, "adapters": args.adapters,
                       "ref_idx": args.ref_idx, "procrustes": args.procrustes,
                       "lora_r": r, "lora_scaling": scaling, "n_matrices_merged": applied},
                      f, indent=2)
        del model, sd
        print(f"      saved to {outdir}")

    print("dry run passed" if args.dry_run else "done")


if __name__ == "__main__":
    main()
