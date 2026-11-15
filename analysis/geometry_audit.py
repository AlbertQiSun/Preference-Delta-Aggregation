#!/usr/bin/env python3
"""Geometry audit of the GAM merge (CPU only, fp64, no model evaluation).

For every adapted matrix this script measures

  * how much there is to align: principal angles between the adapters'
    singular subspaces (invariant under the Procrustes rotation), the deviation
    ||R - I|| of the Procrustes solutions, and the conditioning of the problem;
  * which step of GAM changes the merged update, using a ladder in which each
    variant adds one ingredient:
        AVG --(+truncation)--> AVG-r --(+factor-wise avg)--> GAM-id --(+alignment)--> GAM
  * the Frobenius norm of every variant (the implicit merging coefficient);
  * one-sided variants (align U only, as in KnOTS, or V only) and the
    AVG-factor baseline that averages lora_B and lora_A directly.

Outputs <out>/geometry_audit.json (per matrix) and <out>/merge_prescreen.md.

Usage
    python analysis/geometry_audit.py \
        --adapters ckpt/delta_a ckpt/delta_b ckpt/delta_c ckpt/delta_d \
        --ref-idx 0 --out results/geometry
"""

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pda.gam import procrustes, thin_svd_from_factors  # noqa: E402
from pda.lora_io import load_adapters  # noqa: E402

DTYPE = torch.float64


class LowRank:
    """dW = L @ R with L (d, m) and R (m, k). All arithmetic stays in the factors."""

    def __init__(self, L: torch.Tensor, R: torch.Tensor, name: str = ""):
        self.L, self.R, self.name = L, R, name

    def inner(self, other: "LowRank") -> float:
        # <X, Y>_F = tr((L_x^T L_y)(R_y R_x^T))
        return torch.einsum("ij,ji->", self.L.T @ other.L, other.R @ self.R.T).item()

    def fro(self) -> float:
        return math.sqrt(max(self.inner(self), 0.0))

    def rel_dist(self, other: "LowRank") -> float:
        """||X - Y||_F / ||X||_F"""
        xx, yy, xy = self.inner(self), other.inner(other), self.inner(other)
        return math.sqrt(max(xx + yy - 2 * xy, 0.0)) / math.sqrt(max(xx, 1e-300))

    def cos(self, other: "LowRank") -> float:
        return self.inner(other) / max(self.fro() * other.fro(), 1e-300)


def build_variants(B_list: List[torch.Tensor], A_list: List[torch.Tensor], r: int,
                   ref_idx: int, mode: str) -> Tuple[Dict[str, LowRank], dict]:
    n = len(B_list)
    diag = {}

    L_avg, R_avg = torch.cat(B_list, dim=1) / n, torch.cat(A_list, dim=0)
    avg = LowRank(L_avg, R_avg, "AVG")
    U, s, V = thin_svd_from_factors(L_avg, R_avg)
    avg_r = LowRank(U[:, :r] * s[:r], V[:, :r].T, "AVG-r")
    avg_factor = LowRank(torch.stack(B_list).mean(0), torch.stack(A_list).mean(0), "AVG-factor")

    Us, Ss, Vs = [], [], []
    for B, A in zip(B_list, A_list):
        u, s_i, v = thin_svd_from_factors(B, A)
        Us.append(u[:, :r]); Ss.append(torch.diag(s_i[:r])); Vs.append(v[:, :r])

    raw = [LowRank(B, A) for B, A in zip(B_list, A_list)]
    pair_cos = [raw[i].cos(raw[j]) for i in range(n) for j in range(i + 1, n)]

    U_ref, V_ref = Us[ref_idx], Vs[ref_idx]
    Ub, Sb, Vb = torch.stack(Us).mean(0), torch.stack(Ss).mean(0), torch.stack(Vs).mean(0)
    gam_id = LowRank(Ub @ Sb, Vb.T, "GAM-id")

    aU, aS, aV = [], [], []      # both sides   -> GAM
    uU, uS, uV = [], [], []      # U side only  -> GAM-Uonly
    vU, vS, vV = [], [], []      # V side only  -> GAM-Vonly
    refl, dev_U, dev_V, cond, pang_U, pang_V = 0, [], [], [], [], []
    eye = torch.eye(r, dtype=DTYPE)
    for i in range(n):
        cU = torch.linalg.svdvals(Us[i].T @ U_ref)   # cosines of principal angles
        cV = torch.linalg.svdvals(Vs[i].T @ V_ref)
        pang_U.append(cU); pang_V.append(cV)
        cond.append(cU.min().item())
        if i == ref_idx:
            for dst in (aU, uU, vU): dst.append(Us[i])
            for dst in (aS, uS, vS): dst.append(Ss[i])
            for dst in (aV, uV, vV): dst.append(Vs[i])
            continue
        RL = procrustes(Us[i], U_ref, mode)
        RR = procrustes(Vs[i], V_ref, mode)
        refl += int(torch.det(Us[i].T @ U_ref) < 0) + int(torch.det(Vs[i].T @ V_ref) < 0)
        dev_U.append((RL - eye).norm().item() / math.sqrt(2 * r))
        dev_V.append((RR - eye).norm().item() / math.sqrt(2 * r))
        aU.append(Us[i] @ RL); aS.append(RL.T @ Ss[i] @ RR); aV.append(Vs[i] @ RR)
        uU.append(Us[i] @ RL); uS.append(RL.T @ Ss[i]);      uV.append(Vs[i])
        vU.append(Us[i]);      vS.append(Ss[i] @ RR);        vV.append(Vs[i] @ RR)

    def assemble(Ul, Sl, Vl, name):
        Um, Sm, Vm = torch.stack(Ul).mean(0), torch.stack(Sl).mean(0), torch.stack(Vl).mean(0)
        return LowRank(Um @ Sm, Vm.T, name), Um, Vm

    gam, Ua, Va = assemble(aU, aS, aV, "GAM")
    gam_u, _, _ = assemble(uU, uS, uV, "GAM-Uonly")
    gam_v, _, _ = assemble(vU, vS, vV, "GAM-Vonly")

    diag["pairwise_cos_dW_mean"] = float(sum(pair_cos) / len(pair_cos)) if pair_cos else 0.0
    diag["pairwise_cos_dW_min"] = float(min(pair_cos)) if pair_cos else 0.0
    diag["pairwise_cos_dW_max"] = float(max(pair_cos)) if pair_cos else 0.0
    diag["random_overlap_baseline"] = float(r) / float(B_list[0].shape[0])
    diag["principal_angle_cos_U_mean"] = float(torch.stack(pang_U).mean())
    diag["principal_angle_cos_U_min"] = float(torch.stack(pang_U).min())
    diag["principal_angle_cos_V_mean"] = float(torch.stack(pang_V).mean())
    diag["subspace_overlap_U"] = float((torch.stack(pang_U) ** 2).mean())
    diag["R_dev_from_identity_U"] = float(sum(dev_U) / max(len(dev_U), 1))
    diag["R_dev_from_identity_V"] = float(sum(dev_V) / max(len(dev_V), 1))
    diag["procrustes_min_sv"] = float(min(cond))
    diag["reflections"] = refl
    diag["energy_retention_U"] = float(Ua.norm() / max(Ub.norm(), 1e-30))
    diag["energy_retention_V"] = float(Va.norm() / max(Vb.norm(), 1e-30))
    diag["Ubar_mean_col_norm"] = float(Ua.norm(dim=0).mean())
    diag["Vbar_mean_col_norm"] = float(Va.norm(dim=0).mean())

    return {"AVG": avg, "AVG-r": avg_r, "AVG-factor": avg_factor, "GAM-id": gam_id,
            "GAM-Uonly": gam_u, "GAM-Vonly": gam_v, "GAM": gam}, diag


def module_type(name: str) -> str:
    for m in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        if m in name:
            return m
    return "other"


def layer_depth(name: str) -> int:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return -1


def run_audit(adapters, ref_idx: int, out_dir: str, mode: str):
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    names = list(adapters[0])
    for li, ln in enumerate(names):
        B_list = [a[ln][0] for a in adapters]
        A_list = [a[ln][1] for a in adapters]
        r = A_list[0].shape[0]
        v, diag = build_variants(B_list, A_list, r, ref_idx, mode)
        row = dict(diag)
        row.update({
            "layer": ln, "module": module_type(ln), "depth": layer_depth(ln), "rank": r,
            "step1_truncation": v["AVG"].rel_dist(v["AVG-r"]),
            "step2_factorwise": v["AVG-r"].rel_dist(v["GAM-id"]),
            "step3_rotation": v["GAM"].rel_dist(v["GAM-id"]),
            "fro_AVG": v["AVG"].fro(), "fro_AVG_r": v["AVG-r"].fro(),
            "fro_GAM_id": v["GAM-id"].fro(), "fro_GAM": v["GAM"].fro(),
            "scale_GAM_over_AVG": v["GAM"].fro() / max(v["AVG"].fro(), 1e-300),
            "scale_GAM_over_AVG_r": v["GAM"].fro() / max(v["AVG-r"].fro(), 1e-300),
            "cos_GAM_AVGr": v["GAM"].cos(v["AVG-r"]),
            "cos_GAM_GAMid": v["GAM"].cos(v["GAM-id"]),
            "rel_AVG_vs_AVGfactor": v["AVG"].rel_dist(v["AVG-factor"]),
            "cos_GAM_AVGfactor": v["GAM"].cos(v["AVG-factor"]),
            "fro_AVG_factor": v["AVG-factor"].fro(),
            "cos_GAM_GAMUonly": v["GAM"].cos(v["GAM-Uonly"]),
            "cos_GAM_GAMVonly": v["GAM"].cos(v["GAM-Vonly"]),
            "rel_GAMVonly_vs_GAMid": v["GAM-Vonly"].rel_dist(v["GAM-id"]),
            "rel_GAMUonly_vs_GAMid": v["GAM-Uonly"].rel_dist(v["GAM-id"]),
        })
        rows.append(row)
        if (li + 1) % 20 == 0:
            print(f"  {li + 1}/{len(names)} matrices")

    with open(os.path.join(out_dir, "geometry_audit.json"), "w") as f:
        json.dump(rows, f, indent=2)
    write_report(rows, os.path.join(out_dir, "merge_prescreen.md"), len(adapters))
    return rows


def _agg(rows, key):
    v = torch.tensor([r[key] for r in rows], dtype=DTYPE)
    return v.mean().item(), v.median().item(), v.min().item(), v.max().item()


def write_report(rows, path, n_adapters):
    out = []
    w = out.append
    w(f"# GAM geometry audit (n = {n_adapters} adapters, {len(rows)} adapted matrices)\n")

    w("## 1. Subspace geometry\n")
    w("| Quantity | mean | median | min | max |")
    w("|---|---|---|---|---|")
    for k, label in [
        ("pairwise_cos_dW_mean", "cos(dW_i, dW_j), pairwise update cosine"),
        ("principal_angle_cos_U_mean", "cos(principal angle), U (invariant under R)"),
        ("principal_angle_cos_V_mean", "cos(principal angle), V (invariant under R)"),
        ("subspace_overlap_U", "subspace overlap (1/r) sum cos^2(theta), U"),
        ("procrustes_min_sv", "sigma_min(U_i^T U_ref), conditioning"),
        ("R_dev_from_identity_U", "||R_U - I||_F / sqrt(2r)"),
        ("R_dev_from_identity_V", "||R_V - I||_F / sqrt(2r)"),
        ("energy_retention_U", "||mean(U_i R_i)|| / ||mean(U_i)||"),
        ("energy_retention_V", "||mean(V_i R_i)|| / ||mean(V_i)||"),
    ]:
        m, md, lo, hi = _agg(rows, k)
        w(f"| {label} | {m:.4f} | {md:.4f} | {lo:.4f} | {hi:.4f} |")
    w("")
    w("> Overlap near the random baseline r/d means the subspaces are close to orthogonal "
      "and alignment has little to act on. Overlap well above that baseline together with "
      "||R - I|| well above 0 is the regime in which alignment changes the merge.\n")

    w("## 2. Ladder: relative Frobenius change added by each step\n")
    w("| Step | Ingredient added | mean | median | min | max |")
    w("|---|---|---|---|---|---|")
    for k, step, var in [("step1_truncation", "1", "rank-r truncation"),
                         ("step2_factorwise", "2", "factor-wise averaging"),
                         ("step3_rotation", "3", "Procrustes alignment")]:
        m, md, lo, hi = _agg(rows, k)
        w(f"| {step} | {var} | {m:.4f} | {md:.4f} | {lo:.4f} | {hi:.4f} |")
    w("")

    w("## 3. Magnitude of the merged update\n")
    w("| Quantity | mean | median | min | max |")
    w("|---|---|---|---|---|")
    for k, label in [("scale_GAM_over_AVG", "||dW_GAM|| / ||dW_AVG||"),
                     ("scale_GAM_over_AVG_r", "||dW_GAM|| / ||dW_AVG-r||"),
                     ("Ubar_mean_col_norm", "mean column norm of U_bar (1.0 = orthonormal)")]:
        m, md, lo, hi = _agg(rows, k)
        w(f"| {label} | {m:.4f} | {md:.4f} | {lo:.4f} | {hi:.4f} |")
    w("")
    w("> A ratio different from 1.0 means GAM also rescales the update relative to the "
      "averaging baselines. `scripts/build_ablation_models.py` provides norm-matched controls "
      "to separate the effect of direction from the effect of magnitude.\n")

    w("## 4. Direction of GAM relative to the averaging variants\n")
    m, md, lo, hi = _agg(rows, "cos_GAM_AVGr")
    w(f"cos(dW_GAM, dW_AVG-r): {m:.4f} (mean), {md:.4f} (median), range [{lo:.4f}, {hi:.4f}]\n")
    m, md, _, _ = _agg(rows, "cos_GAM_GAMid")
    w(f"cos(dW_GAM, dW_GAM-id): {m:.4f} (mean), {md:.4f} (median)\n")
    m, md, _, _ = _agg(rows, "rel_AVG_vs_AVGfactor")
    w(f"AVG vs AVG-factor (average lora_B and lora_A directly): rel. difference "
      f"{m:.4f} (mean), {md:.4f} (median)\n")
    m, md, _, _ = _agg(rows, "cos_GAM_AVGfactor")
    w(f"cos(dW_GAM, dW_AVG-factor): {m:.4f} (mean), {md:.4f} (median)\n")

    w("## 5. One-sided alignment\n")
    w("| Variant | rel. change vs GAM-id (mean) | (median) | cos with GAM (mean) | (median) |")
    w("|---|---|---|---|---|")
    for rel_k, cos_k, label in [("rel_GAMUonly_vs_GAMid", "cos_GAM_GAMUonly", "align U only (KnOTS-style)"),
                                ("rel_GAMVonly_vs_GAMid", "cos_GAM_GAMVonly", "align V only")]:
        a = _agg(rows, rel_k)
        c = _agg(rows, cos_k)
        w(f"| {label} | {a[0]:.4f} | {a[1]:.4f} | {c[0]:.4f} | {c[1]:.4f} |")
    w("")

    w("## 6. By module type\n")
    w("| module | n | step 3 (alignment) | cos(GAM, AVG-r) | ||GAM|| / ||AVG|| |")
    w("|---|---|---|---|---|")
    by_mod = defaultdict(list)
    for r in rows:
        by_mod[r["module"]].append(r)
    for mod, rs in sorted(by_mod.items()):
        w(f"| {mod} | {len(rs)} | {_agg(rs, 'step3_rotation')[0]:.4f} | "
          f"{_agg(rs, 'cos_GAM_AVGr')[0]:.4f} | {_agg(rs, 'scale_GAM_over_AVG')[0]:.4f} |")
    w("")

    depths = sorted({r["depth"] for r in rows if r["depth"] >= 0})
    if depths:
        w("## 7. By layer depth (quartiles)\n")
        w("| depth range | n | step 3 (alignment) | cos(GAM, AVG-r) |")
        w("|---|---|---|---|")
        q = max(1, len(depths) // 4)
        for i in range(0, len(depths), q):
            sel = set(depths[i:i + q])
            rs = [r for r in rows if r["depth"] in sel]
            if rs:
                w(f"| {min(sel)}-{max(sel)} | {len(rs)} | {_agg(rs, 'step3_rotation')[0]:.4f} | "
                  f"{_agg(rs, 'cos_GAM_AVGr')[0]:.4f} |")
        w("")

    with open(path, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapters", nargs="+", required=True)
    p.add_argument("--ref-idx", type=int, default=0,
                   help="Procrustes reference adapter (paper: best single delta on validation)")
    p.add_argument("--out", default="results/geometry")
    p.add_argument("--target-modules", nargs="*", default=None)
    p.add_argument("--procrustes", default="orthogonal", choices=("orthogonal", "rotation"))
    args = p.parse_args()

    adapters, _ = load_adapters(args.adapters, dtype=DTYPE, target_modules=args.target_modules)
    print(f"{len(adapters)} adapters, {len(adapters[0])} adapted matrices each")
    run_audit(adapters, args.ref_idx, args.out, args.procrustes)


if __name__ == "__main__":
    main()
