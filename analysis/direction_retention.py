#!/usr/bin/env python3
"""How much of each delta's direction survives the merge, as a function of n.

For each merge operator M and each nested subset of the deltas (first 2, first 3,
first 4) this reports

  1. mean_i cos(dW_i, M): direction retention averaged over the deltas
  2. min_i  cos(dW_i, M): retention of the least preserved delta
  3. for TIES, the fraction of entries zeroed by magnitude trimming and the
     fraction discarded by sign election
  4. optionally (--with-rank) the effective rank (participation ratio)

These are geometric quantities. Lower retention means less of a delta's update
direction is kept. It does not by itself show which directions are useful.

Cosines are computed in factored form, <B_i A_i, L R>_F = tr((B_i^T L)(R A_i^T)),
so only TIES on dense updates forms d x k matrices.

Usage
    python analysis/direction_retention.py \
        --adapters ckpt/delta_a ckpt/delta_b ckpt/delta_c ckpt/delta_d \
        --names A B C D --density 0.5 --all-layers --device cuda
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pda.gam import average_merge, gam_merge  # noqa: E402
from pda.lora_io import load_adapters  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
DTYPE = torch.float32


def fro_ip_ff(B1, A1, B2, A2):
    """<B1 A1, B2 A2>_F = tr((B1^T B2)(A2 A1^T)), computed on r x r matrices."""
    return torch.sum((B1.T @ B2) * (A1 @ A2.T))


def fro_norm_ff(B, A):
    return torch.sqrt(torch.clamp(fro_ip_ff(B, A, B, A), min=0))


def fro_ip_fd(B, A, M):
    """<B A, M>_F = tr((B^T M) A^T) for a dense M."""
    return torch.sum((B.T @ M) * A)


def ties_merge(tensors, density=0.5, sign_method="frequency"):
    """Returns (merged, fraction zeroed by trimming, fraction discarded by sign election)."""
    pruned, zr = [], []
    for t in tensors:
        flat = t.reshape(-1)
        kk = max(int(density * flat.numel()), 1)
        idx = torch.topk(flat.abs(), k=kk, largest=True).indices
        mask = torch.zeros_like(flat)
        mask[idx] = 1.0
        pruned.append((flat * mask).reshape(t.shape))
        zr.append(1.0 - kk / flat.numel())
    st = torch.stack(pruned, 0)
    sg = st.sign()
    agg = sg.sum(0) if sign_method == "frequency" else st.sum(0)
    maj = torch.where(agg >= 0, 1.0, -1.0)
    keep = (sg == maj) & (st != 0)
    merged = (st * keep).sum(0) / keep.sum(0).clamp(min=1.0)
    nz = (st != 0).sum().item()
    disc = 1.0 - keep.sum().item() / max(nz, 1)
    return merged, sum(zr) / len(zr), disc


def eff_rank_ff(L, R):
    Rl = torch.linalg.qr(L, mode="r")[1]
    Rr = torch.linalg.qr(R.T, mode="r")[1]
    s2 = torch.linalg.svdvals(Rl @ Rr.T) ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum().clamp(min=1e-30))


def eff_rank_dense(M):
    s2 = torch.linalg.svdvals(M) ** 2
    return float((s2.sum() ** 2) / (s2 ** 2).sum().clamp(min=1e-30))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapters", nargs="+", required=True,
                   help="adapters in the order in which deltas are added")
    p.add_argument("--names", nargs="+", default=None)
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--sign-method", default="frequency", choices=["frequency", "total"])
    p.add_argument("--n-layers", type=int, default=32, help="matrices to sample")
    p.add_argument("--all-layers", action="store_true")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--with-rank", action="store_true",
                   help="also compute effective rank (slower)")
    args = p.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    ads, cfgs = load_adapters(args.adapters, dtype=DTYPE)
    names = args.names or [f"d{i}" for i in range(len(ads))]
    r = cfgs[0]["r"]
    print(f"device={dev}  {len(ads)} adapters  rank={r}  TIES density={args.density}")
    print(f"order: {' -> '.join(names)}\n")

    common = list(ads[0])
    sel = common if args.all_layers else common[::max(1, len(common) // args.n_layers)][:args.n_layers]
    print(f"{len(common)} matrices, using {len(sel)}\n")

    subsets = [list(range(m)) for m in range(2, len(ads) + 1)]
    methods = ("AVG_r", "TIES_factor", "TIES_dW", "GAM")
    results = {}
    for idx in subsets:
        n = len(idx)
        acc = {m: {"cos_mean": [], "cos_min": [], "pr": []} for m in methods}
        tz, td = [], []
        for li, ln in enumerate(sel):
            Bs = [ads[j][ln][0].to(dev) for j in idx]
            As = [ads[j][ln][1].to(dev) for j in idx]
            aL, aR = average_merge(Bs, As, r)
            gL, gR = gam_merge(Bs, As, r)
            tA, z1, d1 = ties_merge(As, args.density, args.sign_method)
            tB, z2, d2 = ties_merge(Bs, args.density, args.sign_method)
            tW, _, _ = ties_merge([B @ A for B, A in zip(Bs, As)], args.density, args.sign_method)
            tz.append((z1 + z2) / 2)
            td.append((d1 + d2) / 2)

            norms = [fro_norm_ff(B, A) for B, A in zip(Bs, As)]
            for mname, obj in (("AVG_r", (aL, aR)), ("GAM", (gL, gR)),
                               ("TIES_factor", (tB, tA)), ("TIES_dW", tW)):
                if isinstance(obj, tuple):
                    nm = fro_norm_ff(*obj)
                    cs = [float(fro_ip_ff(B, A, *obj) / (ni * nm).clamp(min=1e-30))
                          for B, A, ni in zip(Bs, As, norms)]
                else:
                    nm = obj.norm()
                    cs = [float(fro_ip_fd(B, A, obj) / (ni * nm).clamp(min=1e-30))
                          for B, A, ni in zip(Bs, As, norms)]
                acc[mname]["cos_mean"].append(sum(cs) / len(cs))
                acc[mname]["cos_min"].append(min(cs))

            if args.with_rank:
                acc["AVG_r"]["pr"].append(eff_rank_ff(aL, aR))
                acc["GAM"]["pr"].append(eff_rank_ff(gL, gR))
                acc["TIES_factor"]["pr"].append(eff_rank_ff(tB, tA))
                acc["TIES_dW"]["pr"].append(eff_rank_dense(tW))
            del tW
            if (li + 1) % 40 == 0:
                print(f"  n={n}: {li + 1}/{len(sel)}")

        results[n] = {m: {k: (sum(v) / len(v) if v else None) for k, v in d.items()}
                      for m, d in acc.items()}
        results[n]["_zeroed"] = sum(tz) / len(tz)
        results[n]["_discarded"] = sum(td) / len(td)

    ns = sorted(results)
    header = f"{'operator':<14}" + "".join(f"{f'n={n}':>11}" for n in ns)
    print("=" * 74)
    print("1. Direction retention, mean_i cos(dW_i, merged)")
    print("=" * 74)
    print(header)
    for m in methods:
        print(f"{m:<14}" + "".join(f"{results[n][m]['cos_mean']:>11.4f}" for n in ns))
    print("\n" + "=" * 74)
    print("2. Retention of the least preserved delta, min_i cos")
    print("=" * 74)
    print(header)
    for m in methods:
        print(f"{m:<14}" + "".join(f"{results[n][m]['cos_min']:>11.4f}" for n in ns))
    if args.with_rank:
        print("\n" + "=" * 74)
        print("3. Effective rank (participation ratio)")
        print("=" * 74)
        print(header)
        for m in methods:
            print(f"{m:<14}" + "".join(f"{results[n][m]['pr']:>11.2f}" for n in ns))
    print("\n" + "=" * 74)
    print("4. TIES pruning (on factors)")
    print("=" * 74)
    print(f"{'':<26}" + "".join(f"{f'n={n}':>11}" for n in ns))
    print(f"{'zeroed by trimming':<26}" + "".join(f"{results[n]['_zeroed']:>11.3f}" for n in ns))
    print(f"{'discarded by sign vote':<26}" + "".join(f"{results[n]['_discarded']:>11.3f}" for n in ns))
    print("\nTIES_factor corresponds to PEFT add_weighted_adapter(combination_type='ties'),")
    print("the TIES baseline used in the paper.")


if __name__ == "__main__":
    main()
