#!/usr/bin/env python3
"""Numerical accuracy of the dense and factored SVD paths used by GAM.

The question is not how far the two paths are from each other, but how far each
is from the exact update dW = B @ A, which is known. In float64 both reach
machine precision. In float32 the factored path has a reconstruction error that
does not grow with the matrix size, because it never accumulates over d x k
entries. The script also reports whether TF32 matmul was active.

Usage
    python analysis/gam_numerics.py           # uses CUDA if available
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pda.gam import thin_svd_from_factors  # noqa: E402

print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
      + (f"  gpu={torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))
print(f"matmul.allow_tf32 (default) = {torch.backends.cuda.matmul.allow_tf32}")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
print(f"matmul.allow_tf32 (now)     = {torch.backends.cuda.matmul.allow_tf32}")

dev = "cuda" if torch.cuda.is_available() else "cpu"


def reconstruction_errors(d, k, r, dtype):
    torch.manual_seed(0)
    B = torch.randn(d, r, dtype=dtype, device=dev) * 0.02
    A = torch.randn(r, k, dtype=dtype, device=dev) * 0.02
    dW = B @ A
    nrm = dW.norm()
    U1, S1, Vt1 = torch.linalg.svd(dW, full_matrices=False)
    e_dense = (((U1[:, :r] * S1[:r]) @ Vt1[:r, :] - dW).norm() / nrm).item()
    U2, S2, V2 = thin_svd_from_factors(B, A)
    e_fact = (((U2[:, :r] * S2[:r]) @ V2[:, :r].T - dW).norm() / nrm).item()
    return e_dense, e_fact


print("\n1. float64 (checks the algorithm)")
ed, ef = reconstruction_errors(1024, 3072, 64, torch.float64)
print(f"   1024x3072  dense={ed:.3e}  factored={ef:.3e}")

print("\n2. float32: relative reconstruction error of the rank-r SVD")
print(f"{'shape':>16}{'dense':>14}{'factored':>14}")
for d, k in [(1024, 3072), (2048, 6144), (4096, 12288)]:
    t0 = time.time()
    ed, ef = reconstruction_errors(d, k, 64, torch.float32)
    print(f"{f'{d}x{k}':>16}{ed:>14.3e}{ef:>14.3e}   ({time.time() - t0:.1f}s)")
