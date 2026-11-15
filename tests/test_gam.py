"""Numerical checks for the GAM implementations. Run with `python -m pytest tests/`."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pda import average_merge, gam_merge, gam_merge_dense, thin_svd_from_factors  # noqa: E402

torch.set_default_dtype(torch.float64)


def _adapters(d=192, k=256, r=8, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    B0 = torch.randn(d, r, generator=g) * 0.02
    A0 = torch.randn(r, k, generator=g) * 0.02
    Bs = [B0 + 0.01 * torch.randn(d, r, generator=g) for _ in range(n)]
    As = [A0 + 0.01 * torch.randn(r, k, generator=g) for _ in range(n)]
    return Bs, As


def _rel(x, y):
    return ((x - y).norm() / y.norm()).item()


def test_thin_svd_from_factors_is_exact():
    Bs, As = _adapters()
    U, s, V = thin_svd_from_factors(Bs[0], As[0])
    assert _rel(U @ torch.diag(s) @ V.T, Bs[0] @ As[0]) < 1e-12


def test_fast_gam_matches_dense_reference():
    for seed, n in [(0, 2), (1, 3), (2, 4)]:
        Bs, As = _adapters(n=n, seed=seed)
        r = As[0].shape[0]
        L, R = gam_merge(Bs, As, r)
        dense = gam_merge_dense([b @ a for b, a in zip(Bs, As)], r)
        assert _rel(L @ R, dense) < 1e-10


def test_gam_is_invariant_to_svd_sign_convention():
    # Flipping the sign of a singular pair gives another valid SVD of the same
    # matrix. With Procrustes over O(r) the merged update must not change.
    Bs, As = _adapters(n=3, seed=3)
    r = As[0].shape[0]
    L, R = gam_merge(Bs, As, r)
    flip = torch.ones(r)
    flip[0] = -1.0
    # B A = (B D)(D A) for D = diag(+-1); this changes the SVD signs internally
    Bs2 = [b * flip for b in Bs]
    As2 = [(a.T * flip).T for a in As]
    L2, R2 = gam_merge(Bs2, As2, r)
    assert _rel(L2 @ R2, L @ R) < 1e-10


def test_gam_output_rank_and_single_adapter():
    Bs, As = _adapters(n=4)
    r = As[0].shape[0]
    L, R = gam_merge(Bs, As, r)
    assert L.shape == (Bs[0].shape[0], r) and R.shape == (r, As[0].shape[1])
    L1, R1 = gam_merge(Bs[:1], As[:1], r)
    assert _rel(L1 @ R1, Bs[0] @ As[0]) < 1e-10


def test_gam_on_identical_adapters_returns_that_adapter():
    Bs, As = _adapters(n=1)
    L, R = gam_merge(Bs * 3, As * 3)
    assert _rel(L @ R, Bs[0] @ As[0]) < 1e-10


def test_average_matches_dense_truncated_average():
    Bs, As = _adapters(n=3)
    r = As[0].shape[0]
    L, R = average_merge(Bs, As, r)
    avg = torch.stack([b @ a for b, a in zip(Bs, As)]).mean(0)
    U, S, Vt = torch.linalg.svd(avg, full_matrices=False)
    assert _rel(L @ R, (U[:, :r] * S[:r]) @ Vt[:r]) < 1e-10


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
