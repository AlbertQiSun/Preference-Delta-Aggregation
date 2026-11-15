"""Geometric Alignment Merging (GAM) and the averaging baseline.

GAM merges n rank-r LoRA updates dW_i = B_i A_i of the same module:

    1. Decompose each update with a thin SVD,  dW_i = U_i S_i V_i^T  (rank r).
    2. Align every adapter to a reference adapter (index `ref_idx`) with
       orthogonal Procrustes on both sides:
           R_L,i = argmin_{R in O(r)} ||U_i R - U_ref||_F
           R_R,i = argmin_{R in O(r)} ||V_i R - V_ref||_F
       and rotate the factors:
           U_i <- U_i R_L,i,   S_i <- R_L,i^T S_i R_R,i,   V_i <- V_i R_R,i
       The product U_i S_i V_i^T is unchanged by this step.
    3. Average the aligned factors separately:  U_bar, S_bar, V_bar.
    4. Re-orthonormalize U_bar and V_bar with a thin QR and absorb the
       triangular factors into the core matrix.
    5. Reconstruct  dW_GAM = U_hat S_hat V_hat^T, which has rank <= r and can be
       stored exactly as a rank-r LoRA adapter.

Two implementations are provided and agree to machine precision:

* `gam_merge` (fast, used by default). Step 1 is computed from the LoRA factors
  with two thin QRs and an r x r SVD, so the d x k update is never formed.
  Step 4 is skipped because U_hat S_hat V_hat^T = U_bar S_bar V_bar^T exactly.
  Cost per module is O((d + k) r^2) instead of O(d k min(d, k)).
* `gam_merge_dense` (reference). Forms each dW_i and follows the five steps
  literally. This is the original implementation of GAM and is kept for
  verification.

Procrustes mode
---------------
`procrustes="orthogonal"` (default) solves over O(r), as defined in the paper.
The result is then invariant to the sign convention of the SVD routine.
`procrustes="rotation"` additionally forces det(R) = +1 (SO(r)). This was the
behaviour of the earliest GAM code. It makes the result depend on the
arbitrary signs returned by the SVD, so it is only kept to reproduce old runs.
"""

from typing import List, Optional, Tuple

import torch

PROCRUSTES_MODES = ("orthogonal", "rotation")


# -----------------------------------------------------------------------------
# Primitives
# -----------------------------------------------------------------------------

def thin_svd_from_factors(B: torch.Tensor, A: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact thin SVD of dW = B @ A without forming dW.

    B (d, m), A (m, k)  ->  U (d, m), s (m,), V (k, m)  with  dW = U diag(s) V^T.
    """
    Qb, Rb = torch.linalg.qr(B)        # B   = Qb Rb
    Qa, Ra = torch.linalg.qr(A.T)      # A^T = Qa Ra
    Um, s, Vmt = torch.linalg.svd(Rb @ Ra.T)
    return Qb @ Um, s, Qa @ Vmt.T


def procrustes(X: torch.Tensor, Y: torch.Tensor, mode: str = "orthogonal") -> torch.Tensor:
    """argmin_R ||X R - Y||_F over O(r) ("orthogonal") or SO(r) ("rotation")."""
    if mode not in PROCRUSTES_MODES:
        raise ValueError(f"procrustes mode must be one of {PROCRUSTES_MODES}")
    U, _, Vt = torch.linalg.svd(X.T @ Y)
    R = U @ Vt
    if mode == "rotation" and torch.det(R) < 0:
        U = U.clone()
        U[:, -1] *= -1
        R = U @ Vt
    return R


def _align_and_average(Us, Ss, Vs, ref_idx: int, mode: str):
    """Steps 2 and 3. Us: list of (d, r), Ss: list of (r, r), Vs: list of (k, r)."""
    U_ref, V_ref = Us[ref_idx], Vs[ref_idx]
    aU, aS, aV = [], [], []
    for i in range(len(Us)):
        if i == ref_idx:
            aU.append(Us[i]); aS.append(Ss[i]); aV.append(Vs[i])
            continue
        RL = procrustes(Us[i], U_ref, mode)
        RR = procrustes(Vs[i], V_ref, mode)
        aU.append(Us[i] @ RL)
        aS.append(RL.T @ Ss[i] @ RR)
        aV.append(Vs[i] @ RR)
    return torch.stack(aU).mean(0), torch.stack(aS).mean(0), torch.stack(aV).mean(0)


# -----------------------------------------------------------------------------
# GAM
# -----------------------------------------------------------------------------

def gam_merge(B_list: List[torch.Tensor], A_list: List[torch.Tensor],
              rank: Optional[int] = None, ref_idx: int = 0,
              procrustes_mode: str = "orthogonal") -> Tuple[torch.Tensor, torch.Tensor]:
    """Fast GAM on LoRA factors.

    Args:
        B_list: n tensors (d, r).  A_list: n tensors (r, k).
        rank: rank of the merged update (default: the adapter rank r).
        ref_idx: index of the reference adapter for Procrustes alignment.
    Returns:
        (B_merged (d, rank), A_merged (rank, k)) with dW_GAM = B_merged @ A_merged.
    """
    r = rank if rank is not None else A_list[0].shape[0]
    Us, Ss, Vs = [], [], []
    for B, A in zip(B_list, A_list):
        u, s, v = thin_svd_from_factors(B, A)
        Us.append(u[:, :r]); Ss.append(torch.diag(s[:r])); Vs.append(v[:, :r])
    Ub, Sb, Vb = _align_and_average(Us, Ss, Vs, ref_idx, procrustes_mode)
    return Ub @ Sb, Vb.T


def gam_merge_dense(delta_list: List[torch.Tensor], rank: int, ref_idx: int = 0,
                    procrustes_mode: str = "orthogonal") -> torch.Tensor:
    """Reference GAM on dense updates, following the five steps literally.

    Args:
        delta_list: n dense updates dW_i = B_i A_i, each (d, k).
    Returns:
        merged dense update (d, k), rank <= `rank`.
    """
    if len(delta_list) < 2:
        return delta_list[0].clone()
    Us, Ss, Vs = [], [], []
    for dW in delta_list:                                     # step 1
        U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
        Us.append(U[:, :rank]); Ss.append(torch.diag(S[:rank])); Vs.append(Vt[:rank].T)
    Ub, Sb, Vb = _align_and_average(Us, Ss, Vs, ref_idx, procrustes_mode)   # steps 2-3
    U_hat, R_U = torch.linalg.qr(Ub)                          # step 4
    V_hat, R_V = torch.linalg.qr(Vb)
    S_hat = R_U @ Sb @ R_V.T
    return U_hat @ S_hat @ V_hat.T                            # step 5


# -----------------------------------------------------------------------------
# Averaging baseline
# -----------------------------------------------------------------------------

def average_merge(B_list: List[torch.Tensor], A_list: List[torch.Tensor],
                  rank: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average the updates, (1/n) sum_i B_i A_i, then truncate to rank r.

    Computed in factored form: the average equals [B_1 .. B_n]/n @ [A_1; ..; A_n].
    """
    r = rank if rank is not None else A_list[0].shape[0]
    n = len(B_list)
    U, s, V = thin_svd_from_factors(torch.cat(B_list, dim=1) / n, torch.cat(A_list, dim=0))
    return U[:, :r] * s[:r], V[:, :r].T


def dense_to_factors(dW: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rank-`rank` factorization of a dense update: dW ~= B @ A."""
    U, S, Vt = torch.linalg.svd(dW, full_matrices=False)
    return U[:, :rank] * S[:rank], Vt[:rank]
