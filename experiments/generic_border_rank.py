#!/usr/bin/env python3
"""
Track 1 Bridge: Positive-Measure Generic Border-Rank Singularity,
the New beta^(1/4) / gamma^(3/4) Universality Class, Higher-Dimensional Scaling,
and Real Bilinear vs. SwiGLU MLPs.

Answers the #1 SLT objection to Track 1 ("Is the W-tensor a measure-zero
2x2x2 curiosity?"):
  Part 1 — Positive-Measure Generic Real 2x2x2 Tensors (de Silva-Lim 2008):
    1A. Exact Cayley hyperdeterminant Det(T) verified on I (+1), W (0), T3 (-4),
        plus empirical measure of Det(T*) < 0 on Gaussian T* ~ N(0, I_8)
        matching Bergqvist's exact constant 1 - pi/4 = 0.214602.
    1B. Unseeded multi-start L-BFGS weight-decay scaling across gamma in [1e-1, 1e-6]
        on generic Det < 0 vs. Det > 0 Gaussian tensors and canonical T3 (k=2 vs k=3):
        establishes the POSITIVE-MEASURE generic universality class
          Delta L*(gamma) = L*(gamma) - L_wall ~ C_L * gamma^(3/4),
          ||theta*(gamma)||_2^2 ~ C_R * gamma^(-1/4),
          virial gamma * ||theta*||^2 / Delta L* -> 3.0,
          cancelling neuron pairs |cos(a_1, a_2)|, |cos(b_1, b_2)|, |cos(c_1, c_2)| -> 1.
        (Distinct from the measure-zero Det = 0 W-tensor class gamma^(6/7), virial -> 6.)
    1C. Exact 4-scale Laplace Hessian spectrum & equilibrated SMC + HMC on a
        generic Gaussian Det(T_g) < 0 tensor (zero continuous symmetries!) up to
        beta = 1e6, with 2x2 mode-wise SPD preconditioning, exact GL(2)^3 Lie-group
        border-rank moves S_u = P_m diag(e^u, e^{-2u}) P_m^{-1} (preserving T_wall
        to 1e-15 with exact Jacobian -3*k*u), torus gauge moves, and independent
        cold-start MCMC verification at beta in {1e4, 1e5, 1e6}:
          k = 2: lambda_exc(beta) = beta * E_beta[L - L_wall] tracks exact Laplace
                 beta * Delta L*(beta) + 33/8 (33/8 = 4.125),
                 E_beta[||theta||^2] = ||theta*(beta)||^2 + 15/4 (15/4 = 3.75),
                 excess free energy F_exc(beta) ~ C_sad * beta^(1/4) + (33/8) log beta.
          k = 3: lambda(beta) plateaus to finite Watanabe RLCT ~ 4.0, bounded R^2.

  Part 2 — Scaling to Higher Dimensions (d x d x d Gaussian Tensors, d in {2, 3, 4}):
    Measures empirical border-rank degeneration frequency, cancellation ratio,
    norm growth, and pairwise factor alignment across (d=2, k=2), (d=3, k=2,3,4),
    and (d=4, k=4,5,6).

  Part 3 — Real Bilinear MLP & SwiGLU MLP on Actual Tasks:
    3A. Real single-input Bilinear MLP f(x) = W_out ((W_1 x) * (W_2 x)) on generic
        d=4 -> 4 quadratic tasks: empirical degeneration rate across 20 tasks,
        Adam & AdamW training trajectories (k=4 vs k=6) tracking cancellation ratio
        and pairwise neuron alignment, and L-BFGS WD sweep (k=4, 6, 8).
    3B. Real SwiGLU MLP f(x) = W_out (SiLU(W_1 x) * (W_2 x)): honest head-to-head
        comparison on BOTH the same pure quadratic target (discovering the
        gate-linearization singularity W_1 -> 0, MSE ~ gamma^(2/3), R^2 ~ gamma^(-1/3),
        virial -> 2 at k < 2 r*, and exact antisymmetric pairing SiLU(z)-SiLU(-z)=z
        collapsing SwiGLU width 2k onto Bilinear width k at 2x norm^2) AND a
        SwiGLU-native teacher control (where non-homogeneous SiLU breaks multilinear
        homogeneity and yields finite-norm best approximations at k < k*).

"Unseeded" above means the L-BFGS multistarts use i.i.d. random Gaussian inits with no
analytic or symmetry-based warm-start seeding (no centered-curve or exact-realization
inits). The random number generators themselves use fixed seeds, so runs are reproducible.

Usage:
  python generic_border_rank.py            # full run (SMC+HMC etc.; ~11 min on CPU)
  python generic_border_rank.py --replot   # redraw the figure from the saved JSON only

Outputs (relative to this file):
  results/generic_border_rank.json
  results/generic_border_rank.png
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core tensor algebra & Cayley hyperdeterminant
# ---------------------------------------------------------------------------
def cayley_hyperdet(T: torch.Tensor) -> torch.Tensor:
    """
    Cayley's hyperdeterminant Det(T) for 2x2x2 tensors T: (..., 2, 2, 2).
    Sign convention (de Silva-Lim 2008):
      Det(T) > 0  <=> generic real rank 2 (e.g. I = e0^3 + e1^3 has Det = +1),
      Det(T) = 0  <=> tangential / border-rank wall (e.g. W has Det = 0),
      Det(T) < 0  <=> generic real rank 3 (e.g. T3 has Det = -4).
    """
    M0 = T[..., :, :, 0]
    M1 = T[..., :, :, 1]
    det0 = M0[..., 0, 0] * M0[..., 1, 1] - M0[..., 0, 1] * M0[..., 1, 0]
    det1 = M1[..., 0, 0] * M1[..., 1, 1] - M1[..., 0, 1] * M1[..., 1, 0]
    tr_adj = (
        M0[..., 1, 1] * M1[..., 0, 0]
        - M0[..., 0, 1] * M1[..., 1, 0]
        - M0[..., 1, 0] * M1[..., 0, 1]
        + M0[..., 0, 0] * M1[..., 1, 1]
    )
    return tr_adj ** 2 - 4.0 * det0 * det1


def make_canonical_tensors(dtype=torch.float64) -> dict[str, torch.Tensor]:
    I = torch.zeros(2, 2, 2, dtype=dtype)
    I[0, 0, 0] = 1.0
    I[1, 1, 1] = 1.0

    W = torch.zeros(2, 2, 2, dtype=dtype)
    W[0, 0, 1] = 1.0
    W[0, 1, 0] = 1.0
    W[1, 0, 0] = 1.0

    # T3 = Re((e0 + i e1)^{otimes 3}) = e0^3 - e0 e1 e1 - e1 e0 e1 - e1 e1 e0
    T3 = torch.zeros(2, 2, 2, dtype=dtype)
    T3[0, 0, 0] = 1.0
    T3[0, 1, 1] = -1.0
    T3[1, 0, 1] = -1.0
    T3[1, 1, 0] = -1.0
    return {"I": I, "W": W, "T3": T3}


def cp_tensor(theta: torch.Tensor) -> torch.Tensor:
    """theta: (..., 3, d, k) -> T: (..., d, d, d)."""
    return torch.einsum(
        "...ir,...jr,...lr->...ijl",
        theta[..., 0, :, :],
        theta[..., 1, :, :],
        theta[..., 2, :, :],
    )


def fit_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    return float(np.polyfit(np.log(xs), np.log(ys), 1)[0])


def cp_diagnostics(theta: torch.Tensor) -> tuple[float, list[float], float]:
    """
    Compute cancellation ratio sum_r ||a_r|| ||b_r|| ||c_r|| / ||T||_F,
    per-mode max pairwise |cos|, and max pairwise min-across-modes |cos|.
    """
    with torch.no_grad():
        T = cp_tensor(theta)
        norms = theta.norm(dim=1)  # (3, k)
        r1_sum = float((norms[0] * norms[1] * norms[2]).sum().item())
        t_norm = max(float(T.norm().item()), 1e-12)
        cancel_ratio = r1_sum / t_norm

        k = theta.shape[2]
        mode_max_cos = [0.0, 0.0, 0.0]
        max_min_cos = 0.0
        for i in range(k):
            for j in range(i + 1, k):
                cs = [
                    abs(
                        float(
                            torch.nn.functional.cosine_similarity(
                                theta[m, :, i], theta[m, :, j], dim=0
                            ).item()
                        )
                    )
                    for m in range(3)
                ]
                for m in range(3):
                    if cs[m] > mode_max_cos[m]:
                        mode_max_cos[m] = cs[m]
                if min(cs) > max_min_cos:
                    max_min_cos = min(cs)
    return cancel_ratio, mode_max_cos, max_min_cos


def run_cp_lbfgs(
    init_theta: torch.Tensor,
    target: torch.Tensor,
    gam: float,
    max_iter: int = 300,
    n_outer: int = 6,
    tol_grad: float = 1e-13,
) -> tuple[torch.Tensor, float, float, float]:
    """Minimize ||T(theta) - target||_F^2 + gam * ||theta||_2^2 with strong-Wolfe L-BFGS."""
    theta = init_theta.clone().detach().requires_grad_(True)
    opt = torch.optim.LBFGS(
        [theta],
        lr=0.5,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=tol_grad,
        tolerance_change=1e-16,
    )

    def closure():
        opt.zero_grad()
        T = cp_tensor(theta)
        loss = ((T - target) ** 2).sum() + gam * (theta ** 2).sum()
        loss.backward()
        return loss

    prev_val = float("inf")
    for _ in range(n_outer):
        val = float(opt.step(closure).item())
        gnorm = float(theta.grad.norm().item()) if theta.grad is not None else float("inf")
        if gnorm < tol_grad * 10:
            break
        if abs(prev_val - val) < 1e-16 and gnorm < 1e-9:
            break
        prev_val = val

    with torch.no_grad():
        T = cp_tensor(theta)
        L = float(((T - target) ** 2).sum().item())
        R2 = float((theta ** 2).sum().item())
        obj = L + gam * R2
    return theta.detach().clone(), obj, L, R2


def project_onto_det_wall(
    target: torch.Tensor, init_T: torch.Tensor
) -> tuple[torch.Tensor, float, float]:
    """
    Compute exact Euclidean projection T_wall = argmin_{Det(T)=0} ||T - target||_F^2
    via quadratic penalty continuation up to pen = 1e14.
    """
    T_proj = init_T.clone().detach().requires_grad_(True)
    for pen in (1e3, 1e6, 1e9, 1e12, 1e14):
        opt = torch.optim.LBFGS(
            [T_proj],
            lr=0.5,
            max_iter=300,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-14,
            tolerance_change=1e-16,
        )

        def closure():
            opt.zero_grad()
            val = ((T_proj - target) ** 2).sum() + pen * (cayley_hyperdet(T_proj) ** 2)
            val.backward()
            return val

        for _ in range(6):
            opt.step(closure)
            if T_proj.grad is not None and float(T_proj.grad.norm().item()) < 1e-12:
                break

    T_wall = T_proj.detach().clone()
    L_wall = float(((T_wall - target) ** 2).sum().item())
    det_wall = float(cayley_hyperdet(T_wall).item())
    return T_wall, L_wall, det_wall


def extract_gl2_basis_from_wall(T_wall: torch.Tensor) -> torch.Tensor:
    """
    Given any smooth tangential wall tensor T_wall with Det(T_wall) = 0,
    extract exact invertible matrices (P, Q, S) in GL(2)^3 such that
    T_wall is in the W-basis: (P^{-1} x Q^{-1} x S^{-1}) T_wall = W_scaled.
    Then S_u^{(m)} = P_m diag(e^u, e^{-2u}) P_m^{-1} leaves T_wall invariant
    to machine precision (~1e-15) with exact Jacobian det(S_u^{(m)}) = e^{-u}.
    """
    M0, M1 = T_wall[:, :, 0], T_wall[:, :, 1]
    d0 = float((M0[0, 0] * M0[1, 1] - M0[0, 1] * M0[1, 0]).item())
    d1 = float((M1[0, 0] * M1[1, 1] - M1[0, 1] * M1[1, 0]).item())
    tr = float(
        (
            M0[1, 1] * M1[0, 0]
            - M0[0, 1] * M1[1, 0]
            - M0[1, 0] * M1[0, 1]
            + M0[0, 0] * M1[1, 1]
        ).item()
    )
    if abs(d0) >= abs(d1):
        u = torch.tensor([-tr, 2.0 * d0], dtype=torch.float64)
    else:
        u = torch.tensor([2.0 * d1, -tr], dtype=torch.float64)
    u = u / u.norm()
    s0 = torch.tensor([-u[1], u[0]], dtype=torch.float64)
    M_rank1 = u[0] * M0 + u[1] * M1
    U_svd, _, Vt_svd = torch.linalg.svd(M_rank1)
    p0 = U_svd[:, 0]
    q0 = Vt_svd[0, :]

    p1_0 = torch.tensor([-p0[1], p0[0]], dtype=torch.float64)
    q1_0 = torch.tensor([-q0[1], q0[0]], dtype=torch.float64)
    s1_0 = torch.tensor([-s0[1], s0[0]], dtype=torch.float64)
    P0 = torch.stack([p0, p1_0], dim=1)
    Q0 = torch.stack([q0, q1_0], dim=1)
    S0 = torch.stack([s0, s1_0], dim=1)

    T_hat = torch.einsum(
        "ai,bj,cl,ijl->abc",
        torch.linalg.inv(P0),
        torch.linalg.inv(Q0),
        torch.linalg.inv(S0),
        T_wall,
    )
    a0 = float(T_hat[0, 0, 0].item())
    a1 = float(T_hat[1, 0, 0].item())
    b1 = float(T_hat[0, 1, 0].item())
    c1 = float(T_hat[0, 0, 1].item())
    p1 = a1 * p1_0 + (a0 / 3.0) * p0
    q1 = b1 * q1_0 + (a0 / 3.0) * q0
    s1 = c1 * s1_0 + (a0 / 3.0) * s0
    return torch.stack(
        [
            torch.stack([p0, p1 / p1.norm()], dim=1),
            torch.stack([q0, q1 / q1.norm()], dim=1),
            torch.stack([s0, s1 / s1.norm()], dim=1),
        ],
        dim=0,
    )


# ---------------------------------------------------------------------------
# Part 1A & 1B: Cayley Det Measure + Generic 2x2x2 Weight-Decay Universality
# ---------------------------------------------------------------------------
def run_part1_generic_2x2x2() -> tuple[dict, torch.Tensor, torch.Tensor, float]:
    print("=== Part 1A/1B: Positive-Measure Generic Real 2x2x2 Tensors ===")
    canon = make_canonical_tensors()
    canon_dets = {name: float(cayley_hyperdet(T).item()) for name, T in canon.items()}
    print(f"  Canonical Cayley Det verification: {canon_dets}")
    assert canon_dets["I"] > 0 and abs(canon_dets["W"]) < 1e-15 and canon_dets["T3"] < 0

    # Sample M = 500 Gaussian tensors + 500,000-draw high-precision MC verification
    torch.manual_seed(42)
    M_batch = 500
    G_batch = torch.randn(M_batch, 2, 2, 2, dtype=torch.float64)
    dets_batch = cayley_hyperdet(G_batch)
    n_neg_500 = int((dets_batch < 0).sum().item())
    p_neg_500 = n_neg_500 / M_batch

    torch.manual_seed(12345)
    n_mc = 500_000
    G_mc = torch.randn(n_mc, 2, 2, 2, dtype=torch.float64)
    p_neg_mc = float((cayley_hyperdet(G_mc) < 0).double().mean().item())
    se_mc = math.sqrt(p_neg_mc * (1.0 - p_neg_mc) / n_mc)
    p_theory = 1.0 - math.pi / 4.0
    print(
        f"  Empirical P(Det < 0): M={M_batch} -> {n_neg_500}/{M_batch} ({100*p_neg_500:.2f}%) | "
        f"MC {n_mc:,} -> {p_neg_mc:.5f}±{se_mc:.5f} vs exact 1-pi/4 = {p_theory:.5f}"
    )

    neg_indices = [i for i in range(M_batch) if float(dets_batch[i].item()) < 0][:6]
    pos_indices = [i for i in range(M_batch) if float(dets_batch[i].item()) > 0][:6]
    Tg = G_batch[neg_indices[0]].clone()  # Primary headline generic Det < 0 tensor (idx 4)
    T3 = canon["T3"]

    # Standalone purely unseeded multi-start discovery check on generic Tg (k=2)
    print("  [1B-disc] Purely random i.i.d. Gaussian multi-start check on generic Tg (16 starts each):")
    disc_gammas = [1e-2, 1e-3, 1e-4, 1e-5]
    n_rand = 16
    pure_random_Tg = []
    for gam in disc_gammas:
        trials = []
        seed_base = 20000 + int(round(-1000 * math.log10(gam)))
        for tr in range(n_rand):
            torch.manual_seed(seed_base + tr)
            sc = 0.7 if tr % 2 == 0 else 1.4
            init = sc * torch.randn(3, 2, 2, dtype=torch.float64)
            th, obj, L, R2 = run_cp_lbfgs(init, Tg, gam)
            cr, mcos, minc = cp_diagnostics(th)
            trials.append((obj, L, R2, cr, minc))
        best = min(trials, key=lambda r: r[0])
        n_hit = sum(1 for r in trials if abs(r[0] - best[0]) / max(best[0], 1e-12) < 1e-4)
        pure_random_Tg.append(
            {
                "gamma": gam,
                "n_starts": n_rand,
                "n_global_opt": n_hit,
                "discovery_rate": n_hit / n_rand,
                "best_obj": best[0],
                "best_loss": best[1],
                "best_norm_sq": best[2],
                "cancel_ratio": best[3],
                "min_mode_cos": best[4],
            }
        )
        print(
            f"    Tg k=2 gam={gam:.0e}: {n_hit:2d}/{n_rand} hit global opt | "
            f"R2={best[2]:6.2f} cr={best[3]:5.2f} min|cos|={best[4]:.5f}"
        )

    # Full weight-decay sweeps gamma in [1e-1, 1e-6]
    gammas = np.geomspace(1e-1, 1e-6, 16)

    def sweep_tensor(target: torch.Tensor, k: int, seed_offset: int, n_starts: int = 8):
        prev_best = None
        rows = []
        best_thetas = []
        for idx, gam in enumerate(gammas):
            gam_f = float(gam)
            best = None
            for tr in range(n_starts):
                torch.manual_seed(seed_offset + 100 * idx + tr)
                if tr == 0 and prev_best is not None:
                    init = prev_best.clone()
                else:
                    sc = 0.7 if tr < 5 else 1.4
                    init = sc * torch.randn(3, 2, k, dtype=torch.float64)
                th, obj, L, R2 = run_cp_lbfgs(init, target, gam_f)
                if best is None or obj < best[1]:
                    best = (th, obj, L, R2)
            prev_best = best[0]
            cr, mcos, minc = cp_diagnostics(best[0])
            best_thetas.append(best[0])
            rows.append(
                {
                    "gamma": gam_f,
                    "loss": best[2],
                    "norm_sq": best[3],
                    "obj": best[1],
                    "cancel_ratio": cr,
                    "mode_cos": mcos,
                    "min_mode_cos": minc,
                }
            )
        return rows, best_thetas

    print("  [1B-neg] Running unseeded k=2 WD sweeps on 6 generic Det<0 Gaussian tensors + T3:")
    neg_records = []
    Tg_wall = None
    Tg_L_wall = None
    for rank_i, g_idx in enumerate(neg_indices):
        T_tar = G_batch[g_idx]
        rows, thetas = sweep_tensor(
            T_tar, k=2, seed_offset=30000 + 2000 * rank_i, n_starts=5 if rank_i == 0 else 3
        )
        T_wall, L_wall, det_wall = project_onto_det_wall(T_tar, cp_tensor(thetas[-1]))
        if rank_i == 0:
            Tg_wall = T_wall
            Tg_L_wall = L_wall
        # Richardson extrapolation check from last two points assuming gamma^(3/4)
        g1, L1 = rows[-2]["gamma"], rows[-2]["loss"]
        g2, L2 = rows[-1]["gamma"], rows[-1]["loss"]
        L_wall_rich = (L2 * (g1 ** 0.75) - L1 * (g2 ** 0.75)) / ((g1 ** 0.75) - (g2 ** 0.75))

        for r in rows:
            dL = max(r["loss"] - L_wall, 1e-16)
            r["delta_loss"] = dL
            r["virial"] = (r["gamma"] * r["norm_sq"]) / dL

        # Fit slopes on tail gamma in [1e-3, 1e-6] (last 8 points)
        tail = rows[-8:]
        g_tail = np.array([r["gamma"] for r in tail])
        dL_tail = np.array([r["delta_loss"] for r in tail])
        R2_tail = np.array([r["norm_sq"] for r in tail])
        dL_slope = fit_slope(g_tail, dL_tail)
        R2_slope = fit_slope(g_tail, R2_tail)
        rec = {
            "sample_idx": g_idx,
            "det": float(dets_batch[g_idx].item()),
            "target_norm_sq": float((T_tar ** 2).sum().item()),
            "L_wall_exact": L_wall,
            "L_wall_richardson": L_wall_rich,
            "det_T_wall": det_wall,
            "dL_slope": dL_slope,
            "R2_slope": R2_slope,
            "virial_final": rows[-1]["virial"],
            "cancel_ratio_final": rows[-1]["cancel_ratio"],
            "min_mode_cos_final": rows[-1]["min_mode_cos"],
            "sweep": rows,
        }
        neg_records.append(rec)
        print(
            f"    Gauss#{g_idx:2d} (Det={rec['det']:7.2f}, L_wall={L_wall:.5f}): "
            f"dL_slope={dL_slope:.4f} R2_slope={R2_slope:.4f} vir={rows[-1]['virial']:.3f} "
            f"cr={rows[-1]['cancel_ratio']:5.1f} cos={rows[-1]['min_mode_cos']:.5f}"
        )

    # Canonical symmetric rank-3 tensor T3 (k=2 and k=3)
    rows_T3_k2, thetas_T3_k2 = sweep_tensor(T3, k=2, seed_offset=60000, n_starts=5)
    T3_wall, T3_L_wall, T3_det_wall = project_onto_det_wall(T3, cp_tensor(thetas_T3_k2[-1]))
    for r in rows_T3_k2:
        dL = max(r["loss"] - T3_L_wall, 1e-16)
        r["delta_loss"] = dL
        r["virial"] = (r["gamma"] * r["norm_sq"]) / dL
        r["ratio_dL_to_sqrt2_g34"] = dL / (math.sqrt(2.0) * (r["gamma"] ** 0.75))
        r["ratio_R2_to_3sqrt2_gm14"] = r["norm_sq"] / (3.0 * math.sqrt(2.0) * (r["gamma"] ** -0.25))
    tail_T3 = rows_T3_k2[-8:]
    g_tail = np.array([r["gamma"] for r in tail_T3])
    T3_k2_summary = {
        "det": float(cayley_hyperdet(T3).item()),
        "L_wall_exact": T3_L_wall,
        "L_wall_analytic": 1.0,
        "det_T_wall": T3_det_wall,
        "dL_slope": fit_slope(g_tail, np.array([r["delta_loss"] for r in tail_T3])),
        "R2_slope": fit_slope(g_tail, np.array([r["norm_sq"] for r in tail_T3])),
        "virial_final": rows_T3_k2[-1]["virial"],
        "ratio_dL_final": rows_T3_k2[-1]["ratio_dL_to_sqrt2_g34"],
        "ratio_R2_final": rows_T3_k2[-1]["ratio_R2_to_3sqrt2_gm14"],
        "sweep": rows_T3_k2,
    }
    print(
        f"    Canonical T3 (Det=-4.00, L_wall={T3_L_wall:.6f}): "
        f"dL_slope={T3_k2_summary['dL_slope']:.4f} R2_slope={T3_k2_summary['R2_slope']:.4f} "
        f"vir={T3_k2_summary['virial_final']:.3f} dL/sqrt(2)g^(3/4)={T3_k2_summary['ratio_dL_final']:.5f}"
    )

    # Width k=3 on Tg and T3 (realizable contrast + high-gamma branch collapse)
    rows_Tg_k3, _ = sweep_tensor(Tg, k=3, seed_offset=70000, n_starts=4)
    rows_T3_k3, _ = sweep_tensor(T3, k=3, seed_offset=75000, n_starts=4)
    Tg_k3_slope = fit_slope(g_tail, np.array([r["loss"] for r in rows_Tg_k3[-8:]]))
    T3_k3_slope = fit_slope(g_tail, np.array([r["loss"] for r in rows_T3_k3[-8:]]))
    print(
        f"    Width k=3 contrast: Tg L_slope={Tg_k3_slope:.4f} (R2_final={rows_Tg_k3[-1]['norm_sq']:.3f}) | "
        f"T3 L_slope={T3_k3_slope:.4f} (R2_final={rows_T3_k3[-1]['norm_sq']:.3f})"
    )

    # Positive-Det control batch (6 generic Det > 0 Gaussian tensors at k=2)
    print("  [1B-pos] Running unseeded k=2 WD sweeps on 6 generic Det>0 Gaussian tensors:")
    pos_records = []
    for rank_i, g_idx in enumerate(pos_indices):
        T_tar = G_batch[g_idx]
        rows, _ = sweep_tensor(T_tar, k=2, seed_offset=80000 + 2000 * rank_i, n_starts=3)
        L_slope = fit_slope(g_tail, np.array([max(r["loss"], 1e-25) for r in rows[-8:]]))
        R2_slope = fit_slope(g_tail, np.array([r["norm_sq"] for r in rows[-8:]]))
        pos_records.append(
            {
                "sample_idx": g_idx,
                "det": float(dets_batch[g_idx].item()),
                "L_slope": L_slope,
                "R2_slope": R2_slope,
                "loss_final": rows[-1]["loss"],
                "norm_sq_final": rows[-1]["norm_sq"],
                "cancel_ratio_final": rows[-1]["cancel_ratio"],
                "min_mode_cos_final": rows[-1]["min_mode_cos"],
                "sweep": rows,
            }
        )
    print(
        f"    6 Det>0 tensors: mean L_slope={np.mean([r['L_slope'] for r in pos_records]):.4f} "
        f"(theory 2.000), mean R2_slope={np.mean([r['R2_slope'] for r in pos_records]):.5f}, "
        f"max cr={max(r['cancel_ratio_final'] for r in pos_records):.2f}"
    )

    summary_neg = {
        "mean_dL_slope": float(np.mean([r["dL_slope"] for r in neg_records])),
        "std_dL_slope": float(np.std([r["dL_slope"] for r in neg_records], ddof=1)),
        "theory_dL_slope": 0.75,
        "mean_R2_slope": float(np.mean([r["R2_slope"] for r in neg_records])),
        "std_R2_slope": float(np.std([r["R2_slope"] for r in neg_records], ddof=1)),
        "theory_R2_slope": -0.25,
        "mean_virial_final": float(np.mean([r["virial_final"] for r in neg_records])),
        "theory_virial": 3.0,
        "min_cos_across_all_neg": float(min(r["min_mode_cos_final"] for r in neg_records)),
    }
    print("  Summary across 6 generic Det<0 tensors:", json.dumps(summary_neg, indent=2))

    out = {
        "canonical_dets": canon_dets,
        "gaussian_measure": {
            "M_batch": M_batch,
            "n_det_neg": n_neg_500,
            "p_det_neg_batch": p_neg_500,
            "n_mc": n_mc,
            "p_det_neg_mc": p_neg_mc,
            "se_mc": se_mc,
            "theory_1_minus_pi_over_4": p_theory,
        },
        "gammas": [float(g) for g in gammas],
        "pure_random_Tg_k2": pure_random_Tg,
        "generic_det_neg_k2": neg_records,
        "generic_det_neg_summary": summary_neg,
        "generic_det_pos_k2": pos_records,
        "canonical_T3_k2": T3_k2_summary,
        "Tg_k3": {
            "L_slope": Tg_k3_slope,
            "norm_sq_final": rows_Tg_k3[-1]["norm_sq"],
            "sweep": rows_Tg_k3,
        },
        "T3_k3": {
            "L_slope": T3_k3_slope,
            "norm_sq_final": rows_T3_k3[-1]["norm_sq"],
            "sweep": rows_T3_k3,
        },
    }
    return out, Tg, Tg_wall, Tg_L_wall


# ---------------------------------------------------------------------------
# Part 1C: Exact Laplace Spectrum & Equilibrated SMC + HMC on Generic Tg
# ---------------------------------------------------------------------------
def solve_generic_saddle(
    beta: float,
    target: torch.Tensor,
    L_wall: float,
    prev_th: torch.Tensor | None = None,
    prev_beta: float | None = None,
    P_stack: torch.Tensor | None = None,
    P_inv_stack: torch.Tensor | None = None,
) -> dict:
    """
    High-precision damped Newton saddle point of
      U_exc(theta) = beta * (||T(theta) - target||_F^2 - L_wall) + 0.5 * ||theta||_2^2.
    When prev_th and (P_stack, P_inv_stack) are supplied, extrapolates along the exact
    GL(2)^3 border-rank tube S_u with u = (1/8) * log(beta / prev_beta) so Newton starts
    strictly inside the positive-definite quadratic basin up to beta = 1e6 and beyond.
    """
    if prev_th is None:
        gam = 0.5 / beta
        best = None
        for tr in range(8):
            torch.manual_seed(91000 + tr)
            init = 0.8 * torch.randn(3, 2, 2, dtype=torch.float64)
            th_c, obj_c, _, _ = run_cp_lbfgs(init, target, gam, max_iter=350, n_outer=6, tol_grad=1e-13)
            if best is None or obj_c < best[1]:
                best = (th_c, obj_c)
        th = best[0]
    else:
        th = prev_th.detach().clone()
        if prev_beta is not None and P_stack is not None and P_inv_stack is not None:
            u_ext = 0.125 * math.log(beta / prev_beta)
            tmp = torch.einsum("mij,mjr->mir", P_inv_stack, th)
            sc = torch.tensor([math.exp(u_ext), math.exp(-2.0 * u_ext)], dtype=torch.float64).view(1, 2, 1)
            th = torch.einsum("mij,mjr->mir", P_stack, tmp * sc)

    x = th.detach().reshape(-1).clone()

    def U_exc_fn(v: torch.Tensor) -> torch.Tensor:
        T = cp_tensor(v.reshape(3, 2, 2))
        return beta * (((T - target) ** 2).sum() - L_wall) + 0.5 * (v ** 2).sum()

    eye12 = torch.eye(12, dtype=torch.float64)
    for _ in range(30):
        v = x.detach().requires_grad_(True)
        u = U_exc_fn(v)
        g = torch.autograd.grad(u, v)[0]
        if float(g.norm().item()) < 1e-11:
            break
        H = torch.autograd.functional.hessian(U_exc_fn, x)
        H = 0.5 * (H + H.T)
        ev_h = torch.linalg.eigvalsh(H)
        min_ev = float(ev_h[0].item())
        damp = (abs(min_ev) + 1e-2) if min_ev <= 1e-6 else 1e-11
        step = torch.linalg.solve(H + damp * eye12, g)
        u0 = float(u.detach().item())
        alpha = 1.0
        accepted = False
        while alpha > 1e-8:
            xn = x - alpha * step
            if float(U_exc_fn(xn).item()) <= u0 + 1e-12 * max(abs(u0), 1.0):
                x = xn
                accepted = True
                break
            alpha *= 0.5
        if not accepted:
            break

    th = x.reshape(3, 2, 2)
    T = cp_tensor(th)
    L = float(((T - target) ** 2).sum().item())
    dL = L - L_wall
    R2 = float((th ** 2).sum().item())
    U_exc = beta * dL + 0.5 * R2
    U_val = beta * L + 0.5 * R2
    H_U = torch.autograd.functional.hessian(U_exc_fn, x)
    H_U = 0.5 * (H_U + H_U.T)
    evals_U = torch.linalg.eigvalsh(H_U).numpy()
    logdet_U = float(np.sum(np.log(np.maximum(evals_U, 1e-30))))
    F_exc_lap = U_exc + 0.5 * logdet_U - math.log(32.0)
    F_lap = F_exc_lap + beta * L_wall
    return {
        "theta": th,
        "loss": L,
        "delta_loss": dL,
        "norm_sq": R2,
        "U": U_val,
        "H_U": H_U,
        "evals_U": evals_U,
        "F_laplace": F_lap,
        "F_exc_laplace": F_exc_lap,
    }


def run_part1c_smc_hmc(Tg: torch.Tensor, Tg_wall: torch.Tensor, L_wall: float) -> dict:
    print("\n=== Part 1C: Exact Laplace & Equilibrated SMC+HMC on Generic Det<0 Tensor Tg ===")
    P_stack = extract_gl2_basis_from_wall(Tg_wall)  # (3, 2, 2)
    P_inv_stack = torch.linalg.inv(P_stack)

    # Verify exact GL(2)^3 Lie-group invariance of T_wall
    u_chk = 0.5
    Su_chk = [
        P_stack[m]
        @ torch.diag(torch.tensor([math.exp(u_chk), math.exp(-2.0 * u_chk)], dtype=torch.float64))
        @ P_inv_stack[m]
        for m in range(3)
    ]
    Tw_chk = torch.einsum("ai,bj,cl,ijl->abc", Su_chk[0], Su_chk[1], Su_chk[2], Tg_wall)
    lie_inv_err = float((Tw_chk - Tg_wall).norm().item())
    print(f"  GL(2)^3 Lie move invariance ||S_u(T_wall) - T_wall|| at u=0.5: {lie_inv_err:.3e}")

    betas = np.concatenate(
        [
            np.geomspace(0.5, 1e4, 28),
            np.geomspace(1e4, 1e5, 9)[1:],
            np.geomspace(1e5, 1e6, 9)[1:],
        ]
    )

    # Exact Laplace curve for k=2 on generic Tg
    laplace_curve = []
    laplace_diag = {}
    laplace_sols = {}
    th_prev = None
    beta_prev_lap = None
    diag_betas = {1e3, 1e4, 1e5, 1e6}
    # Exact derivative of F_exc_laplace avoids subtracting huge beta*L_wall terms
    for beta in betas:
        bf = float(beta)
        if bf < 20.0:
            continue
        sol = solve_generic_saddle(bf, Tg, L_wall, th_prev, beta_prev_lap, P_stack, P_inv_stack)
        th_prev = sol["theta"]
        beta_prev_lap = bf
        db = 1e-3
        sol_hi = solve_generic_saddle(bf * math.exp(db), Tg, L_wall, th_prev, bf, P_stack, P_inv_stack)
        sol_lo = solve_generic_saddle(bf * math.exp(-db), Tg, L_wall, th_prev, bf, P_stack, P_inv_stack)
        lam_exc_lap = (sol_hi["F_exc_laplace"] - sol_lo["F_exc_laplace"]) / (2.0 * db)
        saddle_b_dL = bf * sol["delta_loss"]
        fluct = lam_exc_lap - saddle_b_dL
        r2_lap = sol["norm_sq"] + 15.0 / 4.0
        rec = {
            "beta": bf,
            "saddle_beta_dL": saddle_b_dL,
            "fluct_lambda": fluct,
            "lambda_exc_laplace": lam_exc_lap,
            "norm_sq_saddle": sol["norm_sq"],
            "norm_sq_laplace": r2_lap,
            "F_exc_laplace": sol["F_exc_laplace"],
            "F_laplace": sol["F_laplace"],
            "hess_min": float(sol["evals_U"][0]),
            "hess_radial": float(sol["evals_U"][4]),
            "hess_max": float(sol["evals_U"][-1]),
            "cond_number": float(sol["evals_U"][-1] / max(sol["evals_U"][0], 1e-12)),
        }
        laplace_curve.append(rec)
        for target_b in diag_betas:
            if abs(bf - target_b) / target_b < 1e-6:
                laplace_sols[round(target_b)] = sol
                laplace_diag[f"beta_{int(target_b)}"] = {
                    **rec,
                    "evals_HU": [float(v) for v in sol["evals_U"]],
                }
    if "beta_1000" not in laplace_diag:
        sol = solve_generic_saddle(1000.0, Tg, L_wall)
        db = 1e-3
        F_hi = solve_generic_saddle(1000.0 * math.exp(db), Tg, L_wall, sol["theta"], 1000.0, P_stack, P_inv_stack)["F_exc_laplace"]
        F_lo = solve_generic_saddle(1000.0 * math.exp(-db), Tg, L_wall, sol["theta"], 1000.0, P_stack, P_inv_stack)["F_exc_laplace"]
        lam_exc_lap = (F_hi - F_lo) / (2.0 * db)
        laplace_diag["beta_1000"] = {
            "beta": 1000.0,
            "saddle_beta_dL": 1000.0 * sol["delta_loss"],
            "fluct_lambda": lam_exc_lap - 1000.0 * sol["delta_loss"],
            "lambda_exc_laplace": lam_exc_lap,
            "norm_sq_saddle": sol["norm_sq"],
            "norm_sq_laplace": sol["norm_sq"] + 15.0 / 4.0,
            "F_exc_laplace": sol["F_exc_laplace"],
            "F_laplace": sol["F_laplace"],
            "evals_HU": [float(v) for v in sol["evals_U"]],
        }

    # Also record T3 Hessian spectrum at beta = 1e5 to document the 2 SO(2)^2 Goldstone modes
    T3 = make_canonical_tensors()["T3"]
    sol_T3_1e3 = solve_generic_saddle(1e3, T3, 1.0)
    T3_wall_exact, _, _ = project_onto_det_wall(T3, cp_tensor(sol_T3_1e3["theta"]))
    P_T3 = extract_gl2_basis_from_wall(T3_wall_exact)
    P_T3_inv = torch.linalg.inv(P_T3)
    sol_T3_1e4 = solve_generic_saddle(1e4, T3, 1.0, sol_T3_1e3["theta"], 1e3, P_T3, P_T3_inv)
    sol_T3_1e5 = solve_generic_saddle(1e5, T3, 1.0, sol_T3_1e4["theta"], 1e4, P_T3, P_T3_inv)
    T3_hess_1e5 = [float(v) for v in sol_T3_1e5["evals_U"]]

    print("  Exact 4-scale Laplace benchmarks for generic Tg (k=2):")
    for key in ("beta_1000", "beta_10000", "beta_100000", "beta_1000000"):
        d = laplace_diag[key]
        print(
            f"    beta={d['beta']:9.0f}: b*dL*={d['saddle_beta_dL']:6.3f} + "
            f"fluct={d['fluct_lambda']:.4f} (33/8=4.125) = {d['lambda_exc_laplace']:6.3f} | "
            f"R2*+15/4={d['norm_sq_laplace']:6.2f} | F_exc={d['F_exc_laplace']:6.2f}"
        )

    # SMC + 2x2 mode-wise SPD preconditioned HMC + GL(2)^3 Lie move + torus gauge move
    n_chains = 192
    n_leapfrog = 8
    burnin_hmc = 65
    sample_hmc = 110

    def energy_and_grad(theta: torch.Tensor, beta_f: float):
        th = theta.detach().requires_grad_(True)
        T = cp_tensor(th)
        loss = ((T - Tg) ** 2).sum(dim=(-3, -2, -1))
        norm_sq = (th ** 2).sum(dim=(-3, -2, -1))
        U = beta_f * loss + 0.5 * norm_sq
        grad = torch.autograd.grad(U.sum(), th)[0]
        return U.detach(), loss.detach(), norm_sq.detach(), grad.detach()

    def compute_mode_spd_mass(grad: torch.Tensor):
        C, _, _, k = grad.shape
        M = torch.einsum("cmir,cmjr->mij", grad, grad) / (C * k)
        M = 0.5 * (M + M.transpose(-1, -2))
        evals, evecs = torch.linalg.eigh(M)
        evals = evals.clamp(min=1.0)
        inv_M = torch.einsum("mik,mk,mjk->mij", evecs, 1.0 / evals, evecs)
        sqrt_M = torch.einsum("mik,mk,mjk->mij", evecs, torch.sqrt(evals), evecs)
        return inv_M, sqrt_M

    def hmc_step_spd(theta, U, loss, norm_sq, grad, beta_f: float, eps: float, inv_M, sqrt_M):
        C = theta.shape[0]
        z = torch.randn_like(theta)
        p0 = torch.einsum("mij,cmjr->cmir", sqrt_M, z)
        K0 = 0.5 * (p0 * torch.einsum("mij,cmjr->cmir", inv_M, p0)).sum(dim=(-3, -2, -1))
        q = theta.clone()
        p = p0 - 0.5 * eps * grad
        for lf in range(n_leapfrog):
            q = q + eps * torch.einsum("mij,cmjr->cmir", inv_M, p)
            U_q, loss_q, norm_q, grad_q = energy_and_grad(q, beta_f)
            p = p - (eps if lf < n_leapfrog - 1 else 0.5 * eps) * grad_q
        K1 = 0.5 * (p * torch.einsum("mij,cmjr->cmir", inv_M, p)).sum(dim=(-3, -2, -1))
        log_alpha = (U + K0) - (U_q + K1)
        accept = torch.isfinite(log_alpha) & (
            torch.log(torch.rand(C, dtype=torch.float64)) < log_alpha
        )
        mask = accept.view(C, 1, 1, 1)
        return (
            torch.where(mask, q, theta),
            torch.where(accept, U_q, U),
            torch.where(accept, loss_q, loss),
            torch.where(accept, norm_q, norm_sq),
            torch.where(mask, grad_q, grad),
            float(accept.float().mean().item()),
        )

    def lie_stretch_mh_step(theta, U, loss, norm_sq, grad, beta_f: float, k: int, sigma: float = 0.05):
        C = theta.shape[0]
        u_s = sigma * torch.randn(C, dtype=torch.float64)
        tmp = torch.einsum("mij,cmjr->cmir", P_inv_stack, theta)
        sc = torch.stack([torch.exp(u_s), torch.exp(-2.0 * u_s)], dim=1).view(C, 1, 2, 1)
        prop = torch.einsum("mij,cmjr->cmir", P_stack, tmp * sc)
        U_p, loss_p, norm_p, grad_p = energy_and_grad(prop, beta_f)
        log_alpha = (U - U_p) - (3.0 * k) * u_s
        accept = torch.isfinite(log_alpha) & (
            torch.log(torch.rand(C, dtype=torch.float64)) < log_alpha
        )
        mask = accept.view(C, 1, 1, 1)
        return (
            torch.where(mask, prop, theta),
            torch.where(accept, U_p, U),
            torch.where(accept, loss_p, loss),
            torch.where(accept, norm_p, norm_sq),
            torch.where(mask, grad_p, grad),
            float(accept.float().mean().item()),
        )

    def gauge_balance_mh_step(theta, U, loss, norm_sq, grad, beta_f: float, k: int, sigma: float = 0.15):
        C = theta.shape[0]
        u_s = sigma * torch.randn(C, 1, k, dtype=torch.float64)
        v_s = sigma * torch.randn(C, 1, k, dtype=torch.float64)
        scales = torch.stack([torch.exp(u_s), torch.exp(v_s), torch.exp(-(u_s + v_s))], dim=1)
        prop = theta * scales
        norm_p = (prop ** 2).sum(dim=(-3, -2, -1))
        log_alpha = 0.5 * (norm_sq - norm_p)
        accept = torch.isfinite(log_alpha) & (
            torch.log(torch.rand(C, dtype=torch.float64)) < log_alpha
        )
        th_new = torch.where(accept.view(C, 1, 1, 1), prop, theta)
        U_n, l_n, n_n, g_n = energy_and_grad(th_new, beta_f)
        return th_new, U_n, l_n, n_n, g_n

    def run_rejuv_and_sample(
        theta: torch.Tensor,
        beta_f: float,
        k: int,
        init_eps: float,
        n_burn: int,
        n_samp: int,
        init_mass: tuple[torch.Tensor, torch.Tensor] | None = None,
    ):
        C = theta.shape[0]
        U, loss, norm_sq, grad = energy_and_grad(theta, beta_f)
        if init_mass is not None:
            inv_M, sqrt_M = init_mass
        else:
            inv_M, sqrt_M = compute_mode_spd_mass(grad)

        eps = init_eps
        acc_ema = 0.70
        for step in range(n_burn):
            theta, U, loss, norm_sq, grad, r = hmc_step_spd(
                theta, U, loss, norm_sq, grad, beta_f, eps, inv_M, sqrt_M
            )
            if (step + 1) % 3 == 0:
                if k == 2:
                    theta, U, loss, norm_sq, grad, _ = lie_stretch_mh_step(
                        theta, U, loss, norm_sq, grad, beta_f, k
                    )
                theta, U, loss, norm_sq, grad = gauge_balance_mh_step(
                    theta, U, loss, norm_sq, grad, beta_f, k
                )
            if step == n_burn // 3:
                inv_M, sqrt_M = compute_mode_spd_mass(grad)
            acc_ema = 0.8 * acc_ema + 0.2 * r
            if step < n_burn - 15 and (step + 1) % 3 == 0:
                if acc_ema < 0.55:
                    eps *= 0.85
                elif acc_ema > 0.80:
                    eps *= 1.15
                eps = float(np.clip(eps, 1e-4, 1.0))

        loss_samps, norm_samps = [], []
        acc_sum, su_sum, su_cnt = 0.0, 0.0, 0
        for step in range(n_samp):
            theta, U, loss, norm_sq, grad, r = hmc_step_spd(
                theta, U, loss, norm_sq, grad, beta_f, eps, inv_M, sqrt_M
            )
            if (step + 1) % 4 == 0:
                if k == 2:
                    theta, U, loss, norm_sq, grad, sr = lie_stretch_mh_step(
                        theta, U, loss, norm_sq, grad, beta_f, k
                    )
                    su_sum += sr
                    su_cnt += 1
                theta, U, loss, norm_sq, grad = gauge_balance_mh_step(
                    theta, U, loss, norm_sq, grad, beta_f, k
                )
            acc_sum += r
            loss_samps.append(loss.numpy().copy())
            norm_samps.append(norm_sq.numpy().copy())

        loss_arr = np.stack(loss_samps, axis=0)
        norm_arr = np.stack(norm_samps, axis=0)
        cm_L = loss_arr.mean(axis=0)
        cm_R = norm_arr.mean(axis=0)
        mean_L = float(cm_L.mean())
        se_L = float(cm_L.std(ddof=1) / math.sqrt(C))
        mean_R2 = float(cm_R.mean())
        se_R2 = float(cm_R.std(ddof=1) / math.sqrt(C))
        ref_wall = L_wall if k == 2 else 0.0
        return {
            "theta": theta,
            "mean_loss": mean_L,
            "se_loss": se_L,
            "mean_norm_sq": mean_R2,
            "se_norm_sq": se_R2,
            "lambda_exc": beta_f * (mean_L - ref_wall),
            "lambda_se": beta_f * se_L,
            "acc_rate": acc_sum / n_samp,
            "su_acc_rate": su_sum / max(su_cnt, 1),
            "step_size": eps,
        }

    out = {
        "betas": [float(b) for b in betas],
        "L_wall": L_wall,
        "lie_invariance_err": lie_inv_err,
        "k2": [],
        "k3": [],
        "k2_laplace": laplace_curve,
        "k2_laplace_diagnostics": laplace_diag,
        "T3_hessian_1e5": T3_hess_1e5,
        "cold_start_verification": {"k2": [], "k3": []},
    }

    for k in (2, 3):
        torch.manual_seed(3026 + k)
        theta = torch.randn(n_chains, 3, 2, k, dtype=torch.float64)
        beta_prev = 0.0
        F_smc = 0.0
        step_size = 0.35
        ref_wall = L_wall if k == 2 else 0.0

        for idx, beta in enumerate(betas):
            bf = float(beta)
            with torch.no_grad():
                L_cur = ((cp_tensor(theta) - Tg) ** 2).sum(dim=(-3, -2, -1))
                log_w = -(bf - beta_prev) * L_cur
                lw_max = float(log_w.max().item())
                w_unnorm = torch.exp(log_w - lw_max)
                F_smc -= lw_max + math.log(float(w_unnorm.mean().item()))
                probs = w_unnorm / w_unnorm.sum()
                ess = float(1.0 / (probs ** 2).sum().item())
                cdf = torch.cumsum(probs, dim=0)
                cdf[-1] = 1.0
                u0 = torch.rand(1, dtype=torch.float64).item() / n_chains
                pos = u0 + torch.arange(n_chains, dtype=torch.float64) / n_chains
                anc = torch.searchsorted(cdf, pos).clamp(0, n_chains - 1)
                theta = theta[anc].clone()
            beta_prev = bf

            stats = run_rejuv_and_sample(theta, bf, k, step_size, burnin_hmc, sample_hmc)
            theta = stats["theta"]
            step_size = stats["step_size"]
            rec = {
                "beta": bf,
                "mean_loss": stats["mean_loss"],
                "se_loss": stats["se_loss"],
                "mean_norm_sq": stats["mean_norm_sq"],
                "se_norm_sq": stats["se_norm_sq"],
                "lambda_exc": stats["lambda_exc"],
                "lambda_se": stats["lambda_se"],
                "free_energy_smc": float(F_smc),
                "free_energy_exc": float(F_smc - bf * ref_wall),
                "ess": ess,
                "acc_rate": stats["acc_rate"],
                "su_acc_rate": stats["su_acc_rate"],
                "step_size": step_size,
            }
            out[f"k{k}"].append(rec)
            if idx % 7 == 0 or idx in (27, 35, len(betas) - 1):
                print(
                    f"  k={k} beta={bf:10.1f}  lam_exc={rec['lambda_exc']:6.3f}±{rec['lambda_se']:.3f}  "
                    f"E[R2]={rec['mean_norm_sq']:6.2f}  F_exc={rec['free_energy_exc']:7.2f}  "
                    f"ESS={ess:5.1f}  acc={rec['acc_rate']:.2f}  Su={rec['su_acc_rate']:.2f}"
                )

    # Mandatory independent cold-start MCMC equilibration checks at beta in {1e4, 1e5, 1e6}
    print("  [Equilibration Check] Independent cold-start MCMC at beta in {1e4, 1e5, 1e6}:")
    check_betas = [1e4, 1e5, 1e6]
    n_cold = 128
    # Precompute 16 distinct rank-3 fiber solutions so k=3 cold-start covers the 10D fiber
    # weighted by the Gaussian prior exp(-0.5 ||theta||^2) rather than a single arbitrary fiber point.
    th3_fiber = []
    for f_i in range(16):
        torch.manual_seed(8800 + f_i)
        # Use gam = 5e-5 (matching beta=1e4 prior balance) so initial points lie in typical prior mass
        th3_f, _, _, _ = run_cp_lbfgs(0.7 * torch.randn(3, 2, 3, dtype=torch.float64), Tg, 5e-5)
        th3_fiber.append(th3_f)
    th3_stack = torch.stack(th3_fiber, dim=0)  # (16, 3, 2, 3)

    for k in (2, 3):
        smc_by_beta = {round(r["beta"]): r for r in out[f"k{k}"]}
        for b_chk in check_betas:
            torch.manual_seed(85000 + 1000 * k + int(round(math.log10(b_chk))))
            if k == 2:
                sol = laplace_sols[round(b_chk)]
                ev_c, evec_c = torch.linalg.eigh(sol["H_U"])
                inv_sqrt_H = evec_c @ torch.diag(1.0 / torch.sqrt(ev_c.clamp(min=1.0))) @ evec_c.T
                z = torch.randn(n_cold, 12, dtype=torch.float64)
                dx = z @ inv_sqrt_H.T
                th_init = sol["theta"].unsqueeze(0) + 0.15 * dx.reshape(n_cold, 3, 2, 2)
                cold = run_rejuv_and_sample(th_init, b_chk, k, 0.50, 180, 240)
            else:
                # Tile the 16 independent fiber points to 128 chains + O(beta^{-1/2}) perturbation
                base_idx = torch.arange(n_cold) % 16
                th_init = th3_stack[base_idx] + (0.25 / math.sqrt(b_chk)) * torch.randn(
                    n_cold, 3, 2, 3, dtype=torch.float64
                )
                cold = run_rejuv_and_sample(th_init, b_chk, k, 0.45, 220, 200)
            smc = smc_by_beta[round(b_chk)]
            z_lam = abs(smc["lambda_exc"] - cold["lambda_exc"]) / math.sqrt(
                smc["lambda_se"] ** 2 + cold["lambda_se"] ** 2
            )
            z_r2 = abs(smc["mean_norm_sq"] - cold["mean_norm_sq"]) / math.sqrt(
                smc["se_norm_sq"] ** 2 + cold["se_norm_sq"] ** 2
            )
            vrec = {
                "beta": b_chk,
                "smc_lambda_exc": smc["lambda_exc"],
                "smc_lambda_se": smc["lambda_se"],
                "cold_lambda_exc": cold["lambda_exc"],
                "cold_lambda_se": cold["lambda_se"],
                "z_lambda": z_lam,
                "smc_norm_sq": smc["mean_norm_sq"],
                "smc_norm_sq_se": smc["se_norm_sq"],
                "cold_norm_sq": cold["mean_norm_sq"],
                "cold_norm_sq_se": cold["se_norm_sq"],
                "z_norm_sq": z_r2,
            }
            if k == 2:
                vrec["laplace_lambda_exc"] = laplace_diag[f"beta_{int(b_chk)}"]["lambda_exc_laplace"]
                vrec["laplace_norm_sq"] = laplace_diag[f"beta_{int(b_chk)}"]["norm_sq_laplace"]
            out["cold_start_verification"][f"k{k}"].append(vrec)
            print(
                f"    k={k} beta={b_chk:8.0f}: SMC lam={smc['lambda_exc']:.3f}±{smc['lambda_se']:.3f} "
                f"vs Cold={cold['lambda_exc']:.3f}±{cold['lambda_se']:.3f} (z={z_lam:.2f}) | "
                f"SMC R2={smc['mean_norm_sq']:.2f}±{smc['se_norm_sq']:.2f} "
                f"vs Cold={cold['mean_norm_sq']:.2f}±{cold['se_norm_sq']:.2f} (z={z_r2:.2f})"
            )

    # High-beta scaling summary (beta >= 1000)
    hi_k2 = [r for r in out["k2"] if r["beta"] >= 1000.0]
    hi_k3 = [r for r in out["k3"] if r["beta"] >= 1000.0]
    b_hi = np.array([r["beta"] for r in hi_k2])
    lam_hi = np.array([r["lambda_exc"] for r in hi_k2])
    r2_hi = np.array([r["mean_norm_sq"] for r in hi_k2])

    out["high_beta_summary"] = {
        "k2_excess_lambda_minus_33_8_slope": fit_slope(b_hi, lam_hi - 33.0 / 8.0),
        "k2_excess_lambda_theory": 0.25,
        "k2_excess_norm_sq_minus_15_4_slope": fit_slope(b_hi, r2_hi - 15.0 / 4.0),
        "k2_excess_norm_sq_theory": 0.25,
        "k3_high_beta_mean_lambda": float(np.mean([r["lambda_exc"] for r in hi_k3])),
        "k3_high_beta_std_lambda": float(np.std([r["lambda_exc"] for r in hi_k3], ddof=1)),
        "beta_1e5_k2_F_exc_smc": next(r["free_energy_exc"] for r in out["k2"] if abs(r["beta"] - 1e5) < 1),
        "beta_1e5_k2_F_exc_lap": laplace_diag["beta_100000"]["F_exc_laplace"],
        "beta_1e6_k2_F_exc_smc": out["k2"][-1]["free_energy_exc"],
        "beta_1e6_k2_F_exc_lap": laplace_diag["beta_1000000"]["F_exc_laplace"],
        "beta_1e6_k3_F_smc": out["k3"][-1]["free_energy_smc"],
    }
    print("  High-beta SMC+HMC summary:", json.dumps(out["high_beta_summary"], indent=2))
    return out


# ---------------------------------------------------------------------------
# Part 2: Scaling to Higher Dimensions (d x d x d Gaussian Tensors, d=2,3,4)
# ---------------------------------------------------------------------------
def run_part2_higher_dimensions() -> dict:
    print("\n=== Part 2: Higher-Dimensional Scaling (d x d x d Gaussian Tensors, d in {2,3,4}) ===")
    configs = [(2, 2), (3, 2), (3, 3), (3, 4), (4, 4), (4, 5), (4, 6)]
    n_tensors = 40
    gam_hi, gam_lo = 1e-2, 1e-4
    results = []

    for d, k in configs:
        t0 = time.time()
        cr_list, gr_list, cos_list = [], [], []
        n_degen_std = 0
        n_degen_strict = 0
        n_high_cos = 0
        n_det_neg_d2 = 0
        n_det_concordant_d2 = 0
        for idx in range(n_tensors):
            torch.manual_seed(20000 * d + 1000 * k + idx)
            T = torch.randn(d, d, d, dtype=torch.float64)
            is_det_neg = (d == 2 and float(cayley_hyperdet(T).item()) < 0.0)
            if is_det_neg:
                n_det_neg_d2 += 1

            best_hi = None
            for tr in range(2):
                torch.manual_seed(50000 * d + 5000 * k + 20 * idx + tr)
                init = 0.6 * torch.randn(3, d, k, dtype=torch.float64)
                th, obj, L, R2 = run_cp_lbfgs(init, T, gam_hi, max_iter=120, n_outer=3, tol_grad=1e-10)
                if best_hi is None or obj < best_hi[1]:
                    best_hi = (th, obj, L, R2)

            # Warm-start continuation + 1 fresh start at gam_lo
            best_lo = run_cp_lbfgs(best_hi[0], T, gam_lo, max_iter=160, n_outer=3, tol_grad=1e-10)
            torch.manual_seed(90000 * d + 9000 * k + 20 * idx)
            fresh_lo = run_cp_lbfgs(
                0.6 * torch.randn(3, d, k, dtype=torch.float64),
                T,
                gam_lo,
                max_iter=160,
                n_outer=3,
                tol_grad=1e-10,
            )
            if fresh_lo[1] < best_lo[1]:
                best_lo = fresh_lo

            gr = best_lo[3] / max(best_hi[3], 1e-12)
            cr, _, min_cos = cp_diagnostics(best_lo[0])
            cr_list.append(cr)
            gr_list.append(gr)
            cos_list.append(min_cos)
            is_deg_now = (cr > 3.0 and gr > 1.35)
            if is_deg_now:
                n_degen_std += 1
            if d == 2 and is_deg_now == is_det_neg:
                n_det_concordant_d2 += 1
            if cr > 5.0 and gr > 1.50:
                n_degen_strict += 1
            if min_cos > 0.90:
                n_high_cos += 1

        rec = {
            "d": d,
            "k": k,
            "n_tensors": n_tensors,
            "degen_rate": n_degen_std / n_tensors,
            "degen_count": n_degen_std,
            "strict_degen_rate": n_degen_strict / n_tensors,
            "high_cos_rate": n_high_cos / n_tensors,
            "median_cancel_ratio": float(np.median(cr_list)),
            "iqr_cancel_ratio": [float(np.percentile(cr_list, 25)), float(np.percentile(cr_list, 75))],
            "max_cancel_ratio": float(np.max(cr_list)),
            "median_R2_growth": float(np.median(gr_list)),
            "median_min_cos": float(np.median(cos_list)),
            "cancel_ratios": [float(x) for x in cr_list],
            "elapsed_sec": round(time.time() - t0, 1),
        }
        if d == 2:
            rec["exact_det_neg_count"] = n_det_neg_d2
            rec["det_concordance"] = n_det_concordant_d2 / n_tensors
        results.append(rec)
        print(
            f"  d={d}, k={k}: degen={n_degen_std:2d}/{n_tensors} ({100*rec['degen_rate']:5.1f}%) "
            f"[strict {100*rec['strict_degen_rate']:4.1f}%, cos>0.9 {100*rec['high_cos_rate']:4.1f}%] | "
            f"med cr={rec['median_cancel_ratio']:5.2f} med R2gr={rec['median_R2_growth']:.2f} "
            f"med cos={rec['median_min_cos']:.3f} ({rec['elapsed_sec']}s)"
        )

    return {"gam_hi": gam_hi, "gam_lo": gam_lo, "n_tensors": n_tensors, "configs": results}


# ---------------------------------------------------------------------------
# Part 3: Real Bilinear MLP & SwiGLU MLP on Actual Tasks
# ---------------------------------------------------------------------------
def mlp_forward(X: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, Wo: torch.Tensor, arch: str):
    h1 = X @ W1.T
    if arch == "swiglu":
        h1 = torch.nn.functional.silu(h1)
    return (h1 * (X @ W2.T)) @ Wo.T


def mlp_cancellation_metrics(
    X: torch.Tensor, W1: torch.Tensor, W2: torch.Tensor, Wo: torch.Tensor, arch: str
) -> dict:
    with torch.no_grad():
        k = W1.shape[0]
        h1 = X @ W1.T
        if arch == "swiglu":
            h1 = torch.nn.functional.silu(h1)
        h_prod = h1 * (X @ W2.T)  # (N, k)
        pred = h_prod @ Wo.T  # (N, d_out)

        # Functional cancellation ratio on the data distribution:
        # sum_r E[||f_r(x)||_2^2]^{1/2} / E[||f(x)||_2^2]^{1/2}
        per_r_rms = torch.sqrt((h_prod ** 2).mean(dim=0)) * Wo.norm(dim=0)
        tot_rms = math.sqrt(float((pred ** 2).sum(dim=1).mean().item()))
        cr_func = float(per_r_rms.sum().item() / max(tot_rms, 1e-12))

        # Exact algebraic tensor cancellation ratio & max pairwise tensor cosine (for bilinear)
        if arch == "bilinear":
            S_r = 0.5 * (
                torch.einsum("ri,rj->rij", W1, W2) + torch.einsum("rj,ri->rij", W1, W2)
            )
            T_r = torch.einsum("or,rij->roij", Wo, S_r)  # (k, d_out, d, d)
            T_tot = T_r.sum(dim=0)
            cr_tensor = float(
                T_r.flatten(1).norm(dim=1).sum().item() / max(float(T_tot.norm().item()), 1e-12)
            )
            flat = T_r.flatten(1)
            fn = flat / flat.norm(dim=1, keepdim=True).clamp(min=1e-12)
            gram = (fn @ fn.T).abs()
            gram.fill_diagonal_(0.0)
            max_pair_cos = float(gram.max().item()) if k > 1 else 0.0
        else:
            cr_tensor = cr_func
            # Pairwise functional output cosine across neurons (catches both border-rank & SiLU antisymmetric pairs!)
            # f_r(x) = h_prod[:, r:r+1] * Wo[:, r]
            # Wait: also track exact weight-space min-mode |cos| for SiLU antisymmetric pairing
            max_pair_cos = 0.0
            for i in range(k):
                for j in range(i + 1, k):
                    c1 = abs(float(torch.nn.functional.cosine_similarity(W1[i], W1[j], dim=0).item()))
                    c2 = abs(float(torch.nn.functional.cosine_similarity(W2[i], W2[j], dim=0).item()))
                    co = abs(float(torch.nn.functional.cosine_similarity(Wo[:, i], Wo[:, j], dim=0).item()))
                    max_pair_cos = max(max_pair_cos, min(c1, c2, co))

        return {
            "cancel_ratio_func": cr_func,
            "cancel_ratio_tensor": cr_tensor,
            "max_pair_cos": max_pair_cos,
            "W1_norm_sq": float((W1 ** 2).sum().item()),
            "W2_norm_sq": float((W2 ** 2).sum().item()),
            "Wo_norm_sq": float((Wo ** 2).sum().item()),
        }


def fit_mlp_lbfgs(
    X: torch.Tensor,
    Y: torch.Tensor,
    k: int,
    gam: float,
    arch: str,
    init: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    seed: int = 0,
    max_iter: int = 260,
    n_outer: int = 5,
):
    d_in, d_out = X.shape[1], Y.shape[1]
    if init is None:
        torch.manual_seed(seed)
        W1 = (0.5 * torch.randn(k, d_in, dtype=torch.float64)).requires_grad_(True)
        W2 = (0.5 * torch.randn(k, d_in, dtype=torch.float64)).requires_grad_(True)
        Wo = (0.5 * torch.randn(d_out, k, dtype=torch.float64)).requires_grad_(True)
    else:
        W1, W2, Wo = [p.clone().detach().requires_grad_(True) for p in init]

    opt = torch.optim.LBFGS(
        [W1, W2, Wo],
        lr=0.5,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )

    def closure():
        opt.zero_grad()
        pred = mlp_forward(X, W1, W2, Wo, arch)
        mse = ((pred - Y) ** 2).mean()
        reg = (W1 ** 2).sum() + (W2 ** 2).sum() + (Wo ** 2).sum()
        obj = mse + gam * reg
        obj.backward()
        return obj

    prev_val = float("inf")
    for _ in range(n_outer):
        val = float(opt.step(closure).item())
        gnorm = max(
            float(p.grad.norm().item()) for p in (W1, W2, Wo) if p.grad is not None
        )
        if gnorm < 1e-10 or abs(prev_val - val) <= 1e-13 * max(abs(val), 1.0):
            break
        prev_val = val

    with torch.no_grad():
        pred = mlp_forward(X, W1, W2, Wo, arch)
        mse = float(((pred - Y) ** 2).mean().item())
        reg = float(((W1 ** 2).sum() + (W2 ** 2).sum() + (Wo ** 2).sum()).item())
        obj = mse + gam * reg
        metrics = mlp_cancellation_metrics(X, W1, W2, Wo, arch)
    return (W1.detach().clone(), W2.detach().clone(), Wo.detach().clone()), obj, mse, reg, metrics


def run_part3_real_mlps() -> dict:
    print("\n=== Part 3: Real Single-Input Bilinear MLP & SwiGLU MLP on Actual Tasks ===")
    torch.manual_seed(42)
    N_data = 4096
    X4 = torch.randn(N_data, 4, dtype=torch.float64)

    # 3A-i: Empirical prevalence of border-rank cancellation across 20 random d=4->4 generic quadratic tasks (k=4)
    print("  [3A-prev] Prevalence across 20 generic d=4->4 quadratic tasks (Bilinear MLP k=4):")
    n_tasks = 20
    task_prevalence = []
    n_degen_tasks = 0
    for s in range(n_tasks):
        torch.manual_seed(300 + s)
        Q = torch.randn(4, 4, 4, dtype=torch.float64)
        Qs = 0.5 * (Q + Q.transpose(1, 2))
        Ys = torch.einsum("ni,oij,nj->no", X4, Qs, X4)
        p_hi = min(
            [fit_mlp_lbfgs(X4, Ys, 4, 1e-2, "bilinear", seed=10 * s + tr, max_iter=160, n_outer=4) for tr in range(3)],
            key=lambda r: r[1],
        )
        p_lo = fit_mlp_lbfgs(X4, Ys, 4, 1e-5, "bilinear", init=p_hi[0], max_iter=240, n_outer=5)
        gr = p_lo[3] / max(p_hi[3], 1e-12)
        cr = p_lo[4]["cancel_ratio_tensor"]
        mc = p_lo[4]["max_pair_cos"]
        is_deg = (cr > 4.0 and gr > 1.5 and mc > 0.85)
        if is_deg:
            n_degen_tasks += 1
        task_prevalence.append(
            {
                "seed": 300 + s,
                "mse_hi": p_hi[2],
                "mse_lo": p_lo[2],
                "R2_hi": p_hi[3],
                "R2_lo": p_lo[3],
                "R2_growth": gr,
                "cancel_ratio": cr,
                "max_pair_cos": mc,
                "degenerate": is_deg,
            }
        )
    print(
        f"    {n_degen_tasks}/{n_tasks} ({100*n_degen_tasks/n_tasks:.1f}%) generic d=4->4 quadratic tasks "
        f"exhibit strong border-rank cancellation (cr>4, R2gr>1.5, cos>0.85) at k=4"
    )

    # 3A-ii: Deep-dive on generic task seed=305 (strong border-rank degeneration at k=4, realizable at k=6)
    torch.manual_seed(305)
    Q_main = torch.randn(4, 4, 4, dtype=torch.float64)
    Q_main = 0.5 * (Q_main + Q_main.transpose(1, 2))
    Y_main = torch.einsum("ni,oij,nj->no", X4, Q_main, X4)

    # Adam (gamma = 0) & AdamW (gamma = 1e-3) training trajectories for k = 4 vs k = 6
    print("  [3A-adam] Unregularized Adam & AdamW trajectories on generic d=4->4 task (seed=305, k=4 vs k=6):")
    n_steps = 12000
    rec_steps = set(np.unique(np.geomspace(1, n_steps, 55).astype(int)).tolist()) | {0, n_steps}
    adam_runs = {}
    for opt_name, wd in [("adam_unreg", 0.0), ("adamw_1e3", 1e-3)]:
        adam_runs[opt_name] = {}
        for k in (4, 6):
            torch.manual_seed(777 + k)
            W1 = (0.35 * torch.randn(k, 4, dtype=torch.float64)).requires_grad_(True)
            W2 = (0.35 * torch.randn(k, 4, dtype=torch.float64)).requires_grad_(True)
            Wo = (0.35 * torch.randn(4, k, dtype=torch.float64)).requires_grad_(True)
            opt = (
                torch.optim.Adam([W1, W2, Wo], lr=1.5e-2)
                if wd == 0.0
                else torch.optim.AdamW([W1, W2, Wo], lr=1.5e-2, weight_decay=wd)
            )
            steps_l, mse_l, r2_l, cr_l, cos_l = [], [], [], [], []
            for step in range(n_steps + 1):
                pred = mlp_forward(X4, W1, W2, Wo, "bilinear")
                mse = ((pred - Y_main) ** 2).mean()
                if step in rec_steps:
                    m = mlp_cancellation_metrics(X4, W1, W2, Wo, "bilinear")
                    steps_l.append(step)
                    mse_l.append(float(mse.item()))
                    r2_l.append(m["W1_norm_sq"] + m["W2_norm_sq"] + m["Wo_norm_sq"])
                    cr_l.append(m["cancel_ratio_tensor"])
                    cos_l.append(m["max_pair_cos"])
                if step == n_steps:
                    break
                opt.zero_grad()
                mse.backward()
                opt.step()
            adam_runs[opt_name][f"k{k}"] = {
                "steps": steps_l,
                "mse": mse_l,
                "norm_sq": r2_l,
                "cancel_ratio": cr_l,
                "max_pair_cos": cos_l,
            }
            print(
                f"    {opt_name:10s} k={k}: mse {mse_l[0]:.2e}->{mse_l[-1]:.2e} | "
                f"R2 {r2_l[0]:4.1f}->{r2_l[-1]:6.1f} | cr={cr_l[-1]:5.2f} | cos={cos_l[-1]:.4f}"
            )

    # 3A-iii & 3B-i: Head-to-head L-BFGS WD sweep on seed=305 for Bilinear (k=4, 6, 8) & SwiGLU (k=4, 6, 8, 12)
    print("  [3A/3B] Head-to-head L-BFGS WD sweep on generic d=4->4 quadratic task (Bilinear vs SwiGLU):")
    gammas_mlp = np.geomspace(1e-1, 1e-5, 11)
    wd_compare = {"gammas": [float(g) for g in gammas_mlp], "bilinear": {}, "swiglu": {}}
    best_b4_params = None

    for arch, ks in [("bilinear", [4, 6, 8]), ("swiglu", [4, 6, 8, 12])]:
        for k in ks:
            prev = None
            rows = []
            for idx, gam in enumerate(gammas_mlp):
                gam_f = float(gam)
                best = None
                n_tr = 3
                for tr in range(n_tr):
                    init = prev if (tr == 0 and prev is not None) else None
                    res = fit_mlp_lbfgs(
                        X4, Y_main, k, gam_f, arch, init=init, seed=4000 * k + 50 * idx + tr,
                        max_iter=200, n_outer=4,
                    )
                    if best is None or res[1] < best[1]:
                        best = res
                prev = best[0]
                rows.append(
                    {
                        "gamma": gam_f,
                        "mse": best[2],
                        "norm_sq": best[3],
                        "obj": best[1],
                        **best[4],
                    }
                )
            if arch == "bilinear" and k == 4:
                best_b4_params = prev
            wd_compare[arch][f"k{k}"] = rows
            g_tail = np.array([r["gamma"] for r in rows[-6:]])
            mse_s = fit_slope(g_tail, np.array([max(r["mse"], 1e-20) for r in rows[-6:]]))
            r2_s = fit_slope(g_tail, np.array([r["norm_sq"] for r in rows[-6:]]))
            vir_last = (rows[-1]["gamma"] * rows[-1]["norm_sq"]) / max(rows[-1]["mse"], 1e-20)
            print(
                f"    {arch:8s} k={k:2d}: mse(1e-5)={rows[-1]['mse']:.3e} (slope {mse_s:5.3f}) | "
                f"R2={rows[-1]['norm_sq']:7.1f} (slope {r2_s:6.3f}, W1={rows[-1]['W1_norm_sq']:5.2f}) | "
                f"cr={rows[-1]['cancel_ratio_tensor']:5.2f} cos={rows[-1]['max_pair_cos']:.4f} vir={vir_last:5.2f}"
            )

    # Continue Bilinear k=4 smoothly from best_b4_params down to gamma=1e-7 to measure L_wall & Delta MSE
    b4 = wd_compare["bilinear"]["k4"]
    g_b4 = np.array([r["gamma"] for r in b4[-6:]])
    r2_b4 = np.array([r["norm_sq"] for r in b4[-6:]])
    p_b4_prev = (best_b4_params, 0.0, b4[-1]["mse"], b4[-1]["norm_sq"], {})
    for g_ex in (1e-6, 1e-7):
        p_b4_prev = fit_mlp_lbfgs(X4, Y_main, 4, g_ex, "bilinear", init=p_b4_prev[0], max_iter=350, n_outer=6)
    b4_L_wall_est = p_b4_prev[2]
    b4_slopes = {
        "bilinear_k4_R2_slope": fit_slope(g_b4, r2_b4),
        "bilinear_k4_mse_at_1e7": b4_L_wall_est,
        "bilinear_k4_R2_at_1e7": p_b4_prev[3],
        "bilinear_k4_delta_mse_slope": fit_slope(
            g_b4, np.array([max(r["mse"] - b4_L_wall_est, 1e-12) for r in b4[-6:]])
        ),
        "bilinear_k6_mse_slope": fit_slope(
            g_b4, np.array([r["mse"] for r in wd_compare["bilinear"]["k6"][-6:]])
        ),
        "swiglu_k4_mse_slope": fit_slope(
            g_b4, np.array([r["mse"] for r in wd_compare["swiglu"]["k4"][-6:]])
        ),
        "swiglu_k4_R2_slope": fit_slope(
            g_b4, np.array([r["norm_sq"] for r in wd_compare["swiglu"]["k4"][-6:]])
        ),
        "swiglu_k6_mse_slope": fit_slope(
            g_b4, np.array([r["mse"] for r in wd_compare["swiglu"]["k6"][-6:]])
        ),
        "swiglu_k6_virial_1e5": (
            wd_compare["swiglu"]["k6"][-1]["gamma"] * wd_compare["swiglu"]["k6"][-1]["norm_sq"]
        )
        / wd_compare["swiglu"]["k6"][-1]["mse"],
        "swiglu_k12_vs_bilinear_k6_R2_ratio": wd_compare["swiglu"]["k12"][-1]["norm_sq"]
        / wd_compare["bilinear"]["k6"][-1]["norm_sq"],
        "swiglu_k12_mse_slope": fit_slope(
            g_b4, np.array([r["mse"] for r in wd_compare["swiglu"]["k12"][-6:]])
        ),
    }
    print("  Bilinear vs SwiGLU summary ratios:", json.dumps(b4_slopes, indent=2))

    # 3B-ii: Honest SwiGLU-native teacher control (d=3->3, k*=4 SwiGLU teacher, students k=2, 3, 4)
    print("  [3B-ctrl] SwiGLU-native teacher control (d=3->3, k*=4 SwiGLU teacher, students k=2, 3, 4):")
    X3 = torch.randn(N_data, 3, dtype=torch.float64)
    torch.manual_seed(99)
    W1_t = torch.randn(4, 3, dtype=torch.float64)
    W2_t = torch.randn(4, 3, dtype=torch.float64)
    Wo_t = torch.randn(3, 4, dtype=torch.float64)
    Y_swi_native = mlp_forward(X3, W1_t, W2_t, Wo_t, "swiglu")
    swiglu_native = {"gammas": [float(g) for g in gammas_mlp], "students": {}}

    for k in (2, 3, 4):
        prev = None
        rows = []
        for idx, gam in enumerate(gammas_mlp):
            gam_f = float(gam)
            best = None
            n_tr = 5 if k == 4 else 4
            for tr in range(n_tr):
                init = prev if (tr == 0 and prev is not None) else None
                res = fit_mlp_lbfgs(
                    X3, Y_swi_native, k, gam_f, "swiglu", init=init, seed=7000 * k + 50 * idx + tr
                )
                if best is None or res[1] < best[1]:
                    best = res
            prev = best[0]
            rows.append(
                {
                    "gamma": gam_f,
                    "mse": best[2],
                    "norm_sq": best[3],
                    "obj": best[1],
                    **best[4],
                }
            )
        swiglu_native["students"][f"k{k}"] = rows
        r2_gr = rows[-1]["norm_sq"] / max(rows[2]["norm_sq"], 1e-12)
        print(
            f"    SwiGLU->SwiGLU k={k}: mse(1.6e-2->1e-5)={rows[2]['mse']:.3e}->{rows[-1]['mse']:.3e} | "
            f"R2={rows[2]['norm_sq']:.1f}->{rows[-1]['norm_sq']:.1f} (gr={r2_gr:.2f}, W1={rows[-1]['W1_norm_sq']:.2f}) | "
            f"cr={rows[-1]['cancel_ratio_func']:.2f} cos={rows[-1]['max_pair_cos']:.3f}"
        )

    return {
        "prevalence_d4_k4": {
            "n_tasks": n_tasks,
            "n_degenerate": n_degen_tasks,
            "rate": n_degen_tasks / n_tasks,
            "tasks": task_prevalence,
        },
        "adam_trajectories_seed305": adam_runs,
        "wd_compare_seed305": wd_compare,
        "wd_compare_summary": b4_slopes,
        "swiglu_native_control": swiglu_native,
    }


# ---------------------------------------------------------------------------
# Publication-grade 6-panel figure
# ---------------------------------------------------------------------------
def make_figure(p1: dict, p1c: dict, p2: dict, p3: dict) -> Path:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 7.6,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6), dpi=180)
    c_red = "#d62728"
    c_blue = "#1f77b4"
    c_green = "#2ca02c"
    c_purple = "#9467bd"
    c_orange = "#ff7f0e"
    c_gray = "#555555"

    # Panel A: Part 1B — Positive-measure generic Det<0 scaling (gamma^{3/4}) vs W (gamma^{6/7}) vs k=3 (gamma^2)
    ax = axes[0, 0]
    g = np.array(p1["gammas"])
    for idx_n, rec in enumerate(p1["generic_det_neg_k2"]):
        dL = np.array([r["delta_loss"] for r in rec["sweep"]])
        ax.loglog(
            g,
            dL,
            "-",
            color=c_red,
            lw=2.0 if idx_n == 0 else 1.0,
            alpha=1.0 if idx_n == 0 else 0.40,
            label=(
                rf"Generic $\mathrm{{Det}}<0$ ($T_g$, slope {rec['dL_slope']:.3f})"
                if idx_n == 0
                else ("5 more Gaussian Det<0" if idx_n == 1 else None)
            ),
        )
    dL_T3 = np.array([r["delta_loss"] for r in p1["canonical_T3_k2"]["sweep"]])
    ax.loglog(
        g,
        dL_T3,
        "o",
        color=c_orange,
        ms=3.8,
        mfc="white",
        mew=1.2,
        label=rf"Canonical $T_3$ (theory $\sqrt{{2}}\gamma^{{3/4}}$)",
    )
    ax.loglog(
        g,
        math.sqrt(2.0) * (g ** 0.75),
        "--",
        color="black",
        lw=1.3,
        label=r"Generic theory $\propto \gamma^{3/4}$ (0.750)",
    )
    # Reference W-tensor gamma^(6/7) slope
    ax.loglog(
        g,
        (2.0 ** (-4.0 / 7.0)) * (g ** (6.0 / 7.0)),
        ":",
        color=c_purple,
        lw=1.4,
        label=r"Boundary $W$ ($\mathrm{Det}=0$, $\gamma^{6/7}=0.857$)",
    )
    L_k3 = np.array([r["loss"] for r in p1["Tg_k3"]["sweep"]])
    ax.loglog(
        g,
        L_k3,
        "s-",
        color=c_blue,
        lw=1.7,
        ms=3.5,
        label=rf"$T_g$ width $k=3$ (slope {p1['Tg_k3']['L_slope']:.2f})",
    )
    ax.invert_xaxis()
    ax.set_xlabel(r"Weight decay $\gamma$ (decreasing $\rightarrow$)")
    ax.set_ylabel(r"Excess loss $\Delta L^*(\gamma) = L^*(\gamma) - L_{\mathrm{wall}}$")
    ax.set_title(r"A. Generic $\mathrm{Det}<0$ (21.5%): New $\gamma^{3/4}$ Universality Class")
    ax.legend(loc="lower left", frameon=True)

    # Panel B: Part 1C — SMC+HMC on generic Gaussian Tg (k=2 vs k=3) + exact Laplace + cold-start
    ax = axes[0, 1]
    betas = np.array(p1c["betas"])
    lam2 = np.array([r["lambda_exc"] for r in p1c["k2"]])
    se2 = np.array([r["lambda_se"] for r in p1c["k2"]])
    lam3 = np.array([r["lambda_exc"] for r in p1c["k3"]])
    se3 = np.array([r["lambda_se"] for r in p1c["k3"]])

    b_lap = np.array([r["beta"] for r in p1c["k2_laplace"]])
    lam_lap = np.array([r["lambda_exc_laplace"] for r in p1c["k2_laplace"]])
    cold2 = p1c["cold_start_verification"]["k2"]
    b_c2 = np.array([r["beta"] for r in cold2])
    l_c2 = np.array([r["cold_lambda_exc"] for r in cold2])
    s_c2 = np.array([r["cold_lambda_se"] for r in cold2])

    ax.errorbar(
        betas, lam2, yerr=2 * se2, fmt="o-", color=c_red, lw=1.7, ms=3.5, capsize=1.8,
        label=r"k=2 SMC+HMC on $T_g$ ($\pm 2\,$SE)",
    )
    ax.plot(
        b_lap, lam_lap, "--", color="black", lw=1.6, zorder=4,
        label=r"k=2 exact Laplace $\beta\Delta L^* + \frac{33}{8}$",
    )
    ax.errorbar(
        b_c2, l_c2, yerr=2 * s_c2, fmt="D", color="black", mfc="white", ms=5.2, mew=1.3, capsize=2.2,
        zorder=5, label="k=2 cold-start check",
    )
    ax.errorbar(
        betas, lam3, yerr=2 * se3, fmt="s-", color=c_blue, lw=1.6, ms=3.2, capsize=1.8,
        label=f"k=3 SMC+HMC (plateau {p1c['high_beta_summary']['k3_high_beta_mean_lambda']:.2f})",
    )
    ax.axhline(4.0, ls=":", color=c_blue, lw=1.2, alpha=0.8)
    ax.set_xscale("log")
    ax.set_ylim(0.5, 35.0)
    ax.set_xlabel(r"Inverse temperature $\beta = N$")
    ax.set_ylabel(r"Excess LLC $\hat{\lambda}_{\mathrm{exc}}(\beta) = \beta\,\mathbb{E}_\beta[L - L_{\mathrm{wall}}]$")
    ax.set_title(r"B. Generic $T_g$ SMC+HMC: Diverging $\beta^{1/4}$ LLC vs. $k=3$")
    ax.legend(loc="upper left", frameon=True)

    # Inset showing pure beta^(1/4) scaling of lam_exc - 33/8 (placed in open middle-left space,
    # far enough right that its y tick labels clear the main axis tick labels)
    axins = ax.inset_axes([0.19, 0.25, 0.35, 0.30])
    hi = betas >= 1000.0
    b_hi = betas[hi]
    exc = lam2[hi] - 33.0 / 8.0
    axins.errorbar(b_hi, exc, yerr=2 * se2[hi], fmt="o", color=c_red, ms=2.8, capsize=1.2, lw=0.9)
    ref_14 = (exc[-1] / (b_hi[-1] ** 0.25)) * (b_hi ** 0.25)
    axins.loglog(b_hi, ref_14, "--", color="black", lw=1.2)
    axins.set_xscale("log")
    axins.set_yscale("log")
    axins.set_title(
        rf"$\hat\lambda_{{\mathrm{{exc}}}} - \frac{{33}}{{8}}$ (slope {p1c['high_beta_summary']['k2_excess_lambda_minus_33_8_slope']:.3f} vs $0.25$)",
        fontsize=7.0, pad=2,
    )
    axins.tick_params(labelsize=6.5)
    axins.grid(True, alpha=0.25)

    # Panel C: Part 2 — Higher-Dimensional Scaling (d in {2, 3, 4})
    ax = axes[0, 2]
    cfgs = p2["configs"]
    labels = [f"d={c['d']}\nk={c['k']}" for c in cfgs]
    rates = [100.0 * c["degen_rate"] for c in cfgs]
    med_cr = [c["median_cancel_ratio"] for c in cfgs]
    x_pos = np.arange(len(cfgs))
    colors = [c_orange if c["d"] == 2 else (c_blue if c["d"] == 3 else c_red) for c in cfgs]
    bars = ax.bar(x_pos, rates, color=colors, alpha=0.82, edgecolor="black", linewidth=0.7)
    p_exact = 100.0 * (1.0 - math.pi / 4.0)
    for bar, r_val in zip(bars, rates):
        # Tall bars: label inside the bar just above the 1 - pi/4 line, well below the dashed
        # median-cancellation line (which climbs steeply into the d=4 bars).
        # Short bars: label above the bar, lifted over the dashed 1 - pi/4 reference line.
        inside = r_val >= 30.0
        if inside:
            y_txt = p_exact + 4.0
        elif r_val < p_exact < r_val + 8.0:
            y_txt = p_exact + 2.0
        else:
            y_txt = r_val + 2.0
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y_txt,
            f"{r_val:.0f}%",
            ha="center",
            va="bottom",
            color="white" if inside else "black",
            fontsize=8.0,
            fontweight="bold",
        )
    ax.axhline(p_exact, ls="--", color=c_orange, lw=1.1, label=r"$d=2,k=2$ exact $1-\pi/4=21.5\%$")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 122)
    ax.set_ylabel("Border-rank degeneration rate (%)")
    ax.set_title(r"C. Higher Dimensions ($d\times d\times d$ Gaussian, $M=40$ each)")

    ax2 = ax.twinx()
    ax2.plot(x_pos, med_cr, "ko--", lw=1.5, ms=5.0, mfc="white", label="Median cancellation ratio")
    ax2.set_ylim(0, max(med_cr) * 1.38)
    ax2.set_ylabel(r"Median cancellation ratio $\kappa_{\mathrm{cancel}}$")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

    # Panel D: Part 3A — Real Bilinear MLP Adam/AdamW trajectories (k=4 vs k=6)
    ax = axes[1, 0]
    tr_u4 = p3["adam_trajectories_seed305"]["adam_unreg"]["k4"]
    tr_u6 = p3["adam_trajectories_seed305"]["adam_unreg"]["k6"]
    tr_w4 = p3["adam_trajectories_seed305"]["adamw_1e3"]["k4"]
    st = np.array(tr_u4["steps"])[1:]

    ax.loglog(st, np.array(tr_u4["norm_sq"])[1:], "-", color=c_red, lw=2.0, label=r"Adam ($\gamma=0$) $k=4$: $\|\theta\|_2^2$")
    ax.loglog(st, np.array(tr_u6["norm_sq"])[1:], "-", color=c_blue, lw=2.0, label=r"Adam ($\gamma=0$) $k=6$: $\|\theta\|_2^2$")
    ax.loglog(st, np.array(tr_w4["norm_sq"])[1:], ":", color=c_orange, lw=1.8, label=r"AdamW ($10^{-3}$) $k=4$: $\|\theta\|_2^2$")
    ax.set_xlabel("Training step $t$")
    ax.set_ylabel(r"Weight norm squared $\|\theta(t)\|_2^2$ (solid/dotted)")
    ax.set_title(r"D. Real Bilinear MLP ($d=4\to 4$): Cancelling Pairs in Adam")

    ax2 = ax.twinx()
    ax2.loglog(st, np.array(tr_u4["cancel_ratio"])[1:], "--", color=c_red, lw=1.7, alpha=0.85, label=r"Adam $k=4$: $\kappa_{\mathrm{cancel}}$")
    ax2.loglog(st, np.array(tr_u6["cancel_ratio"])[1:], "--", color=c_blue, lw=1.7, alpha=0.85, label=r"Adam $k=6$: $\kappa_{\mathrm{cancel}}$")
    ax2.set_ylabel(r"Tensor cancellation ratio $\kappa_{\mathrm{cancel}}$ (dashed)")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

    # Panel E: Part 3A/B — Head-to-head Bilinear vs SwiGLU on same d=4->4 pure quadratic task
    ax = axes[1, 1]
    gm = np.array(p3["wd_compare_seed305"]["gammas"])
    bil = p3["wd_compare_seed305"]["bilinear"]
    swi = p3["wd_compare_seed305"]["swiglu"]

    ax.loglog(gm, [r["mse"] for r in bil["k4"]], "o-", color=c_red, lw=2.0, ms=4.2, label="Bilinear $k=4$ (border-rank)")
    ax.loglog(gm, [r["mse"] for r in bil["k6"]], "s-", color=c_blue, lw=2.0, ms=4.2, label="Bilinear $k=6$ (realizable, $\\gamma^2$)")
    ax.loglog(
        gm,
        [r["mse"] for r in swi["k4"]],
        "^-",
        color=c_purple,
        lw=1.9,
        ms=4.5,
        label=rf"SwiGLU $k=4$ ($W_1\to 0$, gate-lin + wall)",
    )
    ax.loglog(
        gm,
        [r["mse"] for r in swi["k8"]],
        "D--",
        color=c_orange,
        lw=1.7,
        ms=4.2,
        mfc="white",
        label=r"SwiGLU $k=8$ (2 exact pairs + 4 linearizing, $\sim\gamma^{2/3}$)",
    )
    ax.loglog(
        gm,
        [r["mse"] for r in swi["k12"]],
        "v--",
        color=c_green,
        lw=1.7,
        ms=4.2,
        mfc="white",
        label=r"SwiGLU $k=12$ (6 exact $\mathrm{SiLU}$ pairs $\to$ Bilin $k=6$, $\gamma^2$)",
    )
    ax.invert_xaxis()
    ax.set_xlabel(r"Weight decay $\gamma$ (decreasing $\rightarrow$)")
    ax.set_ylabel("Training MSE")
    ax.set_title(r"E. Bilinear vs. SwiGLU on Pure Quadratic Task ($d=4\to 4$)")
    ax.legend(loc="lower left", frameon=True, fontsize=6.8)

    # Panel F: Part 3B — Honest SwiGLU contrast: weight norm² on Pure Quadratic vs SwiGLU-Native teacher
    ax = axes[1, 2]
    ax.loglog(gm, [r["norm_sq"] for r in swi["k4"]], "^-", color=c_purple, lw=2.0, ms=4.5, label=r"SwiGLU $k=4$ on quadratic ($W_1\to 0$, explodes)")
    ax.loglog(gm, [r["norm_sq"] for r in swi["k8"]], "D-", color=c_orange, lw=1.8, ms=4.0, label=r"SwiGLU $k=8$ on quadratic (hybrid, explodes)")
    ax.loglog(gm, [r["norm_sq"] for r in bil["k4"]], "o--", color=c_red, lw=1.6, ms=3.8, label=r"Bilinear $k=4$ on quadratic ($R^2\propto\gamma^{-1/4}$)")
    ax.loglog(gm, [r["norm_sq"] for r in swi["k12"]], "v--", color=c_blue, lw=1.6, ms=3.8, label=r"SwiGLU $k=12$ on quadratic ($2\times$ Bilin $k=6$, bounded)")

    nat = p3["swiglu_native_control"]["students"]
    ax.loglog(gm, [r["norm_sq"] for r in nat["k2"]], "s-", color=c_green, lw=1.9, ms=4.2, label=r"SwiGLU $k=2$ on SwiGLU teacher (bounded)")
    ax.loglog(gm, [r["norm_sq"] for r in nat["k3"]], "p-", color=c_gray, lw=1.9, ms=4.2, label=r"SwiGLU $k=3$ on SwiGLU teacher (bounded)")
    ax.invert_xaxis()
    ax.set_xlabel(r"Weight decay $\gamma$ (decreasing $\rightarrow$)")
    ax.set_ylabel(r"Optimal weight norm squared $\|\theta^*(\gamma)\|_2^2$")
    ax.set_title("F. SwiGLU Gate Effect: Pure Quadratic vs. SwiGLU-Native")
    ax.legend(loc="upper left", frameon=True, fontsize=6.6, handlelength=1.6, labelspacing=0.3)

    fig.suptitle(
        "Generic Positive-Measure Border-Rank Singularity: "
        r"New $\gamma^{3/4}$ / $\beta^{1/4}$ Class, Higher Dimensions, and Real Bilinear / SwiGLU MLPs",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = RESULTS_DIR / "generic_border_rank.png"
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"\nSaved figure to {fig_path}")
    return fig_path


def replot_from_json() -> Path:
    """Redraw the figure from the saved JSON (no SMC+HMC / L-BFGS rerun)."""
    payload = json.loads((RESULTS_DIR / "generic_border_rank.json").read_text())
    return make_figure(
        payload["part1_generic_2x2x2"],
        payload["part1c_smc_hmc"],
        payload["part2_higher_dimensions"],
        payload["part3_real_mlps"],
    )


def main():
    if "--replot" in sys.argv[1:]:
        replot_from_json()
        return
    t0 = time.time()
    p1, Tg, Tg_wall, Tg_L_wall = run_part1_generic_2x2x2()
    p1c = run_part1c_smc_hmc(Tg, Tg_wall, Tg_L_wall)
    p2 = run_part2_higher_dimensions()
    p3 = run_part3_real_mlps()
    fig_path = make_figure(p1, p1c, p2, p3)

    payload = {
        "runtime_sec": round(time.time() - t0, 2),
        "part1_generic_2x2x2": p1,
        "part1c_smc_hmc": p1c,
        "part2_higher_dimensions": p2,
        "part3_real_mlps": p3,
        "figure": fig_path.relative_to(ROOT.parent).as_posix(),
    }
    json_path = RESULTS_DIR / "generic_border_rank.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Saved JSON results to {json_path} (total {payload['runtime_sec']}s)")


if __name__ == "__main__":
    main()
