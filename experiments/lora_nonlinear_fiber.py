"""
Track 2b Empiricism: Nonlinear MLP & Small-Attention LoRA Fiber Dynamics.

Investigates whether the LoRA fiber phase transition (subcritical Aoyagi-Watanabe
plateau for q <= q_c; supercritical parabola +0.5*(q - q_c)_+^2 for q > q_c) and
weight-decay-driven plateau descent survive in realistic nonlinear architectures:

  Part 1: Nonlinear MLP Exact Fiber Sweep (M=N=H=8, r*=0)
    - Compares GELU, healthy ReLU, SwiGLU, and ReLU with 2 dead neurons (m_eff=6,
      testing the exact threshold shift q_c: 4 -> 5 and plateau drop 24.0 -> 19.5).
    - Includes LLC estimator stability diagnostics across (beta, gamma) and MALA vs SGLD.

  Part 2: Nonlinear GELU MLP LoRA Training Dynamics
    - Part 2a: Teacher-Student exact realizable fiber (M=N=H=10, r*=2 -> excess (8,8,8),
      q_c=4, plateau=42.0, LoRA init=50.0). Compares Adam, SGD, AdamW, SGD+WD,
      subcritical rank-deficient LoRA init (q=2), and small balanced init.
      Tracks nonlinear breaking of U_perp^T B(t) = 0, singular spectra of A, B, BA,
      excess nullspace rank, and MALA hat_lambda along training.
    - Part 2b: Realistic low-sample noisy fine-tuning (d=16, H=8, S_train=48, sigma=0.25,
      5 seeds). Compares test MSE, weight recovery, excess fiber rank, functional KL LLC,
      and centered empirical LLC across Adam, AdamW (wd sweep), SGD+WD, subcritical init,
      and small balanced init.

  Part 3: Small Attention Block — Q/K/V/O Coupling Geometry & Sequence Dynamics
    - Part 3a: Exact fiber sweep (d=8, H=4, T=6) on V-only, Q-only (through softmax),
      joint V+O (same-side stacking -> 24 + 2*(q-2)_+^2, 4x steeper penalty), and
      joint Q+K (two-sided transposed (q,q) fiber with exact GL(q) gauge symmetry ->
      flat lambda = 24.0!), plus 2-head attention check.
    - Part 3b: Full Q,K,V,O LoRA training on a residual self-attention sequence task
      (d=10, H=5, T=8, r*=1 per projection). Compares Adam, AdamW, SGD+WD, subcritical
      init, and small balanced init; tracks per-projection excess rank and decomposed
      LLCs (joint QKVO, V+O channel, Q+K channel).

Outputs:
  - experiments/results/lora_nonlinear_fiber.json
  - experiments/results/lora_nonlinear_fiber.png
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# =============================================================================
# Theory & Spectral Utilities
# =============================================================================

def aw_rlct(m: int, n: int, k: int) -> tuple[float, int]:
    """Exact Aoyagi-Watanabe (2005) RLCT (lambda, multiplicity) for ||B A||_F^2."""
    if k == 0 or m == 0 or n == 0:
        return (0.0, 1)
    if k >= m + n:
        return (m * n / 2.0, 1)
    if m >= k + n:
        return (k * n / 2.0, 1)
    if n >= k + m:
        return (k * m / 2.0, 1)
    even_val = (2 * (m * n + n * k + k * m) - m * m - n * n - k * k) / 8.0
    if (m + n + k) % 2 == 0:
        return (even_val, 1)
    return (even_val + 0.125, 2)


def theoretical_lora_lambda(M: int, N: int, H: int, r_star: int, q: int) -> float:
    """Exact local learning coefficient at rank-q one-sided fiber point (B_in=0, rank(A_in)=q)."""
    m, n, k = M - r_star, N - r_star, H - r_star
    lam_act = r_star * (M + N - r_star) / 2.0
    lam_rem, _ = aw_rlct(m, n - q, k - q)
    return lam_act + m * q / 2.0 + lam_rem


def entropy_effective_rank(sv: np.ndarray, eps: float = 1e-15) -> float:
    """Roy-Vetterli (2007) entropy effective rank exp(-sum p_i log p_i), p_i = s_i^2 / sum s_j^2."""
    s2 = sv ** 2
    total = float(np.sum(s2))
    if total <= eps:
        return 0.0
    p = s2 / total
    p = p[p > eps]
    return float(np.exp(-np.sum(p * np.log(p))))


def participation_ratio(sv: np.ndarray, eps: float = 1e-15) -> float:
    """Participation ratio (sum s_i^2)^2 / sum s_i^4."""
    s2 = sv ** 2
    num = float(np.sum(s2)) ** 2
    den = float(np.sum(s2 ** 2))
    return float(num / den) if den > eps else 0.0


@torch.no_grad()
def adapter_spectral_metrics(
    B: torch.Tensor,
    A: torch.Tensor,
    r_ref: int | None = None,
    U_ref: torch.Tensor | None = None,
    V_ref: torch.Tensor | None = None,
    gamma: float = 20.0,
) -> dict:
    """
    Compute model-agnostic spectral & nullspace fiber metrics for a LoRA adapter (B, A),
    plus exact SVD-projected inactive blocks (A_in, B_in) when reference (U_ref, V_ref, r_ref)
    for Delta W* is supplied.
    """
    sv_A = torch.linalg.svdvals(A).cpu().numpy()
    sv_B = torch.linalg.svdvals(B).cpu().numpy()
    BA = B @ A
    sv_BA = torch.linalg.svdvals(BA).cpu().numpy()

    erank_A = entropy_effective_rank(sv_A)
    erank_B = entropy_effective_rank(sv_B)
    erank_BA = entropy_effective_rank(sv_BA)
    excess_erank = erank_A - erank_BA

    # Model-agnostic nullspace projection: project A onto orthogonal complement of top-r_eff row space of B
    # Use r_proj = r_ref if provided, else round(erank_BA) clamped to [0, min(H-1, ...)]
    H = B.shape[1]
    _, sB_full, VhB_full = torch.linalg.svd(B, full_matrices=True)
    if r_ref is not None:
        r_proj = min(r_ref, H)
    else:
        r_proj = int(np.sum(sv_BA > 0.05 * max(float(sv_BA[0]), 1e-12)))
        r_proj = min(r_proj, H)

    Q_kerB = VhB_full[r_proj:, :].T  # H x (H - r_proj)
    A_perp_B = Q_kerB.T @ A
    sv_A_perp = torch.linalg.svdvals(A_perp_B).cpu().numpy() if A_perp_B.numel() > 0 else np.array([])
    norm_A_perp = float(np.linalg.norm(sv_A_perp)) if sv_A_perp.size > 0 else 0.0
    rank_A_perp_005 = int(np.sum(sv_A_perp > 0.05)) if sv_A_perp.size > 0 else 0
    q_eff_loc = float(np.sum((sv_A_perp ** 2) / (sv_A_perp ** 2 + 1.0 / gamma))) if sv_A_perp.size > 0 else 0.0

    # Also project B onto kernel of A^T (nullspace of A's column space)
    UA_full, _, _ = torch.linalg.svd(A, full_matrices=True)
    Q_kerAT = UA_full[:, r_proj:]
    B_perp_A = B @ Q_kerAT
    sv_B_perp = torch.linalg.svdvals(B_perp_A).cpu().numpy() if B_perp_A.numel() > 0 else np.array([])
    norm_B_perp = float(np.linalg.norm(sv_B_perp)) if sv_B_perp.size > 0 else 0.0

    out = {
        "sv_A": sv_A.tolist(),
        "sv_B": sv_B.tolist(),
        "sv_BA": sv_BA.tolist(),
        "erank_A": erank_A,
        "erank_B": erank_B,
        "erank_BA": erank_BA,
        "pr_A": participation_ratio(sv_A),
        "pr_BA": participation_ratio(sv_BA),
        "excess_erank": excess_erank,
        "norm_A": float(np.linalg.norm(sv_A)),
        "norm_B": float(np.linalg.norm(sv_B)),
        "norm_A_perp_B": norm_A_perp,
        "norm_B_perp_A": norm_B_perp,
        "sv_A_perp_B": sv_A_perp.tolist(),
        "sv_B_perp_A": sv_B_perp.tolist(),
        "rank_A_perp_005": rank_A_perp_005,
        "q_eff_loc": q_eff_loc,
    }

    if U_ref is not None and V_ref is not None and r_ref is not None and r_ref > 0:
        U_act, U_perp = U_ref[:, :r_ref], U_ref[:, r_ref:]
        V_act, V_perp = V_ref[:, :r_ref], V_ref[:, r_ref:]
        _, _, Vh_act = torch.linalg.svd(U_act.T @ B, full_matrices=True)
        Q_kB = Vh_act[r_ref:, :].T
        U_actA, _, _ = torch.linalg.svd(A @ V_act, full_matrices=True)
        Q_kAT = U_actA[:, r_ref:]
        A_in = Q_kB.T @ (A @ V_perp)
        B_in = (U_perp.T @ B) @ Q_kAT
        sv_Ain = torch.linalg.svdvals(A_in).cpu().numpy()
        sv_Bin = torch.linalg.svdvals(B_in).cpu().numpy()
        out.update({
            "sv_A_in": sv_Ain.tolist(),
            "sv_B_in": sv_Bin.tolist(),
            "norm_A_in": float(np.linalg.norm(sv_Ain)),
            "norm_B_in": float(np.linalg.norm(sv_Bin)),
            "rank_A_in_005": int(np.sum(sv_Ain > 0.05)),
            "rank_B_in_005": int(np.sum(sv_Bin > 0.05)),
            "q_eff_in_loc": float(np.sum((sv_Ain ** 2) / (sv_Ain ** 2 + 1.0 / gamma))),
        })
    return out


# =============================================================================
# Generic Vectorized MALA & Unadjusted SGLD LLC Estimators (float64)
# =============================================================================

def mala_estimate_llc_generic(
    params0: list[torch.Tensor],
    loss_fn,
    beta: float = 1e5,
    gamma: float = 20.0,
    n_chains: int = 32,
    n_burn: int = 2500,
    n_samples: int = 3500,
    init_step: float = 1e-4,
    seed: int = 0,
    subtract_base_loss: bool = False,
    precond: bool = False,
) -> dict:
    """
    Vectorized Metropolis-Adjusted Langevin Algorithm (MALA) on localized Gibbs posterior:
        U(w) = beta * K(w) + (gamma/2) * ||w - w0||_2^2
    where K(w) = loss_fn(w) [- loss_fn(w0) if subtract_base_loss].
    When precond=True, estimates a fixed diagonal preconditioner s_i ~ H_ii^{-1/2}
    once from 64 local probes before burn-in and runs exact MALA in z = (w - w0) / s
    (constant linear bijection -> exact detailed balance, dramatically faster mixing
    across stiff active vs soft singular directions).
    Returns hat_lambda = beta * E[K(w)] with inter-chain SE and acceptance diagnostics.
    """
    torch.manual_seed(seed)
    p0s = [p.detach().clone().to(torch.float64) for p in params0]

    with torch.no_grad():
        base_loss = float(loss_fn([p.unsqueeze(0) for p in p0s])[0].item()) if subtract_base_loss else 0.0

    init_scale = (beta ** -0.35) * 0.08

    if precond:
        # Estimate fixed diagonal coordinate scale s_i = H_ii^{-1/2} once before burn-in
        probes = [
            p0.unsqueeze(0).expand(64, *p0.shape) + init_scale * torch.randn(64, *p0.shape, dtype=torch.float64)
            for p0 in p0s
        ]
        req_pr = [p.detach().requires_grad_(True) for p in probes]
        K_pr = loss_fn(req_pr) - base_loss
        loc_pr = sum(
            0.5 * gamma * ((p - p0.unsqueeze(0)) ** 2).flatten(1).sum(1)
            for p, p0 in zip(req_pr, p0s)
        )
        U_pr = beta * K_pr + loc_pr
        g_pr = torch.autograd.grad(U_pr.sum(), req_pr)
        scales = [
            (g.detach().pow(2).mean(dim=0).sqrt() / init_scale).clamp(min=gamma).rsqrt()
            for g in g_pr
        ]
        # Work in z-space: w = w0 + s * z
        ps = [0.2 * torch.randn(n_chains, *p0.shape, dtype=torch.float64) for p0 in p0s]
        step0 = 0.03
    else:
        scales = None
        ps = [
            p0.unsqueeze(0).expand(n_chains, *p0.shape).clone()
            + init_scale * torch.randn(n_chains, *p0.shape, dtype=torch.float64)
            for p0 in p0s
        ]
        step0 = init_step

    def eval_eg(cur_ps: list[torch.Tensor]):
        req = [p.detach().requires_grad_(True) for p in cur_ps]
        if precond:
            ws = [p0.unsqueeze(0) + s.unsqueeze(0) * z for p0, s, z in zip(p0s, scales, req)]
            K = loss_fn(ws) - base_loss
            loc = torch.zeros(n_chains, dtype=torch.float64)
            for s, z in zip(scales, req):
                loc = loc + 0.5 * gamma * ((s.unsqueeze(0) * z) ** 2).flatten(1).sum(1)
        else:
            K = loss_fn(req) - base_loss
            loc = torch.zeros(n_chains, dtype=torch.float64)
            for p, p0 in zip(req, p0s):
                loc = loc + 0.5 * gamma * ((p - p0.unsqueeze(0)) ** 2).flatten(1).sum(1)
        U = beta * K + loc
        grads = torch.autograd.grad(U.sum(), req)
        return U.detach(), K.detach(), [g.detach() for g in grads]

    U, K, gs = eval_eg(ps)
    eps = torch.full((n_chains,), step0, dtype=torch.float64)
    k_sum = torch.zeros(n_chains, dtype=torch.float64)
    k_first_half = torch.zeros(n_chains, dtype=torch.float64)
    k_second_half = torch.zeros(n_chains, dtype=torch.float64)
    acc_post = torch.zeros(n_chains, dtype=torch.float64)
    half = n_samples // 2

    for step in range(n_burn + n_samples):
        sqrt_eps = torch.sqrt(eps)
        means = []
        props = []
        for p, g in zip(ps, gs):
            view = [n_chains] + [1] * (p.ndim - 1)
            m = p - 0.5 * eps.view(*view) * g
            pr = m + sqrt_eps.view(*view) * torch.randn_like(p)
            means.append(m)
            props.append(pr)

        U_p, K_p, gs_p = eval_eg(props)

        fwd_sq = torch.zeros(n_chains, dtype=torch.float64)
        rev_sq = torch.zeros(n_chains, dtype=torch.float64)
        for p, pr, m, g_p in zip(ps, props, means, gs_p):
            view = [n_chains] + [1] * (p.ndim - 1)
            rm = pr - 0.5 * eps.view(*view) * g_p
            fwd_sq = fwd_sq + ((pr - m) ** 2).flatten(1).sum(1)
            rev_sq = rev_sq + ((p - rm) ** 2).flatten(1).sum(1)

        log_alpha = (U - U_p) + (fwd_sq - rev_sq) / (2.0 * eps)
        # Guard NaN/Inf proposals
        log_alpha = torch.where(torch.isfinite(log_alpha), log_alpha, torch.full_like(log_alpha, -1e9))
        accept = torch.log(torch.rand(n_chains, dtype=torch.float64)) < log_alpha

        for i in range(len(ps)):
            view = [n_chains] + [1] * (ps[i].ndim - 1)
            mask = accept.view(*view)
            ps[i] = torch.where(mask, props[i], ps[i])
            gs[i] = torch.where(mask, gs_p[i], gs[i])
        U = torch.where(accept, U_p, U)
        K = torch.where(accept, K_p, K)

        if step < n_burn:
            rate = 0.03 / (1.0 + step / 350.0)
            max_eps = 1.0 if precond else 1e-1
            eps = torch.clamp(eps * torch.exp(rate * (accept.to(torch.float64) - 0.574)), 1e-9, max_eps)
        else:
            idx = step - n_burn
            k_sum = k_sum + K
            if idx < half:
                k_first_half = k_first_half + K
            else:
                k_second_half = k_second_half + K
            acc_post = acc_post + accept.to(torch.float64)

    chain_lam = (beta * (k_sum / n_samples)).numpy()
    h1 = float(np.mean((beta * (k_first_half / max(half, 1))).numpy()))
    h2 = float(np.mean((beta * (k_second_half / max(n_samples - half, 1))).numpy()))
    return {
        "mean": float(np.mean(chain_lam)),
        "se": float(np.std(chain_lam, ddof=1) / math.sqrt(n_chains)),
        "acc": float((acc_post / n_samples).mean().item()),
        "eps_median": float(torch.median(eps).item()),
        "half_diff": float(h2 - h1),
        "base_loss": base_loss,
    }


def sgld_estimate_llc_generic(
    params0: list[torch.Tensor],
    loss_fn,
    beta: float = 5e4,
    gamma: float = 20.0,
    eps: float = 5e-6,
    n_chains: int = 32,
    n_burn: int = 2000,
    n_samples: int = 3000,
    seed: int = 0,
) -> dict:
    """Standard unadjusted Langevin (DevInterp-style full-batch SGLD) for estimator cross-check."""
    torch.manual_seed(seed)
    p0s = [p.detach().clone().to(torch.float64) for p in params0]
    ps = [p0.unsqueeze(0).expand(n_chains, *p0.shape).clone() for p0 in p0s]
    sqrt_eps = math.sqrt(eps)
    k_sum = torch.zeros(n_chains, dtype=torch.float64)

    for step in range(n_burn + n_samples):
        req = [p.detach().requires_grad_(True) for p in ps]
        K = loss_fn(req)
        loc = torch.zeros(n_chains, dtype=torch.float64)
        for p, p0 in zip(req, p0s):
            loc = loc + 0.5 * gamma * ((p - p0.unsqueeze(0)) ** 2).flatten(1).sum(1)
        U = beta * K + loc
        grads = torch.autograd.grad(U.sum(), req)
        with torch.no_grad():
            for i in range(len(ps)):
                ps[i] = ps[i] - 0.5 * eps * grads[i] + sqrt_eps * torch.randn_like(ps[i])
            if step >= n_burn:
                k_sum = k_sum + K.detach()

    chain_lam = (beta * (k_sum / n_samples)).numpy()
    return {
        "mean": float(np.mean(chain_lam)),
        "se": float(np.std(chain_lam, ddof=1) / math.sqrt(n_chains)),
        "eps": eps,
    }


# =============================================================================
# Part 1: Nonlinear MLP Exact Fiber Phase Transition Sweep + Stability Checks
# =============================================================================

def run_part1_mlp_fiber_sweep() -> dict:
    print("\n" + "=" * 72)
    print("PART 1: Nonlinear MLP Exact Fiber Sweep (M=N=H=8, r*=0)")
    print("=" * 72)

    M, N, H = 8, 8, 8
    S = 256
    beta, gamma = 1e5, 20.0
    gen = torch.Generator().manual_seed(7)
    W1, _ = torch.linalg.qr(torch.randn(M, N, generator=gen, dtype=torch.float64))
    W2, _ = torch.linalg.qr(torch.randn(M, M, generator=gen, dtype=torch.float64))
    Wgate, _ = torch.linalg.qr(torch.randn(M, N, generator=gen, dtype=torch.float64))
    X = torch.randn(S, N, generator=gen, dtype=torch.float64)
    gate_x = X @ Wgate.T  # (S, M)
    b_dead = torch.zeros(M, dtype=torch.float64)
    b_dead[:2] = -20.0  # 2 dead neurons -> m_eff = 6, q_c shifts 4 -> 5

    def make_A_rank_q(q: int, scale: float = 1.5) -> torch.Tensor:
        A = torch.zeros(H, N, dtype=torch.float64)
        for j in range(q):
            A[j, j] = scale
        return A

    B0 = torch.zeros(M, H, dtype=torch.float64)

    activations = {
        "GELU": (
            lambda Bc, Ac: F.gelu(torch.einsum("sn,cmn->csm", X, W1.unsqueeze(0) + Bc @ Ac)) @ W2.T,
            M,
        ),
        "ReLU": (
            lambda Bc, Ac: torch.relu(torch.einsum("sn,cmn->csm", X, W1.unsqueeze(0) + Bc @ Ac)) @ W2.T,
            M,
        ),
        "SwiGLU": (
            lambda Bc, Ac: (F.silu(torch.einsum("sn,cmn->csm", X, W1.unsqueeze(0) + Bc @ Ac)) * gate_x.unsqueeze(0)) @ W2.T,
            M,
        ),
        "DeadReLU2": (
            lambda Bc, Ac: torch.relu(torch.einsum("sn,cmn->csm", X, W1.unsqueeze(0) + Bc @ Ac) + b_dead) @ W2.T,
            M - 2,
        ),
    }

    # Check empirical firing rate for DeadReLU2
    with torch.no_grad():
        pre_dead = X @ W1.T + b_dead
        dead_fire_rates = (pre_dead > 0).to(torch.float64).mean(dim=0).numpy().tolist()
        print(f"  DeadReLU2 neuron firing rates: {[round(r, 3) for r in dead_fire_rates]}")

    sweeps = {}
    for name, (fwd, m_eff) in activations.items():
        with torch.no_grad():
            Y0 = fwd(B0.unsqueeze(0), make_A_rank_q(0).unsqueeze(0))[0]

        def make_loss_fn(fwd_fn, y_ref):
            def lfn(ps):
                Yp = fwd_fn(ps[0], ps[1])
                return ((Yp - y_ref.unsqueeze(0)) ** 2).sum(dim=2).mean(dim=1)
            return lfn

        lfn = make_loss_fn(fwd, Y0)
        qc_eff = (N + H - m_eff) / 2.0
        print(f"\n  --- {name} (m_eff={m_eff}, q_c={qc_eff:g}) ---")
        rows = []
        for q in range(0, H + 1):
            A0 = make_A_rank_q(q)
            th = theoretical_lora_lambda(m_eff, N, H, 0, q)
            est = mala_estimate_llc_generic(
                [B0, A0], lfn, beta=beta, gamma=gamma,
                n_chains=28, n_burn=2200, n_samples=3200, seed=100 + q,
            )
            err = est["mean"] - th
            print(
                f"    q={q} | theory={th:5.2f} | MALA={est['mean']:5.2f}±{est['se']:.2f} "
                f"(err={err:+.2f}, acc={est['acc']:.2f}, hdiff={est['half_diff']:+.2f})"
            )
            rows.append({
                "q": q,
                "theory_lambda": th,
                "mala_mean": est["mean"],
                "mala_se": est["se"],
                "acc": est["acc"],
                "half_diff": est["half_diff"],
            })
        sweeps[name] = {"m_eff": m_eff, "q_c": qc_eff, "rows": rows}

    # Estimator stability check on GELU at q=0 (plateau) and q=8 (supercritical)
    print("\n  --- Estimator Stability Diagnostic (GELU MLP, q=0 vs q=8) ---")
    fwd_gelu = activations["GELU"][0]
    with torch.no_grad():
        Y0_gelu = fwd_gelu(B0.unsqueeze(0), make_A_rank_q(0).unsqueeze(0))[0]

    def lfn_gelu(ps):
        return ((fwd_gelu(ps[0], ps[1]) - Y0_gelu.unsqueeze(0)) ** 2).sum(dim=2).mean(dim=1)

    stability = []
    mala_grid = [
        (2e4, 20.0),
        (5e4, 20.0),
        (1e5, 20.0),
        (1e5, 10.0),
        (1e5, 40.0),
    ]
    for b_val, g_val in mala_grid:
        e0 = mala_estimate_llc_generic([B0, make_A_rank_q(0)], lfn_gelu, beta=b_val, gamma=g_val, n_chains=24, n_burn=2000, n_samples=2500, seed=701)
        e8 = mala_estimate_llc_generic([B0, make_A_rank_q(8)], lfn_gelu, beta=b_val, gamma=g_val, n_chains=24, n_burn=2000, n_samples=2500, seed=708)
        row = {
            "method": "MALA", "beta": b_val, "gamma": g_val,
            "q0_mean": e0["mean"], "q0_se": e0["se"], "q0_acc": e0["acc"],
            "q8_mean": e8["mean"], "q8_se": e8["se"], "q8_acc": e8["acc"],
            "delta": e8["mean"] - e0["mean"],
        }
        stability.append(row)
        print(
            f"    MALA beta={b_val:7.0f}, gamma={g_val:4.1f} | "
            f"q=0: {e0['mean']:.2f}±{e0['se']:.2f} | q=8: {e8['mean']:.2f}±{e8['se']:.2f} | delta={row['delta']:+.2f}"
        )

    for eps_sgld in [5e-6, 1.5e-5]:
        s0 = sgld_estimate_llc_generic([B0, make_A_rank_q(0)], lfn_gelu, beta=5e4, gamma=20.0, eps=eps_sgld, n_chains=24, n_burn=2000, n_samples=2500, seed=801)
        s8 = sgld_estimate_llc_generic([B0, make_A_rank_q(8)], lfn_gelu, beta=5e4, gamma=20.0, eps=eps_sgld, n_chains=24, n_burn=2000, n_samples=2500, seed=808)
        row = {
            "method": "SGLD", "beta": 5e4, "gamma": 20.0, "eps": eps_sgld,
            "q0_mean": s0["mean"], "q0_se": s0["se"],
            "q8_mean": s8["mean"], "q8_se": s8["se"],
            "delta": s8["mean"] - s0["mean"],
        }
        stability.append(row)
        print(
            f"    SGLD beta=  50000, eps={eps_sgld:.1e} | "
            f"q=0: {s0['mean']:.2f}±{s0['se']:.2f} | q=8: {s8['mean']:.2f}±{s8['se']:.2f} | delta={row['delta']:+.2f}"
        )

    return {
        "config": {"M": M, "N": N, "H": H, "S": S, "beta": beta, "gamma": gamma},
        "dead_fire_rates": dead_fire_rates,
        "sweeps": sweeps,
        "stability": stability,
    }


# =============================================================================
# Part 2a: Nonlinear GELU MLP Teacher-Student LoRA Training Dynamics
# =============================================================================

def run_part2a_mlp_teacher_student() -> dict:
    print("\n" + "=" * 72)
    print("PART 2a: Nonlinear GELU MLP Teacher-Student Dynamics (M=N=H=10, r*=2)")
    print("=" * 72)

    M, N, H, r_star = 10, 10, 10, 2
    S_train, S_test = 256, 512
    beta, gamma = 1e5, 20.0
    lam_plateau = theoretical_lora_lambda(M, N, H, r_star, 0)  # 42.0
    lam_lora = theoretical_lora_lambda(M, N, H, r_star, H - r_star)  # 50.0
    print(f"  Theory: excess (m,n,k)=(8,8,8), q_c=4, plateau={lam_plateau:.1f}, LoRA(q=8)={lam_lora:.1f}")

    gen = torch.Generator().manual_seed(17)
    W1, _ = torch.linalg.qr(torch.randn(M, N, generator=gen, dtype=torch.float64))
    W2, _ = torch.linalg.qr(torch.randn(M, M, generator=gen, dtype=torch.float64))
    U_ref, _ = torch.linalg.qr(torch.randn(M, M, generator=gen, dtype=torch.float64))
    V_ref, _ = torch.linalg.qr(torch.randn(N, N, generator=gen, dtype=torch.float64))
    s_star = torch.tensor([2.0, 1.5], dtype=torch.float64)
    dW_star = U_ref[:, :r_star] @ torch.diag(s_star) @ V_ref[:, :r_star].T

    X_train = torch.randn(S_train, N, generator=gen, dtype=torch.float64)
    X_test = torch.randn(S_test, N, generator=gen, dtype=torch.float64)
    Y_train = F.gelu(X_train @ (W1 + dW_star).T) @ W2.T
    Y_test = F.gelu(X_test @ (W1 + dW_star).T) @ W2.T

    def loss_batch_mala(ps):
        W_eff = W1.unsqueeze(0) + ps[0] @ ps[1]  # (C, M, N)
        Yp = F.gelu(torch.einsum("sn,cmn->csm", X_train, W_eff)) @ W2.T
        return ((Yp - Y_train.unsqueeze(0)) ** 2).sum(dim=2).mean(dim=1)

    # Measure exact reference MALA at q=0 (plateau) and q=8 (LoRA top) on this teacher-student MLP
    B_ref0 = U_ref[:, :r_star] @ torch.diag(torch.sqrt(s_star)) @ torch.eye(r_star, H, dtype=torch.float64)
    A_ref0 = torch.eye(H, r_star, dtype=torch.float64) @ torch.diag(torch.sqrt(s_star)) @ V_ref[:, :r_star].T
    A_ref8 = A_ref0.clone()
    A_ref8[r_star:, :] = 1.5 * V_ref[:, r_star:].T
    ref_q0 = mala_estimate_llc_generic([B_ref0, A_ref0], loss_batch_mala, beta=beta, gamma=gamma, n_chains=28, n_burn=2200, n_samples=3200, seed=200)
    ref_q8 = mala_estimate_llc_generic([B_ref0, A_ref8], loss_batch_mala, beta=beta, gamma=gamma, n_chains=28, n_burn=2200, n_samples=3200, seed=208)
    print(f"  MALA Reference: q=0 -> {ref_q0['mean']:.2f}±{ref_q0['se']:.2f} (th {lam_plateau:.1f}) | q=8 -> {ref_q8['mean']:.2f}±{ref_q8['se']:.2f} (th {lam_lora:.1f})")

    # Initializations
    g_init = torch.Generator().manual_seed(123)
    B_lora = torch.zeros(M, H, dtype=torch.float64)
    A_lora = 0.5 * torch.randn(H, N, generator=g_init, dtype=torch.float64)

    # Subcritical rank-deficient LoRA init: B=0, rank(A0)=2 (first 2 rows random, rest 0)
    A_subcrit = torch.zeros(H, N, dtype=torch.float64)
    A_subcrit[:2, :] = 0.5 * torch.randn(2, N, generator=g_init, dtype=torch.float64)

    # Small balanced init
    B_small = 1e-3 * torch.randn(M, H, generator=g_init, dtype=torch.float64)
    A_small = 1e-3 * torch.randn(H, N, generator=g_init, dtype=torch.float64)

    run_specs = [
        dict(name="MLP_LoRA_Adam", label="LoRA (q=10) + Adam", opt="adamw", B0=B_lora, A0=A_lora, steps=8000, lr=0.02, wd=0.0, cosine=True, wd_zero=None, ckpts=[1500, 4000, 8000]),
        dict(name="MLP_LoRA_SGD", label="LoRA (q=10) + SGD", opt="sgd", B0=B_lora, A0=A_lora, steps=12000, lr=0.05, wd=0.0, cosine=False, wd_zero=None, ckpts=[3000, 6000, 12000]),
        dict(name="MLP_LoRA_AdamW", label="LoRA (q=10) + AdamW (wd=0.04)", opt="adamw", B0=B_lora, A0=A_lora, steps=22000, lr=0.02, wd=0.04, cosine=True, wd_zero=None, ckpts=[2000, 6000, 12000, 22000]),
        dict(name="MLP_LoRA_SGD_WD", label="LoRA (q=10) + SGD+WD (wd=0.02->0)", opt="sgd", B0=B_lora, A0=A_lora, steps=24000, lr=0.05, wd=0.02, cosine=False, wd_zero=20000, ckpts=[4000, 12000, 20000, 24000]),
        dict(name="MLP_Subcrit_Adam", label="Subcrit LoRA (q=2) + Adam", opt="adamw", B0=B_lora, A0=A_subcrit, steps=8000, lr=0.02, wd=0.0, cosine=True, wd_zero=None, ckpts=[1500, 4000, 8000]),
        dict(name="MLP_SmallBal_Adam", label="Small Bal (1e-3) + Adam", opt="adamw", B0=B_small, A0=A_small, steps=8000, lr=0.02, wd=0.0, cosine=True, wd_zero=None, ckpts=[2000, 4000, 8000]),
    ]

    runs = []
    for sp in run_specs:
        B = sp["B0"].clone().detach().requires_grad_(True)
        A = sp["A0"].clone().detach().requires_grad_(True)
        if sp["opt"] == "adamw":
            opt = torch.optim.AdamW([B, A], lr=sp["lr"], weight_decay=sp["wd"])
        else:
            opt = torch.optim.SGD([B, A], lr=sp["lr"], weight_decay=sp["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sp["steps"], eta_min=1e-5) if sp["cosine"] else None

        ckpt_set = set(sp["ckpts"])
        trace = []
        checkpoints = []
        for step in range(1, sp["steps"] + 1):
            if sp["wd_zero"] is not None and step == sp["wd_zero"] + 1:
                for pg in opt.param_groups:
                    pg["weight_decay"] = 0.0
            opt.zero_grad()
            Yp = F.gelu(X_train @ (W1 + B @ A).T) @ W2.T
            loss = ((Yp - Y_train) ** 2).sum(dim=1).mean()
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()

            if step == 1 or step % 200 == 0 or step in ckpt_set:
                with torch.no_grad():
                    Yt_p = F.gelu(X_test @ (W1 + B @ A).T) @ W2.T
                    test_loss = float(((Yt_p - Y_test) ** 2).sum(dim=1).mean().item())
                    dW_err = float(((B @ A - dW_star) ** 2).sum().item())
                trace.append({"step": step, "train_loss": float(loss.item()), "test_loss": test_loss, "dW_err": dW_err})

            if step in ckpt_set:
                with torch.no_grad():
                    Yt_p = F.gelu(X_test @ (W1 + B @ A).T) @ W2.T
                    test_loss = float(((Yt_p - Y_test) ** 2).sum(dim=1).mean().item())
                    dW_err = float(((B @ A - dW_star) ** 2).sum().item())
                    # Also compute functional KL loss at checkpoint (identical to loss_batch_mala when train_loss ~ 0)
                    Y_ckpt = F.gelu(X_train @ (W1 + B @ A).T) @ W2.T
                spec = adapter_spectral_metrics(B, A, r_ref=r_star, U_ref=U_ref, V_ref=V_ref, gamma=gamma)

                def func_kl(ps):
                    W_eff = W1.unsqueeze(0) + ps[0] @ ps[1]
                    Yp_c = F.gelu(torch.einsum("sn,cmn->csm", X_train, W_eff)) @ W2.T
                    return ((Yp_c - Y_ckpt.unsqueeze(0)) ** 2).sum(dim=2).mean(dim=1)

                est = mala_estimate_llc_generic(
                    [B.detach(), A.detach()], func_kl, beta=beta, gamma=gamma,
                    n_chains=28, n_burn=2200, n_samples=3200, seed=1000 + step,
                )
                c = {
                    "step": step,
                    "train_loss": float(loss.item()),
                    "test_loss": test_loss,
                    "dW_err": dW_err,
                    **spec,
                    "mala_mean": est["mean"],
                    "mala_se": est["se"],
                    "mala_acc": est["acc"],
                }
                checkpoints.append(c)
                print(
                    f"  [{sp['name']:17s}] step={step:5d} | tr={c['train_loss']:.2e} | te={c['test_loss']:.2e} | "
                    f"||B_in||={c['norm_B_in']:.3f} | ||A_in||={c['norm_A_in']:.3f} | rA005={c['rank_A_in_005']} | "
                    f"ex_erank={c['excess_erank']:.2f} | hat_lam={c['mala_mean']:.2f}±{c['mala_se']:.2f}"
                )

        runs.append({
            "name": sp["name"], "label": sp["label"], "opt": sp["opt"],
            "steps": sp["steps"], "lr": sp["lr"], "wd": sp["wd"],
            "trace": trace, "checkpoints": checkpoints, "final": checkpoints[-1],
        })

    return {
        "config": {
            "M": M, "N": N, "H": H, "r_star": r_star, "S_train": S_train, "S_test": S_test,
            "beta": beta, "gamma": gamma, "lam_plateau": lam_plateau, "lam_lora": lam_lora,
        },
        "ref_q0": ref_q0,
        "ref_q8": ref_q8,
        "runs": runs,
    }


# =============================================================================
# Part 2b: Realistic Low-Sample Noisy Fine-Tuning of GELU MLP (5 Seeds)
# =============================================================================

def run_part2b_mlp_noisy_finetune() -> dict:
    print("\n" + "=" * 72)
    print("PART 2b: Realistic Low-Sample Noisy GELU MLP Fine-Tuning (d=16, H=8, S=48)")
    print("=" * 72)

    d, H, r_star = 16, 8, 2
    S_train, S_test = 48, 512
    noise_std = 0.25
    seeds = [0, 1, 2, 3, 4]
    beta_func, gamma_func = 1e5, 20.0

    # Fixed frozen pretrained base network & rank-2 target task
    g_base = torch.Generator().manual_seed(42)
    W1, _ = torch.linalg.qr(torch.randn(d, d, generator=g_base, dtype=torch.float64))
    W2, _ = torch.linalg.qr(torch.randn(d, d, generator=g_base, dtype=torch.float64))
    U_t, _ = torch.linalg.qr(torch.randn(d, d, generator=g_base, dtype=torch.float64))
    V_t, _ = torch.linalg.qr(torch.randn(d, d, generator=g_base, dtype=torch.float64))
    dW_star = U_t[:, :r_star] @ torch.diag(torch.tensor([2.0, 1.5], dtype=torch.float64)) @ V_t[:, :r_star].T

    X_test = torch.randn(S_test, d, generator=g_base, dtype=torch.float64)
    Y_test = F.gelu(X_test @ (W1 + dW_star).T) @ W2.T
    base_test_mse = float(((F.gelu(X_test @ W1.T) @ W2.T - Y_test) ** 2).sum(dim=1).mean().item())
    print(f"  Frozen base model test MSE (before LoRA): {base_test_mse:.4f}")

    method_specs = [
        dict(name="LoRA_Adam_wd0", label="LoRA + Adam (wd=0)", init="lora", opt="adamw", lr=0.015, wd=0.0, steps=4000),
        dict(name="LoRA_AdamW_5e2", label="LoRA + AdamW (wd=0.05)", init="lora", opt="adamw", lr=0.015, wd=0.05, steps=4000),
        dict(name="LoRA_AdamW_5e1", label="LoRA + AdamW (wd=0.5)", init="lora", opt="adamw", lr=0.015, wd=0.5, steps=4000),
        dict(name="LoRA_AdamW_3p0", label="LoRA + AdamW (wd=3.0)", init="lora", opt="adamw", lr=0.015, wd=3.0, steps=4000),
        dict(name="LoRA_SGD_WD", label="LoRA + SGD+WD (wd=3e-2)", init="lora", opt="sgd", lr=0.04, wd=3e-2, steps=6000),
        dict(name="Subcrit_q2_Adam", label="Subcrit LoRA (q=2) + Adam", init="subcrit", opt="adamw", lr=0.015, wd=0.0, steps=4000),
        dict(name="SmallBal_Adam", label="Small Bal (1e-3) + Adam", init="small", opt="adamw", lr=0.015, wd=0.0, steps=4000),
    ]

    summary = []
    for ms in method_specs:
        seed_rows = []
        rep_llc_func = None
        rep_llc_emp = None
        for sd in seeds:
            g_s = torch.Generator().manual_seed(1000 + sd)
            X_tr = torch.randn(S_train, d, generator=g_s, dtype=torch.float64)
            Y_tr_clean = F.gelu(X_tr @ (W1 + dW_star).T) @ W2.T
            Y_tr = Y_tr_clean + noise_std * torch.randn(S_train, d, generator=g_s, dtype=torch.float64)

            if ms["init"] == "lora":
                B0 = torch.zeros(d, H, dtype=torch.float64)
                A0 = 0.5 * torch.randn(H, d, generator=g_s, dtype=torch.float64)
            elif ms["init"] == "subcrit":
                B0 = torch.zeros(d, H, dtype=torch.float64)
                A0 = torch.zeros(H, d, dtype=torch.float64)
                A0[:2, :] = 0.5 * torch.randn(2, d, generator=g_s, dtype=torch.float64)
            else:
                B0 = 1e-3 * torch.randn(d, H, generator=g_s, dtype=torch.float64)
                A0 = 1e-3 * torch.randn(H, d, generator=g_s, dtype=torch.float64)

            B = B0.clone().detach().requires_grad_(True)
            A = A0.clone().detach().requires_grad_(True)
            if ms["opt"] == "adamw":
                opt = torch.optim.AdamW([B, A], lr=ms["lr"], weight_decay=ms["wd"])
            else:
                opt = torch.optim.SGD([B, A], lr=ms["lr"], weight_decay=ms["wd"])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ms["steps"], eta_min=1e-4)

            steps_to_half = None
            for step in range(1, ms["steps"] + 1):
                opt.zero_grad()
                Yp = F.gelu(X_tr @ (W1 + B @ A).T) @ W2.T
                loss = ((Yp - Y_tr) ** 2).sum(dim=1).mean()
                loss.backward()
                opt.step()
                sched.step()
                if steps_to_half is None and float(loss.item()) < 0.5 * base_test_mse:
                    steps_to_half = step

            with torch.no_grad():
                Yp_tr = F.gelu(X_tr @ (W1 + B @ A).T) @ W2.T
                tr_mse = float(((Yp_tr - Y_tr) ** 2).sum(dim=1).mean().item())
                Yp_te = F.gelu(X_test @ (W1 + B @ A).T) @ W2.T
                te_mse = float(((Yp_te - Y_test) ** 2).sum(dim=1).mean().item())
                dW_frob = float(torch.linalg.matrix_norm(B @ A - dW_star).item())
                spec = adapter_spectral_metrics(B, A, r_ref=r_star, U_ref=U_t, V_ref=V_t, gamma=gamma_func)

            # Check gradient norm of unregularized train loss at final point
            B_g = B.detach().clone().requires_grad_(True)
            A_g = A.detach().clone().requires_grad_(True)
            l_g = ((F.gelu(X_tr @ (W1 + B_g @ A_g).T) @ W2.T - Y_tr) ** 2).sum(dim=1).mean()
            gB_f, gA_f = torch.autograd.grad(l_g, (B_g, A_g))
            grad_norm = float(math.sqrt(float((gB_f ** 2).sum().item() + (gA_f ** 2).sum().item())))

            seed_rows.append({
                "seed": sd,
                "train_mse": tr_mse,
                "test_mse": te_mse,
                "dW_frob": dW_frob,
                "grad_norm": grad_norm,
                "steps_to_half": steps_to_half if steps_to_half is not None else ms["steps"],
                "erank_A": spec["erank_A"],
                "erank_BA": spec["erank_BA"],
                "excess_erank": spec["excess_erank"],
                "norm_A_perp_B": spec["norm_A_perp_B"],
                "norm_A_in": spec["norm_A_in"],
                "norm_B_in": spec["norm_B_in"],
                "rank_A_perp_005": spec["rank_A_perp_005"],
            })

            # On seed 0, compute functional KL LLC (population test distribution) and centered train LLC (when grad_norm small)
            if sd == 0:
                with torch.no_grad():
                    Y_ref_pop = F.gelu(X_test[:256] @ (W1 + B @ A).T) @ W2.T

                def lfn_func_pop(ps):
                    W_eff = W1.unsqueeze(0) + ps[0] @ ps[1]
                    Yp_c = F.gelu(torch.einsum("sn,cmn->csm", X_test[:256], W_eff)) @ W2.T
                    return ((Yp_c - Y_ref_pop.unsqueeze(0)) ** 2).sum(dim=2).mean(dim=1)

                rep_llc_func = mala_estimate_llc_generic(
                    [B.detach(), A.detach()], lfn_func_pop, beta=beta_func, gamma=gamma_func,
                    n_chains=24, n_burn=2000, n_samples=2500, seed=3000 + len(summary),
                )

                # Centered empirical train LLC (report honestly alongside grad_norm)
                def lfn_emp_train(ps):
                    W_eff = W1.unsqueeze(0) + ps[0] @ ps[1]
                    Yp_c = F.gelu(torch.einsum("sn,cmn->csm", X_tr, W_eff)) @ W2.T
                    return ((Yp_c - Y_tr.unsqueeze(0)) ** 2).sum(dim=2).mean(dim=1)

                if grad_norm < 1e-2:
                    rep_llc_emp = mala_estimate_llc_generic(
                        [B.detach(), A.detach()], lfn_emp_train, beta=2e3, gamma=20.0,
                        n_chains=24, n_burn=1500, n_samples=2000, seed=3500 + len(summary),
                        subtract_base_loss=True,
                    )

        def agg(key):
            vals = np.array([r[key] for r in seed_rows], dtype=float)
            return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals)))

        te_m, te_se = agg("test_mse")
        tr_m, tr_se = agg("train_mse")
        dw_m, dw_se = agg("dW_frob")
        ex_m, ex_se = agg("excess_erank")
        ap_m, ap_se = agg("norm_A_perp_B")
        eba_m, _ = agg("erank_BA")
        sh_m, _ = agg("steps_to_half")
        gn_m, _ = agg("grad_norm")

        emp_str = f"{rep_llc_emp['mean']:.2f}±{rep_llc_emp['se']:.2f}" if rep_llc_emp else "N/A (grad!=0)"
        print(
            f"  [{ms['name']:16s}] test={te_m:.4f}±{te_se:.4f} | dW={dw_m:.3f} | "
            f"||A_perp||={ap_m:.3f} | ex_erank={ex_m:.2f} | erank(BA)={eba_m:.2f} | "
            f"hat_lam_func={rep_llc_func['mean']:.2f}±{rep_llc_func['se']:.2f} | emp={emp_str}"
        )
        summary.append({
            "name": ms["name"], "label": ms["label"], "init": ms["init"],
            "opt": ms["opt"], "wd": ms["wd"],
            "train_mse_mean": tr_m, "train_mse_se": tr_se,
            "test_mse_mean": te_m, "test_mse_se": te_se,
            "dW_frob_mean": dw_m, "dW_frob_se": dw_se,
            "excess_erank_mean": ex_m, "excess_erank_se": ex_se,
            "erank_BA_mean": eba_m,
            "norm_A_perp_mean": ap_m, "norm_A_perp_se": ap_se,
            "steps_to_half_mean": sh_m,
            "grad_norm_mean": gn_m,
            "llc_func_seed0": rep_llc_func,
            "llc_emp_seed0": rep_llc_emp,
            "seeds": seed_rows,
        })

    return {
        "config": {
            "d": d, "H": H, "r_star": r_star, "S_train": S_train, "S_test": S_test,
            "noise_std": noise_std, "n_seeds": len(seeds), "base_test_mse": base_test_mse,
            "beta_func": beta_func, "gamma_func": gamma_func,
        },
        "methods": summary,
    }


# =============================================================================
# Part 3a: Small Attention Block Exact Coupling Geometry (V, Q, V+O, Q+K + 2-Head)
# =============================================================================

def run_part3a_attention_coupling() -> dict:
    print("\n" + "=" * 72)
    print("PART 3a: Attention Block Fiber Phase Transition & Coupling (d=8, H=4, T=6)")
    print("=" * 72)

    d, H, T, S = 8, 4, 6, 48
    gamma = 20.0
    gen = torch.Generator().manual_seed(42)
    W_Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    W_K, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    W_V, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    W_O, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    X = torch.randn(S, T, d, generator=gen, dtype=torch.float64)

    # Exact precomputed quantities for single-head attention
    Q0 = X @ W_Q.T
    K0 = X @ W_K.T
    V0 = X @ W_V.T
    VO0 = V0 @ W_O.T
    P0 = torch.softmax(torch.einsum("std,sud->stu", Q0, K0) / math.sqrt(d), dim=-1)
    Y0 = torch.einsum("stu,sud->std", P0, VO0)
    PX = torch.einsum("stu,sud->std", P0, X)
    Cov_PX = torch.einsum("sti,stj->ij", PX, PX) / (S * T)

    # Construct clean orthogonal rank-q bases so [A_V; A_O W_V] has exact conditioning 1.0 (sv = 1.5)
    g_orth = torch.Generator().manual_seed(99)
    Q_full, _ = torch.linalg.qr(torch.randn(d, d, generator=g_orth, dtype=torch.float64))  # 8 orthonormal rows

    def make_A_block(q: int, offset: int = 0, scale: float = 1.5) -> torch.Tensor:
        A = torch.zeros(H, d, dtype=torch.float64)
        if q > 0:
            A[:q, :] = scale * Q_full[offset : offset + q, :]
        return A

    B0 = torch.zeros(d, H, dtype=torch.float64)

    results = {}

    # 1. V_only: theory = 14 + 0.5*(q - 2)_+^2
    print("\n  --- Single-Head V_only (theory: 14 + 0.5*(q-2)_+^2) ---")
    rows_v = []
    for q in range(0, H + 1):
        A0 = make_A_block(q, offset=0)
        def lfn_v(ps):
            M = W_O.unsqueeze(0) @ (ps[0] @ ps[1])
            return torch.einsum("cij,jk,cik->c", M, Cov_PX, M)
        th = theoretical_lora_lambda(d, d, H, 0, q)
        est = mala_estimate_llc_generic([B0, A0], lfn_v, beta=2e5, gamma=gamma, n_chains=28, n_burn=2200, n_samples=3200, seed=410 + q)
        print(f"    q={q} | th={th:5.2f} | MALA={est['mean']:5.2f}±{est['se']:.2f} (acc={est['acc']:.2f})")
        rows_v.append({"q": q, "theory_lambda": th, "mala_mean": est["mean"], "mala_se": est["se"], "acc": est["acc"]})
    results["V_only"] = rows_v

    # 2. VO_joint (well-conditioned orthogonal [A_V; A_O W_V]): theory = 24 + 2*(q - 2)_+^2
    print("\n  --- Single-Head VO_joint (theory: 24 + 2*(q-2)_+^2, 4x single jump) ---")
    rows_vo = []
    for q in range(0, H + 1):
        AV0 = make_A_block(q, offset=0)
        # Choose AO0 so AO0 @ W_V occupies orthogonal rows Q_full[H : H+q]
        AO0 = make_A_block(q, offset=H) @ W_V.T
        def lfn_vo(ps):
            dWV = ps[0] @ ps[1]
            dWO = ps[2] @ ps[3]
            M = W_O.unsqueeze(0) @ dWV + dWO @ W_V.unsqueeze(0) + dWO @ dWV
            return torch.einsum("cij,jk,cik->c", M, Cov_PX, M)
        th = theoretical_lora_lambda(d, d, 2 * H, 0, 2 * q)
        est = mala_estimate_llc_generic([B0, AV0, B0, AO0], lfn_vo, beta=2e5, gamma=gamma, n_chains=28, n_burn=2500, n_samples=3500, seed=420 + q)
        print(f"    q={q} (2q={2*q}) | th={th:5.2f} | MALA={est['mean']:5.2f}±{est['se']:.2f} (acc={est['acc']:.2f})")
        rows_vo.append({"q": q, "theory_lambda": th, "mala_mean": est["mean"], "mala_se": est["se"], "acc": est["acc"]})
    results["VO_joint"] = rows_vo

    # 3. Q_only (inside softmax!): theory = 14 + 0.5*(q - 2)_+^2
    print("\n  --- Single-Head Q_only (inside softmax; theory: 14 + 0.5*(q-2)_+^2) ---")
    rows_q = []
    for q in range(0, H + 1):
        A0 = make_A_block(q, offset=0)
        def lfn_q(ps):
            WQ = W_Q.unsqueeze(0) + ps[0] @ ps[1]
            Q = torch.einsum("std,ced->cste", X, WQ)
            P = torch.softmax(torch.einsum("cstd,sud->cstu", Q, K0) / math.sqrt(d), dim=-1)
            out = torch.einsum("cstu,sud->cstd", P, VO0)
            return ((out - Y0.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3)) * d
        th = theoretical_lora_lambda(d, d, H, 0, q)
        est = mala_estimate_llc_generic([B0, A0], lfn_q, beta=5e5, gamma=gamma, n_chains=24, n_burn=2200, n_samples=3000, seed=430 + q)
        print(f"    q={q} | th={th:5.2f} | MALA={est['mean']:5.2f}±{est['se']:.2f} (acc={est['acc']:.2f})")
        rows_q.append({"q": q, "theory_lambda": th, "mala_mean": est["mean"], "mala_se": est["se"], "acc": est["acc"]})
    results["Q_only"] = rows_q

    # 4. QK_joint (two-sided (q,q) with exact GL(q) gauge symmetry): theory = FLAT 24.0!
    print("\n  --- Single-Head QK_joint (two-sided GL(q) gauge -> theory: FLAT 24.0!) ---")
    rows_qk = []
    for q in range(0, H + 1):
        AQ0 = make_A_block(q, offset=0)
        AK0 = make_A_block(q, offset=H)
        def lfn_qk(ps):
            WQ = W_Q.unsqueeze(0) + ps[0] @ ps[1]
            WK = W_K.unsqueeze(0) + ps[2] @ ps[3]
            Q = torch.einsum("std,ced->cste", X, WQ)
            K = torch.einsum("std,ced->cste", X, WK)
            P = torch.softmax(torch.einsum("cstd,csud->cstu", Q, K) / math.sqrt(d), dim=-1)
            out = torch.einsum("cstu,sud->cstd", P, VO0)
            return ((out - Y0.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3)) * d
        th = float(d * H - (H ** 2) / 2.0)  # Exact flat 24.0
        est = mala_estimate_llc_generic([B0, AQ0, B0, AK0], lfn_qk, beta=5e5, gamma=gamma, n_chains=24, n_burn=2500, n_samples=3200, seed=440 + q)
        print(f"    q={q} | th={th:5.2f} | MALA={est['mean']:5.2f}±{est['se']:.2f} (acc={est['acc']:.2f})")
        rows_qk.append({"q": q, "theory_lambda": th, "mala_mean": est["mean"], "mala_se": est["se"], "acc": est["acc"]})
    results["QK_joint"] = rows_qk

    # 5. 2-Head Attention check (d=8, n_heads=2, d_h=4, H=4) at q in {0, 2, 4}
    print("\n  --- 2-Head Attention Check (d=8, n_heads=2, d_h=4, H=4) ---")
    n_heads, dh = 2, d // 2
    def mha_fwd(WQ, WK, WV, WO):
        C = WQ.shape[0]
        Q = torch.einsum("std,ced->cste", X, WQ).view(C, S, T, n_heads, dh)
        K = torch.einsum("std,ced->cste", X, WK).view(C, S, T, n_heads, dh)
        V = torch.einsum("std,ced->cste", X, WV).view(C, S, T, n_heads, dh)
        P = torch.softmax(torch.einsum("csthd,csuhd->chstu", Q, K) / math.sqrt(dh), dim=-1)
        H_out = torch.einsum("chstu,csuhd->csthd", P, V).reshape(C, S, T, d)
        return torch.einsum("cstd,ced->cste", H_out, WO)

    Y0_mha = mha_fwd(W_Q.unsqueeze(0), W_K.unsqueeze(0), W_V.unsqueeze(0), W_O.unsqueeze(0))[0].detach()
    mha_res = {"VO_joint": [], "QK_joint": []}
    for q in [0, 2, 4]:
        AV0 = make_A_block(q, offset=0)
        AO0 = make_A_block(q, offset=H) @ W_V.T
        def lfn_mha_vo(ps):
            C = ps[0].shape[0]
            WV = W_V.unsqueeze(0) + ps[0] @ ps[1]
            WO = W_O.unsqueeze(0) + ps[2] @ ps[3]
            out = mha_fwd(W_Q.unsqueeze(0).expand(C, -1, -1), W_K.unsqueeze(0).expand(C, -1, -1), WV, WO)
            return ((out - Y0_mha.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3)) * d
        e_vo = mala_estimate_llc_generic([B0, AV0, B0, AO0], lfn_mha_vo, beta=3e5, gamma=gamma, n_chains=20, n_burn=2000, n_samples=2500, seed=460 + q)
        mha_res["VO_joint"].append({"q": q, "mala_mean": e_vo["mean"], "mala_se": e_vo["se"], "acc": e_vo["acc"]})

        AQ0 = make_A_block(q, offset=0)
        AK0 = make_A_block(q, offset=H)
        def lfn_mha_qk(ps):
            C = ps[0].shape[0]
            WQ = W_Q.unsqueeze(0) + ps[0] @ ps[1]
            WK = W_K.unsqueeze(0) + ps[2] @ ps[3]
            out = mha_fwd(WQ, WK, W_V.unsqueeze(0).expand(C, -1, -1), W_O.unsqueeze(0).expand(C, -1, -1))
            return ((out - Y0_mha.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3)) * d
        e_qk = mala_estimate_llc_generic([B0, AQ0, B0, AK0], lfn_mha_qk, beta=5e5, gamma=gamma, n_chains=20, n_burn=2000, n_samples=2500, seed=470 + q)
        mha_res["QK_joint"].append({"q": q, "mala_mean": e_qk["mean"], "mala_se": e_qk["se"], "acc": e_qk["acc"]})
        print(f"    2-Head q={q} | VO_joint={e_vo['mean']:.2f}±{e_vo['se']:.2f} | QK_joint={e_qk['mean']:.2f}±{e_qk['se']:.2f}")

    return {
        "config": {"d": d, "H": H, "T": T, "S": S, "gamma": gamma},
        "single_head": results,
        "two_head": mha_res,
    }


# =============================================================================
# Part 3b: Small Attention Block Full Q,K,V,O LoRA Training Dynamics
# =============================================================================

def run_part3b_attention_training() -> dict:
    print("\n" + "=" * 72)
    print("PART 3b: Attention Block Full Q,K,V,O LoRA Training (d=10, H=5, T=8, r*=1)")
    print("=" * 72)

    d, H, T, r_star = 10, 5, 8, 1
    S_train, S_test = 96, 192
    gamma = 20.0

    gen = torch.Generator().manual_seed(31)
    W_Q, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    W_K, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    W_V, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))
    W_O, _ = torch.linalg.qr(torch.randn(d, d, generator=gen, dtype=torch.float64))

    # Rank-1 structured task shifts on ALL four projections (both routing Q,K and readout V,O adapt!)
    def rand_rank1(scale: float):
        u = torch.randn(d, 1, generator=gen, dtype=torch.float64)
        v = torch.randn(d, 1, generator=gen, dtype=torch.float64)
        u = u / torch.linalg.norm(u)
        v = v / torch.linalg.norm(v)
        return scale * (u @ v.T)

    dWQ_star = rand_rank1(1.8)
    dWK_star = rand_rank1(1.8)
    dWV_star = rand_rank1(1.8)
    dWO_star = rand_rank1(1.8)

    X_train = torch.randn(S_train, T, d, generator=gen, dtype=torch.float64)
    X_test = torch.randn(S_test, T, d, generator=gen, dtype=torch.float64)

    def attn_block_single(X_in, WQ, WK, WV, WO):
        Q = X_in @ WQ.T
        K = X_in @ WK.T
        V = X_in @ WV.T
        P = torch.softmax(torch.einsum("std,sud->stu", Q, K) / math.sqrt(d), dim=-1)
        return X_in + torch.einsum("stu,sud->std", P, V) @ WO.T

    def attn_block_batched(X_in, WQ, WK, WV, WO):
        # WQ..WO: (C, d, d)
        Q = torch.einsum("std,ced->cste", X_in, WQ)
        K = torch.einsum("std,ced->cste", X_in, WK)
        V = torch.einsum("std,ced->cste", X_in, WV)
        P = torch.softmax(torch.einsum("cstd,csud->cstu", Q, K) / math.sqrt(d), dim=-1)
        H_out = torch.einsum("cstu,csud->cstd", P, V)
        return X_in.unsqueeze(0) + torch.einsum("cstd,ced->cste", H_out, WO)

    Y_train = attn_block_single(X_train, W_Q + dWQ_star, W_K + dWK_star, W_V + dWV_star, W_O + dWO_star).detach()
    Y_test = attn_block_single(X_test, W_Q + dWQ_star, W_K + dWK_star, W_V + dWV_star, W_O + dWO_star).detach()
    base_test_mse = float(((attn_block_single(X_test, W_Q, W_K, W_V, W_O) - Y_test) ** 2).mean().item() * d)
    print(f"  Base attention test loss (before LoRA): {base_test_mse:.4f}")
    # Theoretical prediction for d=10, H=5, r*=1 per projection (combined V+O and Q+K have rank-2 effective delta):
    # excess (8,8,8) -> V+O: plateau 42.0, LoRA top 50.0 (+8.0 jump); Q+K: flat 42.0!
    print("  Theory (d=10, H=5, rank-2 net delta per pair): V+O plateau=42.0 -> top=50.0 (+8.0); Q+K flat=42.0")

    # Initializations for (Q, K, V, O)
    g_in = torch.Generator().manual_seed(777)
    def make_inits(kind: str):
        inits = {}
        for proj in ["Q", "K", "V", "O"]:
            if kind == "lora":
                B0 = torch.zeros(d, H, dtype=torch.float64)
                A0 = 0.5 * torch.randn(H, d, generator=g_in, dtype=torch.float64)
            elif kind == "subcrit":
                B0 = torch.zeros(d, H, dtype=torch.float64)
                A0 = torch.zeros(H, d, dtype=torch.float64)
                A0[:2, :] = 0.5 * torch.randn(2, d, generator=g_in, dtype=torch.float64)
            else:  # small
                B0 = 1e-3 * torch.randn(d, H, generator=g_in, dtype=torch.float64)
                A0 = 1e-3 * torch.randn(H, d, generator=g_in, dtype=torch.float64)
            inits[proj] = (B0, A0)
        return inits

    init_lora = make_inits("lora")
    init_subcrit = make_inits("subcrit")
    init_small = make_inits("small")

    specs = [
        dict(name="Attn_LoRA_Adam", label="LoRA (q=5) + Adam (wd=0)", init=init_lora, opt="adamw", steps=5000, lr=0.015, wd=0.0, cosine=True, wd_zero=None),
        dict(name="Attn_LoRA_AdamW", label="LoRA (q=5) + AdamW (wd=0.04)", init=init_lora, opt="adamw", steps=10000, lr=0.015, wd=0.04, cosine=True, wd_zero=None),
        dict(name="Attn_LoRA_SGD_WD", label="LoRA (q=5) + SGD+WD (wd=0.02->0)", init=init_lora, opt="sgd", steps=12000, lr=0.08, wd=0.02, cosine=False, wd_zero=10000),
        dict(name="Attn_Subcrit_Adam", label="Subcrit LoRA (q=2) + Adam", init=init_subcrit, opt="adamw", steps=5000, lr=0.015, wd=0.0, cosine=True, wd_zero=None),
        dict(name="Attn_SmallBal_Adam", label="Small Bal (1e-3) + Adam", init=init_small, opt="adamw", steps=5000, lr=0.015, wd=0.0, cosine=True, wd_zero=None),
    ]

    # Use S_mala=40 subset of X_train for fast high-precision MALA (40*8*10 = 3200 scalar outputs >> 400 params)
    X_mala = X_train[:40]
    runs = []

    for sp in specs:
        params = {}
        param_list = []
        for proj in ["Q", "K", "V", "O"]:
            B0, A0 = sp["init"][proj]
            B = B0.clone().detach().requires_grad_(True)
            A = A0.clone().detach().requires_grad_(True)
            params[proj] = (B, A)
            param_list.extend([B, A])

        if sp["opt"] == "adamw":
            opt = torch.optim.AdamW(param_list, lr=sp["lr"], weight_decay=sp["wd"])
        else:
            opt = torch.optim.SGD(param_list, lr=sp["lr"], weight_decay=sp["wd"])
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sp["steps"], eta_min=1e-5) if sp["cosine"] else None

        trace = []
        for step in range(1, sp["steps"] + 1):
            if sp["wd_zero"] is not None and step == sp["wd_zero"] + 1:
                for pg in opt.param_groups:
                    pg["weight_decay"] = 0.0
            opt.zero_grad()
            WQ = W_Q + params["Q"][0] @ params["Q"][1]
            WK = W_K + params["K"][0] @ params["K"][1]
            WV = W_V + params["V"][0] @ params["V"][1]
            WO = W_O + params["O"][0] @ params["O"][1]
            Yp = attn_block_single(X_train, WQ, WK, WV, WO)
            loss = ((Yp - Y_train) ** 2).mean() * d
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()

            if step == 1 or step % 250 == 0 or step == sp["steps"]:
                with torch.no_grad():
                    Yt_p = attn_block_single(X_test, WQ, WK, WV, WO)
                    te_loss = float(((Yt_p - Y_test) ** 2).mean().item() * d)
                    mean_aperp = float(np.mean([
                        adapter_spectral_metrics(params[p][0], params[p][1], gamma=gamma)["norm_A_perp_B"]
                        for p in ["Q", "K", "V", "O"]
                    ]))
                trace.append({"step": step, "train_loss": float(loss.item()), "test_loss": te_loss, "mean_norm_A_perp": mean_aperp})

        # Final evaluation: per-projection spectra + decomposed MALA LLCs (VO, QK, and joint QKVO)
        with torch.no_grad():
            WQ_f = (W_Q + params["Q"][0] @ params["Q"][1]).detach()
            WK_f = (W_K + params["K"][0] @ params["K"][1]).detach()
            WV_f = (W_V + params["V"][0] @ params["V"][1]).detach()
            WO_f = (W_O + params["O"][0] @ params["O"][1]).detach()
            tr_final = float(((attn_block_single(X_train, WQ_f, WK_f, WV_f, WO_f) - Y_train) ** 2).mean().item() * d)
            te_final = float(((attn_block_single(X_test, WQ_f, WK_f, WV_f, WO_f) - Y_test) ** 2).mean().item() * d)
            Y_ckpt_m = attn_block_single(X_mala, WQ_f, WK_f, WV_f, WO_f).detach()

            # Precompute P_f @ X_mala covariance for blazing-fast VO MALA!
            P_f = torch.softmax(torch.einsum("std,sud->stu", X_mala @ WQ_f.T, X_mala @ WK_f.T) / math.sqrt(d), dim=-1)
            PX_f = torch.einsum("stu,sud->std", P_f, X_mala)
            Cov_f = torch.einsum("sti,stj->ij", PX_f, PX_f) / (X_mala.shape[0] * T)
            M_VO_f = WO_f @ WV_f
            VO_mala_f = (X_mala @ WV_f.T) @ WO_f.T

            proj_metrics = {
                p: adapter_spectral_metrics(params[p][0], params[p][1], gamma=gamma)
                for p in ["Q", "K", "V", "O"]
            }

        # Decomposed V+O LLC (freezing Q, K at checkpoint)
        def lfn_vo_ckpt(ps):
            WV_c = W_V.unsqueeze(0) + ps[0] @ ps[1]
            WO_c = W_O.unsqueeze(0) + ps[2] @ ps[3]
            dM = WO_c @ WV_c - M_VO_f.unsqueeze(0)
            return torch.einsum("cij,jk,cik->c", dM, Cov_f, dM)

        llc_vo = mala_estimate_llc_generic(
            [params["V"][0].detach(), params["V"][1].detach(), params["O"][0].detach(), params["O"][1].detach()],
            lfn_vo_ckpt, beta=2e5, gamma=gamma, n_chains=28, n_burn=3000, n_samples=3500,
            seed=5100 + len(runs), precond=True,
        )

        # Decomposed Q+K LLC (freezing V, O at checkpoint)
        def lfn_qk_ckpt(ps):
            WQ_c = W_Q.unsqueeze(0) + ps[0] @ ps[1]
            WK_c = W_K.unsqueeze(0) + ps[2] @ ps[3]
            Q_c = torch.einsum("std,ced->cste", X_mala, WQ_c)
            K_c = torch.einsum("std,ced->cste", X_mala, WK_c)
            P_c = torch.softmax(torch.einsum("cstd,csud->cstu", Q_c, K_c) / math.sqrt(d), dim=-1)
            out_c = X_mala.unsqueeze(0) + torch.einsum("cstu,sud->cstd", P_c, VO_mala_f)
            return ((out_c - Y_ckpt_m.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3)) * d

        llc_qk = mala_estimate_llc_generic(
            [params["Q"][0].detach(), params["Q"][1].detach(), params["K"][0].detach(), params["K"][1].detach()],
            lfn_qk_ckpt, beta=5e5, gamma=gamma, n_chains=24, n_burn=3000, n_samples=3500,
            seed=5200 + len(runs), precond=True,
        )

        # Full joint Q+K+V+O LLC
        def lfn_all_ckpt(ps):
            WQ_c = W_Q.unsqueeze(0) + ps[0] @ ps[1]
            WK_c = W_K.unsqueeze(0) + ps[2] @ ps[3]
            WV_c = W_V.unsqueeze(0) + ps[4] @ ps[5]
            WO_c = W_O.unsqueeze(0) + ps[6] @ ps[7]
            out_c = attn_block_batched(X_mala, WQ_c, WK_c, WV_c, WO_c)
            return ((out_c - Y_ckpt_m.unsqueeze(0)) ** 2).mean(dim=(1, 2, 3)) * d

        llc_all = mala_estimate_llc_generic(
            [
                params["Q"][0].detach(), params["Q"][1].detach(),
                params["K"][0].detach(), params["K"][1].detach(),
                params["V"][0].detach(), params["V"][1].detach(),
                params["O"][0].detach(), params["O"][1].detach(),
            ],
            lfn_all_ckpt, beta=3e5, gamma=gamma, n_chains=24, n_burn=3000, n_samples=3500,
            seed=5300 + len(runs), precond=True,
        )

        ap_str = " ".join(f"{p}:{proj_metrics[p]['norm_A_perp_B']:.2f}" for p in ["Q", "K", "V", "O"])
        print(
            f"  [{sp['name']:18s}] tr={tr_final:.2e} | te={te_final:.2e} | ||A_perp||({ap_str}) | "
            f"VO={llc_vo['mean']:.2f}±{llc_vo['se']:.2f} | QK={llc_qk['mean']:.2f}±{llc_qk['se']:.2f} | "
            f"ALL={llc_all['mean']:.2f}±{llc_all['se']:.2f}"
        )
        runs.append({
            "name": sp["name"], "label": sp["label"], "opt": sp["opt"], "wd": sp["wd"], "steps": sp["steps"],
            "train_loss": tr_final, "test_loss": te_final,
            "proj_metrics": proj_metrics,
            "llc_vo": llc_vo, "llc_qk": llc_qk, "llc_all": llc_all,
            "trace": trace,
        })

    return {
        "config": {
            "d": d, "H": H, "T": T, "r_star": r_star, "S_train": S_train, "S_test": S_test,
            "base_test_mse": base_test_mse, "gamma": gamma,
        },
        "runs": runs,
    }


# =============================================================================
# Publication-Grade 2x3 Figure
# =============================================================================

def make_figure(res: dict, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2), dpi=160)
    ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
    ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]

    # --- Panel (a): Nonlinear MLP Exact Fiber Sweep (GELU, ReLU, SwiGLU, DeadReLU2) ---
    p1 = res["part1_mlp_sweep"]
    q_dense = np.linspace(0, 8, 200)
    ax1.plot(q_dense, 24.0 + 0.5 * np.maximum(q_dense - 4.0, 0.0) ** 2, "-", lw=2.0, color="#1f4e79", label=r"Theory ($m=8, q_c=4$)")
    ax1.plot(q_dense, 19.5 + 0.5 * np.maximum(q_dense - 5.0, 0.0) ** 2, "--", lw=2.0, color="#7f8c8d", label=r"Theory dead-2 ($m_{\mathrm{eff}}=6, q_c=5$)")
    ax1.axvline(4.0, ls=":", color="#1f4e79", alpha=0.5, lw=1.0)
    ax1.axvline(5.0, ls=":", color="#7f8c8d", alpha=0.6, lw=1.0)

    act_styles = {
        "GELU": ("#c0392b", "o", -0.10),
        "ReLU": ("#27ae60", "s", 0.0),
        "SwiGLU": ("#8e44ad", "^", +0.10),
        "DeadReLU2": ("#2c3e50", "D", 0.0),
    }
    for name, (col, mk, dx) in act_styles.items():
        rows = p1["sweeps"][name]["rows"]
        qs = np.array([r["q"] for r in rows], dtype=float) + dx
        ms = np.array([r["mala_mean"] for r in rows])
        es = np.array([2.0 * r["mala_se"] for r in rows])
        ax1.errorbar(qs, ms, yerr=es, fmt=mk, color=col, capsize=2.5, markersize=5, label=f"{name}")
    ax1.set_title(r"(a) Nonlinear MLP Fiber Sweep ($M=N=H=8$)")
    ax1.set_xlabel(r"Rank $q$ of $A_0$ ($q=0$: dead, $q=8$: LoRA init)")
    ax1.set_ylabel(r"Local learning coefficient $\hat\lambda$ ($\pm 2\,$SE)")
    ax1.set_xlim(-0.5, 8.5)
    ax1.set_ylim(17.5, 33.5)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=7.3, loc="upper left", ncol=2)

    # --- Panel (b): Nonlinear GELU MLP Teacher-Student Trajectories ---
    p2a = res["part2a_mlp_ts"]
    c2a = {
        "MLP_LoRA_Adam": ("#e67e22", "s"),
        "MLP_LoRA_SGD": ("#c0392b", "o"),
        "MLP_LoRA_AdamW": ("#8e44ad", "D"),
        "MLP_LoRA_SGD_WD": ("#27ae60", "P"),
        "MLP_Subcrit_Adam": ("#16a085", "^"),
        "MLP_SmallBal_Adam": ("#2980b9", "v"),
    }
    for run in p2a["runs"]:
        col, mk = c2a[run["name"]]
        ckpts = run["checkpoints"]
        xs = np.array([c["step"] / run["steps"] for c in ckpts])
        ys = np.array([c["mala_mean"] for c in ckpts])
        es = np.array([2.0 * c["mala_se"] for c in ckpts])
        ax2.errorbar(xs, ys, yerr=es, fmt=f"-{mk}", color=col, capsize=2.5, markersize=5, lw=1.7, label=run["label"])
    ax2.axhline(p2a["config"]["lam_lora"], ls="--", color="#c0392b", alpha=0.6, lw=1.1, label=r"Theory LoRA $q=8$ (50)")
    ax2.axhline(p2a["config"]["lam_plateau"], ls="--", color="#1f4e79", alpha=0.6, lw=1.1, label=r"Theory plateau (42)")
    ax2.axhline(p2a["ref_q0"]["mean"], ls=":", color="#1f4e79", alpha=0.5, lw=1.0, label=rf"MALA $q=0$ ref ({p2a['ref_q0']['mean']:.1f})")
    ax2.set_title(r"(b) GELU MLP Training $\hat\lambda(t)$ ($r^*=2, q_c=4$)")
    ax2.set_xlabel(r"Normalized training progress $t / T$")
    ax2.set_ylabel(r"Local learning coefficient $\hat\lambda$")
    ax2.set_ylim(36.8, 52.0)
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=6.8, loc="upper right")

    # --- Panel (c): Realistic Low-Sample Noisy Fine-Tuning (Generalization vs Fiber Complexity) ---
    p2b = res["part2b_mlp_noisy"]
    methods = p2b["methods"]
    names_short = [
        "Adam\nwd=0", "AdamW\n0.05", "AdamW\n0.5", "AdamW\n3.0",
        "SGD+WD\n3e-2", "Subcrit\nq=2", "Small\n1e-3",
    ]
    x_pos = np.arange(len(methods))
    te_ms = np.array([m["test_mse_mean"] for m in methods])
    te_ses = np.array([m["test_mse_se"] for m in methods])
    lam_fs = np.array([m["llc_func_seed0"]["mean"] for m in methods])

    ax3.bar(x_pos - 0.16, te_ms, width=0.38, yerr=te_ses, capsize=3, color="#2980b9", alpha=0.85, label="Test MSE (5 seeds)")
    ax3.set_ylabel("Held-out Test MSE", color="#2980b9")
    ax3.tick_params(axis="y", labelcolor="#2980b9")
    ax3.set_ylim(0.0, 1.32)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names_short, fontsize=7.5)
    ax3.grid(alpha=0.2, axis="y")

    ax3_r = ax3.twinx()
    ax3_r.plot(x_pos + 0.12, lam_fs, "o-", color="#c0392b", lw=1.8, markersize=5.5, label=r"Functional $\hat\lambda$ (seed 0)")
    ax3_r.set_ylabel(r"Functional LLC $\hat\lambda$", color="#c0392b")
    ax3_r.tick_params(axis="y", labelcolor="#c0392b")
    ax3_r.set_ylim(55.0, 106.0)
    ax3.set_title(r"(c) Noisy Fine-Tuning ($d=16, H=8, S=48$)")
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_r.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, fontsize=7.2, loc="lower left")

    # --- Panel (d): Attention Coupling Dichotomy (V, Q, V+O, Q+K) ---
    p3a = res["part3a_attn_coupling"]["single_head"]
    q4_dense = np.linspace(0, 4, 120)
    ax4.plot(q4_dense, 14.0 + 0.5 * np.maximum(q4_dense - 2.0, 0.0) ** 2, "-", lw=1.8, color="#1f4e79", label=r"Theory single: $14 + \frac{1}{2}(q-2)_+^2$")
    ax4.plot(q4_dense, 24.0 + 2.0 * np.maximum(q4_dense - 2.0, 0.0) ** 2, "-", lw=2.0, color="#c0392b", label=r"Theory $V{+}O$: $24 + 2(q-2)_+^2$")
    ax4.plot(q4_dense, np.full_like(q4_dense, 24.0), "--", lw=2.0, color="#27ae60", label=r"Theory $Q{+}K$: flat $24$ ($GL(q)$)")
    ax4.axvline(2.0, ls=":", color="#666666", lw=1.0)

    attn_styles = {
        "V_only": ("#2980b9", "o", -0.06, "V only (MALA)"),
        "Q_only": ("#16a085", "^", +0.06, "Q only / softmax (MALA)"),
        "VO_joint": ("#c0392b", "s", 0.0, "V+O joint (MALA)"),
        "QK_joint": ("#27ae60", "D", 0.0, "Q+K joint (MALA)"),
    }
    for key, (col, mk, dx, lbl) in attn_styles.items():
        rows = p3a[key]
        qs = np.array([r["q"] for r in rows], dtype=float) + dx
        ms = np.array([r["mala_mean"] for r in rows])
        es = np.array([2.0 * r["mala_se"] for r in rows])
        ax4.errorbar(qs, ms, yerr=es, fmt=mk, color=col, capsize=3, markersize=5.5, label=lbl)
    ax4.set_title(r"(d) Attention LoRA Coupling ($d=8, H=4, q_c=2$)")
    ax4.set_xlabel(r"Rank $q$ of $A_0$ per adapted projection")
    ax4.set_ylabel(r"Local learning coefficient $\hat\lambda$ ($\pm 2\,$SE)")
    ax4.set_xlim(-0.35, 4.35)
    ax4.set_ylim(12.0, 34.0)
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=7.0, loc="upper left", ncol=1)

    # --- Panel (e): Attention Training — Per-Projection Excess Nullspace Norm ||A_perp_B|| ---
    p3b = res["part3b_attn_train"]
    attn_runs = p3b["runs"]
    projs = ["Q", "K", "V", "O"]
    x_p = np.arange(len(attn_runs))
    w_bar = 0.19
    p_cols = {"Q": "#27ae60", "K": "#2ecc71", "V": "#c0392b", "O": "#e67e22"}
    for j, p in enumerate(projs):
        vals = [r["proj_metrics"][p]["norm_A_perp_B"] for r in attn_runs]
        ax5.bar(x_p + (j - 1.5) * w_bar, vals, width=w_bar, color=p_cols[p], label=f"${p}$ adapter")
    short_attn = ["LoRA\nAdam", "LoRA\nAdamW", "LoRA\nSGD+WD", "Subcrit\n(q=2)", "Small\n(1e-3)"]
    ax5.set_xticks(x_p)
    ax5.set_xticklabels(short_attn, fontsize=8.0)
    ax5.set_title(r"(e) Attention Training: Excess Fiber Norm $\|A_{\perp B}\|_F$")
    ax5.set_ylabel(r"Nullspace component $\|Q_{\ker(B)}^\top A\|_F$")
    ax5.grid(alpha=0.25, axis="y")
    ax5.legend(fontsize=8.0, loc="upper right", ncol=2)

    # --- Panel (f): Attention Training — Decomposed V+O vs Q+K Channel LLCs ---
    vo_m = [r["llc_vo"]["mean"] for r in attn_runs]
    vo_e = [2.0 * r["llc_vo"]["se"] for r in attn_runs]
    qk_m = [r["llc_qk"]["mean"] for r in attn_runs]
    qk_e = [2.0 * r["llc_qk"]["se"] for r in attn_runs]
    ax6.bar(x_p - 0.18, vo_m, width=0.34, yerr=vo_e, capsize=3, color="#c0392b", alpha=0.85, label=r"$V{+}O$ channel $\hat\lambda_{V+O}$")
    ax6.bar(x_p + 0.18, qk_m, width=0.34, yerr=qk_e, capsize=3, color="#27ae60", alpha=0.85, label=r"$Q{+}K$ channel $\hat\lambda_{Q+K}$")
    ax6.axhline(50.0, ls="--", color="#c0392b", alpha=0.6, lw=1.1, label=r"$V{+}O$ LoRA top (50)")
    ax6.axhline(42.0, ls="--", color="#1f4e79", alpha=0.7, lw=1.1, label=r"Plateau / $Q{+}K$ flat (42)")
    ax6.set_xticks(x_p)
    ax6.set_xticklabels(short_attn, fontsize=8.0)
    ax6.set_ylim(34.0, 53.5)
    ax6.set_title(r"(f) Attention Converged LLC: $V{+}O$ vs $Q{+}K$ Channels")
    ax6.set_ylabel(r"Channel LLC $\hat\lambda$ ($\pm 2\,$SE)")
    ax6.grid(alpha=0.25, axis="y")
    ax6.legend(fontsize=7.3, loc="upper right")

    fig.suptitle(
        "Track 2b: Nonlinear MLP & Self-Attention LoRA Fiber Phase Transitions and Training Dynamics",
        fontsize=13.5, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    t0 = time.time()
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    p1 = run_part1_mlp_fiber_sweep()
    p2a = run_part2a_mlp_teacher_student()
    p2b = run_part2b_mlp_noisy_finetune()
    p3a = run_part3a_attention_coupling()
    p3b = run_part3b_attention_training()

    all_res = {
        "part1_mlp_sweep": p1,
        "part2a_mlp_ts": p2a,
        "part2b_mlp_noisy": p2b,
        "part3a_attn_coupling": p3a,
        "part3b_attn_train": p3b,
        "elapsed_seconds": time.time() - t0,
    }

    json_path = out_dir / "lora_nonlinear_fiber.json"
    json_path.write_text(json.dumps(all_res, indent=2))
    fig_path = out_dir / "lora_nonlinear_fiber.png"
    make_figure(all_res, fig_path)
    print(f"\nAll experiments finished in {time.time() - t0:.1f}s.")
    print(f"Saved JSON: {json_path}")
    print(f"Saved PNG:  {fig_path}")


if __name__ == "__main__":
    main()
