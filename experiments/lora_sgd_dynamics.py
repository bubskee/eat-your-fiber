"""
Track 2 Empiricism: Real SGD, Adam, and AdamW Training Dynamics on LoRA Adapters.

Tests where gradient-based optimization lands on the zero-loss fiber {B A = M*}
relative to the LoRA Fiber Phase Transition theory (notes/lora_fiber_phase_transition.md):
  - Main setup: M=16, N=16, H=10, r*=2 -> excess (m, n, k) = (14, 14, 8), q_c = 4.
  - Global minimum (q <= q_c): lambda_AW = 78.00 (MALA ref at q=0: 75.68 +/- 0.19).
  - Standard LoRA init (B=0, A full rank q=8): lambda(8) = 86.00 (MALA ref: 85.96 +/- 0.05).

Experiments:
  1. Full-batch population loss K(B, A) = ||B A - M*||_F^2 across:
     - Init A (Standard LoRA: B=0, A ~ N(0, 0.5^2)): GD, Adam
     - Init B (Small balanced: B, A ~ N(0, 1e-3^2)): GD, Adam
     - Init C (Standard LoRA + weight decay wd=1e-2): AdamW, SGD+WD (with WD anneal polish)
  2. Finite-sample mini-batch SGD / AdamW (S=64, batch size=16):
     - Clean targets (SGD wd=0, Adam wd=0, AdamW wd=1e-2) and noisy targets (SGD wd=0).

Outputs:
  - experiments/results/lora_sgd_dynamics.json
  - experiments/results/lora_sgd_dynamics.png
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lora_fiber_sgld import mala_estimate_llc, theoretical_lora_fiber_lambda


def make_target(M: int, N: int, r_star: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct a non-axis-aligned rank-r* target M* = U_* diag(s*) V_*^T."""
    gen = torch.Generator().manual_seed(seed)
    U, _ = torch.linalg.qr(torch.randn(M, M, generator=gen, dtype=torch.float64))
    V, _ = torch.linalg.qr(torch.randn(N, N, generator=gen, dtype=torch.float64))
    s_star = torch.linspace(2.0, 1.5, r_star, dtype=torch.float64)
    M_star = U[:, :r_star] @ torch.diag(s_star) @ V[:, :r_star].T
    return M_star, U, V


@torch.no_grad()
def analyze_inactive_blocks(
    B: torch.Tensor,
    A: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    r_star: int,
    beta: float = 1e5,
    gamma: float = 20.0,
) -> dict:
    """
    Project (B, A) onto the orthogonal complement of the active rank-r* subspace of M*.

    In the SVD basis of M*:
      B_act_out = U_*^T B  (r* x H),   B_perp = U_perp^T B  (m x H)
      A_act_in  = A V_*    (H x r*),   A_perp = A V_perp    (H x n)

    At BA = M*, B_act_out A_perp = 0 and B_perp A_act_in = 0, so the inactive blocks are:
      A_in = Q_{ker(B_act_out)}^T A_perp   (k x n)
      B_in = B_perp Q_{col(A_act_in)^perp} (m x k)
    """
    U_act, U_perp = U[:, :r_star], U[:, r_star:]
    V_act, V_perp = V[:, :r_star], V[:, r_star:]

    B_act_out = U_act.T @ B
    B_perp = U_perp.T @ B
    A_act_in = A @ V_act
    A_perp = A @ V_perp

    _, _, VhB = torch.linalg.svd(B_act_out, full_matrices=True)
    Q_kerB = VhB[r_star:, :].T  # H x k

    UA, _, _ = torch.linalg.svd(A_act_in, full_matrices=True)
    Q_kerAT = UA[:, r_star:]  # H x k

    A_in = Q_kerB.T @ A_perp  # k x n
    B_in = B_perp @ Q_kerAT  # m x k

    sv_A = torch.linalg.svdvals(A_in).cpu().numpy()
    sv_B = torch.linalg.svdvals(B_in).cpu().numpy()

    # Effective rank measures at MALA scale (beta, gamma) and threshold tau=0.05
    q_eff_mala = float(np.sum((2.0 * beta * sv_A ** 2) / (2.0 * beta * sv_A ** 2 + gamma)))
    q_eff_loc = float(np.sum((sv_A ** 2) / (sv_A ** 2 + 1.0 / gamma)))
    rank_A_005 = int(np.sum(sv_A > 0.05))
    rank_B_005 = int(np.sum(sv_B > 0.05))
    rank_A_0014 = int(np.sum(sv_A > math.sqrt(gamma / beta)))

    return {
        "sv_A": sv_A.tolist(),
        "sv_B": sv_B.tolist(),
        "norm_A_in": float(torch.linalg.matrix_norm(A_in).item()),
        "norm_B_in": float(torch.linalg.matrix_norm(B_in).item()),
        "q_eff_mala": q_eff_mala,
        "q_eff_loc": q_eff_loc,
        "rank_A_005": rank_A_005,
        "rank_B_005": rank_B_005,
        "rank_A_0014": rank_A_0014,
    }


def eval_checkpoint(
    step: int,
    B: torch.Tensor,
    A: torch.Tensor,
    M_star: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    r_star: int,
    beta: float,
    gamma: float,
    mala_seed: int,
) -> dict:
    """Measure loss, inactive singular spectra, and MALA hat_lambda at a parameter state."""
    with torch.no_grad():
        pop_loss = float(((B @ A - M_star) ** 2).sum().item())
    spec = analyze_inactive_blocks(B, A, U, V, r_star, beta=beta, gamma=gamma)
    est = mala_estimate_llc(B.detach(), A.detach(), M_star, beta=beta, gamma=gamma, seed=mala_seed)
    return {
        "step": step,
        "pop_loss": pop_loss,
        **spec,
        "mala_mean": est["mean"],
        "mala_se": est["se"],
        "mala_acc": est["acc"],
    }


def run_full_batch_experiment(
    name: str,
    label: str,
    opt_kind: str,
    B_init: torch.Tensor,
    A_init: torch.Tensor,
    M_star: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    r_star: int,
    steps: int,
    lr: float,
    wd: float,
    cosine: bool,
    wd_zero_after: int | None,
    ckpt_steps: list[int],
    beta: float = 1e5,
    gamma: float = 20.0,
) -> dict:
    """Train on full-batch K(B, A) = ||B A - M*||_F^2 and evaluate MALA at checkpoints."""
    B = B_init.clone().detach().requires_grad_(True)
    A = A_init.clone().detach().requires_grad_(True)

    if opt_kind == "sgd":
        opt = torch.optim.SGD([B, A], lr=lr, weight_decay=wd)
    elif opt_kind == "adamw":
        opt = torch.optim.AdamW([B, A], lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown opt_kind: {opt_kind}")

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=1e-5) if cosine else None

    ckpt_set = set(ckpt_steps)
    loss_trace = []
    checkpoints = []

    for step in range(1, steps + 1):
        if wd_zero_after is not None and step == wd_zero_after + 1:
            for pg in opt.param_groups:
                pg["weight_decay"] = 0.0

        opt.zero_grad()
        loss = ((B @ A - M_star) ** 2).sum()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()

        lv = float(loss.item())
        if step == 1 or step % 100 == 0 or step == steps or step in ckpt_set:
            loss_trace.append({"step": step, "pop_loss": lv})

        if step in ckpt_set:
            c = eval_checkpoint(step, B, A, M_star, U, V, r_star, beta, gamma, mala_seed=1000 + step)
            checkpoints.append(c)
            print(
                f"  [{name}] step={step:5d} | loss={c['pop_loss']:.2e} | "
                f"svB[0]={c['sv_B'][0]:.3f} | svA=[{c['sv_A'][0]:.3f}..{c['sv_A'][-1]:.3f}] | "
                f"rank005={c['rank_A_005']} | hat_lam={c['mala_mean']:.2f}±{c['mala_se']:.2f}"
            )

    final = checkpoints[-1]
    return {
        "name": name,
        "label": label,
        "opt_kind": opt_kind,
        "steps": steps,
        "lr": lr,
        "wd": wd,
        "cosine": cosine,
        "wd_zero_after": wd_zero_after,
        "loss_trace": loss_trace,
        "checkpoints": checkpoints,
        "final": final,
    }


def run_minibatch_experiment(
    name: str,
    label: str,
    opt_kind: str,
    B_init: torch.Tensor,
    A_init: torch.Tensor,
    M_star: torch.Tensor,
    X: torch.Tensor,
    Y_target: torch.Tensor,
    U: torch.Tensor,
    V: torch.Tensor,
    r_star: int,
    epochs: int,
    batch_size: int,
    lr: float,
    wd: float,
    cosine: bool,
    seed: int = 0,
    beta: float = 1e5,
    gamma: float = 20.0,
) -> dict:
    """Train with finite-sample mini-batch SGD or AdamW on (1/bs)||B A X_b - Y_b||_F^2."""
    gen = torch.Generator().manual_seed(seed)
    S = X.shape[1]
    B = B_init.clone().detach().requires_grad_(True)
    A = A_init.clone().detach().requires_grad_(True)

    if opt_kind == "sgd":
        opt = torch.optim.SGD([B, A], lr=lr, weight_decay=wd)
    elif opt_kind == "adamw":
        opt = torch.optim.AdamW([B, A], lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown opt_kind: {opt_kind}")

    total_steps = epochs * (S // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=1e-5) if cosine else None

    for _ in range(epochs):
        perm = torch.randperm(S, generator=gen)
        for i in range(0, S, batch_size):
            idx = perm[i : i + batch_size]
            xb = X[:, idx]
            yb = Y_target[:, idx]
            opt.zero_grad()
            loss = ((B @ A @ xb - yb) ** 2).sum() / batch_size
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()

    with torch.no_grad():
        emp_loss = float((((B @ A @ X - Y_target) ** 2).sum() / S).item())
    final = eval_checkpoint(epochs, B, A, M_star, U, V, r_star, beta, gamma, mala_seed=2000 + epochs)
    final["emp_loss"] = emp_loss
    print(
        f"  [{name}] epochs={epochs:4d} | pop_loss={final['pop_loss']:.2e} | emp_loss={emp_loss:.2e} | "
        f"svB[0]={final['sv_B'][0]:.3f} | svA=[{final['sv_A'][0]:.3f}..{final['sv_A'][-1]:.3f}] | "
        f"hat_lam={final['mala_mean']:.2f}±{final['mala_se']:.2f}"
    )
    return {
        "name": name,
        "label": label,
        "opt_kind": opt_kind,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "wd": wd,
        "cosine": cosine,
        "final": final,
    }


def make_figure(
    results: dict,
    ref_rows: list[dict] | None,
    out_path: Path,
):
    """Generate 2x2 publication-quality summary plot."""
    M, N, H, r_star = results["config"]["M"], results["config"]["N"], results["config"]["H"], results["config"]["r_star"]
    m, n, k = M - r_star, N - r_star, H - r_star
    qc = (n + k - m) / 2.0
    lam_min = theoretical_lora_fiber_lambda(M, N, H, r_star, 0)[0]
    lam_lora = theoretical_lora_fiber_lambda(M, N, H, r_star, k)[0]

    colors = {
        "InitA_GD": "#c0392b",
        "InitA_Adam": "#e67e22",
        "InitB_GD": "#2980b9",
        "InitB_Adam": "#1abc9c",
        "InitC_AdamW": "#8e44ad",
        "InitC_SGD_WD": "#27ae60",
    }
    markers = {
        "InitA_GD": "o",
        "InitA_Adam": "s",
        "InitB_GD": "^",
        "InitB_Adam": "v",
        "InitC_AdamW": "D",
        "InitC_SGD_WD": "P",
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.2), dpi=160)
    ax_loss, ax_traj, ax_sv, ax_curve = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Panel (a): Training loss vs normalized progress / step
    for run in results["full_batch"]:
        nm = run["name"]
        steps = np.array([p["step"] for p in run["loss_trace"]], dtype=float)
        losses = np.maximum(np.array([p["pop_loss"] for p in run["loss_trace"]], dtype=float), 1e-16)
        ax_loss.plot(steps, losses, lw=1.8, color=colors[nm], label=run["label"])
    ax_loss.axhline(1e-10, ls=":", color="#555555", lw=1.0, label=r"Target $10^{-10}$")
    ax_loss.set_xscale("log")
    ax_loss.set_yscale("log")
    ax_loss.set_ylim(1e-16, 20.0)
    ax_loss.set_title(r"(a) Population Loss $K(B, A) = \|BA - M^*\|_F^2$")
    ax_loss.set_xlabel("Optimization step (log scale)")
    ax_loss.set_ylabel(r"$\|BA - M^*\|_F^2$")
    ax_loss.grid(alpha=0.25)
    ax_loss.legend(fontsize=7.5, loc="lower left")

    # Panel (b): hat_lambda(t) trajectories
    for run in results["full_batch"]:
        nm = run["name"]
        ckpts = [c for c in run["checkpoints"] if c["pop_loss"] < 1e-4]
        if not ckpts:
            continue
        xs = np.array([c["step"] / run["steps"] for c in ckpts])
        ys = np.array([c["mala_mean"] for c in ckpts])
        es = np.array([2.0 * c["mala_se"] for c in ckpts])
        ax_traj.errorbar(
            xs, ys, yerr=es, fmt=f"-{markers[nm]}", color=colors[nm],
            capsize=3, markersize=5, lw=1.8, label=run["label"]
        )
    ax_traj.axhline(lam_lora, ls="--", color="#c0392b", alpha=0.6, lw=1.2, label=rf"LoRA init $\lambda(k={k}) = {lam_lora:.0f}$")
    ax_traj.axhline(lam_min, ls="--", color="#1f4e79", alpha=0.7, lw=1.2, label=rf"Global min $\lambda_{{\mathrm{{AW}}}} = {lam_min:.0f}$")
    if ref_rows:
        q0_mala = ref_rows[0]["mala_mean"]
        ax_traj.axhline(q0_mala, ls=":", color="#1f4e79", alpha=0.5, lw=1.1, label=rf"MALA $q=0$ ref ({q0_mala:.2f})")
    ax_traj.set_title(r"(b) Empirical $\hat\lambda(t)$ Along Fiber Training ($\pm 2\,$SE)")
    ax_traj.set_xlabel("Normalized training progress $t / T$")
    ax_traj.set_ylabel(r"Local learning coefficient $\hat\lambda$")
    ax_traj.set_ylim(73.5, 88.0)
    ax_traj.grid(alpha=0.25)
    ax_traj.legend(fontsize=7.2, loc="upper right", ncol=1)

    # Panel (c): Singular values of inactive A_in at convergence
    idx = np.arange(1, k + 1)
    for run in results["full_batch"]:
        nm = run["name"]
        sv = np.maximum(np.array(run["final"]["sv_A"]), 1e-7)
        ax_sv.plot(idx, sv, f"-{markers[nm]}", color=colors[nm], lw=1.7, markersize=5, label=run["label"])
    # Include key mini-batch runs
    mb_styles = {
        "MB_SGD_clean": ("#d35400", "--", "x", "MB-SGD (clean, wd=0)"),
        "MB_AdamW_clean": ("#6c3483", "--", "*", "MB-AdamW (clean, wd=1e-2)"),
    }
    for mb in results["minibatch"]:
        if mb["name"] in mb_styles:
            col, ls, mk, lbl = mb_styles[mb["name"]]
            sv = np.maximum(np.array(mb["final"]["sv_A"]), 1e-7)
            ax_sv.plot(idx, sv, ls=ls, marker=mk, color=col, lw=1.5, markersize=6, label=lbl)

    gamma = results["config"]["gamma"]
    beta = results["config"]["beta"]
    ax_sv.axhline(1.0 / math.sqrt(gamma), ls=":", color="#444444", lw=1.1, label=rf"$\gamma^{{-1/2}} = {1/math.sqrt(gamma):.3f}$")
    ax_sv.axhline(math.sqrt(gamma / beta), ls="-.", color="#777777", lw=1.0, label=rf"$\sqrt{{\gamma/\beta}} = {math.sqrt(gamma/beta):.3f}$")
    ax_sv.set_yscale("log")
    ax_sv.set_xticks(idx)
    ax_sv.set_ylim(1e-5, 6.0)
    ax_sv.set_title(r"(c) Inactive Residual $A_{\mathrm{in}}$ Singular Values at Convergence")
    ax_sv.set_xlabel(r"Singular value index $i \in \{1, \dots, k=8\}$")
    ax_sv.set_ylabel(r"$\sigma_i(A_{\mathrm{in}})$ (log scale)")
    ax_sv.grid(alpha=0.25)
    ax_sv.legend(fontsize=7.0, loc="lower left", ncol=2)

    # Panel (d): Converged hat_lambda overlaid on theoretical (q - q_c)_+^2/2 curve
    q_dense = np.linspace(0, k, 200)
    lam_active = r_star * (M + N - r_star) / 2.0
    lam_base = (2 * (m * n + n * k + k * m) - m * m - n * n - k * k) / 8.0
    th_dense = lam_active + lam_base + 0.5 * np.maximum(q_dense - qc, 0.0) ** 2
    ax_curve.plot(q_dense, th_dense, "-", lw=2.2, color="#1f4e79", label=r"Theory: $78 + \frac{1}{2}(q - 4)_+^2$")
    ax_curve.axvline(qc, ls="--", color="#666666", lw=1.1, label=rf"Critical $q_c = {qc:g}$")

    if ref_rows:
        rqs = [r["q"] for r in ref_rows]
        rms = [r["mala_mean"] for r in ref_rows]
        rse = [2.0 * r["mala_se"] for r in ref_rows]
        ax_curve.errorbar(
            rqs, rms, yerr=rse, fmt="o--", color="#95a5a6", alpha=0.8,
            capsize=2, markersize=4, lw=1.1, label=r"MALA baseline ($q=0..8$)"
        )

    # Plot converged runs using rank_A_005 (with small x-offset for overlapping points)
    x_offsets = {
        "InitA_GD": -0.14,
        "InitA_Adam": 0.0,
        "InitB_GD": -0.22,
        "InitB_Adam": 0.0,
        "InitC_AdamW": 0.0,
        "InitC_SGD_WD": +0.22,
        "MB_SGD_clean": +0.14,
        "MB_Adam_clean": +0.28,
        "MB_SGD_noisy": -0.28,
        "MB_AdamW_clean": +0.44,
    }
    for run in results["full_batch"]:
        nm = run["name"]
        f = run["final"]
        qx = f["rank_A_005"] + x_offsets.get(nm, 0.0)
        ax_curve.errorbar(
            qx, f["mala_mean"], yerr=2.0 * f["mala_se"],
            fmt=markers[nm], color=colors[nm], capsize=3, markersize=7,
            markeredgecolor="black", markeredgewidth=0.6, label=run["label"]
        )
    mb_curve_styles = {
        "MB_SGD_clean": ("#d35400", "x", "MB-SGD (clean)"),
        "MB_AdamW_clean": ("#6c3483", "*", "MB-AdamW (wd=1e-2)"),
    }
    for mb in results["minibatch"]:
        nm = mb["name"]
        if nm in mb_curve_styles:
            col, mk, lbl = mb_curve_styles[nm]
            f = mb["final"]
            qx = f["rank_A_005"] + x_offsets.get(nm, 0.0)
            ax_curve.errorbar(
                qx, f["mala_mean"], yerr=2.0 * f["mala_se"],
                fmt=mk, color=col, capsize=3, markersize=8, label=lbl
            )

    ax_curve.set_title(r"(d) Converged $\hat\lambda$ vs Fiber Phase Transition Curve")
    ax_curve.set_xlabel(r"Effective inactive rank $\mathrm{rank}(A_{\mathrm{in}} > 0.05)$")
    ax_curve.set_ylabel(r"Local learning coefficient $\hat\lambda$")
    ax_curve.set_xlim(-0.6, 8.6)
    ax_curve.set_ylim(74.0, 87.5)
    ax_curve.grid(alpha=0.25)
    ax_curve.legend(fontsize=6.8, loc="upper left", ncol=2)

    fig.suptitle(
        rf"LoRA Training Dynamics on the Zero-Loss Fiber ($M={M}, N={N}, H={H}, r^*={r_star}$, $q_c={qc:g}$)",
        fontsize=13, fontweight="bold", y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    t_start = time.time()
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    M, N, H, r_star = 16, 16, 10, 2
    beta, gamma = 1e5, 20.0
    M_star, U, V = make_target(M, N, r_star, seed=0)

    # Initializations (deterministic seed)
    gen = torch.Generator().manual_seed(123)
    B_lora = torch.zeros(M, H, dtype=torch.float64)
    A_lora = 0.5 * torch.randn(H, N, generator=gen, dtype=torch.float64)
    B_small = 1e-3 * torch.randn(M, H, generator=gen, dtype=torch.float64)
    A_small = 1e-3 * torch.randn(H, N, generator=gen, dtype=torch.float64)

    print("=== Part 1: Full-Batch Training Dynamics on ||B A - M*||_F^2 ===")
    fb_specs = [
        dict(
            name="InitA_GD", label="Init A (LoRA) + GD", opt_kind="sgd",
            B_init=B_lora, A_init=A_lora, steps=6000, lr=0.02, wd=0.0,
            cosine=False, wd_zero_after=None, ckpt_steps=[500, 1500, 3000, 6000],
        ),
        dict(
            name="InitA_Adam", label="Init A (LoRA) + Adam", opt_kind="adamw",
            B_init=B_lora, A_init=A_lora, steps=6000, lr=0.01, wd=0.0,
            cosine=True, wd_zero_after=None, ckpt_steps=[1500, 3000, 4500, 6000],
        ),
        dict(
            name="InitB_GD", label="Init B (Small 1e-3) + GD", opt_kind="sgd",
            B_init=B_small, A_init=A_small, steps=20000, lr=0.02, wd=0.0,
            cosine=False, wd_zero_after=None, ckpt_steps=[4000, 8000, 14000, 20000],
        ),
        dict(
            name="InitB_Adam", label="Init B (Small 1e-3) + Adam", opt_kind="adamw",
            B_init=B_small, A_init=A_small, steps=8000, lr=0.01, wd=0.0,
            cosine=True, wd_zero_after=None, ckpt_steps=[2000, 4000, 8000],
        ),
        dict(
            name="InitC_AdamW", label="Init C (LoRA) + AdamW (wd=1e-2)", opt_kind="adamw",
            B_init=B_lora, A_init=A_lora, steps=50000, lr=0.015, wd=1e-2,
            cosine=True, wd_zero_after=None, ckpt_steps=[2000, 6000, 12000, 20000, 32000, 50000],
        ),
        dict(
            name="InitC_SGD_WD", label="Init C (LoRA) + SGD+WD (wd=1e-2->0)", opt_kind="sgd",
            B_init=B_lora, A_init=A_lora, steps=50000, lr=0.03, wd=1e-2,
            cosine=False, wd_zero_after=45000, ckpt_steps=[2000, 10000, 25000, 45000, 50000],
        ),
    ]

    full_batch_results = []
    for spec in fb_specs:
        res = run_full_batch_experiment(
            M_star=M_star, U=U, V=V, r_star=r_star, beta=beta, gamma=gamma, **spec
        )
        full_batch_results.append(res)

    print("\n=== Part 2: Finite-Sample Mini-Batch Dynamics (S=64, bs=16) ===")
    gen_data = torch.Generator().manual_seed(999)
    S, bs = 64, 16
    X = torch.randn(N, S, generator=gen_data, dtype=torch.float64)
    Y_clean = M_star @ X
    Y_noisy = Y_clean + 0.05 * torch.randn(M, S, generator=gen_data, dtype=torch.float64)

    mb_specs = [
        dict(
            name="MB_SGD_clean", label="MB-SGD (clean, wd=0)", opt_kind="sgd",
            B_init=B_lora, A_init=A_lora, Y_target=Y_clean,
            epochs=1500, batch_size=bs, lr=0.05, wd=0.0, cosine=False, seed=1,
        ),
        dict(
            name="MB_Adam_clean", label="MB-Adam (clean, wd=0)", opt_kind="adamw",
            B_init=B_lora, A_init=A_lora, Y_target=Y_clean,
            epochs=1500, batch_size=bs, lr=0.01, wd=0.0, cosine=True, seed=2,
        ),
        dict(
            name="MB_SGD_noisy", label="MB-SGD (noisy, wd=0)", opt_kind="sgd",
            B_init=B_lora, A_init=A_lora, Y_target=Y_noisy,
            epochs=2000, batch_size=bs, lr=0.05, wd=0.0, cosine=True, seed=3,
        ),
        dict(
            name="MB_AdamW_clean", label="MB-AdamW (clean, wd=1e-2)", opt_kind="adamw",
            B_init=B_lora, A_init=A_lora, Y_target=Y_clean,
            epochs=5000, batch_size=bs, lr=0.015, wd=1e-2, cosine=True, seed=4,
        ),
    ]

    minibatch_results = []
    for spec in mb_specs:
        res = run_minibatch_experiment(
            M_star=M_star, X=X, U=U, V=V, r_star=r_star, beta=beta, gamma=gamma, **spec
        )
        minibatch_results.append(res)

    # Load baseline q=0..8 curve from lora_fiber_phase_transition.json if present
    ref_path = out_dir / "lora_fiber_phase_transition.json"
    ref_rows = None
    if ref_path.exists():
        ref_data = json.loads(ref_path.read_text())
        if "lora_16x16_r2_H10" in ref_data:
            ref_rows = ref_data["lora_16x16_r2_H10"]["rows"]

    all_results = {
        "config": {
            "M": M, "N": N, "H": H, "r_star": r_star,
            "m": M - r_star, "n": N - r_star, "k": H - r_star,
            "q_c": (N - r_star + H - r_star - (M - r_star)) / 2.0,
            "lambda_AW": theoretical_lora_fiber_lambda(M, N, H, r_star, 0)[0],
            "lambda_lora_init": theoretical_lora_fiber_lambda(M, N, H, r_star, H - r_star)[0],
            "beta": beta, "gamma": gamma,
            "S_minibatch": S, "batch_size": bs,
        },
        "full_batch": full_batch_results,
        "minibatch": minibatch_results,
    }

    json_path = out_dir / "lora_sgd_dynamics.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    plot_path = out_dir / "lora_sgd_dynamics.png"
    make_figure(all_results, ref_rows, plot_path)

    print(f"\nDone in {time.time() - t_start:.1f}s. Saved {json_path} and {plot_path}")


if __name__ == "__main__":
    main()
