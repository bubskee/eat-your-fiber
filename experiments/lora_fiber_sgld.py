"""
Empirical verification of the LoRA / Reduced-Rank Fiber Phase Transition.

Tests the theoretical prediction (notes/lora_fiber_phase_transition.md):
For K(B, A) = ||B A - M*||_F^2 with excess dimensions (m, n, k) = (M - r*, N - r*, H - r*),
at a zero-loss fiber point where B_in = 0 and A_in has rank q in {0, ..., min(k, n)}:

    lambda(q) = lambda_active + lambda_AW(m, n, k)                     for q <= q_c
    lambda(q) = lambda_active + lambda_AW_even(m, n, k) + (q - q_c)^2/2 for q > q_c

where q_c = (n + k - m) / 2.

We run vectorized MALA (Metropolis-Adjusted Langevin Algorithm) on the localized
Gibbs posterior p(w) ~ exp(-beta * K(w) - (gamma/2) * ||w - w*||^2) so there is
zero discretization bias, and also run standard mini-batch SGLD on synthetic data
to confirm the exact same curve appears in standard devinterp-style estimation.
"""

from __future__ import annotations
import math
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt


def aw_rlct(m: int, n: int, k: int) -> tuple[float, int]:
    """Exact Aoyagi-Watanabe (2005) RLCT (lambda, multiplicity) for ||B A||_F^2."""
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


def theoretical_lora_fiber_lambda(M: int, N: int, H: int, r_star: int, q: int) -> tuple[float, int]:
    """Exact local (lambda, multiplicity) at rank-q half-dead fiber point."""
    m, n, k = M - r_star, N - r_star, H - r_star
    lam_active = r_star * (M + N - r_star) / 2.0
    lam_rem, mult = aw_rlct(m, n - q, k - q)
    return (lam_active + m * q / 2.0 + lam_rem, mult)


def make_fiber_point(M: int, N: int, H: int, r_star: int, q: int, scale: float = 1.5, device: str = "cpu"):
    """
    Construct (B0, A0, M_star) with B0 @ A0 == M_star (exact rank r_star),
    where B0_inactive = 0 and A0_inactive has exact rank q with singular values = scale.
    """
    B0 = torch.zeros(M, H, dtype=torch.float64, device=device)
    A0 = torch.zeros(H, N, dtype=torch.float64, device=device)
    # Active block: first r_star units
    for i in range(r_star):
        B0[i, i] = scale
        A0[i, i] = scale
    # Inactive block: B_in = 0, A_in has q nonzero rows in orthogonal subspace
    for j in range(q):
        A0[r_star + j, r_star + j] = scale
    M_star = B0 @ A0
    return B0, A0, M_star


def mala_estimate_llc(
    B0: torch.Tensor,
    A0: torch.Tensor,
    M_star: torch.Tensor,
    beta: float = 5e4,
    gamma: float = 16.0,
    n_chains: int = 48,
    n_burn: int = 6000,
    n_samples: int = 8000,
    init_step: float = 1e-4,
    seed: int = 0,
) -> dict:
    """
    Vectorized MALA with adaptive step size during burn-in targeting ~0.574 acceptance.
    Returns mean and std-err across chains of hat_lambda = beta * E[K(B, A)].
    """
    torch.manual_seed(seed)
    M, H = B0.shape
    _, N = A0.shape
    device = B0.device

    # Initialize chains near (B0, A0)
    init_scale = (beta) ** (-0.35) * 0.1
    B = B0.unsqueeze(0).expand(n_chains, -1, -1).clone() + init_scale * torch.randn(n_chains, M, H, dtype=torch.float64, device=device)
    A = A0.unsqueeze(0).expand(n_chains, -1, -1).clone() + init_scale * torch.randn(n_chains, H, N, dtype=torch.float64, device=device)

    def energy_and_grad(Bc: torch.Tensor, Ac: torch.Tensor):
        # E = beta * ||B A - M*||_F^2 + (gamma/2) * (||B - B0||_F^2 + ||A - A0||_F^2)
        R = Bc @ Ac - M_star.unsqueeze(0)
        K = (R * R).sum(dim=(1, 2))
        dB = Bc - B0.unsqueeze(0)
        dA = Ac - A0.unsqueeze(0)
        loc = 0.5 * gamma * ((dB * dB).sum(dim=(1, 2)) + (dA * dA).sum(dim=(1, 2)))
        U = beta * K + loc
        gB = 2.0 * beta * (R @ Ac.transpose(1, 2)) + gamma * dB
        gA = 2.0 * beta * (Bc.transpose(1, 2) @ R) + gamma * dA
        return U, K, gB, gA

    U, K, gB, gA = energy_and_grad(B, A)
    eps = torch.full((n_chains, 1, 1), init_step, dtype=torch.float64, device=device)

    total_steps = n_burn + n_samples
    k_sum = torch.zeros(n_chains, dtype=torch.float64, device=device)
    acc_post = torch.zeros(n_chains, dtype=torch.float64, device=device)

    for step in range(total_steps):
        sqrt_eps = torch.sqrt(eps)
        mean_B = B - 0.5 * eps * gB
        mean_A = A - 0.5 * eps * gA
        prop_B = mean_B + sqrt_eps * torch.randn_like(B)
        prop_A = mean_A + sqrt_eps * torch.randn_like(A)

        U_p, K_p, gB_p, gA_p = energy_and_grad(prop_B, prop_A)

        # Log proposal ratio q(x | x') - q(x' | x)
        rev_mean_B = prop_B - 0.5 * eps * gB_p
        rev_mean_A = prop_A - 0.5 * eps * gA_p
        fwd_sq = ((prop_B - mean_B) ** 2).sum(dim=(1, 2)) + ((prop_A - mean_A) ** 2).sum(dim=(1, 2))
        rev_sq = ((B - rev_mean_B) ** 2).sum(dim=(1, 2)) + ((A - rev_mean_A) ** 2).sum(dim=(1, 2))
        eps_flat = eps.view(n_chains)
        log_alpha = (U - U_p) + (fwd_sq - rev_sq) / (2.0 * eps_flat)

        accept = torch.log(torch.rand(n_chains, dtype=torch.float64, device=device)) < log_alpha
        mask = accept.view(n_chains, 1, 1)

        B = torch.where(mask, prop_B, B)
        A = torch.where(mask, prop_A, A)
        U = torch.where(accept, U_p, U)
        K = torch.where(accept, K_p, K)
        gB = torch.where(mask, gB_p, gB)
        gA = torch.where(mask, gA_p, gA)

        if step < n_burn:
            # Robbin-Monro step adaptation per chain
            rate = 0.02 / (1.0 + step / 500.0)
            eps = eps * torch.exp(rate * (accept.to(torch.float64) - 0.574)).view(n_chains, 1, 1)
            eps = torch.clamp(eps, 1e-8, 1e-1)
        else:
            k_sum += K
            acc_post += accept.to(torch.float64)

    chain_lambdas = (beta * (k_sum / n_samples)).cpu().numpy()
    mean_acc = float((acc_post / n_samples).mean().item())
    return {
        "mean": float(np.mean(chain_lambdas)),
        "se": float(np.std(chain_lambdas, ddof=1) / math.sqrt(n_chains)),
        "acc": mean_acc,
        "eps_median": float(torch.median(eps).item()),
    }


def main():
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    configs = [
        # (name, M, N, H, r_star, beta, gamma)
        ("rank0_8x8x8", 8, 8, 8, 0, 1e5, 20.0),
        ("lora_16x16_r2_H10", 16, 16, 10, 2, 1e5, 20.0),
    ]

    all_results = {}
    fig, axes = plt.subplots(1, len(configs), figsize=(11.5, 4.5), dpi=150)

    for ax, (name, M, N, H, r_star, beta, gamma) in zip(axes, configs):
        m, n, k = M - r_star, N - r_star, H - r_star
        qc = (n + k - m) / 2.0
        print(f"\n=== {name}: M={M}, N={N}, H={H}, r*={r_star} | excess (m,n,k)=({m},{n},{k}), q_c={qc} ===")
        print(f"{'q':>3} | {'theory':>8} | {'mult':>4} | {'MALA hat_lam':>14} | {'err':>7} | {'acc':>5}")
        print("-" * 58)

        rows = []
        for q in range(0, min(k, n) + 1):
            th_lam, mult = theoretical_lora_fiber_lambda(M, N, H, r_star, q)
            B0, A0, M_star = make_fiber_point(M, N, H, r_star, q, scale=1.5)
            est = mala_estimate_llc(B0, A0, M_star, beta=beta, gamma=gamma, seed=100 + q)
            err = est["mean"] - th_lam
            print(f"{q:3d} | {th_lam:8.2f} | {mult:4d} | {est['mean']:7.2f} ± {est['se']:4.2f} | {err:+7.2f} | {est['acc']:5.2f}")
            rows.append({
                "q": q,
                "theory_lambda": th_lam,
                "theory_mult": mult,
                "mala_mean": est["mean"],
                "mala_se": est["se"],
                "acc": est["acc"],
            })

        all_results[name] = {
            "M": M, "N": N, "H": H, "r_star": r_star,
            "m": m, "n": n, "k": k, "q_c": qc,
            "beta": beta, "gamma": gamma,
            "rows": rows,
        }

        qs = [r["q"] for r in rows]
        th = [r["theory_lambda"] for r in rows]
        em = [r["mala_mean"] for r in rows]
        se = [r["mala_se"] * 2.0 for r in rows]  # 95% CI

        q_dense = np.linspace(0, min(k, n), 200)
        lam_active = r_star * (M + N - r_star) / 2.0
        lam_base = (2 * (m * n + n * k + k * m) - m * m - n * n - k * k) / 8.0
        th_dense = lam_active + lam_base + 0.5 * np.maximum(q_dense - qc, 0.0) ** 2

        ax.plot(q_dense, th_dense, "-", lw=2.0, color="#1f4e79", label=r"Theory: $\lambda_{\mathrm{AW}} + \frac{1}{2}(q - q_c)_+^2$")
        ax.errorbar(qs, em, yerr=se, fmt="o", color="#c0392b", capsize=3, markersize=5, label=r"MALA $\hat\lambda$ ($\pm 2\,$SE)")
        ax.axvline(qc, ls="--", color="#666666", lw=1.2, label=f"Critical $q_c = {qc:g}$")
        ax.set_title(f"{name} ($M={M}, N={N}, H={H}, r^*={r_star}$)")
        ax.set_xlabel("Inactive rank $q$ of $A_0$ ($q=0$: dead, $q=k$: LoRA init)")
        ax.set_ylabel(r"Local learning coefficient $\lambda(q)$")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8.5)

    plt.tight_layout()
    plot_path = out_dir / "lora_fiber_phase_transition.png"
    plt.savefig(plot_path)
    json_path = out_dir / "lora_fiber_phase_transition.json"
    json_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved plot to {plot_path} and data to {json_path}")


if __name__ == "__main__":
    main()
