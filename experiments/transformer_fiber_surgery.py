"""
Track 2 Flagship Empirical Bridge: Exact Iso-Functional Fiber Surgery, Continual Learning,
Optimizer Phase-Diagram Trajectories, and Multi-Head Attention Coupling on a Real
Pretrained 2-Layer Causal Transformer.

Tests Lean-certified Track 2 theorems (`RRRFiberStratification.lean`, `AttentionLoRA.lean`)
on a genuine 2-layer causal GPT/LLaMA-style Transformer (token + positional embeddings,
Pre-RMSNorm [plus Pre-LayerNorm architectural comparison], causal multi-head self-attention,
residual stream, GELU MLP, LM head, real next-token cross-entropy & functional KL divergence):

  Part 1: Exact Iso-Functional 2D (p, q) Fiber Surgery on a Pretrained 2-Layer Causal Transformer
    - Sweeps the full 2D (p, q) phase diagram (A-axis, B-axis, balanced diagonal, off-axis
      supercritical points) at function-identical checkpoints (max|Delta logit| < 1e-13),
      verifying the flat 2D plateau (p <= p_c, q <= q_c), A-supercritical parabola +(q-q_c)^2/2,
      symmetric B-supercritical parabola +(p-p_c)^2/2, and mutual exclusivity.
    - Bonus architectural check: Pre-LayerNorm's exact 1_d shift-invariance kills 1 output
      dimension (m_eff = d - 1 = 11, 12 exact zero Fisher eigenvalues), breaking p <-> q symmetry!

  Part 2: Downstream Consequence — Plasticity-Stability Split from Function-Matched
          Task-A Checkpoints
    - Compares 5 function-matched Task-A checkpoints (4 to ~1e-14; Near-Plateau to 6e-4): Exact (0,0), Near-Plateau (0,0)+1e-3,
      Balanced (4,4) [same low LLC, nonzero norms!], A-Supercrit (0,8), B-Supercrit (8,0).
    - Multi-seed (5 seeds) Task B fine-tuning under BOTH SGD and Adam on:
        (2a) Clean shifted real-token associative recall (real next-token Cross-Entropy)
        (2b) Noisy limited-sample real-token Task B (overfitting gap & generalization)
        (2c) Directional V_perp vs U_perp Task-B shift probe (mirror-image plasticity split)

  Part 3: Real Optimizer Trajectories on the 2D (p, q) Phase Diagram + Functional vs.
          Iso-Functional Delta-lambda Decomposition
    - Compares Standard LoRA+Adam (wd=0), LoRA+AdamW (wd>0), LoRA+SGD+WD, LoRA+ (eta_B = 12*eta_A),
      and PiSSA/SVD-Balanced small init on Task A.
    - Explicitly decomposes late-stage Delta hat_lambda (after Task-A KL has converged!) into
      pure functional change vs. pure iso-functional fiber sliding via exact balanced-SVD surgery.

  Part 4: Multi-Head Attention V+O vs. Q+K Coupling Dichotomy inside the Real 2-Layer Causal
          Transformer (n_heads in {1, 2, 4}, d_model=12, H=4)
    - Verifies single-head V+O (4x jump) vs Q+K (GL(q) flat cancellation) and multi-head
      decoupling reversal (both jumping identically at n_heads = 2, 4) through real causal
      softmax + RMSNorm + GELU MLP + LM head + next-token KL divergence.

Usage:
  python transformer_fiber_surgery.py            # full run (pretraining + MALA; ~12 min on CPU)
  python transformer_fiber_surgery.py --replot   # redraw the figure from the saved JSON only

Outputs (relative to this file):
  - results/transformer_fiber_surgery.json
  - results/transformer_fiber_surgery.png
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
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. Exact Theoretical Formulas (Matching Lean RRRFiberStratification & AttentionLoRA)
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


def theoretical_rrr_2d_lambda(M: int, N: int, H: int, r_star: int, p: int, q: int) -> tuple[float, int]:
    """
    Exact Lean-certified 2D (r*, p, q) RRR fiber RLCT from RRRFiberStratification.lean:
      lambda(r*, p, q) = r*(M + N - r*)/2 + (p*n + q*m - p*q)/2 + lambda_AW(m - p, n - q, k - p - q)
    where m = M - r*, n = N - r*, k = H - r*, p + q <= k, p <= m, q <= n.
    """
    m, n, k = M - r_star, N - r_star, H - r_star
    assert p >= 0 and q >= 0 and p + q <= k and p <= m and q <= n
    lam_act = r_star * (M + N - r_star) / 2.0
    lam_pq = (p * n + q * m - p * q) / 2.0
    lam_rem, mult = aw_rlct(m - p, n - q, k - p - q)
    return (lam_act + lam_pq + lam_rem, mult)


def entropy_effective_rank(sv: np.ndarray, eps: float = 1e-15) -> float:
    s2 = sv ** 2
    tot = float(np.sum(s2))
    if tot <= eps:
        return 0.0
    p = s2 / tot
    p = p[p > eps]
    return float(np.exp(-np.sum(p * np.log(p))))


# =============================================================================
# 2. Structured Sequence Dataset & Real 2-Layer Causal Transformer LM
# =============================================================================

def generate_ar_dataset(
    n_seq: int,
    rng: np.random.Generator,
    shift_rule: int = 0,
    corrupt_prob: float = 0.0,
) -> torch.Tensor:
    """
    Structured 3-pair Associative Recall + Shifted Modular Composition (V=24, T=12):
      - Tokens 0..7: 8 Keys
      - Tokens 8..15: 8 Values
      - Tokens 16..23: 8 Modifiers
      Sequence layout (length 12):
        [k0, v0, k1, v1, k2, v2, mod, q1, r1, c1, q2, r2]
      where:
        k0, k1, k2 are 3 distinct keys from 0..7
        v0, v1, v2 are 3 values from 8..15
        mod is in 16..23
        q1, q2 are 2 distinct queries from {k0, k1, k2}
        r1 = 8 + ((val(q1) - 8 + shift_rule) % 8)
        c1 = 8 + ((r1 - 8 + (mod - 16)) % 8)
        r2 = 8 + ((val(q2) - 8 + shift_rule) % 8)
      If corrupt_prob > 0, independently corrupts target tokens r1, c1, r2 to random values
      in 8..15 with probability corrupt_prob (noisy fine-tuning regime).
    """
    ks = np.argsort(rng.random((n_seq, 8)), axis=1)[:, :3]
    vs = rng.integers(8, 16, size=(n_seq, 3))
    mod = rng.integers(16, 24, size=(n_seq,))
    q_perm = np.argsort(rng.random((n_seq, 3)), axis=1)[:, :2]
    ar = np.arange(n_seq)
    q1 = ks[ar, q_perm[:, 0]]
    q2 = ks[ar, q_perm[:, 1]]
    r1 = 8 + ((vs[ar, q_perm[:, 0]] - 8 + shift_rule) % 8)
    c1 = 8 + ((r1 - 8 + (mod - 16)) % 8)
    r2 = 8 + ((vs[ar, q_perm[:, 1]] - 8 + shift_rule) % 8)

    if corrupt_prob > 0.0:
        for arr in (r1, c1, r2):
            m = rng.random(n_seq) < corrupt_prob
            arr[m] = rng.integers(8, 16, size=int(m.sum()))

    seqs = np.empty((n_seq, 12), dtype=np.int64)
    seqs[:, 0], seqs[:, 1] = ks[:, 0], vs[:, 0]
    seqs[:, 2], seqs[:, 3] = ks[:, 1], vs[:, 1]
    seqs[:, 4], seqs[:, 5] = ks[:, 2], vs[:, 2]
    seqs[:, 6] = mod
    seqs[:, 7], seqs[:, 8], seqs[:, 9] = q1, r1, c1
    seqs[:, 10], seqs[:, 11] = q2, r2
    return torch.from_numpy(seqs)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, T: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_Q = nn.Parameter(torch.empty(d_model, d_model))
        self.W_K = nn.Parameter(torch.empty(d_model, d_model))
        self.W_V = nn.Parameter(torch.empty(d_model, d_model))
        self.W_O = nn.Parameter(torch.empty(d_model, d_model))
        for p in (self.W_Q, self.W_K, self.W_V, self.W_O):
            nn.init.orthogonal_(p, gain=0.8)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask)

    def forward(self, x: torch.Tensor, dQ=None, dK=None, dV=None, dO=None) -> torch.Tensor:
        S, T, d = x.shape
        WQ = self.W_Q if dQ is None else self.W_Q + dQ
        WK = self.W_K if dK is None else self.W_K + dK
        WV = self.W_V if dV is None else self.W_V + dV
        WO = self.W_O if dO is None else self.W_O + dO
        Q = (x @ WQ.T).view(S, T, self.n_heads, self.d_head).transpose(1, 2)
        K = (x @ WK.T).view(S, T, self.n_heads, self.d_head).transpose(1, 2)
        V = (x @ WV.T).view(S, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (Q @ K.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.causal_mask[:T, :T], float("-inf"))
        P = F.softmax(att, dim=-1)
        out = (P @ V).transpose(1, 2).contiguous().view(S, T, d)
        return out @ WO.T


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mlp: int, T: int, norm_type: str = "rmsnorm"):
        super().__init__()
        Norm = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
        self.norm_type = norm_type
        self.ln1 = Norm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, T)
        self.ln2 = Norm(d_model)
        self.fc1 = nn.Linear(d_model, d_mlp)
        self.fc2 = nn.Linear(d_mlp, d_model)

    def forward(self, x: torch.Tensor, dQ=None, dK=None, dV=None, dO=None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), dQ=dQ, dK=dK, dV=dV, dO=dO)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int = 24,
        d_model: int = 12,
        n_heads: int = 2,
        d_mlp: int = 24,
        T: int = 12,
        norm_type: str = "rmsnorm",
    ):
        super().__init__()
        Norm = RMSNorm if norm_type == "rmsnorm" else nn.LayerNorm
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_mlp = d_mlp
        self.T = T
        self.norm_type = norm_type
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(T, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_mlp, T, norm_type),
            TransformerBlock(d_model, n_heads, d_mlp, T, norm_type),
        ])
        self.ln_f = Norm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor, layer1_deltas: dict | None = None) -> torch.Tensor:
        S, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.blocks[0](x)
        x = self.blocks[1](x) if layer1_deltas is None else self.blocks[1](x, **layer1_deltas)
        return self.lm_head(self.ln_f(x))


def pretrain_transformer(
    n_heads: int = 2,
    norm_type: str = "rmsnorm",
    steps: int = 3000,
    seed: int = 42,
) -> tuple[CausalTransformerLM, dict]:
    t0 = time.time()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    model = CausalTransformerLM(
        vocab_size=24, d_model=12, n_heads=n_heads, d_mlp=24, T=12, norm_type=norm_type
    )
    X_val = generate_ar_dataset(512, np.random.default_rng(999))
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-2, weight_decay=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=2e-4)

    for _ in range(steps):
        batch = generate_ar_dataset(128, rng)
        logits = model(batch)[:, :-1, :]
        targets = batch[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, 24), targets.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    model = model.double()
    for p in model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        lv = model(X_val)[:, :-1, :]
        tv = X_val[:, 1:]
        val_ce = float(F.cross_entropy(lv.reshape(-1, 24), tv.reshape(-1)).item())
        acc_r1 = float((lv[:, 7].argmax(-1) == tv[:, 7]).double().mean().item())
        acc_c1 = float((lv[:, 8].argmax(-1) == tv[:, 8]).double().mean().item())
        acc_r2 = float((lv[:, 10].argmax(-1) == tv[:, 10]).double().mean().item())

    info = {
        "n_heads": n_heads,
        "norm_type": norm_type,
        "steps": steps,
        "val_ce": val_ce,
        "acc_r1": acc_r1,
        "acc_c1": acc_c1,
        "acc_r2": acc_r2,
        "n_params": sum(p.numel() for p in model.parameters()),
        "elapsed_s": round(time.time() - t0, 2),
    }
    print(
        f"  [Pretrain {norm_type:9s} h={n_heads}] {info['elapsed_s']:.1f}s | "
        f"val_CE={val_ce:.4f} | r1={acc_r1:.3f}, c1={acc_c1:.3f}, r2={acc_r2:.3f}"
    )
    return model, info


# =============================================================================
# 3. Bit-Exact Cached Layer-1 Forward Wrappers (for Fast Float64 MALA & Training)
# =============================================================================

def _apply_norm_batched(x: torch.Tensor, norm_mod: nn.Module) -> torch.Tensor:
    if isinstance(norm_mod, RMSNorm):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + norm_mod.eps) * norm_mod.weight
    # Exact LayerNorm
    mean = x.mean(dim=-1, keepdim=True)
    var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + norm_mod.eps) * norm_mod.weight + norm_mod.bias


class Layer1WOCache:
    """
    Caches frozen pre-W_O activations of Layer 1 on a fixed sequence batch X.
    Bit-identical (max|diff| < 1e-13) to running the full 2-layer causal Transformer,
    evaluating W_O + dW -> residual -> LN2 -> GELU MLP -> residual -> LN_f -> LM head -> KL / CE.
    """

    def __init__(self, model: CausalTransformerLM, X: torch.Tensor):
        self.model = model
        self.X = X
        blk1 = model.blocks[1]
        att1 = blk1.attn
        with torch.no_grad():
            S, T = X.shape
            pos = torch.arange(T)
            x0 = model.blocks[0](model.tok_emb(X) + model.pos_emb(pos))
            x1_norm = blk1.ln1(x0)
            Q = (x1_norm @ att1.W_Q.T).view(S, T, att1.n_heads, att1.d_head).transpose(1, 2)
            K = (x1_norm @ att1.W_K.T).view(S, T, att1.n_heads, att1.d_head).transpose(1, 2)
            V = (x1_norm @ att1.W_V.T).view(S, T, att1.n_heads, att1.d_head).transpose(1, 2)
            att = ((Q @ K.transpose(-2, -1)) / math.sqrt(att1.d_head)).masked_fill(
                att1.causal_mask[:T, :T], float("-inf")
            )
            P = F.softmax(att, dim=-1)
            h_att = (P @ V).transpose(1, 2).contiguous().view(S, T, att1.d_model)
            self.S, self.T = S, T
            self.N_tok = S * (T - 1)
            self.x0_f = x0[:, :-1, :].reshape(self.N_tok, att1.d_model).detach()
            self.h_att_f = h_att[:, :-1, :].reshape(self.N_tok, att1.d_model).detach()
            self.targets_f = X[:, 1:].reshape(self.N_tok).detach()
            self.WO0 = att1.W_O.detach()
            self.ln2 = blk1.ln2
            self.fc1_w, self.fc1_b = blk1.fc1.weight.detach(), blk1.fc1.bias.detach()
            self.fc2_w, self.fc2_b = blk1.fc2.weight.detach(), blk1.fc2.bias.detach()
            self.ln_f = model.ln_f
            self.lm_w, self.lm_b = model.lm_head.weight.detach(), model.lm_head.bias.detach()

    def logits_single(self, dW: torch.Tensor) -> torch.Tensor:
        """Returns (N_tok, V) logits for a single (d, d) weight delta dW."""
        ao = self.h_att_f @ (self.WO0 + dW).T
        x1 = self.x0_f + ao
        x1n = _apply_norm_batched(x1, self.ln2)
        h_m = F.gelu(x1n @ self.fc1_w.T + self.fc1_b)
        x2 = x1 + h_m @ self.fc2_w.T + self.fc2_b
        x2n = _apply_norm_batched(x2, self.ln_f)
        return x2n @ self.lm_w.T + self.lm_b

    def logits_batched(self, dWs: torch.Tensor) -> torch.Tensor:
        """Returns (C, N_tok, V) logits for a batch of (C, d, d) deltas."""
        W_eff = self.WO0.unsqueeze(0) + dWs
        ao = torch.einsum("nd,ced->cne", self.h_att_f, W_eff)
        x1 = self.x0_f.unsqueeze(0) + ao
        x1n = _apply_norm_batched(x1, self.ln2)
        h_m = F.gelu(torch.matmul(x1n, self.fc1_w.T) + self.fc1_b)
        x2 = x1 + torch.matmul(h_m, self.fc2_w.T) + self.fc2_b
        x2n = _apply_norm_batched(x2, self.ln_f)
        return torch.matmul(x2n, self.lm_w.T) + self.lm_b

    def make_kl_loss_fn(self, dW_ref: torch.Tensor):
        with torch.no_grad():
            lg_ref = self.logits_single(dW_ref)
            p_ref = F.softmax(lg_ref, dim=-1)
            log_p_ref = F.log_softmax(lg_ref, dim=-1)

        def kl_fn(ps: list[torch.Tensor]) -> torch.Tensor:
            dWs = ps[0] @ ps[1]
            lp = F.log_softmax(self.logits_batched(dWs), dim=-1)
            return (p_ref.unsqueeze(0) * (log_p_ref.unsqueeze(0) - lp)).sum(dim=-1).mean(dim=1)

        return kl_fn, p_ref, log_p_ref

    def ce_loss_single(self, dW: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(self.logits_single(dW), self.targets_f)

    def kl_to_ref_single(self, dW: torch.Tensor, p_ref: torch.Tensor, log_p_ref: torch.Tensor) -> torch.Tensor:
        lp = F.log_softmax(self.logits_single(dW), dim=-1)
        return (p_ref * (log_p_ref - lp)).sum(dim=-1).mean()


def make_batched_layer1_attn_kl(model: CausalTransformerLM, X_eval: torch.Tensor, mode: str):
    """Exact bit-identical cached Layer-1 Attention (V+O or Q+K) -> LN2 + MLP + LN_f + LM head -> KL."""
    blk1 = model.blocks[1]
    att1 = blk1.attn
    d, nh, dh = att1.d_model, att1.n_heads, att1.d_head
    with torch.no_grad():
        S, T = X_eval.shape
        pos = torch.arange(T)
        x0 = model.blocks[0](model.tok_emb(X_eval) + model.pos_emb(pos))
        x1n = blk1.ln1(x0)
        WQ0, WK0, WV0, WO0 = att1.W_Q.detach(), att1.W_K.detach(), att1.W_V.detach(), att1.W_O.detach()
        cmask = att1.causal_mask[:T, :T]
        ln2, ln_f = blk1.ln2, model.ln_f
        fc1_w, fc1_b = blk1.fc1.weight.detach(), blk1.fc1.bias.detach()
        fc2_w, fc2_b = blk1.fc2.weight.detach(), blk1.fc2.bias.detach()
        lm_w, lm_b = model.lm_head.weight.detach(), model.lm_head.bias.detach()

        Q0 = (x1n @ WQ0.T).view(S, T, nh, dh).transpose(1, 2)
        K0 = (x1n @ WK0.T).view(S, T, nh, dh).transpose(1, 2)
        V0 = (x1n @ WV0.T).view(S, T, nh, dh).transpose(1, 2)
        att0 = ((Q0 @ K0.transpose(-2, -1)) / math.sqrt(dh)).masked_fill(cmask, float("-inf"))
        P0 = F.softmax(att0, dim=-1)
        PX0 = torch.einsum("shtu,sud->shtd", P0[:, :, :-1, :], x1n)
        x0_sub = x0[:, :-1, :]

        lg_ref = model(X_eval)[:, :-1, :]
        p_ref = F.softmax(lg_ref, dim=-1)
        log_p_ref = F.log_softmax(lg_ref, dim=-1)

    def tail_kl(ao: torch.Tensor) -> torch.Tensor:
        x1 = x0_sub.unsqueeze(0) + ao
        x1_n = _apply_norm_batched(x1, ln2)
        h_m = F.gelu(torch.matmul(x1_n, fc1_w.T) + fc1_b)
        x2 = x1 + torch.matmul(h_m, fc2_w.T) + fc2_b
        x2_n = _apply_norm_batched(x2, ln_f)
        lg = torch.matmul(x2_n, lm_w.T) + lm_b
        lp = F.log_softmax(lg, dim=-1)
        return (p_ref.unsqueeze(0) * (log_p_ref.unsqueeze(0) - lp)).sum(dim=-1).mean(dim=(1, 2))

    if mode == "VO":
        def kl_vo(ps):
            C = ps[0].shape[0]
            WV = (WV0.unsqueeze(0) + ps[0] @ ps[1]).view(C, nh, dh, d)
            WO = WO0.unsqueeze(0) + ps[2] @ ps[3]
            pv = torch.einsum("shtd,chkd->csthk", PX0, WV).reshape(C, S, T - 1, d)
            ao = torch.einsum("cstd,ced->cste", pv, WO)
            return tail_kl(ao)
        return kl_vo
    else:
        V0_t = V0.transpose(1, 2)
        def kl_qk(ps):
            C = ps[0].shape[0]
            WQ = (WQ0.unsqueeze(0) + ps[0] @ ps[1]).view(C, nh, dh, d)
            WK = (WK0.unsqueeze(0) + ps[2] @ ps[3]).view(C, nh, dh, d)
            Q = torch.einsum("std,chkd->cshtk", x1n[:, :-1, :], WQ)
            K = torch.einsum("sud,chkd->cshuk", x1n, WK)
            att = torch.einsum("cshtk,cshuk->cshtu", Q, K) / math.sqrt(dh)
            att = att.masked_fill(cmask[:-1, :], float("-inf"))
            P = F.softmax(att, dim=-1)
            pv = torch.einsum("cshtu,suhk->csthk", P, V0_t).reshape(C, S, T - 1, d)
            ao = torch.matmul(pv, WO0.T)
            return tail_kl(ao)
        return kl_qk


# =============================================================================
# 4. Exact Full-Matrix Preconditioned MALA (Constant Linear Map -> Exact Detailed Balance)
# =============================================================================

def mala_full_precond(
    params0: list[torch.Tensor],
    loss_fn,
    beta: float = 1e9,
    gamma: float = 5.0,
    n_chains: int = 20,
    n_burn: int = 1100,
    n_samples: int = 1500,
    seed: int = 0,
    subtract_base_loss: bool = False,
    soft_thresh_mult: float = 10.0,
) -> dict:
    """
    Exact Metropolis-Adjusted Langevin Algorithm in preconditioned coordinates w = w0 + P @ z:
      1. Exact Hessian H_0 of beta*K at w0 (central FD of exact gradients).
      2. Soft/null-subspace (eig < soft_thresh_mult * gamma) probe curvature at quartic scale
         separates quartic singular/transition modes from flat gauge directions while keeping
         P strictly constant across MCMC steps.
      3. Constant invertible bijection P preserves exact detailed balance for the target
         localized Gibbs measure exp(-beta * K(w) - 0.5 * gamma * ||w - w0||_2^2).
    Returns hat_lambda = beta * E[K(w)], inter-chain SE, acceptance, split-half drift,
    and local Hessian Gaussian diagnostic (gauss_lam, n_hess_gt_gamma).
    """
    torch.manual_seed(seed)
    p0s = [p.detach().clone().to(torch.float64) for p in params0]
    shapes = [p.shape for p in p0s]
    sizes = [p.numel() for p in p0s]
    D = sum(sizes)
    w0_flat = torch.cat([p.reshape(-1) for p in p0s])

    def unpack_batch(w_flat: torch.Tensor) -> list[torch.Tensor]:
        C = w_flat.shape[0]
        out = []
        off = 0
        for sh, sz in zip(shapes, sizes):
            out.append(w_flat[:, off : off + sz].view(C, *sh))
            off += sz
        return out

    with torch.no_grad():
        base_loss = float(loss_fn(unpack_batch(w0_flat.unsqueeze(0)))[0].item()) if subtract_base_loss else 0.0

    # 1. Exact local Hessian H_0 of beta*K at w0
    eps_fd = 1e-6
    eye_D = torch.eye(D, dtype=torch.float64)
    H_cols = torch.zeros(D, D, dtype=torch.float64)
    bs_fd = 64
    for start in range(0, D, bs_fd):
        end = min(start + bs_fd, D)
        E = eye_D[start:end]
        W_p = (w0_flat.unsqueeze(0) + eps_fd * E).detach().requires_grad_(True)
        W_m = (w0_flat.unsqueeze(0) - eps_fd * E).detach().requires_grad_(True)
        g_p = torch.autograd.grad((beta * (loss_fn(unpack_batch(W_p)) - base_loss)).sum(), W_p)[0].detach()
        g_m = torch.autograd.grad((beta * (loss_fn(unpack_batch(W_m)) - base_loss)).sum(), W_m)[0].detach()
        H_cols[start:end] = (g_p - g_m) / (2.0 * eps_fd)

    H_sym = 0.5 * (H_cols + H_cols.T)
    eigvals, eigvecs = torch.linalg.eigh(H_sym)
    eig_pos = eigvals.clamp(min=0.0)
    gauss_lam = float(0.5 * (eig_pos / (eig_pos + gamma)).sum().item())
    n_hess_gt_g = int((eig_pos > gamma).sum().item())

    # 2. Soft/null-subspace quartic-scale curvature estimate
    med_pos = float(eig_pos[eig_pos > gamma].median().item()) if (eig_pos > gamma).any() else float(beta * 1e-3)
    delta_q = min(0.05, max(1e-4, 0.5 * (med_pos ** -0.25)))
    null_mask = eig_pos < soft_thresh_mult * gamma
    eig_eff = eig_pos + gamma
    if null_mask.any():
        Q_null = eigvecs[:, null_mask]
        n_pr = 32
        r_null = delta_q * torch.randn(n_pr, Q_null.shape[1], dtype=torch.float64)
        W_pr = (w0_flat.unsqueeze(0) + r_null @ Q_null.T).detach().requires_grad_(True)
        K_pr = loss_fn(unpack_batch(W_pr)) - base_loss
        g_pr = torch.autograd.grad((beta * K_pr).sum(), W_pr)[0].detach()
        g_null = g_pr @ Q_null
        curv_null = (g_null.pow(2).mean(dim=0).sqrt() / delta_q).clamp(min=0.0)
        eig_eff = eig_eff.clone()
        if soft_thresh_mult > 20.0:
            eig_eff[null_mask] = (curv_null + gamma).clamp(min=gamma)
        else:
            eig_eff[null_mask] = (eig_pos[null_mask] + curv_null + gamma).clamp(min=gamma)

    P = (eigvecs * eig_eff.rsqrt().unsqueeze(0)) @ eigvecs.T

    # 3. Run exact MALA in z-space
    z = 0.05 * torch.randn(n_chains, D, dtype=torch.float64)

    def eval_eg(cur_z: torch.Tensor):
        req_z = cur_z.detach().requires_grad_(True)
        dw = req_z @ P.T
        w = w0_flat.unsqueeze(0) + dw
        K = loss_fn(unpack_batch(w)) - base_loss
        loc = 0.5 * gamma * (dw ** 2).sum(dim=1)
        U = beta * K + loc
        g = torch.autograd.grad(U.sum(), req_z)[0].detach()
        return U.detach(), K.detach(), g

    U, K, g = eval_eg(z)
    eps = torch.full((n_chains,), 0.05, dtype=torch.float64)
    k_sum = torch.zeros(n_chains, dtype=torch.float64)
    k1 = torch.zeros(n_chains, dtype=torch.float64)
    k2 = torch.zeros(n_chains, dtype=torch.float64)
    acc_post = torch.zeros(n_chains, dtype=torch.float64)
    half = n_samples // 2

    for step in range(n_burn + n_samples):
        sqrt_eps = torch.sqrt(eps).unsqueeze(1)
        m = z - 0.5 * eps.unsqueeze(1) * g
        z_p = m + sqrt_eps * torch.randn_like(z)
        U_p, K_p, g_p = eval_eg(z_p)

        rm = z_p - 0.5 * eps.unsqueeze(1) * g_p
        fwd_sq = ((z_p - m) ** 2).sum(dim=1)
        rev_sq = ((z - rm) ** 2).sum(dim=1)
        log_alpha = (U - U_p) + (fwd_sq - rev_sq) / (2.0 * eps)
        log_alpha = torch.where(torch.isfinite(log_alpha), log_alpha, torch.full_like(log_alpha, -1e9))
        accept = torch.log(torch.rand(n_chains, dtype=torch.float64)) < log_alpha

        mask = accept.unsqueeze(1)
        z = torch.where(mask, z_p, z)
        g = torch.where(mask, g_p, g)
        U = torch.where(accept, U_p, U)
        K = torch.where(accept, K_p, K)

        if step < n_burn:
            rate = 0.04 / (1.0 + step / 250.0)
            eps = torch.clamp(eps * torch.exp(rate * (accept.to(torch.float64) - 0.574)), 1e-8, 2.0)
        else:
            idx = step - n_burn
            k_sum = k_sum + K
            if idx < half:
                k1 = k1 + K
            else:
                k2 = k2 + K
            acc_post = acc_post + accept.to(torch.float64)

    chain_lam = (beta * (k_sum / n_samples)).numpy()
    h1 = float(np.mean((beta * (k1 / max(half, 1))).numpy()))
    h2 = float(np.mean((beta * (k2 / max(n_samples - half, 1))).numpy()))
    return {
        "mean": float(np.mean(chain_lam)),
        "se": float(np.std(chain_lam, ddof=1) / math.sqrt(n_chains)),
        "acc": float((acc_post / n_samples).mean().item()),
        "eps_med": float(torch.median(eps).item()),
        "hdiff": float(h2 - h1),
        "gauss_lam": gauss_lam,
        "n_hess_gt_gamma": n_hess_gt_g,
        "base_loss": base_loss,
    }


# =============================================================================
# 5. Iso-Functional Fiber Surgery & Spectral Metrics
# =============================================================================

@torch.no_grad()
def construct_pq_checkpoint(
    U: torch.Tensor,
    s_act: torch.Tensor,
    V: torch.Tensor,
    H: int,
    p: int,
    q: int,
    scale: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Construct exact iso-functional fiber checkpoint (B', A') at (r*, p, q) satisfying
      B' A' = U[:, :r*] diag(s_act) V[:, :r*]^T
    identically to float64 machine precision!
    """
    M = U.shape[0]
    N = V.shape[0]
    r_star = s_act.numel()
    assert p + q <= H - r_star
    B = torch.zeros(M, H, dtype=torch.float64)
    A = torch.zeros(H, N, dtype=torch.float64)
    B[:, :r_star] = U[:, :r_star] * s_act.sqrt().unsqueeze(0)
    A[:r_star, :] = s_act.sqrt().unsqueeze(1) * V[:, :r_star].T
    if p > 0:
        B[:, r_star : r_star + p] = scale * U[:, r_star : r_star + p]
    if q > 0:
        A[r_star + p : r_star + p + q, :] = scale * V[:, r_star : r_star + q].T
    return B, A


@torch.no_grad()
def balanced_svd_surgery(B: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Given ANY rank-H LoRA adapter (B, A), return its exact iso-functional balanced-SVD
    factorization (B_bal, A_bal) satisfying B_bal @ A_bal == B @ A to machine precision
    with ZERO inactive fiber norm (sits on the (0,0) subcritical plateau for the same function!).
    """
    M, H = B.shape
    N = A.shape[1]
    dW = B @ A
    U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
    r = min(H, S.numel())
    B_bal = torch.zeros(M, H, dtype=torch.float64)
    A_bal = torch.zeros(H, N, dtype=torch.float64)
    s_rt = S[:r].clamp(min=0.0).sqrt()
    B_bal[:, :r] = U[:, :r] * s_rt.unsqueeze(0)
    A_bal[:r, :] = s_rt.unsqueeze(1) * Vh[:r, :]
    return B_bal, A_bal


@torch.no_grad()
def fiber_pq_metrics(
    B: torch.Tensor,
    A: torch.Tensor,
    U_ref: torch.Tensor,
    V_ref: torch.Tensor,
    r_star: int,
    gamma: float = 5.0,
) -> dict:
    """
    Compute exact inactive fiber blocks (B_in, A_in) in the orthogonal complement of Task A's
    active subspaces, plus smooth localization-scale effective ranks (p_eff, q_eff) and
    threshold ranks for mapping real optimizer trajectories onto the 2D (p, q) phase diagram.
    """
    U_act, U_perp = U_ref[:, :r_star], U_ref[:, r_star:]
    V_act, V_perp = V_ref[:, :r_star], V_ref[:, r_star:]
    _, _, Vh_act = torch.linalg.svd(U_act.T @ B, full_matrices=True)
    Q_kB = Vh_act[r_star:, :].T  # H x (H - r_star)
    U_actA, _, _ = torch.linalg.svd(A @ V_act, full_matrices=True)
    Q_kAT = U_actA[:, r_star:]   # H x (H - r_star)

    A_in = Q_kB.T @ (A @ V_perp)
    B_in = (U_perp.T @ B) @ Q_kAT
    sv_Ain = torch.linalg.svdvals(A_in).cpu().numpy()
    sv_Bin = torch.linalg.svdvals(B_in).cpu().numpy()
    sv_BA = torch.linalg.svdvals(B @ A).cpu().numpy()

    # Smooth localization-scale effective rank: sum_i s_i^2 / (s_i^2 + 1/gamma)
    p_eff = float(np.sum((sv_Bin ** 2) / (sv_Bin ** 2 + 1.0 / gamma)))
    q_eff = float(np.sum((sv_Ain ** 2) / (sv_Ain ** 2 + 1.0 / gamma)))
    return {
        "norm_B_in": float(np.linalg.norm(sv_Bin)),
        "norm_A_in": float(np.linalg.norm(sv_Ain)),
        "sv_B_in": [round(float(x), 5) for x in sv_Bin],
        "sv_A_in": [round(float(x), 5) for x in sv_Ain],
        "p_eff": p_eff,
        "q_eff": q_eff,
        "p_rank_01": int(np.sum(sv_Bin > 0.1)),
        "q_rank_01": int(np.sum(sv_Ain > 0.1)),
        "erank_BA": entropy_effective_rank(sv_BA),
        "norm_B": float(torch.linalg.matrix_norm(B).item()),
        "norm_A": float(torch.linalg.matrix_norm(A).item()),
    }


# =============================================================================
# 6. PART 1 — Exact Iso-Functional 2D (p, q) Fiber Surgery (+ Pre-LayerNorm Check)
# =============================================================================

def run_part1_2d_fiber_surgery(
    model_rms: CausalTransformerLM,
    model_ln: CausalTransformerLM,
    X_val: torch.Tensor,
    U_A: torch.Tensor,
    s_A: torch.Tensor,
    V_A: torch.Tensor,
) -> dict:
    print("\n" + "=" * 78)
    print("PART 1: Exact Iso-Functional 2D (p, q) Fiber Surgery on Real 2-Layer Transformer")
    print("=" * 78)

    M, N, H, r_star = 12, 12, 10, 2
    beta, gamma, s_in = 1e9, 5.0, 2.0
    X_eval = X_val[:32]
    dW_A = U_A[:, :r_star] @ torch.diag(s_A) @ V_A[:, :r_star].T

    cache_rms = Layer1WOCache(model_rms, X_eval)
    kl_fn_rms, p_ref_A, log_p_ref_A = cache_rms.make_kl_loss_fn(dW_A)

    # First verify converged LoRA checkpoint on Task A before performing surgery
    g_tr = torch.Generator().manual_seed(111)
    B_tr = torch.zeros(M, H, dtype=torch.float64, requires_grad=True)
    A_tr = (0.4 * torch.randn(H, N, generator=g_tr, dtype=torch.float64)).requires_grad_(True)
    opt = torch.optim.Adam([B_tr, A_tr], lr=0.02)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2500, eta_min=1e-5)
    for _ in range(2500):
        opt.zero_grad()
        loss = cache_rms.kl_to_ref_single(B_tr @ A_tr, p_ref_A, log_p_ref_A)
        loss.backward()
        opt.step()
        sched.step()

    with torch.no_grad():
        conv_kl = float(cache_rms.kl_to_ref_single(B_tr @ A_tr, p_ref_A, log_p_ref_A).item())
        conv_dW_err = float(torch.linalg.matrix_norm(B_tr @ A_tr - dW_A).item())
        base_kl = float(cache_rms.kl_to_ref_single(torch.zeros_like(dW_A), p_ref_A, log_p_ref_A).item())
        lg_full_ref = model_rms(X_val[:64], layer1_deltas={"dO": dW_A})
        lg_cached_ref = cache_rms.logits_single(dW_A)
        lg_full_sub = model_rms(X_eval, layer1_deltas={"dO": dW_A})[:, :-1, :].reshape(cache_rms.N_tok, -1)
        cache_err = float((lg_cached_ref - lg_full_sub).abs().max().item())

    print(f"  Task A shift: KL(base || Task A) = {base_kl:.4f} nats")
    print(f"  Converged LoRA checkpoint: KL_A = {conv_kl:.2e}, ||B*A* - dW_A||_F = {conv_dW_err:.2e}")
    print(f"  Bit-exact cached vs full 2-layer Transformer max|Delta logit| = {cache_err:.2e}")

    # Check exact Gauss-Newton Fisher spectrum of Layer 1 W_O on RMSNorm vs LayerNorm
    def fisher_summary(mod: CausalTransformerLM) -> dict:
        c_tmp = Layer1WOCache(mod, X_eval)
        with torch.no_grad():
            lg0 = c_tmp.logits_single(torch.zeros(12, 12, dtype=torch.float64))
            p0, lp0 = F.softmax(lg0, dim=-1), F.log_softmax(lg0, dim=-1)
        dW0 = torch.zeros(12, 12, dtype=torch.float64, requires_grad=True)
        def kf(dw):
            lp = F.log_softmax(c_tmp.logits_single(dw), dim=-1)
            return (p0 * (lp0 - lp)).sum(dim=-1).mean()
        Hm = torch.autograd.functional.hessian(kf, dW0).reshape(144, 144)
        eigs = torch.linalg.eigvalsh(0.5 * (Hm + Hm.T))
        return {
            "n_zero_lt_1e10": int((eigs < 1e-10).sum().item()),
            "min_eig": float(eigs[0].item()),
            "eig_13": float(eigs[12].item()),
            "med_eig": float(eigs[72].item()),
            "max_eig": float(eigs[-1].item()),
        }

    fish_rms = fisher_summary(model_rms)
    fish_ln = fisher_summary(model_ln)
    print(f"  Fisher(W_O) Pre-RMSNorm:  n_zero={fish_rms['n_zero_lt_1e10']:2d}, min={fish_rms['min_eig']:.2e}, med={fish_rms['med_eig']:.2e}, max={fish_rms['max_eig']:.2e}")
    print(f"  Fisher(W_O) Pre-LayerNorm: n_zero={fish_ln['n_zero_lt_1e10']:2d}, min={fish_ln['min_eig']:.2e}, 13th={fish_ln['eig_13']:.2e}, max={fish_ln['max_eig']:.2e}")

    # 2D (p, q) grid points: A-axis (9), B-axis (8), Balanced diagonal (4), Off-axis supercrit (4) = 25 pts
    grid_pts = (
        [(0, q, "A_axis") for q in range(0, 9)]
        + [(p, 0, "B_axis") for p in range(1, 9)]
        + [(k, k, "diag") for k in [1, 2, 3, 4]]
        + [(1, 6, "off_axis"), (2, 6, "off_axis"), (6, 1, "off_axis"), (6, 2, "off_axis")]
    )

    rows = []
    max_logit_err_all = 0.0
    max_mat_err_all = 0.0
    for idx_pt, (p, q, grp) in enumerate(grid_pts):
        B_pq, A_pq = construct_pq_checkpoint(U_A, s_A, V_A, H, p, q, scale=s_in)
        with torch.no_grad():
            mat_err = float((B_pq @ A_pq - dW_A).abs().max().item())
            lg_pq = model_rms(X_val[:64], layer1_deltas={"dO": B_pq @ A_pq})
            logit_err = float((lg_pq - lg_full_ref).abs().max().item())
            max_mat_err_all = max(max_mat_err_all, mat_err)
            max_logit_err_all = max(max_logit_err_all, logit_err)

        th_lam, th_mult = theoretical_rrr_2d_lambda(M, N, H, r_star, p, q)
        t0 = time.time()
        est = mala_full_precond(
            [B_pq, A_pq], kl_fn_rms, beta=beta, gamma=gamma,
            n_chains=20, n_burn=1100, n_samples=1500, seed=1000 + idx_pt,
        )
        print(
            f"  [{grp:8s}] (p={p}, q={q}) | th={th_lam:4.1f} | MALA={est['mean']:5.2f}±{est['se']:.2f} "
            f"(err={est['mean']-th_lam:+.2f}, acc={est['acc']:.2f}, hd={est['hdiff']:+.2f}, "
            f"g_lam={est['gauss_lam']:4.1f}) | dLogit={logit_err:.1e} | {time.time()-t0:.1f}s"
        )
        rows.append({
            "p": p, "q": q, "group": grp,
            "theory_lambda": th_lam, "theory_mult": th_mult,
            "mala_mean": est["mean"], "mala_se": est["se"],
            "acc": est["acc"], "hdiff": est["hdiff"],
            "gauss_lam": est["gauss_lam"], "n_hess_gt_gamma": est["n_hess_gt_gamma"],
            "max_mat_err": mat_err, "max_logit_err": logit_err,
        })

    # Bonus Architectural Verification: Pre-LayerNorm m_eff = d - 1 = 11 breaks p <-> q symmetry!
    # Construct U_ln whose first column is 1_d / sqrt(d) (exact LN kernel), active cols 1..2,
    # and U_perp cols 3..11 (9 active + 1 dead -> put active U_perp in cols 3..11 so p=8 uses live output dims!)
    print("\n  --- Bonus Architectural Check: Pre-LayerNorm m_eff = 11 Symmetry Breaking ---")
    ones = torch.ones(12, 1, dtype=torch.float64) / math.sqrt(12.0)
    g_ln = torch.Generator().manual_seed(77)
    R_ln = torch.randn(12, 12, generator=g_ln, dtype=torch.float64)
    # Orthogonalize R_ln against ones, putting ones as the LAST column (col 11)
    R_perp = R_ln[:, :11] - ones @ (ones.T @ R_ln[:, :11])
    Q_11, _ = torch.linalg.qr(R_perp)  # 12 x 11 orthonormal, all orthogonal to 1_d!
    U_ln = torch.cat([Q_11, ones], dim=1)  # col 0..1 active (live), col 2..10 live U_perp (9 cols), col 11 = 1_d (dead!)
    dW_ln = U_ln[:, :r_star] @ torch.diag(s_A) @ V_A[:, :r_star].T
    cache_ln = Layer1WOCache(model_ln, X_eval)
    kl_fn_ln, _, _ = cache_ln.make_kl_loss_fn(dW_ln)

    ln_rows = []
    # Theory for M_eff=11, N=12, H=10, r*=2 -> excess (m=9, n=10, k=8):
    # (0,0): 51.0 (m_log=2), (0,8): 57.0, (8,0): 61.0 (4.0-pt symmetry breaking!)
    for idx_ln, (p, q) in enumerate([(0, 0), (0, 8), (8, 0)]):
        B_pq, A_pq = construct_pq_checkpoint(U_ln, s_A, V_A, H, p, q, scale=s_in)
        th_lam, th_mult = theoretical_rrr_2d_lambda(11, 12, H, r_star, p, q)
        est = mala_full_precond(
            [B_pq, A_pq], kl_fn_ln, beta=beta, gamma=gamma,
            n_chains=20, n_burn=1100, n_samples=1500, seed=1500 + idx_ln,
        )
        print(
            f"  [Pre-LN m_eff=11] (p={p}, q={q}) | th={th_lam:4.1f} (m_log={th_mult}) | "
            f"MALA={est['mean']:5.2f}±{est['se']:.2f} (acc={est['acc']:.2f}, hd={est['hdiff']:+.2f}, g_lam={est['gauss_lam']:4.1f})"
        )
        ln_rows.append({
            "p": p, "q": q, "theory_lambda": th_lam, "theory_mult": th_mult,
            "mala_mean": est["mean"], "mala_se": est["se"],
            "acc": est["acc"], "hdiff": est["hdiff"], "gauss_lam": est["gauss_lam"],
        })

    return {
        "config": {
            "M": M, "N": N, "H": H, "r_star": r_star, "p_c": 4, "q_c": 4,
            "beta": beta, "gamma": gamma, "s_in": s_in, "S_eval": 32,
            "base_kl_taskA": base_kl, "conv_kl_taskA": conv_kl,
            "conv_dW_err": conv_dW_err, "cache_max_logit_diff": cache_err,
            "max_surgery_mat_err": max_mat_err_all,
            "max_surgery_logit_err": max_logit_err_all,
        },
        "fisher_rmsnorm": fish_rms,
        "fisher_layernorm": fish_ln,
        "grid_2d": rows,
        "pre_layernorm_asymmetry": ln_rows,
    }


# =============================================================================
# 7. PART 2 — Continual Learning / Plasticity-Stability Split from Function-Matched Checkpoints
# =============================================================================

def run_part2_continual_learning(
    model_rms: CausalTransformerLM,
    X_val_A: torch.Tensor,
    U_A: torch.Tensor,
    s_A: torch.Tensor,
    V_A: torch.Tensor,
) -> dict:
    print("\n" + "=" * 78)
    print("PART 2: Continual Learning / Plasticity-Stability Split from Identical Checkpoints")
    print("=" * 78)

    M, N, H, r_star = 12, 12, 10, 2
    dW_A = U_A[:, :r_star] @ torch.diag(s_A) @ V_A[:, :r_star].T
    seeds = [0, 1, 2, 3, 4]
    s_in = 1.5

    # Fixed held-out evaluation caches for Task A and Task B (clean shifted recall shift_rule=3)
    cache_A_val = Layer1WOCache(model_rms, X_val_A[:256])
    with torch.no_grad():
        lg_A_ref = cache_A_val.logits_single(dW_A)
        p_A_ref = F.softmax(lg_A_ref, dim=-1)
        log_p_A_ref = F.log_softmax(lg_A_ref, dim=-1)
        ce_A_init = float(cache_A_val.ce_loss_single(dW_A).item())

    X_B_test = generate_ar_dataset(256, np.random.default_rng(7777), shift_rule=3, corrupt_prob=0.0)
    cache_B_test = Layer1WOCache(model_rms, X_B_test)
    with torch.no_grad():
        ce_B_init = float(cache_B_test.ce_loss_single(dW_A).item())
        lg_B_init = cache_B_test.logits_single(dW_A).view(256, 11, 24)
        acc_B_init = float((lg_B_init[:, 7].argmax(-1) == X_B_test[:, 8]).double().mean().item())
    print(f"  Initial Task-A checkpoint: CE_A={ce_A_init:.4f}, KL_A=0.0000 | Task-B (shift=3): CE_B={ce_B_init:.4f}, r1_acc={acc_B_init:.3f}")

    ckpt_names = [
        ("exact_0_0", "Exact (0,0) Plateau"),
        ("near_0_0", "Near-Plateau (0,0)+1e-3"),
        ("bal_4_4", "Balanced (4,4) Plateau"),
        ("A_sup_0_8", "A-Supercrit (0,8) LoRA"),
        ("B_sup_8_0", "B-Supercrit (8,0) LoRA+"),
    ]

    def build_checkpoint(name: str, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Rotate inactive orthogonal complements per seed so results don't depend on a single basis!
        g = torch.Generator().manual_seed(2000 + seed)
        R_u, _ = torch.linalg.qr(torch.randn(M - r_star, M - r_star, generator=g, dtype=torch.float64))
        R_v, _ = torch.linalg.qr(torch.randn(N - r_star, N - r_star, generator=g, dtype=torch.float64))
        U_rot = torch.cat([U_A[:, :r_star], U_A[:, r_star:] @ R_u], dim=1)
        V_rot = torch.cat([V_A[:, :r_star], V_A[:, r_star:] @ R_v], dim=1)
        if name == "exact_0_0":
            return construct_pq_checkpoint(U_rot, s_A, V_rot, H, 0, 0, scale=s_in)
        elif name == "near_0_0":
            B, A = construct_pq_checkpoint(U_rot, s_A, V_rot, H, 0, 0, scale=s_in)
            B[:, r_star:] = 1e-3 * torch.randn(M, H - r_star, generator=g, dtype=torch.float64)
            A[r_star:, :] = 1e-3 * torch.randn(H - r_star, N, generator=g, dtype=torch.float64)
            return B, A
        elif name == "bal_4_4":
            return construct_pq_checkpoint(U_rot, s_A, V_rot, H, 4, 4, scale=s_in)
        elif name == "A_sup_0_8":
            return construct_pq_checkpoint(U_rot, s_A, V_rot, H, 0, 8, scale=s_in)
        elif name == "B_sup_8_0":
            return construct_pq_checkpoint(U_rot, s_A, V_rot, H, 8, 0, scale=s_in)
        raise ValueError(name)

    # Verify bitwise identity at t=0
    init_checks = {}
    for cname, _ in ckpt_names:
        B0, A0 = build_checkpoint(cname, 0)
        with torch.no_grad():
            dlog = float((cache_A_val.logits_single(B0 @ A0) - lg_A_ref).abs().max().item())
            kla = float(cache_A_val.kl_to_ref_single(B0 @ A0, p_A_ref, log_p_A_ref).item())
        init_checks[cname] = {"max_logit_diff": dlog, "init_kl_A": kla}
    print("  Init bitwise-identity check (max|Delta logit|):", {k: f"{v['max_logit_diff']:.1e}" for k, v in init_checks.items()})

    def eval_taskB_regime(regime: str, n_train: int, corrupt_prob: float, opt_name: str, lr: float, steps: int, target_ce: float) -> dict:
        out = {}
        for cname, clabel in ckpt_names:
            seed_rows = []
            for sd in seeds:
                rng_b = np.random.default_rng(5000 + 100 * sd + (0 if regime == "clean" else 50))
                X_B_tr = generate_ar_dataset(n_train, rng_b, shift_rule=3, corrupt_prob=corrupt_prob)
                cache_B_tr = Layer1WOCache(model_rms, X_B_tr)

                B0, A0 = build_checkpoint(cname, sd)
                B = B0.clone().detach().requires_grad_(True)
                A = A0.clone().detach().requires_grad_(True)
                if opt_name == "sgd":
                    opt = torch.optim.SGD([B, A], lr=lr)
                else:
                    opt = torch.optim.Adam([B, A], lr=lr)

                ce_step20 = None
                steps_to_target = steps + 1
                curve = []
                for step in range(1, steps + 1):
                    opt.zero_grad()
                    loss = cache_B_tr.ce_loss_single(B @ A)
                    loss.backward()
                    opt.step()

                    if step in (5, 10, 20, 50, 100, 200, steps) or steps_to_target > steps:
                        with torch.no_grad():
                            te_ce = float(cache_B_test.ce_loss_single(B @ A).item())
                        if step == 20:
                            ce_step20 = te_ce
                        if steps_to_target > steps and te_ce <= target_ce:
                            steps_to_target = step
                        if step in (5, 10, 20, 50, 100, 200, steps):
                            with torch.no_grad():
                                kla_s = float(cache_A_val.kl_to_ref_single(B @ A, p_A_ref, log_p_A_ref).item())
                            curve.append({"step": step, "train_ce": float(loss.item()), "test_ce": te_ce, "kl_A": kla_s})

                with torch.no_grad():
                    dW_f = B @ A
                    tr_ce_f = float(cache_B_tr.ce_loss_single(dW_f).item())
                    te_ce_f = float(cache_B_test.ce_loss_single(dW_f).item())
                    lg_te = cache_B_test.logits_single(dW_f).view(256, 11, 24)
                    acc_r1_f = float((lg_te[:, 7].argmax(-1) == X_B_test[:, 8]).double().mean().item())
                    kl_A_f = float(cache_A_val.kl_to_ref_single(dW_f, p_A_ref, log_p_A_ref).item())
                    ce_A_f = float(cache_A_val.ce_loss_single(dW_f).item())
                    delta_mat = dW_f - dW_A
                    sv_upd = torch.linalg.svdvals(delta_mat).cpu().numpy()
                    upd_norm = float(np.linalg.norm(sv_upd))
                    upd_erank = entropy_effective_rank(sv_upd)
                    ba_erank = entropy_effective_rank(torch.linalg.svdvals(dW_f).cpu().numpy())

                seed_rows.append({
                    "seed": sd,
                    "ce_step20": ce_step20,
                    "drop_step20": ce_B_init - ce_step20,
                    "steps_to_target": steps_to_target,
                    "final_train_ce": tr_ce_f,
                    "final_test_ce": te_ce_f,
                    "gen_gap": te_ce_f - tr_ce_f,
                    "final_r1_acc": acc_r1_f,
                    "final_kl_A": kl_A_f,
                    "final_ce_A": ce_A_f,
                    "upd_norm": upd_norm,
                    "upd_erank": upd_erank,
                    "ba_erank": ba_erank,
                    "curve": curve,
                })

            def agg(k):
                vals = np.array([r[k] for r in seed_rows], dtype=float)
                return float(np.mean(vals)), float(np.std(vals, ddof=1) / math.sqrt(len(vals)))

            res_c = {"label": clabel, "seeds": seed_rows}
            for k in [
                "ce_step20", "drop_step20", "steps_to_target",
                "final_train_ce", "final_test_ce", "gen_gap",
                "final_r1_acc", "final_kl_A", "final_ce_A",
                "upd_norm", "upd_erank", "ba_erank",
            ]:
                m_v, se_v = agg(k)
                res_c[f"{k}_mean"] = m_v
                res_c[f"{k}_se"] = se_v
            out[cname] = res_c
        return out

    # Regime 2a: Clean real-token Task B (S_train=192, 0% corruption)
    print("\n  --- Regime 2a: Clean Real-Token Task B (shift_rule=3, S=192, 5 seeds) ---")
    clean_sgd = eval_taskB_regime("clean", 192, 0.0, "sgd", lr=0.8, steps=250, target_ce=2.3)
    clean_adam = eval_taskB_regime("clean", 192, 0.0, "adam", lr=0.025, steps=200, target_ce=2.3)
    for opt_lbl, dct in [("SGD ", clean_sgd), ("Adam", clean_adam)]:
        print(f"  [{opt_lbl}]")
        for cn, _ in ckpt_names:
            r = dct[cn]
            print(
                f"    {cn:10s} | CE@20={r['ce_step20_mean']:.3f}±{r['ce_step20_se']:.3f} | "
                f"CE_fin={r['final_test_ce_mean']:.3f}±{r['final_test_ce_se']:.3f} | "
                f"acc={r['final_r1_acc_mean']:.3f} | KL_A={r['final_kl_A_mean']:.3f}±{r['final_kl_A_se']:.3f} | "
                f"erank(dW)={r['upd_erank_mean']:.2f}"
            )

    # Regime 2b: Noisy limited-sample real-token Task B (S_train=32, 25% corruption)
    print("\n  --- Regime 2b: Noisy Limited-Sample Task B (S=32, 25% corruption, 5 seeds) ---")
    noisy_sgd = eval_taskB_regime("noisy", 32, 0.25, "sgd", lr=0.8, steps=250, target_ce=2.6)
    noisy_adam = eval_taskB_regime("noisy", 32, 0.25, "adam", lr=0.025, steps=200, target_ce=2.6)
    for opt_lbl, dct in [("SGD ", noisy_sgd), ("Adam", noisy_adam)]:
        print(f"  [{opt_lbl}]")
        for cn, _ in ckpt_names:
            r = dct[cn]
            print(
                f"    {cn:10s} | trCE={r['final_train_ce_mean']:.3f} | teCE={r['final_test_ce_mean']:.3f}±{r['final_test_ce_se']:.3f} | "
                f"gap={r['gen_gap_mean']:+.3f}±{r['gen_gap_se']:.3f} | KL_A={r['final_kl_A_mean']:.3f} | erank(dW)={r['upd_erank_mean']:.2f}"
            )

    # Regime 2c: Directional Plasticity Probe (V_perp input-aligned vs U_perp output-aligned rank-2 shift)
    # Using exact Task-B KL distillation toward dW_A + dW_dir so directional alignment is 100% controlled:
    #   - V_perp_input:  U_rot[:, 10:12] @ V_rot[:, 2:4]^T (covered by A_in of (0,8) & (4,4), NOT B_in of (8,0)!)
    #   - U_perp_output: U_rot[:, 2:4] @ V_rot[:, 10:12]^T (covered by B_in of (8,0) & (4,4), NOT A_in of (0,8)!)
    print("\n  --- Regime 2c: Directional Plasticity Split (V_perp-aligned vs U_perp-aligned Task B, 5 seeds) ---")
    X_dir = X_val_A[:64]
    cache_dir = Layer1WOCache(model_rms, X_dir)
    dir_results = {}
    for align_mode in ["V_perp_input", "U_perp_output"]:
        dir_results[align_mode] = {}
        for opt_name, lr, n_steps in [("sgd", 0.25, 100), ("adam", 0.02, 100)]:
            dir_results[align_mode][opt_name] = {}
            for cname, clabel in ckpt_names:
                drops_25, fins, klas = [], [], []
                for sd in seeds:
                    g = torch.Generator().manual_seed(2000 + sd)
                    R_u, _ = torch.linalg.qr(torch.randn(M - r_star, M - r_star, generator=g, dtype=torch.float64))
                    R_v, _ = torch.linalg.qr(torch.randn(N - r_star, N - r_star, generator=g, dtype=torch.float64))
                    U_rot = torch.cat([U_A[:, :r_star], U_A[:, r_star:] @ R_u], dim=1)
                    V_rot = torch.cat([V_A[:, :r_star], V_A[:, r_star:] @ R_v], dim=1)

                    if align_mode == "V_perp_input":
                        dW_dir = 0.8 * (U_rot[:, 10:12] @ V_rot[:, 2:4].T)
                    else:
                        dW_dir = 0.8 * (U_rot[:, 2:4] @ V_rot[:, 10:12].T)

                    dW_target = dW_A + dW_dir
                    with torch.no_grad():
                        lg_t = cache_dir.logits_single(dW_target)
                        p_t, lp_t = F.softmax(lg_t, dim=-1), F.log_softmax(lg_t, dim=-1)
                        kl0 = float(cache_dir.kl_to_ref_single(dW_A, p_t, lp_t).item())

                    B0, A0 = build_checkpoint(cname, sd)
                    B = B0.clone().detach().requires_grad_(True)
                    A = A0.clone().detach().requires_grad_(True)
                    opt = torch.optim.SGD([B, A], lr=lr) if opt_name == "sgd" else torch.optim.Adam([B, A], lr=lr)
                    kl25 = None
                    for st in range(1, n_steps + 1):
                        opt.zero_grad()
                        l_kl = cache_dir.kl_to_ref_single(B @ A, p_t, lp_t)
                        l_kl.backward()
                        opt.step()
                        if st == 25:
                            kl25 = float(l_kl.item())
                    with torch.no_grad():
                        kl_f = float(cache_dir.kl_to_ref_single(B @ A, p_t, lp_t).item())
                        kl_a = float(cache_A_val.kl_to_ref_single(B @ A, p_A_ref, log_p_A_ref).item())
                    drops_25.append((kl0 - kl25) / max(kl0, 1e-12))
                    fins.append(kl_f / max(kl0, 1e-12))
                    klas.append(kl_a)
                dir_results[align_mode][opt_name][cname] = {
                    "rel_drop_25_mean": float(np.mean(drops_25)),
                    "rel_drop_25_se": float(np.std(drops_25, ddof=1) / math.sqrt(len(seeds))),
                    "rel_final_kl_mean": float(np.mean(fins)),
                    "rel_final_kl_se": float(np.std(fins, ddof=1) / math.sqrt(len(seeds))),
                    "final_kl_A_mean": float(np.mean(klas)),
                }
        print(f"  [{align_mode}] SGD rel_drop@25:", {cn: f"{dir_results[align_mode]['sgd'][cn]['rel_drop_25_mean']:.3f}" for cn, _ in ckpt_names})
        print(f"  [{align_mode}] Adam rel_drop@25:", {cn: f"{dir_results[align_mode]['adam'][cn]['rel_drop_25_mean']:.3f}" for cn, _ in ckpt_names})

    return {
        "config": {
            "n_seeds": len(seeds), "s_in": s_in,
            "ce_A_init": ce_A_init, "ce_B_init": ce_B_init, "acc_B_init": acc_B_init,
            "init_identity_checks": init_checks,
        },
        "clean_sgd": clean_sgd,
        "clean_adam": clean_adam,
        "noisy_sgd": noisy_sgd,
        "noisy_adam": noisy_adam,
        "directional": dir_results,
    }


# =============================================================================
# 8. PART 3 — Real Optimizer Trajectories on 2D (p, q) Phase Diagram + Delta-lambda Decomposition
# =============================================================================

def run_part3_optimizer_trajectories(
    model_rms: CausalTransformerLM,
    X_val: torch.Tensor,
    U_A: torch.Tensor,
    s_A: torch.Tensor,
    V_A: torch.Tensor,
) -> dict:
    print("\n" + "=" * 78)
    print("PART 3: Real Optimizer Trajectories on 2D (p, q) Phase Diagram + Delta-lambda Decomp")
    print("=" * 78)

    M, N, H, r_star = 12, 12, 10, 2
    beta, gamma = 5e8, 5.0
    dW_A = U_A[:, :r_star] @ torch.diag(s_A) @ V_A[:, :r_star].T
    X_train = X_val[:96]
    X_eval = X_val[:32]
    cache_tr = Layer1WOCache(model_rms, X_train)
    cache_ev = Layer1WOCache(model_rms, X_eval)
    _, p_tr_A, lp_tr_A = cache_tr.make_kl_loss_fn(dW_A)

    # Exact (0,0) and (0,8) references at beta=5e8 for apples-to-apples Part 3 baseline
    kl_ref_A, _, _ = cache_ev.make_kl_loss_fn(dW_A)
    B_ref0, A_ref0 = construct_pq_checkpoint(U_A, s_A, V_A, H, 0, 0, scale=2.0)
    B_ref8, A_ref8 = construct_pq_checkpoint(U_A, s_A, V_A, H, 0, 8, scale=2.0)
    ref_p0 = mala_full_precond([B_ref0, A_ref0], kl_ref_A, beta=beta, gamma=gamma, n_chains=20, n_burn=1200, n_samples=1500, seed=2900, soft_thresh_mult=500.0)
    ref_p8 = mala_full_precond([B_ref8, A_ref8], kl_ref_A, beta=beta, gamma=gamma, n_chains=20, n_burn=1200, n_samples=1500, seed=2908, soft_thresh_mult=500.0)
    print(f"  Part 3 Reference (beta=5e8): (0,0)={ref_p0['mean']:.2f}±{ref_p0['se']:.2f} (hd={ref_p0['hdiff']:+.2f}) | (0,8)={ref_p8['mean']:.2f}±{ref_p8['se']:.2f} (hd={ref_p8['hdiff']:+.2f})")

    # Shared deterministic initializations
    g_in = torch.Generator().manual_seed(321)
    B_lora0 = torch.zeros(M, H, dtype=torch.float64)
    A_lora0 = 0.7 * torch.randn(H, N, generator=g_in, dtype=torch.float64)
    B_bal0 = 2e-3 * torch.randn(M, H, generator=g_in, dtype=torch.float64)
    A_bal0 = 2e-3 * torch.randn(H, N, generator=g_in, dtype=torch.float64)

    # 5 optimizer trajectories + post-convergence AdamW fiber-sliding diagnostic:
    #   - LoRA_Adam:   wd=0 (remains A-supercritical; control for decomposition)
    #   - LoRA_AdamW:  2-phase post-convergence WD (converge to KL~1e-8 at t1=2500, then AdamW wd=0.12
    #                  slides down the fiber to (0,0) while KL stays ~1e-8 -> pure fiber-sliding decomposition!)
    #   - LoRA_SGD_WD: SGD+momentum(0.9) + coupled WD=0.0025 -> 0 polish (converges cleanly onto (0,0)!)
    #   - LoRA_Plus:   Hayou et al. (2024) eta_B = 12 * eta_A, wd=0
    #   - PiSSA_Bal:   Balanced small init + Adam (starts and stays inside the 2D plateau!)
    run_specs = [
        dict(
            name="LoRA_Adam", label="LoRA + Adam (wd=0)",
            opt="adamw", B0=B_lora0, A0=A_lora0,
            lr_B=0.025, lr_A=0.025, wd=0.0, steps=4500, cosine=True,
            wd_start=None, wd_zero=None, mom=0.0,
            ckpts=[1000, 2500, 4500], decomp_pair=(2500, 4500),
        ),
        dict(
            name="LoRA_AdamW", label="LoRA + AdamW (wd=0.12)",
            opt="adamw_2stage", B0=B_lora0, A0=A_lora0,
            lr_B=0.025, lr_A=0.025, wd=0.12, steps=8000, cosine=True,
            wd_start=2500, wd_zero=7500, mom=0.0,
            ckpts=[1000, 2500, 4500, 8000], decomp_pair=(2500, 8000),
        ),
        dict(
            name="LoRA_SGD_WD", label="LoRA + SGD+WD (mom=0.9)",
            opt="sgd", B0=B_lora0, A0=A_lora0,
            lr_B=0.25, lr_A=0.25, wd=0.0025, steps=9000, cosine=False,
            wd_start=1500, wd_zero=7500, mom=0.9,
            ckpts=[1500, 4500, 9000], decomp_pair=(1500, 9000),
        ),
        dict(
            name="LoRA_Plus", label="LoRA+ (eta_B=12*eta_A, wd=0)",
            opt="adamw", B0=B_lora0, A0=A_lora0,
            lr_B=0.036, lr_A=0.003, wd=0.0, steps=4500, cosine=True,
            wd_start=None, wd_zero=None, mom=0.0,
            ckpts=[1000, 2500, 4500], decomp_pair=None,
        ),
        dict(
            name="PiSSA_Bal", label="Balanced Small Init + Adam",
            opt="adamw", B0=B_bal0, A0=A_bal0,
            lr_B=0.025, lr_A=0.025, wd=0.0, steps=4500, cosine=True,
            wd_start=None, wd_zero=None, mom=0.0,
            ckpts=[1000, 2500, 4500], decomp_pair=None,
        ),
    ]

    runs = []
    decompositions = {}

    for r_idx, sp in enumerate(run_specs):
        B = sp["B0"].clone().detach().requires_grad_(True)
        A = sp["A0"].clone().detach().requires_grad_(True)
        init_wd = 0.0 if sp["wd_start"] is not None else sp["wd"]
        pgs = [
            {"params": [B], "lr": sp["lr_B"], "weight_decay": init_wd},
            {"params": [A], "lr": sp["lr_A"], "weight_decay": init_wd},
        ]
        if sp["opt"] == "sgd":
            opt = torch.optim.SGD(pgs, momentum=sp["mom"])
            sched = None
        elif sp["opt"] == "adamw_2stage":
            opt = torch.optim.AdamW(pgs)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sp["wd_start"], eta_min=1e-4)
        else:
            opt = torch.optim.AdamW(pgs)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sp["steps"], eta_min=1e-5)

        ckpt_set = set(sp["ckpts"])
        decomp_steps = set(sp["decomp_pair"]) if sp["decomp_pair"] is not None else set()
        trace = []
        checkpoints = []
        saved_for_decomp = {}

        with torch.no_grad():
            kl0 = float(cache_tr.kl_to_ref_single(B @ A, p_tr_A, lp_tr_A).item())
            m0 = fiber_pq_metrics(B, A, U_A, V_A, r_star, gamma=gamma)
            trace.append({"step": 0, "kl_A": kl0, "p_eff": m0["p_eff"], "q_eff": m0["q_eff"], "norm_B_in": m0["norm_B_in"], "norm_A_in": m0["norm_A_in"]})

        for step in range(1, sp["steps"] + 1):
            if sp["wd_start"] is not None and step == sp["wd_start"] + 1:
                if sp["opt"] == "adamw_2stage":
                    # Stage 2: continue training with AdamW weight decay from the converged checkpoint
                    opt = torch.optim.AdamW([
                        {"params": [B], "lr": 0.015, "weight_decay": sp["wd"]},
                        {"params": [A], "lr": 0.015, "weight_decay": sp["wd"]},
                    ])
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=sp["steps"] - sp["wd_start"], eta_min=1e-5)
                else:
                    for pg in opt.param_groups:
                        pg["weight_decay"] = sp["wd"]
            if sp["wd_zero"] is not None and step == sp["wd_zero"] + 1:
                for pg in opt.param_groups:
                    pg["weight_decay"] = 0.0

            opt.zero_grad()
            loss = cache_tr.kl_to_ref_single(B @ A, p_tr_A, lp_tr_A)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()

            if step % 150 == 0 or step in ckpt_set:
                with torch.no_grad():
                    mf = fiber_pq_metrics(B, A, U_A, V_A, r_star, gamma=gamma)
                trace.append({
                    "step": step, "kl_A": float(loss.item()),
                    "p_eff": mf["p_eff"], "q_eff": mf["q_eff"],
                    "norm_B_in": mf["norm_B_in"], "norm_A_in": mf["norm_A_in"],
                })

            if step in ckpt_set:
                with torch.no_grad():
                    dW_c = (B @ A).detach()
                    dW_err = float(torch.linalg.matrix_norm(dW_c - dW_A).item())
                    mf = fiber_pq_metrics(B, A, U_A, V_A, r_star, gamma=gamma)
                kl_ckpt_fn, _, _ = cache_ev.make_kl_loss_fn(dW_c)
                est = mala_full_precond(
                    [B.detach(), A.detach()], kl_ckpt_fn, beta=beta, gamma=gamma,
                    n_chains=20, n_burn=1300, n_samples=1600, seed=3000 + 100 * r_idx + step,
                    soft_thresh_mult=500.0,
                )
                c_rec = {
                    "step": step, "kl_A": float(loss.item()), "dW_err": dW_err,
                    **mf,
                    "mala_mean": est["mean"], "mala_se": est["se"],
                    "acc": est["acc"], "hdiff": est["hdiff"], "gauss_lam": est["gauss_lam"],
                }
                checkpoints.append(c_rec)
                print(
                    f"  [{sp['name']:11s}] step={step:4d} | KL_A={c_rec['kl_A']:.2e} | "
                    f"(p_eff={mf['p_eff']:.2f}, q_eff={mf['q_eff']:.2f}) | "
                    f"||B_in||={mf['norm_B_in']:.3f}, ||A_in||={mf['norm_A_in']:.3f} | "
                    f"hat_lam={est['mean']:5.2f}±{est['se']:.2f} (acc={est['acc']:.2f}, hd={est['hdiff']:+.2f})"
                )

                if step in decomp_steps:
                    B_bal, A_bal = balanced_svd_surgery(B.detach(), A.detach())
                    with torch.no_grad():
                        dlog_bal = float((cache_ev.logits_single(B_bal @ A_bal) - cache_ev.logits_single(dW_c)).abs().max().item())
                    est_bal = mala_full_precond(
                        [B_bal, A_bal], kl_ckpt_fn, beta=beta, gamma=gamma,
                        n_chains=20, n_burn=1300, n_samples=1600, seed=4000 + 100 * r_idx + step,
                        soft_thresh_mult=500.0,
                    )
                    saved_for_decomp[step] = {
                        "step": step,
                        "kl_A": float(loss.item()),
                        "raw_mean": est["mean"], "raw_se": est["se"],
                        "raw_acc": est["acc"], "raw_hdiff": est["hdiff"],
                        "bal_mean": est_bal["mean"], "bal_se": est_bal["se"],
                        "bal_acc": est_bal["acc"], "bal_hdiff": est_bal["hdiff"],
                        "excess_mean": est["mean"] - est_bal["mean"],
                        "max_logit_diff_bal": dlog_bal,
                    }

        if sp["decomp_pair"] is not None:
            t1, t2 = sp["decomp_pair"]
            d1, d2 = saved_for_decomp[t1], saved_for_decomp[t2]
            d_total = d2["raw_mean"] - d1["raw_mean"]
            d_func = d2["bal_mean"] - d1["bal_mean"]
            d_fiber = d2["excess_mean"] - d1["excess_mean"]
            se_total = math.sqrt(d1["raw_se"] ** 2 + d2["raw_se"] ** 2)
            se_func = math.sqrt(d1["bal_se"] ** 2 + d2["bal_se"] ** 2)
            se_fiber = math.sqrt(d1["raw_se"] ** 2 + d1["bal_se"] ** 2 + d2["raw_se"] ** 2 + d2["bal_se"] ** 2)
            frac_fiber = float(d_fiber / d_total) if abs(d_total) > 0.5 else float("nan")
            decompositions[sp["name"]] = {
                "t1": d1, "t2": d2,
                "delta_total": d_total, "se_total": se_total,
                "delta_func": d_func, "se_func": se_func,
                "delta_fiber": d_fiber, "se_fiber": se_fiber,
                "fiber_fraction": frac_fiber,
            }
            print(
                f"    --> [DECOMP {sp['name']} {t1}->{t2}] "
                f"Total={d_total:+.2f}±{se_total:.2f} = Func({d_func:+.2f}±{se_func:.2f}) "
                f"+ Fiber({d_fiber:+.2f}±{se_fiber:.2f}) | bal_dLogit={d2['max_logit_diff_bal']:.1e}"
            )

        runs.append({
            "name": sp["name"], "label": sp["label"], "opt": sp["opt"],
            "steps": sp["steps"], "lr_B": sp["lr_B"], "lr_A": sp["lr_A"], "wd": sp["wd"],
            "trace": trace, "checkpoints": checkpoints, "final": checkpoints[-1],
        })

    return {
        "config": {"M": M, "N": N, "H": H, "r_star": r_star, "beta": beta, "gamma": gamma, "p_c": 4, "q_c": 4},
        "ref_p0": ref_p0,
        "ref_p8": ref_p8,
        "runs": runs,
        "decompositions": decompositions,
    }


# =============================================================================
# 9. PART 4 — Multi-Head Attention V+O vs Q+K Coupling Dichotomy inside Real Transformer
# =============================================================================

def run_part4_multihead_coupling(
    models_by_head: dict[int, CausalTransformerLM],
    X_val: torch.Tensor,
) -> dict:
    print("\n" + "=" * 78)
    print("PART 4: Multi-Head Attention V+O vs Q+K Dichotomy in Real 2-Layer Causal Transformer")
    print("=" * 78)

    d, H = 12, 4
    s_in = 4.5
    gamma = 60.0
    beta_vo = 1e8
    beta_qk = 8e8
    X_eval = X_val[:32]

    Q_orth, _ = torch.linalg.qr(torch.randn(d, d, generator=torch.Generator().manual_seed(99), dtype=torch.float64))
    B0 = torch.zeros(d, H, dtype=torch.float64)

    # Exact theoretical predictions for d=12, H=4:
    # - n_heads=1: V+O: 40.0 + 2*(q-2)_+^2 (q=0: 40, q=2: 40, q=4: 48, jump +8)
    #              Q+K: flat 40.0 (GL(q) cancellation, jump 0!)
    # - n_heads=2, 4 (d_h + q <= 12): decoupled 2 * (22 + 0.5*(q-2)_+^2) = 44 + (q-2)_+^2 (q=0: 44, q=2: 44, q=4: 48, jump +4 BOTH!)
    results = {}
    for nh in [1, 2, 4]:
        m_h = models_by_head[nh]
        WV_d = m_h.blocks[1].attn.W_V.detach()
        Q_ao, _ = torch.linalg.qr((Q_orth[H : 2 * H] @ torch.linalg.pinv(WV_d)).T)
        AO_basis = Q_ao.T
        kl_vo = make_batched_layer1_attn_kl(m_h, X_eval, "VO")
        kl_qk = make_batched_layer1_attn_kl(m_h, X_eval, "QK")

        rows_vo, rows_qk = [], []
        print(f"\n  --- n_heads = {nh} (d_head = {d // nh}) ---")
        for q in [0, 2, 4]:
            AV = torch.zeros(H, d, dtype=torch.float64)
            AO = torch.zeros(H, d, dtype=torch.float64)
            AQ = torch.zeros(H, d, dtype=torch.float64)
            AK = torch.zeros(H, d, dtype=torch.float64)
            if q > 0:
                AV[:q] = s_in * Q_orth[:q]
                AO[:q] = s_in * AO_basis[:q]
                AQ[:q] = s_in * Q_orth[:q]
                AK[:q] = s_in * Q_orth[H : H + q]

            th_vo = (40.0 + 2.0 * max(q - 2, 0) ** 2) if nh == 1 else (44.0 + float(max(q - 2, 0) ** 2))
            th_qk = 40.0 if nh == 1 else (44.0 + float(max(q - 2, 0) ** 2))

            t0 = time.time()
            e_vo = mala_full_precond(
                [B0, AV, B0, AO], kl_vo, beta=beta_vo, gamma=gamma,
                n_chains=20, n_burn=1100, n_samples=1500, seed=6000 + 20 * nh + q,
            )
            e_qk = mala_full_precond(
                [B0, AQ, B0, AK], kl_qk, beta=beta_qk, gamma=gamma,
                n_chains=20, n_burn=1100, n_samples=1500, seed=7000 + 20 * nh + q,
            )
            print(
                f"    q={q} | VO: th={th_vo:4.1f}, MALA={e_vo['mean']:5.2f}±{e_vo['se']:.2f} "
                f"(g={e_vo['gauss_lam']:4.1f}, #H={e_vo['n_hess_gt_gamma']:2d}, acc={e_vo['acc']:.2f}, hd={e_vo['hdiff']:+.2f}) | "
                f"QK: th={th_qk:4.1f}, MALA={e_qk['mean']:5.2f}±{e_qk['se']:.2f} "
                f"(g={e_qk['gauss_lam']:4.1f}, #H={e_qk['n_hess_gt_gamma']:2d}, acc={e_qk['acc']:.2f}, hd={e_qk['hdiff']:+.2f}) | {time.time()-t0:.1f}s"
            )
            rows_vo.append({
                "q": q, "theory_lambda": th_vo,
                "mala_mean": e_vo["mean"], "mala_se": e_vo["se"],
                "acc": e_vo["acc"], "hdiff": e_vo["hdiff"],
                "gauss_lam": e_vo["gauss_lam"], "n_hess_gt_gamma": e_vo["n_hess_gt_gamma"],
            })
            rows_qk.append({
                "q": q, "theory_lambda": th_qk,
                "mala_mean": e_qk["mean"], "mala_se": e_qk["se"],
                "acc": e_qk["acc"], "hdiff": e_qk["hdiff"],
                "gauss_lam": e_qk["gauss_lam"], "n_hess_gt_gamma": e_qk["n_hess_gt_gamma"],
            })
        results[str(nh)] = {"n_heads": nh, "d_head": d // nh, "VO": rows_vo, "QK": rows_qk}

    return {
        "config": {"d": d, "H": H, "s_in": s_in, "gamma": gamma, "beta_vo": beta_vo, "beta_qk": beta_qk},
        "by_heads": results,
    }


# =============================================================================
# 10. Publication-Grade 2x3 Multi-Panel Figure
# =============================================================================

def make_figure(res: dict, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(18.2, 10.6), dpi=170)
    ax1, ax2, ax3 = axes[0, 0], axes[0, 1], axes[0, 2]
    ax4, ax5, ax6 = axes[1, 0], axes[1, 1], axes[1, 2]

    # --- Panel (a): Part 1 — 2D (p, q) Fiber Surgery on Real 2-Layer Causal Transformer ---
    p1 = res["part1_2d_surgery"]
    g2d = p1["grid_2d"]
    x_dense = np.linspace(0, 8, 200)
    ax1.plot(x_dense, 54.0 + 0.5 * np.maximum(x_dense - 4.0, 0.0) ** 2, "-", lw=2.2, color="#1f4e79", label=r"Lean [LC] RRR theory: $54 + \frac{1}{2}(r-4)_+^2$")
    ax1.axvline(4.0, ls=":", color="#1f4e79", alpha=0.55, lw=1.1)

    a_rows = [r for r in g2d if r["group"] == "A_axis"]
    b_rows = [r for r in g2d if r["group"] == "B_axis"]
    d_rows = [r for r in g2d if r["group"] == "diag"]
    o_rows = [r for r in g2d if r["group"] == "off_axis"]

    ax1.errorbar([r["q"] - 0.10 for r in a_rows], [r["mala_mean"] for r in a_rows], yerr=[2 * r["mala_se"] for r in a_rows],
                 fmt="o", color="#c0392b", capsize=2.5, markersize=5.2, label=r"$A$-axis $(0, q)$ (LoRA side)")
    ax1.errorbar([r["p"] + 0.10 for r in b_rows], [r["mala_mean"] for r in b_rows], yerr=[2 * r["mala_se"] for r in b_rows],
                 fmt="s", color="#2980b9", capsize=2.5, markersize=5.0, label=r"$B$-axis $(p, 0)$ (LoRA+ side)")
    ax1.errorbar([r["p"] for r in d_rows], [r["mala_mean"] for r in d_rows], yerr=[2 * r["mala_se"] for r in d_rows],
                 fmt="D", color="#27ae60", capsize=2.5, markersize=5.5, label=r"Balanced diag $(k, k)$ (2D plateau)")
    ax1.errorbar([max(r["p"], r["q"]) + (0.22 if r["q"] > r["p"] else -0.22) for r in o_rows],
                 [r["mala_mean"] for r in o_rows], yerr=[2 * r["mala_se"] for r in o_rows],
                 fmt="^", color="#8e44ad", capsize=2.5, markersize=5.2, label=r"Off-axis $(1,6),(2,6),(6,1),(6,2)$")

    # Pre-LayerNorm asymmetry markers at r=8
    ln_rows = p1["pre_layernorm_asymmetry"]
    ln_08 = [r for r in ln_rows if r["p"] == 0 and r["q"] == 8][0]
    ln_80 = [r for r in ln_rows if r["p"] == 8 and r["q"] == 0][0]
    ax1.errorbar([7.78], [ln_08["mala_mean"]], yerr=[2 * ln_08["mala_se"]], fmt="v", color="#d35400", capsize=2.5, markersize=6,
                 label=rf"Pre-LN $(0,8)$={ln_08['mala_mean']:.1f} (th 57)")
    ax1.errorbar([8.22], [ln_80["mala_mean"]], yerr=[2 * ln_80["mala_se"]], fmt="P", color="#7f8c8d", capsize=2.5, markersize=6,
                 label=rf"Pre-LN $(8,0)$={ln_80['mala_mean']:.1f} (th 61)")

    ax1.set_title(r"(a) Exact Iso-Functional 2D $(p,q)$ Surgery (Real 2L Transformer)")
    ax1.set_xlabel(r"Inactive fiber rank $\max(p, q)$ (all $\max|\Delta\mathrm{logit}| < 10^{-13}$)")
    ax1.set_ylabel(r"Functional KL LLC $\hat\lambda$ ($\pm 2\,$SE)")
    ax1.set_xlim(-0.5, 8.6)
    ax1.set_ylim(48.5, 64.2)
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=6.7, loc="upper left", ncol=1)

    # --- Panel (b): Part 2a — Clean Task B Plasticity-Stability Tradeoff (SGD & Adam) ---
    p2 = res["part2_continual"]
    ck_order = ["exact_0_0", "near_0_0", "bal_4_4", "A_sup_0_8", "B_sup_8_0"]
    ck_short = ["Exact\n(0,0)", "Near\n(0,0)+1e-3", "Bal\n(4,4)", "A-Sup\n(0,8)", "B-Sup\n(8,0)"]
    x_ck = np.arange(len(ck_order))

    sgd_ce = [p2["clean_sgd"][k]["final_test_ce_mean"] for k in ck_order]
    sgd_ce_se = [p2["clean_sgd"][k]["final_test_ce_se"] for k in ck_order]
    adam_ce = [p2["clean_adam"][k]["final_test_ce_mean"] for k in ck_order]
    adam_ce_se = [p2["clean_adam"][k]["final_test_ce_se"] for k in ck_order]
    sgd_kla = [p2["clean_sgd"][k]["final_kl_A_mean"] for k in ck_order]
    adam_kla = [p2["clean_adam"][k]["final_kl_A_mean"] for k in ck_order]

    ax2.bar(x_ck - 0.18, sgd_ce, width=0.34, yerr=sgd_ce_se, capsize=2.5, color="#2980b9", alpha=0.88, label="Task-B Test CE (SGD)")
    ax2.bar(x_ck + 0.18, adam_ce, width=0.34, yerr=adam_ce_se, capsize=2.5, color="#e67e22", alpha=0.88, label="Task-B Test CE (Adam)")
    ax2.set_ylabel("Task-B Test CE (lower = better)", color="#2c3e50")
    ax2.set_ylim(1.45, 1.88)
    ax2.set_xticks(x_ck)
    ax2.set_xticklabels(ck_short, fontsize=7.8)
    ax2.grid(alpha=0.2, axis="y")
    ax2_r = ax2.twinx()
    ax2_r.plot(x_ck - 0.18, sgd_kla, "o--", color="#1a5276", lw=1.7, markersize=5.2, label=r"Task-A Forgetting $\mathrm{KL}_A$ (SGD)")
    ax2_r.plot(x_ck + 0.18, adam_kla, "s-", color="#c0392b", lw=1.7, markersize=5.2, label=r"Task-A Forgetting $\mathrm{KL}_A$ (Adam)")
    ax2_r.set_ylabel(r"Task-A Forgetting $\mathrm{KL}_A$ (nats)", color="#c0392b")
    ax2_r.set_ylim(0.32, 0.65)
    ax2_r.tick_params(axis="y", labelcolor="#c0392b")
    ax2.set_title(r"(b) Continual Learning from Function-Matched Checkpoints (5 seeds)")
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2_r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, fontsize=6.5, loc="upper left", ncol=2)

    # --- Panel (c): Part 2b & 2c — Noisy Task B Overfitting Gap + Directional Split ---
    ng_sgd = [p2["noisy_sgd"][k]["gen_gap_mean"] for k in ck_order]
    ng_sgd_se = [p2["noisy_sgd"][k]["gen_gap_se"] for k in ck_order]
    ng_adam = [p2["noisy_adam"][k]["gen_gap_mean"] for k in ck_order]
    ng_adam_se = [p2["noisy_adam"][k]["gen_gap_se"] for k in ck_order]
    dir_v = [p2["directional"]["V_perp_input"]["sgd"][k]["rel_drop_25_mean"] for k in ck_order]
    dir_u = [p2["directional"]["U_perp_output"]["sgd"][k]["rel_drop_25_mean"] for k in ck_order]

    ax3.bar(x_ck - 0.18, ng_sgd, width=0.34, yerr=ng_sgd_se, capsize=2.5, color="#8e44ad", alpha=0.82, label="Noisy Task-B Overfit Gap (SGD)")
    ax3.bar(x_ck + 0.18, ng_adam, width=0.34, yerr=ng_adam_se, capsize=2.5, color="#c0392b", alpha=0.82, label="Noisy Task-B Overfit Gap (Adam)")
    ax3.set_ylabel("Noisy Task-B Test $-$ Train CE Gap", color="#2c3e50")
    ax3.set_ylim(0.0, 0.56)
    ax3.set_xticks(x_ck)
    ax3.set_xticklabels(ck_short, fontsize=7.8)
    ax3.grid(alpha=0.2, axis="y")

    ax3_r = ax3.twinx()
    ax3_r.plot(x_ck, dir_v, "^-", color="#16a085", lw=1.9, markersize=6.2, label=r"SGD $V_\perp$-Input Drop@25")
    ax3_r.plot(x_ck, dir_u, "v--", color="#d35400", lw=1.9, markersize=6.2, label=r"SGD $U_\perp$-Output Drop@25")
    ax3_r.set_ylabel("Directional Rel. KL Drop @ step 25", color="#16a085")
    ax3_r.set_ylim(0.35, 0.98)
    ax3.set_title(r"(c) Noisy Generalization Gap & Mirror-Image Directional Split")
    l1, lb1 = ax3.get_legend_handles_labels()
    l2, lb2 = ax3_r.get_legend_handles_labels()
    ax3.legend(l1 + l2, lb1 + lb2, fontsize=6.5, loc="upper left")

    # --- Panel (d): Part 3a — Real Optimizer Trajectories on the 2D (p_eff, q_eff) Phase Diagram ---
    p3 = res["part3_trajectories"]
    # Draw theoretical 2D phase diagram regions: simplex p+q <= 8, plateau p<=4 & q<=4
    ax4.fill([0, 4, 4, 0], [0, 0, 4, 4], color="#d6eaf8", alpha=0.55, label=r"Subcritical 2D plateau ($\lambda=54$)")
    ax4.fill([0, 4, 0], [4, 4, 8], color="#fadbd8", alpha=0.50, label=r"$A$-supercritical ($q>4$)")
    ax4.fill([4, 8, 4], [0, 0, 4], color="#e8daef", alpha=0.50, label=r"$B$-supercritical ($p>4$)")
    ax4.plot([0, 8], [8, 0], "k--", lw=1.0, alpha=0.45)
    ax4.plot([0, 4], [4, 4], "-", color="#1f4e79", lw=1.2, alpha=0.7)
    ax4.plot([4, 4], [0, 4], "-", color="#1f4e79", lw=1.2, alpha=0.7)

    opt_styles = {
        "LoRA_Adam": ("#c0392b", "o"),
        "LoRA_AdamW": ("#2980b9", "s"),
        "LoRA_SGD_WD": ("#27ae60", "P"),
        "LoRA_Plus": ("#8e44ad", "^"),
        "PiSSA_Bal": ("#e67e22", "D"),
    }
    for run in p3["runs"]:
        col, mk = opt_styles[run["name"]]
        tr = run["trace"]
        ps_tr = [t["p_eff"] for t in tr]
        qs_tr = [t["q_eff"] for t in tr]
        ax4.plot(ps_tr, qs_tr, "-", color=col, lw=1.8, alpha=0.85, label=f"{run['label']} ($\\hat\\lambda$={run['final']['mala_mean']:.1f})")
        ax4.scatter([ps_tr[0]], [qs_tr[0]], marker="x", color=col, s=38, zorder=5)
        ax4.scatter([ps_tr[-1]], [qs_tr[-1]], marker=mk, color=col, s=55, edgecolor="k", lw=0.6, zorder=6)

    ax4.set_title(r"(d) Optimizer Trajectories on 2D $(p_{\mathrm{eff}}, q_{\mathrm{eff}})$ Phase Diagram")
    ax4.set_xlabel(r"Effective $B$-inactive rank $p_{\mathrm{eff}}$")
    ax4.set_ylabel(r"Effective $A$-inactive rank $q_{\mathrm{eff}}$")
    ax4.set_xlim(-0.3, 8.3)
    ax4.set_ylim(-0.3, 8.3)
    ax4.grid(alpha=0.25)
    ax4.legend(fontsize=6.5, loc="upper right")

    # --- Panel (e): Part 3b — Functional vs. Iso-Functional Delta-lambda Decomposition ---
    decomps = p3["decompositions"]
    d_names = ["LoRA_AdamW", "LoRA_SGD_WD", "LoRA_Adam"]
    d_labels = ["LoRA + AdamW\n(2500→8000)", "LoRA + SGD+WD\n(1500→9000)", "LoRA + Adam (ctrl)\n(2500→4500)"]
    xd = np.arange(len(d_names))
    w_d = 0.25

    d_tot = [decomps[n]["delta_total"] for n in d_names]
    d_tot_se = [decomps[n]["se_total"] for n in d_names]
    d_fun = [decomps[n]["delta_func"] for n in d_names]
    d_fun_se = [decomps[n]["se_func"] for n in d_names]
    d_fib = [decomps[n]["delta_fiber"] for n in d_names]
    d_fib_se = [decomps[n]["se_fiber"] for n in d_names]

    ax5.axhline(0.0, color="k", lw=0.9, alpha=0.6)
    ax5.bar(xd - w_d, d_tot, width=w_d, yerr=d_tot_se, capsize=2.5, color="#34495e", alpha=0.88, label=r"Total $\Delta\hat\lambda_{\mathrm{total}}$")
    ax5.bar(xd, d_fun, width=w_d, yerr=d_fun_se, capsize=2.5, color="#27ae60", alpha=0.88, label=r"Functional $\Delta\hat\lambda_{\mathrm{func}}$")
    ax5.bar(xd + w_d, d_fib, width=w_d, yerr=d_fib_se, capsize=2.5, color="#c0392b", alpha=0.88, label=r"Iso-Functional Fiber $\Delta\hat\lambda_{\mathrm{fiber}}$")
    ax5.set_xticks(xd)
    ax5.set_xticklabels(d_labels, fontsize=8.0)
    ax5.set_ylabel(r"Late-stage LLC change $\Delta\hat\lambda$ (converged $\mathrm{KL}_A < 10^{-5}$)")
    ax5.set_title(r"(e) Late-Stage $\Delta\hat\lambda$: Functional vs. Pure Fiber Sliding")
    ax5.grid(alpha=0.25, axis="y")
    ax5.legend(fontsize=7.3, loc="lower right")

    # --- Panel (f): Part 4 — Multi-Head Attention V+O vs Q+K Dichotomy in Real 2L Transformer ---
    p4 = res["part4_multihead"]["by_heads"]
    nhs = [1, 2, 4]
    x_h = np.arange(len(nhs))
    w_h = 0.19

    vo_q0 = [p4[str(h)]["VO"][0]["mala_mean"] for h in nhs]
    vo_q0_se = [2 * p4[str(h)]["VO"][0]["mala_se"] for h in nhs]
    vo_q4 = [p4[str(h)]["VO"][2]["mala_mean"] for h in nhs]
    vo_q4_se = [2 * p4[str(h)]["VO"][2]["mala_se"] for h in nhs]
    qk_q0 = [p4[str(h)]["QK"][0]["mala_mean"] for h in nhs]
    qk_q0_se = [2 * p4[str(h)]["QK"][0]["mala_se"] for h in nhs]
    qk_q4 = [p4[str(h)]["QK"][2]["mala_mean"] for h in nhs]
    qk_q4_se = [2 * p4[str(h)]["QK"][2]["mala_se"] for h in nhs]

    ax6.bar(x_h - 1.5 * w_h, vo_q0, width=w_h, yerr=vo_q0_se, capsize=2, color="#f1948a", label=r"$V{+}O$ $q=0$ (plateau)")
    ax6.bar(x_h - 0.5 * w_h, vo_q4, width=w_h, yerr=vo_q4_se, capsize=2, color="#c0392b", label=r"$V{+}O$ $q=4$ (LoRA top)")
    ax6.bar(x_h + 0.5 * w_h, qk_q0, width=w_h, yerr=qk_q0_se, capsize=2, color="#82e0aa", label=r"$Q{+}K$ $q=0$ (plateau)")
    ax6.bar(x_h + 1.5 * w_h, qk_q4, width=w_h, yerr=qk_q4_se, capsize=2, color="#1e8449", label=r"$Q{+}K$ $q=4$ (LoRA top)")
    ax6.axhline(48.0, ls="--", color="#c0392b", alpha=0.55, lw=1.1, label=r"Supercrit top theory ($48$)")
    ax6.axhline(40.0, ls=":", color="#1e8449", alpha=0.75, lw=1.2, label=r"Single-head $GL(4)$ ceiling ($40$)")
    ax6.set_xticks(x_h)
    ax6.set_xticklabels([f"{h}-Head\n($d_h={12//h}$)" for h in nhs], fontsize=8.3)
    ax6.set_ylim(22.0, 53.0)
    ax6.set_ylabel(r"Functional KL LLC $\hat\lambda$ ($\pm 2\,$SE)")
    ax6.set_title(r"(f) Real Transformer Attention: $V{+}O$ vs. $Q{+}K$ ($d=12, H=4$)")
    ax6.grid(alpha=0.25, axis="y")
    ax6.legend(fontsize=6.6, loc="lower right", ncol=2)

    fig.suptitle(
        "Track 2 Flagship: Exact LoRA Fiber Surgery, Plasticity-Stability Split, Optimizer Phase Diagram & Multi-Head Coupling on a Real 2-Layer Causal Transformer",
        fontsize=12.6, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.965])
    plt.savefig(out_path, dpi=170)
    plt.close(fig)


# =============================================================================
# 11. Main Entrypoint
# =============================================================================

def main():
    t_start = time.time()
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    if "--replot" in sys.argv[1:]:
        # Redraw the figure from the saved JSON (no pretraining / MALA rerun).
        saved = json.loads((out_dir / "transformer_fiber_surgery.json").read_text())
        make_figure(saved, out_dir / "transformer_fiber_surgery.png")
        return

    print("=" * 78)
    print("PRETRAINING REAL 2-LAYER CAUSAL TRANSFORMERS (3-Pair AR + Composition)")
    print("=" * 78)
    model_rms_2h, info_rms_2h = pretrain_transformer(n_heads=2, norm_type="rmsnorm", steps=3000, seed=42)
    model_ln_2h, info_ln_2h = pretrain_transformer(n_heads=2, norm_type="layernorm", steps=3000, seed=42)
    model_rms_1h, info_rms_1h = pretrain_transformer(n_heads=1, norm_type="rmsnorm", steps=3000, seed=42)
    model_rms_4h, info_rms_4h = pretrain_transformer(n_heads=4, norm_type="rmsnorm", steps=3000, seed=42)

    X_val = generate_ar_dataset(512, np.random.default_rng(999))

    # Fixed rank-2 Task A target shift on Layer 1 W_O
    g_taskA = torch.Generator().manual_seed(7)
    U_A, _ = torch.linalg.qr(torch.randn(12, 12, generator=g_taskA, dtype=torch.float64))
    V_A, _ = torch.linalg.qr(torch.randn(12, 12, generator=g_taskA, dtype=torch.float64))
    s_A = torch.tensor([0.8, 0.6], dtype=torch.float64)

    part1 = run_part1_2d_fiber_surgery(model_rms_2h, model_ln_2h, X_val, U_A, s_A, V_A)
    part2 = run_part2_continual_learning(model_rms_2h, X_val, U_A, s_A, V_A)
    part3 = run_part3_optimizer_trajectories(model_rms_2h, X_val, U_A, s_A, V_A)
    part4 = run_part4_multihead_coupling({1: model_rms_1h, 2: model_rms_2h, 4: model_rms_4h}, X_val)

    all_res = {
        "pretrain_models": {
            "rmsnorm_2h": info_rms_2h,
            "layernorm_2h": info_ln_2h,
            "rmsnorm_1h": info_rms_1h,
            "rmsnorm_4h": info_rms_4h,
        },
        "part1_2d_surgery": part1,
        "part2_continual": part2,
        "part3_trajectories": part3,
        "part4_multihead": part4,
        "elapsed_seconds": round(time.time() - t_start, 2),
    }

    json_path = out_dir / "transformer_fiber_surgery.json"
    json_path.write_text(json.dumps(all_res, indent=2))
    fig_path = out_dir / "transformer_fiber_surgery.png"
    make_figure(all_res, fig_path)

    print("\n" + "=" * 78)
    print(f"ALL 4 PARTS COMPLETED IN {all_res['elapsed_seconds']:.1f}s")
    print(f"Saved JSON: {json_path}")
    print(f"Saved PNG:  {fig_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
