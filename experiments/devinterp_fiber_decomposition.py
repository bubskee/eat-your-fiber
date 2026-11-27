"""
Track A Empirical Study: "Is that LLC drop development, or is it fiber sliding?"

Applies exact full-model iso-functional canonicalization (balancing all exact continuous
symmetries of the Transformer in float64 to max|Delta output| ~ 1e-13) to Hoogland, Wang,
Farrugia-Roberts, Carroll, Wei & Murfet (2024, TMLR, arXiv:2402.02364),
"Loss Landscape Degeneracy and Stagewise Development in Transformers" — specifically their
51,717-parameter in-context linear regression Transformer (`L2H4Minf` from
`github.com/timaeus-research/icl`).

Parts:
  Part 1: Multi-seed (3 seeds) exact `L2H4Minf` reproduction (`Adam, wd=0.0`, OneCycleLR)
          with stagewise Delta hat_lambda_total = Delta hat_lambda_func + Delta hat_lambda_fiber
          decomposition across LR1-LR5, plus behavioral & structural diagnostics (ID MSE,
          E[y_hat^2], ICL score, OOD g=3,5 magnitude fragility, all 5 LayerNorm stats &
          effective-weight norms, float64 max|Delta y| at every checkpoint).
  Part 2: Weight-decay counterfactual (`AdamW, wd=0.08`, 3 seeds) testing whether bounding
          weight norms prevents anti-correlated LN <-> Linear fiber sliding.
  Part 3: Exact per-channel symmetry ablation at LR2 peak and LR4 trough (seed 0):
          `LN_mean_proj`, `QK`, `VO`, `MLP`, `LNf_unembed`, `LN1_mlp`, `LN0_attn`,
          `LN_all_unit_gamma`, and joint norm-minimizing `FULL`.
  Part 4: SGLD hyperparameter, 5,000-step trace, and full-batch sensitivity suite at LR2 peak
          and LR4 trough (seed 0): gamma in {3.3, 13.3, 40.0}, eps in {1e-4, 3e-4},
          m_batch in {256, 1024}, and zero-noise full-batch (`FB = 1024`) SGLD.

Usage:
  python devinterp_fiber_decomposition.py            # full multi-seed run + plot
  python devinterp_fiber_decomposition.py --replot   # redraw figure from saved JSON only

Outputs (relative to this file):
  - results/devinterp_fiber_decomposition.json
  - results/devinterp_fiber_decomposition.png
"""

from __future__ import annotations

import copy
import json
import math
import multiprocessing as mp
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


RESULTS_DIR = Path(__file__).resolve().parent / "results"
JSON_PATH = RESULTS_DIR / "devinterp_fiber_decomposition.json"
PNG_PATH = RESULTS_DIR / "devinterp_fiber_decomposition.png"


# =============================================================================
# 1. Exact Replica of Hoogland et al. (2024) `L2H4Minf` Architecture (51,717 params)
#    Matches `timaeus-research/icl/src/icl/regression/{dtransformer,model}.py`
#    to float64 machine precision (max|Delta y| < 1e-15).
# =============================================================================

def to_token_sequence(xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    B, K, D = xs.shape
    xys = torch.cat(
        [
            torch.zeros_like(ys),
            xs,
            ys,
            torch.zeros_like(xs),
        ],
        dim=-1,
    )
    return xys.reshape(B, 2 * K, D + 1)


def from_predicted_token_sequence(toks: torch.Tensor) -> torch.Tensor:
    return toks[:, 0::2, :1]


class MultiHeadedCausalSelfAttention(nn.Module):
    def __init__(self, embed_size: int, max_tokens: int, num_heads: int, include_output: bool = True):
        super().__init__()
        assert embed_size % num_heads == 0
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.attention = nn.Linear(embed_size, 3 * embed_size, bias=False)
        self.attention_softmax = nn.Softmax(dim=-1)
        causal_mask = torch.log(torch.tril(torch.ones((max_tokens, max_tokens))))
        self.register_buffer("causal_mask", causal_mask)
        self.attention_scale = self.head_size ** 0.5
        self.include_output = include_output
        if include_output:
            self.output = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        H, c = self.num_heads, self.head_size
        Q, K, V = (
            self.attention(x)
            .view(B, T, H, 3 * c)
            .transpose(-2, -3)
            .split(c, dim=-1)
        )
        A = (Q @ K.transpose(-2, -1)) / self.attention_scale + self.causal_mask[:T, :T]
        p = self.attention_softmax(A)
        y = (p @ V).transpose(-3, -2).contiguous().view(B, T, C)
        if self.include_output:
            y = self.output(y)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int, mlp_size: int, max_tokens: int, num_heads: int, layer_norm: bool = True, include_output: bool = True):
        super().__init__()
        self.attention = MultiHeadedCausalSelfAttention(embed_size, max_tokens, num_heads, include_output=include_output)
        self.compute = nn.Sequential(
            nn.Linear(embed_size, mlp_size),
            nn.ReLU(),
            nn.Linear(mlp_size, embed_size),
        )
        if layer_norm:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_size) for _ in range(2)])
        else:
            self.layer_norms = nn.ModuleList([nn.Identity() for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.layer_norms[0](x))
        x = x + self.compute(self.layer_norms[1](x))
        return x


class DTransformer(nn.Module):
    def __init__(self, token_size: int, max_tokens: int, embed_size: int, mlp_size: int, num_heads: int, num_layers: int, layer_norm: bool = True, include_output: bool = True):
        super().__init__()
        self.token_embedding = nn.Linear(token_size, embed_size, bias=False)
        self.postn_embedding = nn.Linear(max_tokens, embed_size, bias=False)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_size, mlp_size, max_tokens, num_heads, layer_norm=layer_norm, include_output=include_output)
            for _ in range(num_layers)
        ])
        if layer_norm:
            self.unembedding = nn.Sequential(nn.LayerNorm(embed_size), nn.Linear(embed_size, token_size))
        else:
            self.unembedding = nn.Linear(embed_size, token_size)
        self.max_tokens = max_tokens

    def forward(self, toks: torch.Tensor) -> torch.Tensor:
        _, T, _ = toks.shape
        x = self.token_embedding(toks) + self.postn_embedding.weight.T[:T, :]
        for block in self.blocks:
            # Exact match to `timaeus-research/icl` DTransformer.forward line 103
            x = x + block(x)
        return self.unembedding(x)


class InContextRegressionTransformer(nn.Module):
    def __init__(
        self,
        task_size: int = 4,
        max_examples: int = 8,
        embed_size: int = 64,
        mlp_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        layer_norm: bool = True,
        include_output: bool = True,
    ):
        super().__init__()
        self.token_sequence_transformer = DTransformer(
            token_size=1 + task_size,
            max_tokens=2 * max_examples,
            embed_size=embed_size,
            mlp_size=mlp_size,
            num_heads=num_heads,
            num_layers=num_layers,
            layer_norm=layer_norm,
            include_output=include_output,
        )
        self.task_size = task_size
        self.max_examples = max_examples
        self.embed_size = embed_size
        self.mlp_size = mlp_size
        self.num_heads = num_heads
        self.num_layers = num_layers

    def forward(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        toks = to_token_sequence(xs, ys)
        return from_predicted_token_sequence(self.token_sequence_transformer(toks))


def make_model(dtype: torch.dtype = torch.float32, device: str = "cpu") -> InContextRegressionTransformer:
    return InContextRegressionTransformer(
        task_size=4, max_examples=8, embed_size=64, mlp_size=64,
        num_heads=4, num_layers=2, layer_norm=True, include_output=True,
    ).to(device=device, dtype=dtype)


# =============================================================================
# 2. Exact Full-Model Iso-Functional Canonicalization (Float64)
# =============================================================================

@torch.no_grad()
def apply_symmetry_channel_(model: InContextRegressionTransformer, channel: str, n_iters: int = 15) -> None:
    """
    Apply an exact iso-functional symmetry transformation in-place (run in float64).
    Every channel preserves ys_pred = model(xs, ys) to float64 machine precision (~1e-14).
    """
    dt = model.token_sequence_transformer

    if channel == "LN_mean_proj":
        # Pre-LN exact 1_d kernel: project out mean along residual dim from every residual write.
        dt.token_embedding.weight.sub_(dt.token_embedding.weight.mean(dim=0, keepdim=True))
        dt.postn_embedding.weight.sub_(dt.postn_embedding.weight.mean(dim=0, keepdim=True))
        for blk in dt.blocks:
            blk.attention.output.weight.sub_(blk.attention.output.weight.mean(dim=0, keepdim=True))
            blk.compute[2].weight.sub_(blk.compute[2].weight.mean(dim=0, keepdim=True))
            blk.compute[2].bias.sub_(blk.compute[2].bias.mean())

    elif channel == "QK":
        # Per-head exact rank-d_h balanced SVD on W_Q^{(h)T} W_K^{(h)}.
        for blk in dt.blocks:
            att = blk.attention
            H, dh, C = att.num_heads, att.head_size, att.embed_size
            W = att.attention.weight.view(H, 3, dh, C)
            for h in range(H):
                M = W[h, 0].T @ W[h, 1]
                U, S, Vh = torch.linalg.svd(M, full_matrices=False)
                srt = S[:dh].clamp(min=0.0).sqrt()
                W[h, 0].copy_(srt[:, None] * U[:, :dh].T)
                W[h, 1].copy_(srt[:, None] * Vh[:dh, :])

    elif channel == "VO":
        # Per-head exact rank-d_h balanced SVD on W_O^{(h)} W_V^{(h)}.
        for blk in dt.blocks:
            att = blk.attention
            H, dh, C = att.num_heads, att.head_size, att.embed_size
            W = att.attention.weight.view(H, 3, dh, C)
            WO = att.output.weight.view(C, H, dh)
            for h in range(H):
                M = WO[:, h, :] @ W[h, 2]
                U, S, Vh = torch.linalg.svd(M, full_matrices=False)
                srt = S[:dh].clamp(min=0.0).sqrt()
                WO[:, h, :].copy_(U[:, :dh] * srt[None, :])
                W[h, 2].copy_(srt[:, None] * Vh[:dh, :])

    elif channel == "MLP":
        # Exact ReLU positive neuronwise rescaling + canonical descending norm permutation.
        for blk in dt.blocks:
            fc1, fc2 = blk.compute[0], blk.compute[2]
            W1, b1, W2 = fc1.weight, fc1.bias, fc2.weight
            a = torch.sqrt(W1.pow(2).sum(dim=1) + b1.pow(2))
            c = W2.pow(2).sum(dim=0).sqrt()
            prod = a * c
            dead = prod == 0
            s = torch.where(dead, torch.ones_like(a), torch.sqrt(c.clamp(min=1e-30) / a.clamp(min=1e-30)))
            W1.mul_(s[:, None])
            b1.mul_(s)
            W2.div_(s[None, :])
            if dead.any():
                W1[dead] = 0.0
                b1[dead] = 0.0
                W2[:, dead] = 0.0
            order = torch.argsort(prod, descending=True)
            W1.copy_(W1[order])
            b1.copy_(b1[order])
            W2.copy_(W2[:, order])

    elif channel == "LNf_unembed":
        # Exact bias absorption + coordinatewise (gamma_f, W_U[0,:]) balancing.
        lnf, un = dt.unembedding[0], dt.unembedding[1]
        w0 = un.weight[0]
        un.bias[0].add_(torch.dot(w0, lnf.bias))
        lnf.bias.zero_()
        p = lnf.weight * w0
        sp = p.abs().sqrt()
        lnf.weight.copy_(sp)
        w0.copy_(torch.sign(p) * sp)

    elif channel == "LN1_mlp":
        # Exact bias absorption + coordinatewise (gamma_1, W_fc1[:, i]) balancing.
        for blk in dt.blocks:
            ln1, fc1 = blk.layer_norms[1], blk.compute[0]
            fc1.bias.add_(fc1.weight @ ln1.bias)
            ln1.bias.zero_()
            g = ln1.weight.abs()
            col_n = fc1.weight.pow(2).sum(dim=0).sqrt()
            dead = (g * col_n) == 0
            s = torch.where(dead, torch.ones_like(g), torch.sqrt(col_n.clamp(min=1e-30) / g.clamp(min=1e-30)))
            sgn = torch.where(ln1.weight < 0, -torch.ones_like(g), torch.ones_like(g))
            ln1.weight.copy_(torch.where(dead, torch.zeros_like(g), g * s))
            fc1.weight.mul_(torch.where(dead, torch.zeros_like(g), sgn / s)[None, :])

    elif channel == "LN0_attn":
        # Exact simultaneous scaling of (gamma_0[i], beta_0[i]) by s_i and W_QKV[:, i] by 1/s_i.
        for blk in dt.blocks:
            ln0, qkv = blk.layer_norms[0], blk.attention.attention
            gn = torch.sqrt(ln0.weight.pow(2) + ln0.bias.pow(2))
            col_n = qkv.weight.pow(2).sum(dim=0).sqrt()
            dead = (gn * col_n) == 0
            s = torch.where(dead, torch.ones_like(gn), torch.sqrt(col_n.clamp(min=1e-30) / gn.clamp(min=1e-30)))
            ln0.weight.mul_(torch.where(dead, torch.zeros_like(gn), s))
            ln0.bias.mul_(torch.where(dead, torch.zeros_like(gn), s))
            qkv.weight.mul_(torch.where(dead, torch.zeros_like(gn), 1.0 / s)[None, :])

    elif channel == "LN_all_unit_gamma":
        # Control gauge: unit-norm LN affine parameters (gamma=1 for biased layers; ||(gamma,beta)||=1 for LN0).
        lnf, un = dt.unembedding[0], dt.unembedding[1]
        w0 = un.weight[0]
        un.bias[0].add_(torch.dot(w0, lnf.bias))
        lnf.bias.zero_()
        w0.mul_(lnf.weight)
        lnf.weight.fill_(1.0)
        for blk in dt.blocks:
            ln1, fc1 = blk.layer_norms[1], blk.compute[0]
            fc1.bias.add_(fc1.weight @ ln1.bias)
            ln1.bias.zero_()
            fc1.weight.mul_(ln1.weight[None, :])
            ln1.weight.fill_(1.0)
            ln0, qkv = blk.layer_norms[0], blk.attention.attention
            gn = torch.sqrt(ln0.weight.pow(2) + ln0.bias.pow(2)).clamp(min=1e-12)
            ln0.weight.div_(gn)
            ln0.bias.div_(gn)
            qkv.weight.mul_(gn[None, :])

    elif channel == "FULL":
        apply_symmetry_channel_(model, "LN_mean_proj")
        for it in range(n_iters):
            apply_symmetry_channel_(model, "LNf_unembed")
            apply_symmetry_channel_(model, "LN1_mlp")
            apply_symmetry_channel_(model, "LN0_attn")
            apply_symmetry_channel_(model, "QK")
            apply_symmetry_channel_(model, "VO")
            # Skip permutation during intermediate alternating iterations for speed
            for blk in dt.blocks:
                fc1, fc2 = blk.compute[0], blk.compute[2]
                W1, b1, W2 = fc1.weight, fc1.bias, fc2.weight
                a = torch.sqrt(W1.pow(2).sum(dim=1) + b1.pow(2))
                c = W2.pow(2).sum(dim=0).sqrt()
                prod = a * c
                dead = prod == 0
                s = torch.where(dead, torch.ones_like(a), torch.sqrt(c.clamp(min=1e-30) / a.clamp(min=1e-30)))
                W1.mul_(s[:, None])
                b1.mul_(s)
                W2.div_(s[None, :])
                if dead.any():
                    W1[dead] = 0.0
                    b1[dead] = 0.0
                    W2[:, dead] = 0.0
                if it == n_iters - 1:
                    order = torch.argsort(prod, descending=True)
                    W1.copy_(W1[order])
                    b1.copy_(b1[order])
                    W2.copy_(W2[:, order])
    else:
        raise ValueError(f"Unknown symmetry channel: {channel}")


@torch.no_grad()
def canonicalize_state_dict(
    sd_raw: dict[str, torch.Tensor],
    x_check64: torch.Tensor,
    y_check64: torch.Tensor,
    channel: str = "FULL",
) -> tuple[dict[str, torch.Tensor], float, float, float]:
    m_raw = make_model(dtype=torch.float64, device="cpu")
    m_raw.load_state_dict({k: v.double() for k, v in sd_raw.items()})
    y_ref = m_raw(x_check64, y_check64)
    norm_raw = math.sqrt(sum(p.pow(2).sum().item() for p in m_raw.parameters()))

    m_can = copy.deepcopy(m_raw)
    if channel != "RAW":
        apply_symmetry_channel_(m_can, channel)
    max_dy = float((m_can(x_check64, y_check64) - y_ref).abs().max().item())
    norm_can = math.sqrt(sum(p.pow(2).sum().item() for p in m_can.parameters()))
    sd_can = {k: v.float().cpu() for k, v in m_can.state_dict().items()}
    return sd_can, max_dy, norm_raw, norm_can


# =============================================================================
# 3. Structural, Behavioral & OOD Diagnostics (Matching Hoogland et al. Fig. 5)
# =============================================================================

@torch.no_grad()
def evaluate_checkpoint_metrics(
    sd: dict[str, torch.Tensor],
    x_ev: torch.Tensor,
    y_ev: torch.Tensor,
    w_ev: torch.Tensor,
    e_ev: torch.Tensor,
) -> dict:
    m = make_model(dtype=torch.float32, device="cpu")
    m.load_state_dict(sd)
    m.eval()

    yp = m(x_ev, y_ev)
    mse_tok = (yp - y_ev).pow(2).mean(dim=(0, 2))
    mse_id = float(mse_tok.mean().item())
    y_hat_sq = float(yp.pow(2).mean().item())
    icl_0_to_7 = float((mse_tok[-1] - mse_tok[0]).item())
    icl_D_to_K = float((mse_tok[-1] - mse_tok[3]).item())

    ood = {}
    for g in (3, 5):
        xg = x_ev * float(g)
        yg = torch.bmm(xg, w_ev.unsqueeze(-1)) + e_ev
        ypg = m(xg, yg)
        mt = (ypg - yg).pow(2).mean(dim=(0, 2))
        ood[f"g{g}_mse_norm"] = float(mt.mean().item() / (g ** 2))
        ood[f"g{g}_tok7_norm"] = float(mt[-1].item() / (g ** 2))

    dt = m.token_sequence_transformer
    ln_stats = {}
    for name, w_ln, b_ln, w_down in [
        ("blk0_ln0", dt.blocks[0].layer_norms[0].weight, dt.blocks[0].layer_norms[0].bias, dt.blocks[0].attention.attention.weight),
        ("blk0_ln1", dt.blocks[0].layer_norms[1].weight, dt.blocks[0].layer_norms[1].bias, dt.blocks[0].compute[0].weight),
        ("blk1_ln0", dt.blocks[1].layer_norms[0].weight, dt.blocks[1].layer_norms[0].bias, dt.blocks[1].attention.attention.weight),
        ("blk1_ln1", dt.blocks[1].layer_norms[1].weight, dt.blocks[1].layer_norms[1].bias, dt.blocks[1].compute[0].weight),
        ("lnf",      dt.unembedding[0].weight,           dt.unembedding[0].bias,           dt.unembedding[1].weight[0:1]),
    ]:
        col_norm = w_down.pow(2).sum(dim=0).sqrt()
        eff = w_ln.abs() * col_norm
        ln_stats[name] = {
            "mean_abs_gamma": float(w_ln.abs().mean().item()),
            "frac_gamma_lt_01": float((w_ln.abs() < 0.1).float().mean().item()),
            "bias_norm": float(b_ln.norm().item()),
            "downstream_w_norm": float(w_down.norm().item()),
            "mean_eff": float(eff.mean().item()),
            "frac_eff_lt_005": float((eff < 0.05).float().mean().item()),
        }

    return {
        "mse_id": mse_id,
        "y_hat_sq": y_hat_sq,
        "icl_0_to_7": icl_0_to_7,
        "icl_D_to_K": icl_D_to_K,
        "ood": ood,
        "ln_stats": ln_stats,
    }


# =============================================================================
# 4. Multiprocessing CPU SGLD LLC Estimator (Exact Hoogland et al. Algorithm)
# =============================================================================

_X_POOL: torch.Tensor | None = None
_Y_POOL: torch.Tensor | None = None


def _init_sgld_worker(pool_seed: int, n_pool: int) -> None:
    global _X_POOL, _Y_POOL
    torch.set_num_threads(1)
    sigma = math.sqrt(0.125)
    g = torch.Generator().manual_seed(pool_seed)
    w = torch.randn(n_pool, 4, generator=g)
    x = torch.randn(n_pool, 8, 4, generator=g)
    y = torch.bmm(x, w.unsqueeze(-1)) + torch.randn(n_pool, 8, 1, generator=g) * sigma
    _X_POOL, _Y_POOL = x, y


def _run_sgld_chain(args: tuple) -> dict:
    sd, init_loss, eps, gamma, nbeta, n_burn, n_draws, m_batch, full_batch_n, seed, want_windows = args
    torch.manual_seed(seed)
    m = make_model(dtype=torch.float32, device="cpu")
    m.load_state_dict(sd)
    w0 = [p.detach().clone() for p in m.parameters()]
    params = list(m.parameters())
    sqrt_eps, half_eps = math.sqrt(eps), 0.5 * eps
    n_total = n_burn + n_draws

    if full_batch_n is not None:
        xb_fb = _X_POOL[:full_batch_n]
        yb_fb = _Y_POOL[:full_batch_n]
        with torch.no_grad():
            init_loss = float(F.mse_loss(m(xb_fb, yb_fb), yb_fb).item())
        b_order = None
    else:
        n_blocks = _X_POOL.shape[0] // m_batch
        rng_b = np.random.default_rng(seed + 7)
        b_order = rng_b.integers(0, n_blocks, size=n_total)

    losses = np.empty(n_total, dtype=np.float64)
    for t in range(n_total):
        if full_batch_n is not None:
            xb, yb = xb_fb, yb_fb
        else:
            b0 = int(b_order[t]) * m_batch
            xb, yb = _X_POOL[b0 : b0 + m_batch], _Y_POOL[b0 : b0 + m_batch]
        for p in params:
            p.grad = None
        loss = F.mse_loss(m(xb, yb), yb)
        loss.backward()
        losses[t] = float(loss.item())
        with torch.no_grad():
            for p, p0 in zip(params, w0):
                p.add_(nbeta * p.grad + gamma * (p - p0), alpha=-half_eps)
                p.add_(torch.randn_like(p), alpha=sqrt_eps)

    post = losses[n_burn:]
    half = n_draws // 2
    llc = float(nbeta * (post.mean() - init_loss))
    hdiff = float(nbeta * (post[half:].mean() - post[:half].mean()))
    dist = math.sqrt(sum((p - p0).pow(2).sum().item() for p, p0 in zip(params, w0)))

    out = {"llc": llc, "hdiff": hdiff, "dist": dist}
    if want_windows:
        wins = {}
        for a, b in [(300, 900), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (1000, 5000)]:
            if b <= n_total:
                wins[f"{a}:{b}"] = float(nbeta * (losses[a:b].mean() - init_loss))
        out["windows"] = wins
    return out


def compute_pool_init_loss(sd: dict[str, torch.Tensor], x_pool: torch.Tensor, y_pool: torch.Tensor) -> float:
    m = make_model(dtype=torch.float32, device="cpu")
    m.load_state_dict(sd)
    m.eval()
    n_pool = x_pool.shape[0]
    with torch.no_grad():
        tot = sum(
            F.mse_loss(m(x_pool[i : i + 4096], y_pool[i : i + 4096]), y_pool[i : i + 4096], reduction="sum").item()
            for i in range(0, n_pool, 4096)
        )
    return float(tot / (n_pool * 8))


def agg_chains(res: list[dict]) -> dict:
    ll = np.array([r["llc"] for r in res], dtype=np.float64)
    hd = np.array([r["hdiff"] for r in res], dtype=np.float64)
    ds = np.array([r["dist"] for r in res], dtype=np.float64)
    out = {
        "mean": float(ll.mean()),
        "se": float(ll.std(ddof=1) / math.sqrt(len(ll))),
        "hdiff": float(hd.mean()),
        "dist": float(ds.mean()),
    }
    if "windows" in res[0]:
        out["windows"] = {}
        for k in res[0]["windows"]:
            wvals = np.array([r["windows"][k] for r in res], dtype=np.float64)
            out["windows"][k] = {
                "mean": float(wvals.mean()),
                "se": float(wvals.std(ddof=1) / math.sqrt(len(wvals))),
            }
    return out


# =============================================================================
# 5. Training Worker (MPS or CPU)
# =============================================================================

def train_single_run(
    opt_type: str,
    wd: float,
    seed: int,
    t_steps: int,
    ckpt_steps: list[int],
    device: str,
) -> tuple[dict[int, dict[str, torch.Tensor]], float]:
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    model = make_model(dtype=torch.float32, device=device)
    B = 256
    sigma = math.sqrt(0.125)
    if opt_type == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=0.003, betas=(0.9, 0.999), weight_decay=wd)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=0.003, betas=(0.9, 0.999), weight_decay=wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=0.003,
        total_steps=t_steps,
        pct_start=0.5,
        anneal_strategy="linear",
        cycle_momentum=False,
        div_factor=t_steps / 2 - 1,
        final_div_factor=t_steps / 2 - 1,
    )

    ckpt_set = set(ckpt_steps)
    ckpts: dict[int, dict[str, torch.Tensor]] = {}
    if 0 in ckpt_set:
        ckpts[0] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Use a dedicated CPU/device generator seeded cleanly per seed for data generation
    for step in range(1, t_steps + 1):
        ws = torch.randn(B, 4, device=device)
        xs = torch.randn(B, 8, 4, device=device)
        ys = torch.bmm(xs, ws.unsqueeze(-1)) + torch.randn(B, 8, 1, device=device) * sigma
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(xs, ys), ys)
        loss.backward()
        opt.step()
        if step < t_steps:
            sched.step()
        if step in ckpt_set:
            ckpts[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  [Train {opt_type:5s} wd={wd:.2f} seed={seed}] {t_steps} steps finished in {elapsed:.1f}s", flush=True)
    return ckpts, float(elapsed)


# =============================================================================
# 6. Multi-Seed Stage Decomposition Helper
# =============================================================================

def compute_stage_decompositions(seed_runs: list[dict], stage_pairs: list[tuple[str, int, int]]) -> dict:
    out = {}
    for sname, t1, t2 in stage_pairs:
        d_tot_list, d_func_list, d_fib_list, share_list = [], [], [], []
        per_seed = []
        for sr in seed_runs:
            by_step = {c["step"]: c for c in sr["checkpoints"]}
            c1, c2 = by_step[t1], by_step[t2]
            r1, r2 = c1["sgld_raw"]["mean"], c2["sgld_raw"]["mean"]
            b1, b2 = c1["sgld_full"]["mean"], c2["sgld_full"]["mean"]
            dtot = r2 - r1
            dfunc = b2 - b1
            dfib = dtot - dfunc
            share = (100.0 * dfib / dtot) if abs(dtot) > 1e-6 else float("nan")
            d_tot_list.append(dtot)
            d_func_list.append(dfunc)
            d_fib_list.append(dfib)
            share_list.append(share)
            per_seed.append({
                "seed": sr["seed"],
                "raw_t1": r1, "raw_t2": r2,
                "bal_t1": b1, "bal_t2": b2,
                "d_total": dtot, "d_func": dfunc, "d_fiber": dfib,
                "fiber_share_pct": share,
            })

        def m_se(arr):
            a = np.array(arr, dtype=np.float64)
            return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0

        mt, st = m_se(d_tot_list)
        mf, sf = m_se(d_func_list)
        mb, sb = m_se(d_fib_list)
        ms, ss = m_se(share_list)
        out[sname] = {
            "t1": t1, "t2": t2,
            "d_total_mean": mt, "d_total_se": st,
            "d_func_mean": mf, "d_func_se": sf,
            "d_fiber_mean": mb, "d_fiber_se": sb,
            "fiber_share_pct_mean": ms, "fiber_share_pct_se": ss,
            "ratio_of_means_fiber_share_pct": float(100.0 * mb / mt) if abs(mt) > 1e-6 else float("nan"),
            "per_seed": per_seed,
        }
    return out


# =============================================================================
# 7. Publication-Quality 6-Panel Figure (`--replot` reads JSON only)
# =============================================================================

def plot_results(data: dict) -> None:
    plt.rcParams.update({
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 150,
    })

    fig, axes = plt.subplots(2, 3, figsize=(17.2, 9.6))
    fig.subplots_adjust(hspace=0.35, wspace=0.36, top=0.90, bottom=0.08, left=0.055, right=0.98)

    adam_runs = data["part1_adam_wd0"]["runs"]
    adamw_runs = data["part2_adamw"]["runs"]
    steps_a = [c["step"] for c in adam_runs[0]["checkpoints"]]
    # Plot step=0 at x=100 on log scale with clean tick label "0"
    x_a = np.array([max(s, 100) for s in steps_a], dtype=float)

    def stack_curve(runs, key_fn):
        mat = np.array([[key_fn(c) for c in r["checkpoints"]] for r in runs], dtype=float)
        return mat.mean(axis=0), mat.std(axis=0, ddof=1) / math.sqrt(len(runs))

    raw_m, raw_se = stack_curve(adam_runs, lambda c: c["sgld_raw"]["mean"])
    bal_m, bal_se = stack_curve(adam_runs, lambda c: c["sgld_full"]["mean"])
    mse_m, mse_se = stack_curve(adam_runs, lambda c: c["metrics"]["mse_id"])

    # --- Panel A: Raw vs Balanced LLC over training (Adam wd=0, 3 seeds) ---
    ax = axes[0, 0]
    ax.axvspan(100, 1000, color="#e8f1fa", alpha=0.55, label="LR1 (zero pred)")
    ax.axvspan(1000, 25000, color="#e8f6ec", alpha=0.55, label="LR2 (ICL acq.)")
    ax.axvspan(25000, 80000, color="#fde8e8", alpha=0.55, label="LR3–LR4 (drop)")
    ax.axvspan(80000, 100000, color="#f4eef8", alpha=0.55, label="LR5")

    ax.errorbar(x_a, raw_m, yerr=raw_se, fmt="-o", color="#c0392b", lw=2.2, ms=5, capsize=3, label=r"Raw $\hat\lambda$ (Hoogland recipe)")
    ax.errorbar(x_a, bal_m, yerr=bal_se, fmt="-s", color="#1f618d", lw=2.2, ms=5, capsize=3, label=r"Exact Balanced $\hat\lambda$ ($\max|\Delta y|\leq 1.1\times 10^{-12}$)")
    ax.set_xscale("log")
    ax.set_xticks([100, 1000, 10000, 25000, 80000])
    ax.set_xticklabels(["0", "1k", "10k", "25k", "80k"])
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel(r"Estimated LLC $\hat\lambda$ ($\gamma=13.3, n\beta=66.7$)")
    ax.set_title("A. Raw vs. Iso-Functional Balanced LLC (Adam, 3 seeds)")
    ax.set_ylim(-5, 185)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(x_a, mse_m, "--", color="#555555", lw=1.4, alpha=0.8, label="ID MSE")
    ax2.set_ylabel("ID MSE", color="#555555")
    ax2.tick_params(axis="y", labelcolor="#555555")
    ax2.set_ylim(1.5, 4.6)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1[-2:] + h2, l1[-2:] + l2, loc="lower right", frameon=True, framealpha=0.92)

    # --- Panel B: Stagewise Delta hat_lambda decomposition ---
    ax = axes[0, 1]
    stages = data["part1_adam_wd0"]["stage_decompositions"]
    s_keys = ["LR1_0_to_1k", "LR2_1k_to_25k", "LR34_25k_to_80k", "LR5_80k_to_100k"]
    s_labels = ["LR1\n(0→1k)", "LR2\n(1k→25k)", "LR3–LR4\n(25k→80k)", "LR5\n(80k→100k)"]
    idx = np.arange(len(s_keys))
    w = 0.25
    t_m = [stages[k]["d_total_mean"] for k in s_keys]
    t_se = [stages[k]["d_total_se"] for k in s_keys]
    f_m = [stages[k]["d_func_mean"] for k in s_keys]
    f_se = [stages[k]["d_func_se"] for k in s_keys]
    b_m = [stages[k]["d_fiber_mean"] for k in s_keys]
    b_se = [stages[k]["d_fiber_se"] for k in s_keys]

    ax.axhline(0, color="black", lw=0.8)
    ax.bar(idx - w, t_m, w, yerr=t_se, capsize=3, color="#34495e", label=r"$\Delta\hat\lambda_{\mathrm{total}}$ (Raw)")
    ax.bar(idx, f_m, w, yerr=f_se, capsize=3, color="#1f618d", label=r"$\Delta\hat\lambda_{\mathrm{func}}$ (Balanced)")
    ax.bar(idx + w, b_m, w, yerr=b_se, capsize=3, color="#e67e22", label=r"$\Delta\hat\lambda_{\mathrm{fiber}}$ (Fiber slide)")
    ax.set_xticks(idx)
    ax.set_xticklabels(s_labels)
    ax.set_ylabel(r"Stagewise $\Delta\hat\lambda$ (mean ± SE, 3 seeds)")
    ax.set_title(r"B. $\Delta\hat\lambda_{\mathrm{total}} = \Delta\hat\lambda_{\mathrm{func}} + \Delta\hat\lambda_{\mathrm{fiber}}$ by Stage")
    ax.set_ylim(-64, 152)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)

    # Annotate LR3-LR4 fiber share cleanly inside the plot
    sh_34 = stages["LR34_25k_to_80k"]["fiber_share_pct_mean"]
    sh_34_se = stages["LR34_25k_to_80k"]["fiber_share_pct_se"]
    ax.annotate(
        f"LR3–LR4 fiber share:\n{sh_34:.1f}% ± {sh_34_se:.1f}%",
        xy=(2 + w, b_m[2] * 0.55),
        xytext=(2.05, 36),
        ha="center",
        fontsize=8.3,
        fontweight="bold",
        color="#a04000",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef5e7", edgecolor="#e67e22", lw=0.9),
        arrowprops=dict(arrowstyle="->", color="#a04000", lw=1.2),
    )

    # --- Panel C: Per-channel exact symmetry ablation at LR4 trough (80k) ---
    ax = axes[0, 2]
    abl80 = data["part3_channel_ablation"]["step_80000"]
    ch_order = ["LN_mean_proj", "LNf_unembed", "MLP", "QK", "VO", "LN1_mlp", "LN0_attn", "LN_all_unit_gamma", "FULL"]
    ch_labels = [
        "Pre-LN $1_d$ proj",
        "LN$_f$ ↔ Unembed",
        "MLP ReLU scale",
        "Attn Q–K GL($d_h$)",
        "Attn V–O GL($d_h$)",
        "Pre-MLP LN$_1$ ↔ FC1",
        "Pre-Attn LN$_0$ ↔ QKV",
        "Unit-$\\gamma$ LN (ctrl)",
        "FULL Canonical",
    ]
    raw80 = abl80["RAW"]["sgld"]["mean"]
    vals = [abl80[c]["sgld"]["mean"] - raw80 for c in ch_order]
    errs = [math.hypot(abl80[c]["sgld"]["se"], abl80["RAW"]["sgld"]["se"]) for c in ch_order]
    colors = ["#95a5a6"] * 5 + ["#f39c12", "#c0392b", "#7d3c98", "#1f618d"]
    y_pos = np.arange(len(ch_order))
    ax.axvline(0, color="black", lw=0.8)
    ax.barh(y_pos, vals, xerr=errs, color=colors, capsize=2.5, alpha=0.92)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ch_labels)
    ax.set_xlim(-5, 67)
    ax.set_xlabel(r"$\hat\lambda_{\mathrm{channel}} - \hat\lambda_{\mathrm{raw}}$ at LR4 trough (step 80k)")
    ax.set_title("C. Exact Symmetry Channel Ablation (Seed 0, 80k)")
    ax.grid(True, axis="x", alpha=0.25)
    for yp_i, v, e in zip(y_pos, vals, errs):
        if abs(v) > 3.0:
            ax.text(v + e + 1.2, yp_i, f"{v:+.1f}", va="center", fontsize=8.0, fontweight="bold")

    # --- Panel D: Mechanistic LN0 <-> QKV anti-correlated sliding vs genuine LNf collapse ---
    ax = axes[1, 0]
    ln00_m, ln00_se = stack_curve(adam_runs, lambda c: c["metrics"]["ln_stats"]["blk0_ln0"]["mean_abs_gamma"])
    lnf_m, lnf_se = stack_curve(adam_runs, lambda c: c["metrics"]["ln_stats"]["lnf"]["mean_abs_gamma"])
    qkv0_m, qkv0_se = stack_curve(adam_runs, lambda c: c["metrics"]["ln_stats"]["blk0_ln0"]["downstream_w_norm"])
    lnf_eff_m, _ = stack_curve(adam_runs, lambda c: c["metrics"]["ln_stats"]["lnf"]["frac_eff_lt_005"])
    ln00_eff_m, _ = stack_curve(adam_runs, lambda c: c["metrics"]["ln_stats"]["blk0_ln0"]["frac_eff_lt_005"])

    ax.errorbar(x_a, ln00_m, yerr=ln00_se, fmt="-o", color="#c0392b", lw=2.0, ms=4.5, label=r"Blk0 LN$_0$ mean $|\gamma|$")
    ax.errorbar(x_a, lnf_m, yerr=lnf_se, fmt="-^", color="#27ae60", lw=2.0, ms=4.5, label=r"Final LN$_f$ mean $|\gamma|$")
    ax.plot(x_a, lnf_eff_m, ":^", color="#1e8449", lw=1.5, label=r"LN$_f$ frac $|\gamma W_U|<0.05$ (real)")
    ax.plot(x_a, ln00_eff_m, ":o", color="#922b21", lw=1.5, label=r"LN$_0$ frac $|\gamma|\|W_{QKV}\|<0.05$")
    ax.set_xscale("log")
    ax.set_xticks([100, 1000, 10000, 25000, 80000])
    ax.set_xticklabels(["0", "1k", "10k", "25k", "80k"])
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel(r"Mean $|\gamma|$ / collapsed fraction")
    ax.set_ylim(-0.05, 1.35)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.errorbar(x_a, qkv0_m, yerr=qkv0_se, fmt="--s", color="#2980b9", lw=2.0, ms=4.5, label=r"$\|W_{\mathrm{QKV},0}\|_F$")
    ax2.set_ylabel(r"$\|W_{\mathrm{QKV},0}\|_F$ (downstream norm)", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=7.3, frameon=True, framealpha=0.92)
    ax.set_title("D. LN$_0$↔QKV Fiber Blowup vs. Genuine LN$_f$ Collapse")

    # --- Panel E: Weight-decay counterfactual (Adam wd=0 vs AdamW wd=0.08, 3 seeds) ---
    ax = axes[1, 1]
    steps_w = [c["step"] for c in adamw_runs[0]["checkpoints"]]
    x_w = np.array([max(s, 100) for s in steps_w], dtype=float)
    w_raw_m, w_raw_se = stack_curve(adamw_runs, lambda c: c["sgld_raw"]["mean"])
    w_bal_m, w_bal_se = stack_curve(adamw_runs, lambda c: c["sgld_full"]["mean"])

    ax.errorbar(x_a, raw_m, yerr=raw_se, fmt="-o", color="#c0392b", lw=2.0, ms=4.5, alpha=0.85, label="Adam (wd=0) Raw")
    ax.errorbar(x_a, bal_m, yerr=bal_se, fmt="--o", color="#e67e22", lw=1.8, ms=4.0, alpha=0.85, label="Adam (wd=0) Balanced")
    ax.errorbar(x_w, w_raw_m, yerr=w_raw_se, fmt="-s", color="#1f618d", lw=2.2, ms=5.0, label="AdamW (wd=0.08) Raw")
    ax.errorbar(x_w, w_bal_m, yerr=w_bal_se, fmt="--s", color="#16a085", lw=1.8, ms=4.5, label="AdamW (wd=0.08) Balanced")
    ax.set_xscale("log")
    ax.set_xticks([100, 1000, 10000, 25000, 80000])
    ax.set_xticklabels(["0", "1k", "10k", "25k", "80k"])
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel(r"Estimated LLC $\hat\lambda$ (3 seeds mean ± SE)")
    ax.set_title("E. Weight Decay Eliminates the Fake LR3–LR4 Drop")
    ax.set_ylim(-5, 185)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    # --- Panel F: Finite-gamma equilibrium distortion & full-batch SGLD check ---
    ax = axes[1, 2]
    sens = data["part4_sgld_sensitivity"]
    s80 = sens["step_80000"]
    s25 = sens["step_25000"]

    gammas = [3.3, 13.3, 40.0]
    g_keys = ["g3.3_e3e-4_m256", "g13.3_e3e-4_m256", "g40.0_e3e-4_m256"]
    r80 = [s80[k]["raw"]["mean"] for k in g_keys]
    r80_e = [s80[k]["raw"]["se"] for k in g_keys]
    f80 = [s80[k]["full"]["mean"] for k in g_keys]
    f80_e = [s80[k]["full"]["se"] for k in g_keys]
    r25 = [s25[k]["raw"]["mean"] for k in g_keys]
    f25 = [s25[k]["full"]["mean"] for k in g_keys]

    ax.errorbar(gammas, r80, yerr=r80_e, fmt="-o", color="#c0392b", lw=2.2, ms=6, capsize=3, label="Step 80k Raw (minibatch)")
    ax.errorbar(gammas, f80, yerr=f80_e, fmt="-s", color="#1f618d", lw=2.2, ms=6, capsize=3, label="Step 80k Balanced (minibatch)")
    ax.plot(gammas, r25, ":o", color="#e74c3c", lw=1.4, ms=4.5, alpha=0.75, label="Step 25k Raw (LR2 peak)")
    ax.plot(gammas, f25, ":s", color="#5dade2", lw=1.4, ms=4.5, alpha=0.75, label="Step 25k Balanced (LR2 peak)")

    # Overlay m=1024 and zero-noise Full-Batch FB=1024 at gamma=13.3 (offset slightly for clarity)
    fb_r = s80["g13.3_e3e-4_FB1024"]["raw"]["mean"]
    fb_f = s80["g13.3_e3e-4_FB1024"]["full"]["mean"]
    m1k_r = s80["g13.3_e3e-4_m1024"]["raw"]["mean"]
    m1k_f = s80["g13.3_e3e-4_m1024"]["full"]["mean"]
    ax.scatter([14.8], [fb_r], marker="D", s=48, color="#922b21", zorder=6, label="Step 80k Full-Batch (FB=1024)")
    ax.scatter([14.8], [fb_f], marker="D", s=48, color="#154360", zorder=6)
    ax.scatter([11.9], [m1k_r], marker="^", s=44, color="#c0392b", alpha=0.85, zorder=6, label="Step 80k m=1024 (Hoogland)")
    ax.scatter([11.9], [m1k_f], marker="^", s=44, color="#1f618d", alpha=0.85, zorder=6)

    ax.set_xscale("log")
    ax.set_xticks(gammas)
    ax.set_xticklabels(["3.3", "13.3\n(default)", "40.0"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(2.7, 48.0)
    ax.set_xlabel(r"SGLD localization strength $\gamma$ ($T_{\mathrm{SGLD}}=5{,}000$)")
    ax.set_ylabel(r"Converged LLC $\hat\lambda$ (1000:5000 draws)")
    ax.set_title(r"F. Finite-$\gamma$ Fiber Distortion & Full-Batch Control")
    ax.set_ylim(40, 180)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=7.3, frameon=True, framealpha=0.92)

    fig.suptitle(
        "Is That LLC Drop Development or Fiber Sliding? Exact Iso-Functional Decomposition of Hoogland et al. (2024) ICL Transformers",
        fontsize=12.5,
        fontweight="bold",
        y=0.965,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PNG_PATH, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {PNG_PATH}", flush=True)


# =============================================================================
# 8. Main Experiment Orchestration
# =============================================================================

def main() -> None:
    if "--replot" in sys.argv:
        data = json.loads(JSON_PATH.read_text())
        plot_results(data)
        return

    t_global = time.perf_counter()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Starting Track A production experiment | train device={device} | SGLD=8-core CPU spawn", flush=True)

    # Experiment configuration
    t_steps = 100_000
    seeds = [0, 1, 2]
    adam_ckpts_steps = [0, 500, 1000, 2500, 10000, 25000, 50000, 80000, 100000]
    adamw_ckpts_steps = [0, 1000, 10000, 25000, 60000, 80000, 100000]
    adamw_wd = 0.08

    # Fixed SGLD pool and held-out eval set
    pool_seed = 12345
    n_pool = 32768
    sigma = math.sqrt(0.125)
    g_p = torch.Generator().manual_seed(pool_seed)
    w_pool = torch.randn(n_pool, 4, generator=g_p)
    x_pool = torch.randn(n_pool, 8, 4, generator=g_p)
    y_pool = torch.bmm(x_pool, w_pool.unsqueeze(-1)) + torch.randn(n_pool, 8, 1, generator=g_p) * sigma
    x_check64, y_check64 = x_pool[:64].double(), y_pool[:64].double()

    g_ev = torch.Generator().manual_seed(999)
    w_ev = torch.randn(2048, 4, generator=g_ev)
    x_ev = torch.randn(2048, 8, 4, generator=g_ev)
    e_ev = torch.randn(2048, 8, 1, generator=g_ev) * sigma
    y_ev = torch.bmm(x_ev, w_ev.unsqueeze(-1)) + e_ev

    # Exact Hoogland et al. SGLD defaults
    eps_def, gamma_def, nbeta_def = 0.0003, 13.3, 66.7
    n_chains_main = 4
    n_burn_main, n_draws_main = 1000, 2500  # 3,500 steps for multi-seed grid; Part 4 verifies exact 5,000 steps!
    n_burn_exact, n_draws_exact = 1000, 4000  # Exact Hoogland et al. T_SGLD = 5,000

    n_params = sum(p.numel() for p in make_model().parameters())
    print(f"Model parameter count: {n_params} (Hoogland et al. Table F.2.1 exact: 51,717)", flush=True)

    ctx = mp.get_context("spawn")
    with ctx.Pool(8, initializer=_init_sgld_worker, initargs=(pool_seed, n_pool)) as sgld_pool:

        def eval_run_checkpoints(ckpts: dict[int, dict[str, torch.Tensor]], step_list: list[int], label: str) -> list[dict]:
            records = []
            for st in step_list:
                sd_raw = ckpts[st]
                sd_full, max_dy, nr, nf = canonicalize_state_dict(sd_raw, x_check64, y_check64, channel="FULL")
                metrics = evaluate_checkpoint_metrics(sd_raw, x_ev, y_ev, w_ev, e_ev)
                init_loss = compute_pool_init_loss(sd_raw, x_pool, y_pool)
                tasks = (
                    [(sd_raw, init_loss, eps_def, gamma_def, nbeta_def, n_burn_main, n_draws_main, 256, None, 42 + c * 1009, False) for c in range(n_chains_main)]
                    + [(sd_full, init_loss, eps_def, gamma_def, nbeta_def, n_burn_main, n_draws_main, 256, None, 42 + c * 1009, False) for c in range(n_chains_main)]
                )
                res = sgld_pool.map(_run_sgld_chain, tasks)
                sgld_raw = agg_chains(res[:n_chains_main])
                sgld_full = agg_chains(res[n_chains_main:])
                rec = {
                    "step": st,
                    "init_loss_pool": init_loss,
                    "norm_raw": nr,
                    "norm_full": nf,
                    "max_delta_y_full": max_dy,
                    "sgld_raw": sgld_raw,
                    "sgld_full": sgld_full,
                    "excess_raw_minus_full": float(sgld_raw["mean"] - sgld_full["mean"]),
                    "metrics": metrics,
                }
                records.append(rec)
                print(
                    f"  [{label}] step={st:6d} | MSE={metrics['mse_id']:.4f} | "
                    f"RAW={sgld_raw['mean']:6.2f}±{sgld_raw['se']:4.2f} (hd={sgld_raw['hdiff']:+4.1f}) | "
                    f"FULL={sgld_full['mean']:6.2f}±{sgld_full['se']:4.2f} (hd={sgld_full['hdiff']:+4.1f}) | "
                    f"R-F={rec['excess_raw_minus_full']:+6.2f} | ||w|| {nr:5.1f}->{nf:5.1f} | max|dy|={max_dy:.2e}",
                    flush=True,
                )
            return records

        # ---------------------------------------------------------------------
        # PART 1 & PART 2: Pipelined MPS training + 8-core CPU SGLD
        # (Trains run k+1 on MPS while 8 CPU processes run SGLD on run k!)
        # ---------------------------------------------------------------------
        run_queue = (
            [("adam", 0.0, sd_i, adam_ckpts_steps, f"Adam-s{sd_i}") for sd_i in seeds]
            + [("adamw", adamw_wd, sd_i, adamw_ckpts_steps, f"AdamW-s{sd_i}") for sd_i in seeds]
        )

        print("\n" + "=" * 84)
        print(f"PARTS 1 & 2: Multi-Seed Adam (wd=0.0) & AdamW (wd={adamw_wd}) — 3 seeds each, T={t_steps:,} steps")
        print("=" * 84, flush=True)

        adam_runs = []
        adamw_runs = []
        seed0_adam_ckpts = None

        with ThreadPoolExecutor(max_workers=1) as train_exec:
            o0, wd0, s0, st0, lbl0 = run_queue[0]
            fut = train_exec.submit(train_single_run, o0, wd0, s0, t_steps, st0, device)
            for idx_q, (otype, wd_v, sd_i, st_list, lbl) in enumerate(run_queue):
                ckpts, train_s = fut.result()
                # Immediately kick off next training run on MPS while CPU runs SGLD!
                if idx_q + 1 < len(run_queue):
                    on, wdn, sn, stn, _ = run_queue[idx_q + 1]
                    fut = train_exec.submit(train_single_run, on, wdn, sn, t_steps, stn, device)
                if otype == "adam" and sd_i == 0:
                    seed0_adam_ckpts = ckpts
                recs = eval_run_checkpoints(ckpts, st_list, lbl)
                entry = {"seed": sd_i, "train_time_s": train_s, "checkpoints": recs}
                if otype == "adam":
                    adam_runs.append(entry)
                else:
                    adamw_runs.append(entry)

        stage_pairs = [
            ("LR1_0_to_1k", 0, 1000),
            ("LR2_1k_to_25k", 1000, 25000),
            ("LR34_25k_to_80k", 25000, 80000),
            ("LR5_80k_to_100k", 80000, 100000),
        ]
        adam_stages = compute_stage_decompositions(adam_runs, stage_pairs)
        adamw_stages = compute_stage_decompositions(adamw_runs, stage_pairs)

        print("\n  --- Part 1 Multi-Seed Stage Decomposition (Adam, wd=0.0, 3 seeds mean ± SE) ---", flush=True)
        for sn, info in adam_stages.items():
            print(
                f"  {sn:16s} | dTot={info['d_total_mean']:+7.2f}±{info['d_total_se']:4.2f} | "
                f"dFunc={info['d_func_mean']:+7.2f}±{info['d_func_se']:4.2f} | "
                f"dFib={info['d_fiber_mean']:+7.2f}±{info['d_fiber_se']:4.2f} | "
                f"FiberShare={info['fiber_share_pct_mean']:6.1f}±{info['fiber_share_pct_se']:4.1f}%",
                flush=True,
            )
        print(f"\n  --- Part 2 Multi-Seed Stage Decomposition (AdamW, wd={adamw_wd}, 3 seeds mean ± SE) ---", flush=True)
        for sn, info in adamw_stages.items():
            print(
                f"  {sn:16s} | dTot={info['d_total_mean']:+7.2f}±{info['d_total_se']:4.2f} | "
                f"dFunc={info['d_func_mean']:+7.2f}±{info['d_func_se']:4.2f} | "
                f"dFib={info['d_fiber_mean']:+7.2f}±{info['d_fiber_se']:4.2f}",
                flush=True,
            )

        # ---------------------------------------------------------------------
        # PART 3: Per-Channel Exact Symmetry Ablation (Seed 0, steps 25k & 80k)
        #         Dispatched in pairs (8 chains/wave) to saturate all 8 cores!
        # ---------------------------------------------------------------------
        print("\n" + "=" * 84)
        print("PART 3: Per-Channel Exact Symmetry Ablation (Adam seed 0, steps 25k & 80k, T_SGLD=5,000)")
        print("=" * 84, flush=True)
        channels = [
            "RAW", "LN_mean_proj", "QK", "VO", "MLP",
            "LNf_unembed", "LN1_mlp", "LN0_attn", "LN_all_unit_gamma", "FULL",
        ]
        part3 = {}
        for st in (25000, 80000):
            sd_raw = seed0_adam_ckpts[st]
            init_loss = compute_pool_init_loss(sd_raw, x_pool, y_pool)
            ch_res = {}
            print(f"\n  --- Step {st} (exact T_SGLD=5,000: 1000 burn + 4000 draws, g=13.3, eps=3e-4, nb=66.7) ---", flush=True)
            prepped = []
            for ch in channels:
                sd_c, dy, nr, nc = canonicalize_state_dict(sd_raw, x_check64, y_check64, channel=ch)
                prepped.append((ch, sd_c, dy, nc))
            all_tasks = []
            for ch, sd_c, _, _ in prepped:
                for c in range(4):
                    all_tasks.append((sd_c, init_loss, eps_def, gamma_def, nbeta_def, n_burn_exact, n_draws_exact, 256, None, 42 + c * 1009, False))
            all_res = sgld_pool.map(_run_sgld_chain, all_tasks)
            raw_m = None
            for i_ch, (ch, _, dy, nc) in enumerate(prepped):
                agg = agg_chains(all_res[4 * i_ch : 4 * (i_ch + 1)])
                if ch == "RAW":
                    raw_m = agg["mean"]
                ch_res[ch] = {
                    "sgld": agg,
                    "d_vs_raw": float(agg["mean"] - raw_m),
                    "norm": nc,
                    "max_delta_y": dy,
                }
                print(
                    f"    {ch:18s} | LLC={agg['mean']:6.2f}±{agg['se']:4.2f} | "
                    f"d_raw={agg['mean']-raw_m:+7.2f} | hd={agg['hdiff']:+5.1f} | ||w||={nc:5.1f} | max|dy|={dy:.2e}",
                    flush=True,
                )
            part3[f"step_{st}"] = ch_res

        # ---------------------------------------------------------------------
        # PART 4: SGLD Hyperparameter, 5,000-Step Trace & Full-Batch Sensitivity
        # ---------------------------------------------------------------------
        print("\n" + "=" * 84)
        print("PART 4: SGLD Localization (gamma), Step-Size (eps), m=1024 & Full-Batch Controls (T_SGLD=5,000)")
        print("=" * 84, flush=True)
        part4 = {}
        sens_specs_base = [
            ("g3.3_e3e-4_m256",   0.0003,  3.3, 66.7,  256, None),
            ("g13.3_e3e-4_m256",  0.0003, 13.3, 66.7,  256, None),
            ("g40.0_e3e-4_m256",  0.0003, 40.0, 66.7,  256, None),
            ("g13.3_e1e-4_m256",  0.0001, 13.3, 66.7,  256, None),
        ]
        sens_specs_extra80 = [
            ("g13.3_e3e-4_m1024",  0.0003, 13.3, 66.7, 1024, None),
            ("g13.3_e3e-4_FB1024", 0.0003, 13.3, 66.7, 1024, 1024),
        ]
        for st in (25000, 80000):
            sd_raw = seed0_adam_ckpts[st]
            sd_full, dy, _, _ = canonicalize_state_dict(sd_raw, x_check64, y_check64, channel="FULL")
            init_loss = compute_pool_init_loss(sd_raw, x_pool, y_pool)
            st_res = {}
            specs = sens_specs_base + (sens_specs_extra80 if st == 80000 else [])
            print(f"\n  --- Step {st} Sensitivity Suite (T_SGLD=5,000: 1000 burn + 4000 draws) ---", flush=True)
            for name, eps_s, gam_s, nb_s, mb_s, fb_s in specs:
                tasks = (
                    [(sd_raw, init_loss, eps_s, gam_s, nb_s, n_burn_exact, n_draws_exact, mb_s, fb_s, 100 + c * 1009, True) for c in range(4)]
                    + [(sd_full, init_loss, eps_s, gam_s, nb_s, n_burn_exact, n_draws_exact, mb_s, fb_s, 100 + c * 1009, True) for c in range(4)]
                )
                res = sgld_pool.map(_run_sgld_chain, tasks)
                ag_r = agg_chains(res[:4])
                ag_f = agg_chains(res[4:])
                st_res[name] = {
                    "eps": eps_s, "gamma": gam_s, "nbeta": nb_s, "m_batch": mb_s, "full_batch_n": fb_s,
                    "raw": ag_r, "full": ag_f,
                    "raw_minus_full": float(ag_r["mean"] - ag_f["mean"]),
                }
                print(
                    f"    {name:20s} | RAW={ag_r['mean']:6.2f}±{ag_r['se']:4.2f} (hd={ag_r['hdiff']:+5.1f}) | "
                    f"FULL={ag_f['mean']:6.2f}±{ag_f['se']:4.2f} (hd={ag_f['hdiff']:+5.1f}) | R-F={ag_r['mean']-ag_f['mean']:+7.2f}",
                    flush=True,
                )
            part4[f"step_{st}"] = st_res

    total_elapsed = time.perf_counter() - t_global
    payload = {
        "metadata": {
            "script": "experiments/devinterp_fiber_decomposition.py",
            "reference_paper": "Hoogland et al. (2024, TMLR, arXiv:2402.02364) & github.com/timaeus-research/icl",
            "model_config": {
                "name": "L2H4Minf",
                "task_size_D": 4,
                "max_examples_K": 8,
                "noise_variance": 0.125,
                "num_layers_L": 2,
                "num_heads_H": 4,
                "embed_size": 64,
                "mlp_size": 64,
                "layer_norm": True,
                "include_output": True,
                "n_params": n_params,
            },
            "training_config": {
                "steps": t_steps,
                "batch_size": 256,
                "max_lr": 0.003,
                "scheduler": "OneCycleLR(pct_start=0.5, anneal_strategy='linear')",
                "seeds": seeds,
                "adam_wd": 0.0,
                "adamw_wd": adamw_wd,
                "train_device": device,
            },
            "sgld_config": {
                "eps_default": eps_def,
                "gamma_default": gamma_def,
                "nbeta_default": nbeta_def,
                "pool_seed": pool_seed,
                "pool_size": n_pool,
                "n_chains": n_chains_main,
                "part1_part2_burn_draws": [n_burn_main, n_draws_main],
                "part3_part4_burn_draws": [n_burn_exact, n_draws_exact],
            },
            "total_elapsed_s": round(total_elapsed, 2),
        },
        "part1_adam_wd0": {
            "runs": adam_runs,
            "stage_decompositions": adam_stages,
        },
        "part2_adamw": {
            "wd": adamw_wd,
            "runs": adamw_runs,
            "stage_decompositions": adamw_stages,
        },
        "part3_channel_ablation": part3,
        "part4_sgld_sensitivity": part4,
    }

    JSON_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved JSON to {JSON_PATH} (total runtime: {total_elapsed:.1f}s)", flush=True)
    plot_results(payload)


if __name__ == "__main__":
    main()
