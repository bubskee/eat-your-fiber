"""
Track A Empirical Study (Faithful 500k + Remedies):
"Is That LLC Drop Development, or Is the Estimator Sliding Along a Symmetry Fiber?"

Applies exact full-model iso-functional canonicalization (`ORBIT`: pure invertible group
actions preserving all 5 DTransformer output channels; `ORBIT+`: extra norm-reducing moves
reported separately), exact Gauss-Newton spectral analysis, and reparametrization-aware SGLD
remedies to Hoogland, Wang, Farrugia-Roberts, Carroll, Wei & Murfet (2024, TMLR,
arXiv:2402.02364), "Loss Landscape Degeneracy and Stagewise Development in Transformers" —
specifically their 51,717-parameter in-context linear regression Transformer (`L2H4Minf`,
`github.com/timaeus-research/icl` @ `5baf73025d279abd2d56c98e13bf5663608709c0`).

Recipe resolution (explicitly documented):
  - Paper Table 4 & App. F.2: K = 8, D = 4, sigma^2 = 0.125, L = 2, H = 4, d_embed = 64,
    d_mlp = 64 (51,717 params), Adam (wd = 0.0), OneCycleLR(max_lr = 0.003, pct_start = 0.5,
    anneal_strategy = 'linear'), T = 500,000 steps (LR peak at 250k), B = 256.
  - Public `sweeps/regression/training-runs/L2H4Minf.yaml` sets `max_examples: 16` and leaves
    optimizer/scheduler unspecified (`config.py` defaults: SGD, wd = 1e-4).
  - We follow the paper (K = 8, Adam wd = 0, 51,717 params) because `analysis/L2H4Minf/llcs.pt`
    has 8 token positions (0..7) and its MSE/LLC trajectory matches Adam (wd = 0).

CLI Modes (multi-GPU / multi-VM shardable):
  1. Fast end-to-end verification (~60s on GPU):
       python devinterp_fiber_decomposition.py --smoke --out-dir /tmp/smoke_test
  2. Train & save deterministic `.safetensors` checkpoints + `manifest.json`:
       python devinterp_fiber_decomposition.py --train-only --seeds 0,1,2 --optimizers adam,adamw
  3. Evaluate from saved checkpoints (shardable across N GPUs / VMs):
       python devinterp_fiber_decomposition.py --from-checkpoints --shard 0/4
       python devinterp_fiber_decomposition.py --from-checkpoints --shard 1/4
       ...
  4. Merge shards, compute GP stage boundaries & plot 8-panel figure:
       python devinterp_fiber_decomposition.py --merge-shards --n-shards 4
  5. Replot figure directly from saved JSON (< 2s):
       python devinterp_fiber_decomposition.py --replot
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacrev, vmap


def _get_safetensors():
    from safetensors.torch import load_file, save_file
    return load_file, save_file


EXP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXP_DIR / "results"
DEFAULT_CKPT_DIR = RESULTS_DIR / "devinterp_checkpoints"
DEFAULT_JSON_PATH = RESULTS_DIR / "devinterp_fiber_decomposition.json"
DEFAULT_PNG_PATH = RESULTS_DIR / "devinterp_fiber_decomposition.png"
HOOGLAND_REF_PATH = RESULTS_DIR / "hoogland_l2h4minf_reference.json"

UPSTREAM_ICL_COMMIT = "5baf73025d279abd2d56c98e13bf5663608709c0"

# Exact log-spaced checkpoint grid aligned to Hoogland et al.'s released `llcs.pt` (500k steps)
ADAM_CKPT_STEPS_500K = [
    0, 200, 507, 984, 1673, 2843, 5050, 10101, 20202,
    30303, 40292, 46003, 60606, 80808, 101010, 126262,
    161616, 202020, 252525, 318181, 363636, 424242, 500000,
]
ADAMW_CKPT_STEPS_500K = [
    0, 507, 984, 2843, 5050, 10101, 20202,
    40292, 46003, 80808, 126262, 202020, 252525, 363636, 500000,
]


# =============================================================================
# 1. Exact Replica of Hoogland et al. (2024) `L2H4Minf` (51,717 params)
#    Bit-for-bit verified by `experiments/verify_upstream_icl_parity.py`.
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
    def __init__(
        self,
        embed_size: int,
        mlp_size: int,
        max_tokens: int,
        num_heads: int,
        layer_norm: bool = True,
        include_output: bool = True,
    ):
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
    def __init__(
        self,
        token_size: int,
        max_tokens: int,
        embed_size: int,
        mlp_size: int,
        num_heads: int,
        num_layers: int,
        layer_norm: bool = True,
        include_output: bool = True,
    ):
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


def make_model_deterministic(seed: int, dtype: torch.dtype = torch.float32, device: str = "cpu") -> InContextRegressionTransformer:
    """Isolated deterministic initialization on CPU (zero process-global RNG leak).
    Guarantees identical initial weights across optimizers and compute devices."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10_000 + int(seed))
        m = make_model(dtype=torch.float32, device="cpu")
    return m.to(device=device, dtype=dtype)


# Canonical parameter key ordering (excludes non-parameter buffers like causal_mask)
PARAM_KEYS: list[str] = [k for k, _ in make_model().named_parameters()]


class BatchedDTransformerBMM(nn.Module):
    """Vectorized C-instance DTransformer (for C-chain SGLD or C-seed parallel training).
    Uses torch.bmm + fused F.layer_norm + math causal attention (avoids fmha_cutlass 64x64
    padding penalty on T=16). Verified to match DTransformer to < 1e-6 (f32) and < 1e-14 (f64)."""

    def __init__(
        self,
        sds: list[dict[str, torch.Tensor]],
        dtype: torch.dtype = torch.float32,
        device: str = "cuda",
    ):
        super().__init__()
        self.C = len(sds)
        self.H = 4
        self.dh = 16
        self.d = 64
        self.scale = 1.0 / (self.dh ** 0.5)
        self.register_buffer(
            "causal_mask",
            torch.log(torch.tril(torch.ones((16, 16), dtype=dtype, device=device))),
        )
        self.key_to_attr: list[tuple[str, str]] = []
        for k in PARAM_KEYS:
            attr = k.replace("token_sequence_transformer.", "").replace(".", "__")
            self.key_to_attr.append((k, attr))
            stacked = torch.stack([sd[k].to(device=device, dtype=dtype) for sd in sds], dim=0).contiguous()
            self.register_parameter(attr, nn.Parameter(stacked))

    @staticmethod
    def _ln(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (64,), eps=1e-5) * w.unsqueeze(1) + b.unsqueeze(1)

    def forward_toks(self, toks: torch.Tensor) -> torch.Tensor:
        # toks: (C, B, T, 5)
        C, B, T, _ = toks.shape
        N = B * T
        x = torch.bmm(toks.reshape(C, N, 5), self.token_embedding__weight.transpose(1, 2))
        pos = self.postn_embedding__weight.transpose(1, 2)[:, :T, :]
        x = (x.view(C, B, T, self.d) + pos.unsqueeze(1)).reshape(C, N, self.d)
        for l in range(2):
            ln0_w = getattr(self, f"blocks__{l}__layer_norms__0__weight")
            ln0_b = getattr(self, f"blocks__{l}__layer_norms__0__bias")
            qkv_w = getattr(self, f"blocks__{l}__attention__attention__weight")
            out_w = getattr(self, f"blocks__{l}__attention__output__weight")
            ln1_w = getattr(self, f"blocks__{l}__layer_norms__1__weight")
            ln1_b = getattr(self, f"blocks__{l}__layer_norms__1__bias")
            fc1_w = getattr(self, f"blocks__{l}__compute__0__weight")
            fc1_b = getattr(self, f"blocks__{l}__compute__0__bias")
            fc2_w = getattr(self, f"blocks__{l}__compute__2__weight")
            fc2_b = getattr(self, f"blocks__{l}__compute__2__bias")

            h = self._ln(x, ln0_w, ln0_b)
            qkv = torch.bmm(h, qkv_w.transpose(1, 2)).view(C * B, T, self.H, 3 * self.dh).transpose(1, 2)
            Q, K, V = qkv.split(self.dh, dim=-1)
            A = torch.matmul(Q, K.transpose(-2, -1)) * self.scale + self.causal_mask[:T, :T]
            p = torch.softmax(A, dim=-1)
            y = torch.matmul(p, V).transpose(1, 2).reshape(C, N, self.d)
            xb = x + torch.bmm(y, out_w.transpose(1, 2))
            h2 = self._ln(xb, ln1_w, ln1_b)
            mlp = F.relu(torch.bmm(h2, fc1_w.transpose(1, 2)) + fc1_b.unsqueeze(1))
            xb = xb + torch.bmm(mlp, fc2_w.transpose(1, 2)) + fc2_b.unsqueeze(1)
            x = x + xb

        xf = self._ln(x, self.unembedding__0__weight, self.unembedding__0__bias)
        out = torch.bmm(xf, self.unembedding__1__weight.transpose(1, 2)) + self.unembedding__1__bias.unsqueeze(1)
        return out.view(C, B, T, 5)

    def forward_mse_per_chain(self, toks: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
        """Fast regression MSE per chain/seed: reads out channel 0 at even tokens only."""
        C, B, T, _ = toks.shape
        N = B * T
        x = torch.bmm(toks.reshape(C, N, 5), self.token_embedding__weight.transpose(1, 2))
        pos = self.postn_embedding__weight.transpose(1, 2)[:, :T, :]
        x = (x.view(C, B, T, self.d) + pos.unsqueeze(1)).reshape(C, N, self.d)
        for l in range(2):
            ln0_w = getattr(self, f"blocks__{l}__layer_norms__0__weight")
            ln0_b = getattr(self, f"blocks__{l}__layer_norms__0__bias")
            qkv_w = getattr(self, f"blocks__{l}__attention__attention__weight")
            out_w = getattr(self, f"blocks__{l}__attention__output__weight")
            ln1_w = getattr(self, f"blocks__{l}__layer_norms__1__weight")
            ln1_b = getattr(self, f"blocks__{l}__layer_norms__1__bias")
            fc1_w = getattr(self, f"blocks__{l}__compute__0__weight")
            fc1_b = getattr(self, f"blocks__{l}__compute__0__bias")
            fc2_w = getattr(self, f"blocks__{l}__compute__2__weight")
            fc2_b = getattr(self, f"blocks__{l}__compute__2__bias")

            h = self._ln(x, ln0_w, ln0_b)
            qkv = torch.bmm(h, qkv_w.transpose(1, 2)).view(C * B, T, self.H, 3 * self.dh).transpose(1, 2)
            Q, K, V = qkv.split(self.dh, dim=-1)
            A = torch.matmul(Q, K.transpose(-2, -1)) * self.scale + self.causal_mask[:T, :T]
            p = torch.softmax(A, dim=-1)
            y = torch.matmul(p, V).transpose(1, 2).reshape(C, N, self.d)
            xb = x + torch.bmm(y, out_w.transpose(1, 2))
            h2 = self._ln(xb, ln1_w, ln1_b)
            mlp = F.relu(torch.bmm(h2, fc1_w.transpose(1, 2)) + fc1_b.unsqueeze(1))
            xb = xb + torch.bmm(mlp, fc2_w.transpose(1, 2)) + fc2_b.unsqueeze(1)
            x = x + xb

        xf = self._ln(x, self.unembedding__0__weight, self.unembedding__0__bias)
        xf_e = xf.view(C, B, T, self.d)[:, :, 0::2, :].reshape(C, B * 8, self.d)
        pred = torch.bmm(xf_e, self.unembedding__1__weight[:, 0:1, :].transpose(1, 2)) + self.unembedding__1__bias[:, 0:1].unsqueeze(1)
        return (pred.view(C, B, 8, 1) - yb).square().mean(dim=(1, 2, 3))

    def extract_state_dict(self, idx: int) -> dict[str, torch.Tensor]:
        """Extract single-instance CPU float32 state_dict (including causal_mask buffers)."""
        base = make_model(dtype=torch.float32, device="cpu").state_dict()
        for k, attr in self.key_to_attr:
            base[k] = getattr(self, attr)[idx].detach().float().cpu().clone()
        return base


# =============================================================================
# 2. Pure Group Actions (`ORBIT`) vs Extra Moves (`ORBIT+`) + Diffeomorphism Logs
# =============================================================================

def _svd_gl_balance(A: torch.Tensor, B: torch.Tensor, sv_thresh: float = 1e-4) -> tuple[torch.Tensor, torch.Tensor, float, float] | None:
    """
    Compute unique SPD G in GL(d_h) minimizing ||G A||_F^2 + ||G^{-1} B||_F^2
    (solving M (A A^T) M = (B B^T) with G = M^{1/2}) using square-root-free SVDs only
    (no squaring of condition numbers!).
    Returns (G, G_inv, abs_det_G, cond_G), or None if A or B is rank-deficient (< sv_thresh).
    """
    UA, sA, _ = torch.linalg.svd(A, full_matrices=False)
    _, sB, _ = torch.linalg.svd(B, full_matrices=False)
    if float(sA.min().item()) < sv_thresh or float(sB.min().item()) < sv_thresh:
        return None
    # P^{1/2} = UA diag(sA) UA^T, P^{-1/2} = UA diag(1/sA) UA^T
    P_half = (UA * sA.unsqueeze(0)) @ UA.T
    P_mhalf = (UA * (1.0 / sA).unsqueeze(0)) @ UA.T
    Z = P_half @ B
    UZ, sZ, _ = torch.linalg.svd(Z, full_matrices=False)
    if float(sZ.min().item()) < sv_thresh * sv_thresh:
        return None
    # M = P^{-1/2} (UZ diag(sZ) UZ^T) P^{-1/2} = R R^T where R = P^{-1/2} UZ diag(sqrt(sZ))
    R = P_mhalf @ (UZ * sZ.sqrt().unsqueeze(0))
    UR, sR, _ = torch.linalg.svd(R, full_matrices=False)
    if float(sR.min().item()) < 1e-4 or float((sR.max() / sR.min()).item()) > 1e4:
        return None
    G = (UR * sR.unsqueeze(0)) @ UR.T
    G_inv = (UR * (1.0 / sR).unsqueeze(0)) @ UR.T
    # Safety: verify monotone norm reduction
    old_n2 = float((A.pow(2).sum() + B.pow(2).sum()).item())
    new_n2 = float(((G @ A).pow(2).sum() + (G_inv @ B).pow(2).sum()).item())
    if new_n2 > old_n2 + 1e-6:
        return None
    abs_det = float(torch.exp(sR.log().sum()).item())
    cond = float((sR.max() / sR.min()).item())
    return G, G_inv, abs_det, cond


def _init_cert() -> dict[str, Any]:
    return {
        "min_s": float("inf"),
        "max_s": 0.0,
        "min_sv": float("inf"),
        "min_abs_det_G": float("inf"),
        "max_cond_G": 0.0,
        "n_skipped_degenerate": 0,
        "is_pure_diffeomorphism": True,
    }


def _update_scale_cert(cert: dict[str, Any], s: torch.Tensor) -> None:
    if s.numel() > 0:
        cert["min_s"] = min(cert["min_s"], float(s.min().item()))
        cert["max_s"] = max(cert["max_s"], float(s.max().item()))


@torch.no_grad()
def apply_symmetry_channel_(
    model: InContextRegressionTransformer,
    channel: str,
    n_iters: int = 20,
    scale_tracker: dict[str, torch.Tensor] | None = None,
    cert: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Apply an exact iso-functional symmetry transformation in-place (run in float64).

    Pure invertible parameter-space diffeomorphisms (preserving ALL 5 DTransformer output
    channels, ZERO zeroing branches):
      - `LN_mean_proj`, `QK`, `VO`, `MLP`, `LNf_unembed_all5`, `LN1_mlp`, `LN0_attn`,
        `LN_all_unit_gamma`, and joint `ORBIT`.

    Extra norm-reducing moves (preserve regression readout y_pred, NOT invertible on R^{51717}):
      - `LNf_unembed_row0`, `zero_unused_unembed`, and joint `ORBIT+`.
    """
    if cert is None:
        cert = _init_cert()
    dt = model.token_sequence_transformer
    eps_guard = 1e-15

    if channel == "LN_mean_proj":
        dt.token_embedding.weight.sub_(dt.token_embedding.weight.mean(dim=0, keepdim=True))
        dt.postn_embedding.weight.sub_(dt.postn_embedding.weight.mean(dim=0, keepdim=True))
        for blk in dt.blocks:
            blk.attention.output.weight.sub_(blk.attention.output.weight.mean(dim=0, keepdim=True))
            blk.compute[2].weight.sub_(blk.compute[2].weight.mean(dim=0, keepdim=True))
            blk.compute[2].bias.sub_(blk.compute[2].bias.mean())

    elif channel == "QK":
        # Exact invertible per-head GL(d_h) balancing: W_Q' = G W_Q, W_K' = G^{-T} W_K (G SPD => G^{-T}=G^{-1})
        for blk in dt.blocks:
            att = blk.attention
            H, dh, C_emb = att.num_heads, att.head_size, att.embed_size
            W = att.attention.weight.view(H, 3, dh, C_emb)
            for h in range(H):
                WQ, WK = W[h, 0], W[h, 1]
                sv_q = float(torch.linalg.svdvals(WQ).min().item())
                sv_k = float(torch.linalg.svdvals(WK).min().item())
                cert["min_sv"] = min(cert["min_sv"], sv_q, sv_k)
                res_gl = _svd_gl_balance(WQ, WK)
                if res_gl is None:
                    cert["n_skipped_degenerate"] += 1
                    continue
                G, G_inv, abs_det, cond = res_gl
                cert["min_abs_det_G"] = min(cert["min_abs_det_G"], abs_det)
                cert["max_cond_G"] = max(cert["max_cond_G"], cond)
                W[h, 0].copy_(G @ WQ)
                W[h, 1].copy_(G_inv @ WK)

    elif channel == "VO":
        # Exact invertible per-head GL(d_h) balancing: W_V' = G W_V, W_O' = W_O G^{-1}
        for blk in dt.blocks:
            att = blk.attention
            H, dh, C_emb = att.num_heads, att.head_size, att.embed_size
            W = att.attention.weight.view(H, 3, dh, C_emb)
            WO = att.output.weight.view(C_emb, H, dh)
            for h in range(H):
                WV = W[h, 2]
                WOh = WO[:, h, :]
                sv_v = float(torch.linalg.svdvals(WV).min().item())
                sv_o = float(torch.linalg.svdvals(WOh).min().item())
                cert["min_sv"] = min(cert["min_sv"], sv_v, sv_o)
                res_gl = _svd_gl_balance(WV, WOh.T)
                if res_gl is None:
                    cert["n_skipped_degenerate"] += 1
                    continue
                G, G_inv, abs_det, cond = res_gl
                cert["min_abs_det_G"] = min(cert["min_abs_det_G"], abs_det)
                cert["max_cond_G"] = max(cert["max_cond_G"], cond)
                W[h, 2].copy_(G @ WV)
                WO[:, h, :].copy_(WOh @ G_inv)

    elif channel in ("MLP", "MLP_scale_only"):
        # Strictly positive neuronwise ReLU rescaling s_j > 0 (ZERO zeroing branches!)
        for l_idx, blk in enumerate(dt.blocks):
            fc1, fc2 = blk.compute[0], blk.compute[2]
            W1, b1, W2 = fc1.weight, fc1.bias, fc2.weight
            a = torch.sqrt(W1.pow(2).sum(dim=1) + b1.pow(2))
            c = W2.pow(2).sum(dim=0).sqrt()
            valid = (a > eps_guard) & (c > eps_guard)
            cert["n_skipped_degenerate"] += int((~valid).sum().item())
            s = torch.where(valid, torch.sqrt(c.clamp(min=eps_guard) / a.clamp(min=eps_guard)), torch.ones_like(a))
            _update_scale_cert(cert, s)
            W1.mul_(s[:, None])
            b1.mul_(s)
            W2.div_(s[None, :])
            if scale_tracker is not None:
                scale_tracker[f"token_sequence_transformer.blocks.{l_idx}.compute.0.weight"].mul_(s[:, None])
                scale_tracker[f"token_sequence_transformer.blocks.{l_idx}.compute.0.bias"].mul_(s)
                scale_tracker[f"token_sequence_transformer.blocks.{l_idx}.compute.2.weight"].div_(s[None, :])
            if channel == "MLP":
                prod = a * c
                order = torch.argsort(prod, descending=True)
                W1.copy_(W1[order])
                b1.copy_(b1[order])
                W2.copy_(W2[:, order])

    elif channel == "LNf_unembed_all5":
        # Pure diffeomorphism: absorb lnf.bias into ALL 5 unembedding biases and balance
        # gamma_f[i] against ALL 5 rows of unembedding.weight[:, i] with strictly positive s_i > 0!
        lnf, un = dt.unembedding[0], dt.unembedding[1]
        un.bias.add_(un.weight @ lnf.bias)
        lnf.bias.zero_()
        g = lnf.weight.abs()
        col_n = un.weight.pow(2).sum(dim=0).sqrt()
        valid = (g > eps_guard) & (col_n > eps_guard)
        cert["n_skipped_degenerate"] += int((~valid).sum().item())
        s = torch.where(valid, torch.sqrt(col_n.clamp(min=eps_guard) / g.clamp(min=eps_guard)), torch.ones_like(g))
        _update_scale_cert(cert, s)
        lnf.weight.mul_(s)
        un.weight.div_(s[None, :])
        if scale_tracker is not None:
            scale_tracker["token_sequence_transformer.unembedding.0.weight"].mul_(s)
            scale_tracker["token_sequence_transformer.unembedding.1.weight"].div_(s[None, :])

    elif channel == "LN1_mlp":
        # Pure diffeomorphism: exact bias absorption + strictly positive coordinatewise rescaling s_i > 0
        for l_idx, blk in enumerate(dt.blocks):
            ln1, fc1 = blk.layer_norms[1], blk.compute[0]
            fc1.bias.add_(fc1.weight @ ln1.bias)
            ln1.bias.zero_()
            g = ln1.weight.abs()
            col_n = fc1.weight.pow(2).sum(dim=0).sqrt()
            valid = (g > eps_guard) & (col_n > eps_guard)
            cert["n_skipped_degenerate"] += int((~valid).sum().item())
            s = torch.where(valid, torch.sqrt(col_n.clamp(min=eps_guard) / g.clamp(min=eps_guard)), torch.ones_like(g))
            _update_scale_cert(cert, s)
            ln1.weight.mul_(s)
            fc1.weight.div_(s[None, :])
            if scale_tracker is not None:
                scale_tracker[f"token_sequence_transformer.blocks.{l_idx}.layer_norms.1.weight"].mul_(s)
                scale_tracker[f"token_sequence_transformer.blocks.{l_idx}.compute.0.weight"].div_(s[None, :])

    elif channel == "LN0_attn":
        # Pure diffeomorphism: simultaneous strictly positive scaling of (gamma_0[i], beta_0[i]) by s_i
        # and W_QKV[:, i] by 1/s_i (exact even without attention bias!).
        for l_idx, blk in enumerate(dt.blocks):
            ln0, qkv = blk.layer_norms[0], blk.attention.attention
            gn = torch.sqrt(ln0.weight.pow(2) + ln0.bias.pow(2))
            col_n = qkv.weight.pow(2).sum(dim=0).sqrt()
            valid = (gn > eps_guard) & (col_n > eps_guard)
            cert["n_skipped_degenerate"] += int((~valid).sum().item())
            s = torch.where(valid, torch.sqrt(col_n.clamp(min=eps_guard) / gn.clamp(min=eps_guard)), torch.ones_like(gn))
            _update_scale_cert(cert, s)
            ln0.weight.mul_(s)
            ln0.bias.mul_(s)
            qkv.weight.div_(s[None, :])
            if scale_tracker is not None:
                scale_tracker[f"token_sequence_transformer.blocks.{l_idx}.layer_norms.0.weight"].mul_(s)
                scale_tracker[f"token_sequence_transformer.blocks.{l_idx}.layer_norms.0.bias"].mul_(s)
                scale_tracker[f"token_sequence_transformer.blocks.{l_idx}.attention.attention.weight"].div_(s[None, :])

    elif channel == "LN_all_unit_gamma":
        # Pure-diffeomorphism control gauge: ||gamma||=1 on biased LNs (all 5 rows on lnf!),
        # ||(gamma, beta)||=1 on biasless LN0. Strictly positive rescaling only.
        lnf, un = dt.unembedding[0], dt.unembedding[1]
        un.bias.add_(un.weight @ lnf.bias)
        lnf.bias.zero_()
        g = lnf.weight.abs()
        valid = g > eps_guard
        s = torch.where(valid, 1.0 / g.clamp(min=eps_guard), torch.ones_like(g))
        _update_scale_cert(cert, s)
        lnf.weight.mul_(s)
        un.weight.div_(s[None, :])
        for blk in dt.blocks:
            ln1, fc1 = blk.layer_norms[1], blk.compute[0]
            fc1.bias.add_(fc1.weight @ ln1.bias)
            ln1.bias.zero_()
            g1 = ln1.weight.abs()
            v1 = g1 > eps_guard
            s1 = torch.where(v1, 1.0 / g1.clamp(min=eps_guard), torch.ones_like(g1))
            _update_scale_cert(cert, s1)
            ln1.weight.mul_(s1)
            fc1.weight.div_(s1[None, :])

            ln0, qkv = blk.layer_norms[0], blk.attention.attention
            gn = torch.sqrt(ln0.weight.pow(2) + ln0.bias.pow(2))
            v0 = gn > eps_guard
            s0 = torch.where(v0, 1.0 / gn.clamp(min=eps_guard), torch.ones_like(gn))
            _update_scale_cert(cert, s0)
            ln0.weight.mul_(s0)
            ln0.bias.mul_(s0)
            qkv.weight.div_(s0[None, :])

    elif channel == "ORBIT":
        # Pure exact invertible loss-preserving group actions ONLY (diffeomorphism theorem applies!)
        apply_symmetry_channel_(model, "LN_mean_proj", cert=cert)
        for it in range(n_iters):
            apply_symmetry_channel_(model, "LNf_unembed_all5", scale_tracker=scale_tracker, cert=cert)
            apply_symmetry_channel_(model, "LN1_mlp", scale_tracker=scale_tracker, cert=cert)
            apply_symmetry_channel_(model, "LN0_attn", scale_tracker=scale_tracker, cert=cert)
            apply_symmetry_channel_(model, "QK", cert=cert)
            apply_symmetry_channel_(model, "VO", cert=cert)
            apply_symmetry_channel_(
                model,
                "MLP" if it == n_iters - 1 else "MLP_scale_only",
                scale_tracker=scale_tracker,
                cert=cert,
            )

    elif channel == "ORBIT_DIAG":
        # Pure diagonal-rescaling subset of ORBIT (exact diagonal map corresponding to orbit_covariant metric)
        apply_symmetry_channel_(model, "LN_mean_proj", cert=cert)
        for _ in range(n_iters):
            apply_symmetry_channel_(model, "LNf_unembed_all5", scale_tracker=scale_tracker, cert=cert)
            apply_symmetry_channel_(model, "LN1_mlp", scale_tracker=scale_tracker, cert=cert)
            apply_symmetry_channel_(model, "LN0_attn", scale_tracker=scale_tracker, cert=cert)
            apply_symmetry_channel_(model, "MLP_scale_only", scale_tracker=scale_tracker, cert=cert)

    elif channel == "ORBIT+":
        # Extra norm-reducing moves (NOT a pure parameter-space diffeomorphism; reported separately!)
        # 1. Start from pure ORBIT
        apply_symmetry_channel_(model, "ORBIT", n_iters=n_iters, cert=cert)
        cert["is_pure_diffeomorphism"] = False
        # 2. Zero the 4 unused unembedding rows & biases (regression readout uses channel 0 only)
        lnf, un = dt.unembedding[0], dt.unembedding[1]
        un.weight[1:].zero_()
        un.bias[1:].zero_()
        # 3. Re-balance lnf against readout row 0 only (still strictly positive s_i > 0)
        w0 = un.weight[0]
        g = lnf.weight.abs()
        w0a = w0.abs()
        valid = (g > eps_guard) & (w0a > eps_guard)
        s = torch.where(valid, torch.sqrt(w0a.clamp(min=eps_guard) / g.clamp(min=eps_guard)), torch.ones_like(g))
        _update_scale_cert(cert, s)
        lnf.weight.mul_(s)
        w0.div_(s)
        # 4. Polish remaining exact symmetries
        for _ in range(5):
            apply_symmetry_channel_(model, "LN1_mlp", cert=cert)
            apply_symmetry_channel_(model, "LN0_attn", cert=cert)
            apply_symmetry_channel_(model, "QK", cert=cert)
            apply_symmetry_channel_(model, "VO", cert=cert)
            apply_symmetry_channel_(model, "MLP_scale_only", cert=cert)

    else:
        raise ValueError(f"Unknown symmetry channel: {channel}")

    return cert


@torch.no_grad()
def compute_orbit_diagonal_scales(sd_raw: dict[str, torch.Tensor], n_iters: int = 20) -> dict[str, torch.Tensor]:
    """
    Compute the exact cumulative positive coordinatewise scaling factors c_p(w*) > 0
    mapping w* to the diagonal-balanced representative along PURE-MULTIPLICATIVE exact
    symmetries (NO bias absorption, so w* stays untouched and lnf/ln1 biases scale
    jointly with gamma!):
      - All 5 LayerNorms (blk{0,1}.ln0, blk{0,1}.ln1, lnf):
          (gamma_i, beta_i) -> s_i (gamma_i, beta_i),  W_down[:, i] -> s_i^{-1} W_down[:, i]
      - Both ReLU MLPs:
          (W1[j,:], b1[j]) -> s_j (W1[j,:], b1[j]),    W2[:, j] -> s_j^{-1} W2[:, j]
    Setting metric weights d_p = c_p^2 yields an exact linear orbit-covariant metric D(w*)
    such that D(w*)^{1/2}(w - w*) is invariant along all coordinatewise rescalings!
    """
    m = make_model(dtype=torch.float64, device="cpu")
    m.load_state_dict({k: v.double() for k, v in sd_raw.items()})
    sc = {k: torch.ones_like(v, dtype=torch.float64) for k, v in m.named_parameters()}
    dt = m.token_sequence_transformer
    eps_guard = 1e-15

    for _ in range(n_iters):
        ln_pairs = [
            (dt.unembedding[0], dt.unembedding[1], "token_sequence_transformer.unembedding.0", "token_sequence_transformer.unembedding.1.weight"),
        ]
        for l_i, blk in enumerate(dt.blocks):
            ln_pairs.append((
                blk.layer_norms[0], blk.attention.attention,
                f"token_sequence_transformer.blocks.{l_i}.layer_norms.0",
                f"token_sequence_transformer.blocks.{l_i}.attention.attention.weight",
            ))
            ln_pairs.append((
                blk.layer_norms[1], blk.compute[0],
                f"token_sequence_transformer.blocks.{l_i}.layer_norms.1",
                f"token_sequence_transformer.blocks.{l_i}.compute.0.weight",
            ))
        for ln, lin, ln_pref, lin_key in ln_pairs:
            gn = torch.sqrt(ln.weight.pow(2) + ln.bias.pow(2))
            cn = lin.weight.pow(2).sum(dim=0).sqrt()
            valid = (gn > eps_guard) & (cn > eps_guard)
            s = torch.where(valid, torch.sqrt(cn.clamp(min=eps_guard) / gn.clamp(min=eps_guard)), torch.ones_like(gn))
            ln.weight.mul_(s)
            ln.bias.mul_(s)
            lin.weight.div_(s[None, :])
            sc[f"{ln_pref}.weight"].mul_(s)
            sc[f"{ln_pref}.bias"].mul_(s)
            sc[lin_key].div_(s[None, :])

        for l_i, blk in enumerate(dt.blocks):
            fc1, fc2 = blk.compute[0], blk.compute[2]
            a = torch.sqrt(fc1.weight.pow(2).sum(dim=1) + fc1.bias.pow(2))
            c = fc2.weight.pow(2).sum(dim=0).sqrt()
            valid = (a > eps_guard) & (c > eps_guard)
            s = torch.where(valid, torch.sqrt(c.clamp(min=eps_guard) / a.clamp(min=eps_guard)), torch.ones_like(a))
            fc1.weight.mul_(s[:, None])
            fc1.bias.mul_(s)
            fc2.weight.div_(s[None, :])
            sc[f"token_sequence_transformer.blocks.{l_i}.compute.0.weight"].mul_(s[:, None])
            sc[f"token_sequence_transformer.blocks.{l_i}.compute.0.bias"].mul_(s)
            sc[f"token_sequence_transformer.blocks.{l_i}.compute.2.weight"].div_(s[None, :])

    return {k: v.float().cpu() for k, v in sc.items()}


_INIT_TENSOR_RMS_CACHE: dict[str, float] | None = None


def compute_param_rms_scales(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Black-box architecture-agnostic per-tensor parameter-RMS covariant metric:
    for each weight tensor T, c_T = init_rms(T) / max(rms(w_T*), 0.1 * init_rms(T)),
    and companion LayerNorm biases share c_T (linear biases use 1.0).
    Requires ZERO knowledge of model symmetries; cancels layer-wise rescaling blowups!
    """
    global _INIT_TENSOR_RMS_CACHE
    if _INIT_TENSOR_RMS_CACHE is None:
        sd0 = make_model_deterministic(0).state_dict()
        _INIT_TENSOR_RMS_CACHE = {k: float(sd0[k].pow(2).mean().sqrt().item()) for k in PARAM_KEYS}
    sc: dict[str, torch.Tensor] = {}
    for k in PARAM_KEYS:
        if k.endswith(".weight"):
            r0 = _INIT_TENSOR_RMS_CACHE[k]
            r = float(sd[k].pow(2).mean().sqrt().item())
            c_val = r0 / max(r, 0.1 * r0)
            sc[k] = torch.full_like(sd[k], c_val, dtype=torch.float32)
        else:
            wk = k.replace(".bias", ".weight")
            if ("layer_norms" in k or "unembedding.0" in k) and wk in sc:
                sc[k] = torch.full_like(sd[k], float(sc[wk].flatten()[0].item()), dtype=torch.float32)
            else:
                sc[k] = torch.ones_like(sd[k], dtype=torch.float32)
    return sc


def _clean_cert(cert: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in cert.items():
        if isinstance(v, float) and (math.isinf(v) or math.isnan(v)):
            out[k] = None
        else:
            out[k] = v
    return out


@torch.no_grad()
def canonicalize_state_dict(
    sd_raw: dict[str, torch.Tensor],
    x_check64: torch.Tensor,
    y_check64: torch.Tensor,
    channel: str = "ORBIT",
) -> dict[str, Any]:
    """
    Canonicalize `sd_raw` in float64 and verify exactness on `x_check64` (512 fresh sequences)
    BOTH in float64 (before cast) AND in float32 (after casting canonical weights to float32,
    the exact weights SGLD evaluates!), for both readout channel 0 and all 5 DTransformer channels.
    """
    m_raw64 = make_model(dtype=torch.float64, device="cpu")
    m_raw64.load_state_dict({k: v.double() for k, v in sd_raw.items()})
    toks64 = to_token_sequence(x_check64, y_check64)
    all5_ref64 = m_raw64.token_sequence_transformer(toks64)
    y_ref64 = from_predicted_token_sequence(all5_ref64)
    norm_raw = math.sqrt(sum(p.pow(2).sum().item() for p in m_raw64.parameters()))

    m_can64 = copy.deepcopy(m_raw64)
    cert = _init_cert()
    if channel != "RAW":
        cert = apply_symmetry_channel_(m_can64, channel, cert=cert)

    all5_can64 = m_can64.token_sequence_transformer(toks64)
    y_can64 = from_predicted_token_sequence(all5_can64)
    max_dy_f64 = float((y_can64 - y_ref64).abs().max().item())
    max_all5_f64 = float((all5_can64 - all5_ref64).abs().max().item())
    norm_can = math.sqrt(sum(p.pow(2).sum().item() for p in m_can64.parameters()))

    # Cast to float32 and measure POST-float32 discrepancy against raw float32 model!
    sd_can_f32 = {k: v.float().cpu() for k, v in m_can64.state_dict().items()}
    x32, y32 = x_check64.float(), y_check64.float()
    toks32 = to_token_sequence(x32, y32)
    m_raw32 = make_model(dtype=torch.float32, device="cpu")
    m_raw32.load_state_dict({k: v.float() for k, v in sd_raw.items()})
    m_can32 = make_model(dtype=torch.float32, device="cpu")
    m_can32.load_state_dict(sd_can_f32)

    all5_ref32 = m_raw32.token_sequence_transformer(toks32)
    y_ref32 = from_predicted_token_sequence(all5_ref32)
    all5_can32 = m_can32.token_sequence_transformer(toks32)
    y_can32 = from_predicted_token_sequence(all5_can32)
    max_dy_f32 = float((y_can32 - y_ref32).abs().max().item())
    max_all5_f32 = float((all5_can32 - all5_ref32).abs().max().item())

    return {
        "sd": sd_can_f32,
        "norm_raw": norm_raw,
        "norm_can": norm_can,
        "max_delta_y_f64": max_dy_f64,
        "max_delta_y_f32": max_dy_f32,
        "max_delta_all5_f64": max_all5_f64,
        "max_delta_all5_f32": max_all5_f32,
        "diffeomorphism_cert": _clean_cert(cert),
    }


# =============================================================================
# 3. Structural, Behavioral, OOD & Exact Gauss-Newton Diagnostics
# =============================================================================

@torch.no_grad()
def evaluate_checkpoint_metrics(
    sd: dict[str, torch.Tensor],
    x_ev: torch.Tensor,
    y_ev: torch.Tensor,
    w_ev: torch.Tensor,
    e_ev: torch.Tensor,
    device: str = "cpu",
) -> dict[str, Any]:
    m = make_model(dtype=torch.float32, device=device)
    m.load_state_dict({k: v.to(device) for k, v in sd.items()})
    m.eval()

    x_d, y_d, w_d, e_d = x_ev.to(device), y_ev.to(device), w_ev.to(device), e_ev.to(device)
    yp = m(x_d, y_d)
    mse_tok = (yp - y_d).pow(2).mean(dim=(0, 2))
    mse_id = float(mse_tok.mean().item())
    y_hat_sq = float(yp.pow(2).mean().item())
    icl_0_to_7 = float((mse_tok[-1] - mse_tok[0]).item())
    icl_D_to_K = float((mse_tok[-1] - mse_tok[3]).item())

    ood = {}
    for g in (3, 5):
        xg = x_d * float(g)
        yg = torch.bmm(xg, w_d.unsqueeze(-1)) + e_d
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
        ("lnf_row0", dt.unembedding[0].weight,           dt.unembedding[0].bias,           dt.unembedding[1].weight[0:1]),
        ("lnf_all5", dt.unembedding[0].weight,           dt.unembedding[0].bias,           dt.unembedding[1].weight),
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


def compute_gauss_newton_spectrum(
    sd: dict[str, torch.Tensor],
    x_gn: torch.Tensor,
    y_gn: torch.Tensor,
    gamma: float = 13.3,
    nbeta: float = 66.7,
    device: str = "cuda",
    chunk_size: int = 64,
) -> dict[str, Any]:
    """
    Compute the EXACT nonzero Gauss-Newton Hessian spectrum G = (2 / N_res) J^T J of the
    MSE loss L(w) = (1 / N_res) sum (f_w(x) - y)^2 on N_seq = 512 held-out sequences
    (N_res = 4,096 scalar residuals), using dual Gram matrices J J^T in R^{4096 x 4096}
    (which share identical nonzero eigenvalues with J^T J in R^{51717 x 51717})!

    Reports exact spectra & Gaussian-Laplace effective dimensions
        hat_lambda_GN(gamma) = 0.5 * sum_k (n*beta * mu_k) / (n*beta * mu_k + gamma)
    for: full_model (51,717), W_QKV (24,576), LN_0 (128), W_fc1 (8,192), LN_1 (128).
    """
    m = make_model(dtype=torch.float32, device=device)
    m.load_state_dict({k: v.to(device) for k, v in sd.items()})
    m.eval()

    params_d = dict(m.named_parameters())
    buffers_d = dict(m.named_buffers())
    toks = to_token_sequence(x_gn.to(device), y_gn.to(device))
    n_seq = toks.shape[0]
    n_res = n_seq * 8

    def f_single(p, b, tok):
        out = functional_call(m.token_sequence_transformer, (p, b), (tok.unsqueeze(0),))
        return out[0, 0::2, 0]

    dt_params = {k.replace("token_sequence_transformer.", ""): v for k, v in params_d.items()}
    dt_buffers = {k.replace("token_sequence_transformer.", ""): v for k, v in buffers_d.items()}

    blocks_spec = {
        "full_model": list(dt_params.keys()),
        "W_QKV": ["blocks.0.attention.attention.weight", "blocks.1.attention.attention.weight"],
        "LN_0": ["blocks.0.layer_norms.0.weight", "blocks.1.layer_norms.0.weight"],
        "W_fc1": ["blocks.0.compute.0.weight", "blocks.1.compute.0.weight"],
        "LN_1": ["blocks.0.layer_norms.1.weight", "blocks.1.layer_norms.1.weight"],
    }
    J_chunks: dict[str, list[torch.Tensor]] = {bname: [] for bname in blocks_spec}

    for i in range(0, n_seq, chunk_size):
        tb = toks[i : i + chunk_size]
        bs = tb.shape[0]
        jac = vmap(jacrev(f_single), in_dims=(None, None, 0))(dt_params, dt_buffers, tb)
        for bname, keys in blocks_spec.items():
            J_chunks[bname].append(torch.cat([jac[k].reshape(bs * 8, -1) for k in keys], dim=-1))

    tau = gamma / nbeta
    out: dict[str, Any] = {
        "n_sequences": n_seq,
        "n_residuals": n_res,
        "gamma": gamma,
        "nbeta": nbeta,
        "threshold_gamma_over_nbeta": tau,
        "blocks": {},
    }

    for bname in blocks_spec:
        J = torch.cat(J_chunks[bname], dim=0).detach().double() # (n_res, P_block) in float64 for clean eigvalsh
        P_block = J.shape[1]
        if P_block <= n_res:
            G = (2.0 / n_res) * (J.T @ J)
        else:
            G = (2.0 / n_res) * (J @ J.T)
        ev = torch.linalg.eigvalsh(G).clamp(min=0.0).detach().cpu().numpy()[::-1] # descending!

        def laplace_ed(g_val: float) -> float:
            return float(0.5 * np.sum((nbeta * ev) / (nbeta * ev + g_val)))

        out["blocks"][bname] = {
            "n_params": int(P_block),
            "trace": float(ev.sum()),
            "max_eig": float(ev[0]),
            "n_gt_tau": int((ev > tau).sum()),
            "n_gt_0p1_tau": int((ev > 0.1 * tau).sum()),
            "n_gt_1e4": int((ev > 1e-4).sum()),
            "n_between_1e4_and_tau": int(((ev > 1e-4) & (ev <= tau)).sum()),
            "laplace_ed_g3p3": laplace_ed(3.3),
            "laplace_ed_g13p3": laplace_ed(13.3),
            "laplace_ed_g40p0": laplace_ed(40.0),
            "top_20_eigs": [float(x) for x in ev[:20]],
            "top_512_eigs": [float(x) for x in ev[: min(512, len(ev))]],
        }
        del J, G

    return out


# =============================================================================
# 4. Deterministic Evaluation Pool & Batched GPU SGLD + Remedies
# =============================================================================

def generate_sgld_pool(pool_seed: int, n_pool: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate fixed evaluation pool deterministically on CPU in chunks so the exact
    pool is bit-for-bit identical across all hardware/devices, then move to `device`."""
    sigma = math.sqrt(0.125)
    g = torch.Generator(device="cpu").manual_seed(pool_seed)
    chunk = min(n_pool, 65536)
    toks_chunks, y_chunks = [], []
    rem = n_pool
    while rem > 0:
        cur = min(chunk, rem)
        w = torch.randn(cur, 4, generator=g)
        x = torch.randn(cur, 8, 4, generator=g)
        y = torch.bmm(x, w.unsqueeze(-1)) + torch.randn(cur, 8, 1, generator=g) * sigma
        toks_chunks.append(to_token_sequence(x, y))
        y_chunks.append(y)
        rem -= cur
    toks_pool = torch.cat(toks_chunks, dim=0).to(device=device, non_blocking=True)
    y_pool = torch.cat(y_chunks, dim=0).to(device=device, non_blocking=True)
    return toks_pool, y_pool


@torch.no_grad()
def compute_pool_init_loss(
    sd: dict[str, torch.Tensor],
    toks_pool: torch.Tensor,
    y_pool: torch.Tensor,
    device: str,
) -> float:
    m = make_model(dtype=torch.float32, device=device)
    m.load_state_dict({k: v.to(device) for k, v in sd.items()})
    m.eval()
    n_pool = toks_pool.shape[0]
    bs = 8192
    tot = 0.0
    for i in range(0, n_pool, bs):
        pred = from_predicted_token_sequence(m.token_sequence_transformer(toks_pool[i : i + bs]))
        tot += F.mse_loss(pred, y_pool[i : i + bs], reduction="sum").item()
    return float(tot / (n_pool * 8))


def _stateless_batched_mse_per_chain(
    params: tuple[torch.Tensor, ...],
    causal_mask: torch.Tensor,
    toks: torch.Tensor,
    yb: torch.Tensor,
) -> torch.Tensor:
    """Stateless functional forward pass so torch.compile compiles ONCE per shape (C, B)
    and NEVER hits dynamo recompile limits across hundreds of checkpoints!"""
    # Unpack in exact PARAM_KEYS order (24 tensors):
    # 0: token_embedding.weight, 1: postn_embedding.weight,
    # 2..11: block 0 (attn.attn.w, attn.out.w, fc1.w, fc1.b, fc2.w, fc2.b, ln0.w, ln0.b, ln1.w, ln1.b),
    # 12..21: block 1 (same order),
    # 22: unembedding.0.weight, 23: unembedding.0.bias, 24: unembedding.1.weight, 25: unembedding.1.bias
    C, B, T, _ = toks.shape
    N = B * T
    tok_w, pos_w = params[0], params[1]
    x = torch.bmm(toks.reshape(C, N, 5), tok_w.transpose(1, 2))
    pos = pos_w.transpose(1, 2)[:, :T, :]
    x = (x.view(C, B, T, 64) + pos.unsqueeze(1)).reshape(C, N, 64)
    scale = 0.25
    for l in range(2):
        off = 2 + l * 10
        qkv_w, out_w = params[off], params[off + 1]
        fc1_w, fc1_b = params[off + 2], params[off + 3]
        fc2_w, fc2_b = params[off + 4], params[off + 5]
        ln0_w, ln0_b = params[off + 6], params[off + 7]
        ln1_w, ln1_b = params[off + 8], params[off + 9]

        h = F.layer_norm(x, (64,), eps=1e-5) * ln0_w.unsqueeze(1) + ln0_b.unsqueeze(1)
        qkv = torch.bmm(h, qkv_w.transpose(1, 2)).view(C * B, T, 4, 48).transpose(1, 2)
        Q, K, V = qkv.split(16, dim=-1)
        A = torch.matmul(Q, K.transpose(-2, -1)) * scale + causal_mask[:T, :T]
        p = torch.softmax(A, dim=-1)
        y = torch.matmul(p, V).transpose(1, 2).reshape(C, N, 64)
        xb = x + torch.bmm(y, out_w.transpose(1, 2))
        h2 = F.layer_norm(xb, (64,), eps=1e-5) * ln1_w.unsqueeze(1) + ln1_b.unsqueeze(1)
        mlp = F.relu(torch.bmm(h2, fc1_w.transpose(1, 2)) + fc1_b.unsqueeze(1))
        xb = xb + torch.bmm(mlp, fc2_w.transpose(1, 2)) + fc2_b.unsqueeze(1)
        x = x + xb

    lnf_w, lnf_b, un_w, un_b = params[22], params[23], params[24], params[25]
    xf = F.layer_norm(x, (64,), eps=1e-5) * lnf_w.unsqueeze(1) + lnf_b.unsqueeze(1)
    xf_e = xf.view(C, B, T, 64)[:, :, 0::2, :].reshape(C, B * 8, 64)
    pred = torch.bmm(xf_e, un_w[:, 0:1, :].transpose(1, 2)) + un_b[:, 0:1].unsqueeze(1)
    return (pred.view(C, B, 8, 1) - yb).square().mean(dim=(1, 2, 3))


_COMPILED_STATELESS_FWD: Any = None


def _get_fwd_fn(bm: BatchedDTransformerBMM, use_compile: bool):
    global _COMPILED_STATELESS_FWD
    if use_compile and bm.token_embedding__weight.is_cuda:
        if _COMPILED_STATELESS_FWD is None:
            torch._dynamo.config.cache_size_limit = 64
            _COMPILED_STATELESS_FWD = torch.compile(_stateless_batched_mse_per_chain)
        params_tuple = tuple(bm.parameters())
        cm = bm.causal_mask
        return lambda toks, yb: _COMPILED_STATELESS_FWD(params_tuple, cm, toks, yb)
    return bm.forward_mse_per_chain


def run_sgld_multi_variant(
    sds_variants: list[dict[str, torch.Tensor]],
    toks_pool: torch.Tensor,
    y_pool: torch.Tensor,
    init_loss: float,
    *,
    estimator: str = "vanilla",
    n_chains: int = 10,
    n_burn: int = 1000,
    n_draws: int = 4000,
    eps: float = 0.0003,
    gamma: float = 13.3,
    nbeta: float = 66.7,
    m_batch: int = 1024,
    full_batch_n: int | None = None,
    rms_alpha: float = 0.99,
    rms_a: float = 0.01,
    seed: int = 42,
    want_windows: bool = False,
    device: str = "cuda",
    use_compile: bool = True,
) -> list[dict[str, Any]]:
    """
    Super-batched multi-variant SGLD: runs `V = len(sds_variants)` variants x `n_chains` chains
    simultaneously in a single `BatchedDTransformerBMM(C = V * n_chains)` pass (~20% faster on
    wide GPUs like RTX PRO 6000 Blackwell), using matched per-chain minibatch order and Langevin
    noise across variants for paired variance reduction!
    """
    t0 = time.perf_counter()
    V = len(sds_variants)
    C_tot = V * n_chains
    n_total = n_burn + n_draws
    stacked_sds = [sd for sd in sds_variants for _ in range(n_chains)]
    bm = BatchedDTransformerBMM(stacked_sds, dtype=torch.float32, device=device)
    fwd = _get_fwd_fn(bm, use_compile=use_compile)
    params = list(bm.parameters())
    w0 = [p.detach().clone() for p in params]

    D_inv_list: list[torch.Tensor] | None = None
    D_mhalf_list: list[torch.Tensor] | None = None
    D_half_list: list[torch.Tensor] | None = None
    if estimator in ("orbit_covariant", "param_rms_covariant"):
        scale_fn = compute_orbit_diagonal_scales if estimator == "orbit_covariant" else compute_param_rms_scales
        scale_maps = [scale_fn(sd) for sd in sds_variants]
        D_inv_list, D_mhalf_list, D_half_list = [], [], []
        for k, _ in bm.key_to_attr:
            c_chunks = [sm[k].to(device=device, dtype=torch.float32).unsqueeze(0).expand(n_chains, *sm[k].shape) for sm in scale_maps]
            c = torch.cat(c_chunks, dim=0).contiguous()
            D_half_list.append(c)
            D_inv_list.append(1.0 / (c * c))
            D_mhalf_list.append(1.0 / c)

    v_ema: list[torch.Tensor] | None = None
    if estimator in ("rmsprop_psgld", "rmsprop_metric_sgld"):
        v_ema = [torch.zeros_like(p) for p in params]

    n_pool = toks_pool.shape[0]
    if full_batch_n is not None:
        tb_fb = toks_pool[:full_batch_n].unsqueeze(0).expand(C_tot, -1, -1, -1)
        yb_fb = y_pool[:full_batch_n].unsqueeze(0).expand(C_tot, -1, -1, -1)
        with torch.no_grad():
            init_losses = [float(bm.forward_mse_per_chain(tb_fb, yb_fb)[v * n_chains].item()) for v in range(V)]
        b_order = None
    else:
        init_losses = [float(init_loss)] * V
        n_blocks = n_pool // m_batch
        rng_b = np.random.default_rng(seed + 7)
        b_single = torch.from_numpy(rng_b.integers(0, n_blocks, size=(n_total, n_chains))).to(device=device, dtype=torch.long)
        b_order = b_single.repeat(1, V) # (n_total, V * n_chains)
        ar_m = torch.arange(m_batch, device=device, dtype=torch.long)[None, :]

    noise_gen = torch.Generator(device=device).manual_seed(seed + 1009)
    half_eps = 0.5 * eps
    sqrt_eps = math.sqrt(eps)
    eps_rms = eps * rms_a

    losses = torch.empty((n_total, C_tot), dtype=torch.float32, device=device)

    for t in range(n_total):
        if full_batch_n is not None:
            tb, yb = tb_fb, yb_fb
        else:
            starts = b_order[t] * m_batch
            idx = starts[:, None] + ar_m
            tb = toks_pool[idx]
            yb = y_pool[idx]

        for p in params:
            p.grad = None
        per_chain_loss = fwd(tb, yb)
        losses[t] = per_chain_loss.detach()
        per_chain_loss.sum().backward()

        with torch.no_grad():
            grads = [p.grad for p in params]
            diffs = torch._foreach_sub(params, w0)
            noises = [
                torch.randn((n_chains, *p.shape[1:]), device=p.device, dtype=p.dtype, generator=noise_gen).repeat(V, *([1] * (p.ndim - 1)))
                for p in params
            ]
            if estimator == "vanilla":
                upd = torch._foreach_mul(grads, nbeta)
                torch._foreach_add_(upd, diffs, alpha=gamma)
                torch._foreach_add_(params, upd, alpha=-half_eps)
                torch._foreach_add_(params, noises, alpha=sqrt_eps)

            elif estimator in ("orbit_covariant", "param_rms_covariant"):
                assert D_inv_list is not None and D_mhalf_list is not None
                upd = torch._foreach_mul(grads, D_inv_list)
                torch._foreach_mul_(upd, nbeta)
                torch._foreach_add_(upd, diffs, alpha=gamma)
                torch._foreach_add_(params, upd, alpha=-half_eps)
                scaled_n = torch._foreach_mul(noises, D_mhalf_list)
                torch._foreach_add_(params, scaled_n, alpha=sqrt_eps)

            elif estimator == "rmsprop_psgld":
                assert v_ema is not None
                bc = 1.0 - (rms_alpha ** (t + 1))
                torch._foreach_mul_(v_ema, rms_alpha)
                torch._foreach_addcmul_(v_ema, grads, grads, value=1.0 - rms_alpha)
                v_hat = torch._foreach_div(v_ema, bc)
                denom = torch._foreach_sqrt(v_hat)
                torch._foreach_add_(denom, rms_a)
                G_list = [1.0 / d for d in denom]
                upd = torch._foreach_mul(grads, nbeta)
                torch._foreach_add_(upd, diffs, alpha=gamma)
                torch._foreach_mul_(upd, G_list)
                torch._foreach_add_(params, upd, alpha=-0.5 * eps_rms)
                sig_list = [torch.sqrt(eps_rms * g_) for g_ in G_list]
                torch._foreach_addcmul_(params, noises, sig_list)

            elif estimator == "rmsprop_metric_sgld":
                assert v_ema is not None
                bc = 1.0 - (rms_alpha ** (t + 1))
                torch._foreach_mul_(v_ema, rms_alpha)
                torch._foreach_addcmul_(v_ema, grads, grads, value=1.0 - rms_alpha)
                v_hat = torch._foreach_div(v_ema, bc)
                denom = torch._foreach_sqrt(v_hat)
                torch._foreach_add_(denom, rms_a)
                G_list = [1.0 / d for d in denom]
                upd = torch._foreach_mul(grads, G_list)
                torch._foreach_mul_(upd, nbeta)
                torch._foreach_add_(upd, diffs, alpha=gamma / rms_a)
                torch._foreach_add_(params, upd, alpha=-0.5 * eps_rms)
                sig_list = [torch.sqrt(eps_rms * g_) for g_ in G_list]
                torch._foreach_addcmul_(params, noises, sig_list)
            else:
                raise ValueError(f"Unknown estimator: {estimator}")

    losses_np = losses.cpu().numpy().astype(np.float64) # (n_total, V * n_chains)
    elapsed_tot = round(time.perf_counter() - t0, 2)

    with torch.no_grad():
        sq_dist = torch.zeros(C_tot, device=device)
        sq_dist_metric = torch.zeros(C_tot, device=device)
        for idx_p, (p, p0) in enumerate(zip(params, w0)):
            diff = p - p0
            sq_dist += diff.pow(2).reshape(C_tot, -1).sum(dim=1)
            if D_half_list is not None:
                sq_dist_metric += (D_half_list[idx_p] * diff).pow(2).reshape(C_tot, -1).sum(dim=1)
        dists_all = sq_dist.sqrt().cpu().numpy()
        dists_m_all = sq_dist_metric.sqrt().cpu().numpy() if D_half_list is not None else dists_all

    outs: list[dict[str, Any]] = []
    half = n_draws // 2
    for v_idx in range(V):
        sl = slice(v_idx * n_chains, (v_idx + 1) * n_chains)
        il_v = init_losses[v_idx]
        post = losses_np[n_burn:, sl]
        per_chain_llc = nbeta * (post.mean(axis=0) - il_v)
        per_chain_hd = nbeta * (post[half:, :].mean(axis=0) - post[:half, :].mean(axis=0))
        finite = per_chain_llc[np.isfinite(per_chain_llc)]
        mean_v = float(finite.mean()) if len(finite) > 0 else float("nan")
        std_v = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
        se_v = float(std_v / math.sqrt(len(finite))) if len(finite) > 1 else 0.0
        rec: dict[str, Any] = {
            "estimator": estimator,
            "mean": mean_v,
            "std": std_v,
            "se": se_v,
            "n_chains": n_chains,
            "n_nan_chains": int(np.isnan(per_chain_llc).sum()),
            "hdiff": float(np.nanmean(per_chain_hd)),
            "dist": float(np.nanmean(dists_all[sl])),
            "dist_metric": float(np.nanmean(dists_m_all[sl])),
            "init_loss": float(il_v),
            "elapsed_s": round(elapsed_tot / V, 2),
            "per_chain_llcs": [float(x) for x in per_chain_llc],
        }
        if want_windows:
            wins = {}
            win_pairs = [(300, 900), (1000, 2000), (2000, 3000), (3000, 4000), (4000, 5000), (1000, 5000)]
            if n_total < 1000:
                q = n_total // 4
                win_pairs = [(0, q), (q, 2 * q), (2 * q, 3 * q), (3 * q, n_total)]
            for a, b in win_pairs:
                if b <= n_total:
                    wvals = nbeta * (losses_np[a:b, sl].mean(axis=0) - il_v)
                    wins[f"{a}:{b}"] = {
                        "mean": float(np.nanmean(wvals)),
                        "se": float(np.nanstd(wvals, ddof=1) / math.sqrt(n_chains)) if n_chains > 1 else 0.0,
                    }
            rec["windows"] = wins
        outs.append(rec)
    return outs


def run_sgld_batched(
    sd: dict[str, torch.Tensor],
    toks_pool: torch.Tensor,
    y_pool: torch.Tensor,
    init_loss: float,
    **kwargs,
) -> dict[str, Any]:
    return run_sgld_multi_variant([sd], toks_pool, y_pool, init_loss, **kwargs)[0]


# =============================================================================
# 5. Deterministic Batched Training & `.safetensors` Checkpoint Persistence
# =============================================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint_safetensors(sd: dict[str, torch.Tensor], path: Path) -> str:
    _, safetensors_save = _get_safetensors()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save parameter tensors only (causal_mask buffer is deterministic & recreated by make_model)
    clean_sd = {k: sd[k].detach().float().cpu().contiguous() for k in PARAM_KEYS}
    safetensors_save(clean_sd, str(path))
    return sha256_file(path)


def load_checkpoint_safetensors(path: Path) -> dict[str, torch.Tensor]:
    safetensors_load, _ = _get_safetensors()
    loaded = safetensors_load(str(path), device="cpu")
    base = make_model(dtype=torch.float32, device="cpu").state_dict()
    for k in PARAM_KEYS:
        base[k] = loaded[k].float().cpu()
    return base


def train_batched_runs(
    opt_type: str,
    wd: float,
    seeds: list[int],
    t_steps: int,
    ckpt_steps: list[int],
    ckpt_dir: Path,
    device: str,
    use_compile: bool = True,
) -> dict[str, Any]:
    """
    Train all `seeds` for one optimizer (`adam` or `adamw`) simultaneously via BatchedDTransformerBMM,
    with strict per-seed isolated `torch.Generator` for BOTH init AND online data generation.
    Saves every checkpoint directly to `<ckpt_dir>/<run_name>/step_<step:06d>.safetensors`.
    """
    t0 = time.perf_counter()
    C_seeds = len(seeds)
    init_sds = [make_model_deterministic(s, dtype=torch.float32, device="cpu").state_dict() for s in seeds]
    bm = BatchedDTransformerBMM(init_sds, dtype=torch.float32, device=device)
    fwd = _get_fwd_fn(bm, use_compile=use_compile)

    fused_ok = (device == "cuda")
    if opt_type == "adam":
        opt = torch.optim.Adam(bm.parameters(), lr=0.003, betas=(0.9, 0.999), weight_decay=wd, fused=fused_ok)
    elif opt_type == "adamw":
        opt = torch.optim.AdamW(bm.parameters(), lr=0.003, betas=(0.9, 0.999), weight_decay=wd, fused=fused_ok)
    else:
        raise ValueError(f"Unknown opt_type: {opt_type}")

    div_f = max(2.0, t_steps / 2.0 - 1.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=0.003,
        total_steps=t_steps,
        pct_start=0.5,
        anneal_strategy="linear",
        cycle_momentum=False,
        div_factor=div_f,
        final_div_factor=div_f,
    )

    B = 256
    sigma = math.sqrt(0.125)
    # Isolated per-seed data generators (same data stream for Adam and AdamW on each seed!)
    data_gens = [torch.Generator(device=device).manual_seed(20_000 + int(s)) for s in seeds]

    ckpt_set = set(ckpt_steps)
    run_names = [f"{opt_type}_wd{wd:.2f}_seed{s}" for s in seeds]
    saved_records: dict[str, dict[str, Any]] = {
        rn: {"opt_type": opt_type, "wd": wd, "seed": s, "t_steps": t_steps, "checkpoints": {}}
        for rn, s in zip(run_names, seeds)
    }

    def _dump_step(step_num: int):
        for i_s, rn in enumerate(run_names):
            sd_i = bm.extract_state_dict(i_s)
            p = ckpt_dir / rn / f"step_{step_num:06d}.safetensors"
            h = save_checkpoint_safetensors(sd_i, p)
            saved_records[rn]["checkpoints"][str(step_num)] = {
                "path": str(p.relative_to(ckpt_dir)),
                "sha256": h,
            }

    if 0 in ckpt_set:
        _dump_step(0)

    log_every = max(1, t_steps // 10)
    for step in range(1, t_steps + 1):
        # Generate strictly isolated batches per seed
        toks_list, y_list = [], []
        for g_s in data_gens:
            w = torch.randn(B, 4, device=device, generator=g_s)
            x = torch.randn(B, 8, 4, device=device, generator=g_s)
            y = torch.bmm(x, w.unsqueeze(-1)) + torch.randn(B, 8, 1, device=device, generator=g_s) * sigma
            toks_list.append(to_token_sequence(x, y))
            y_list.append(y)
        toks_b = torch.stack(toks_list, dim=0)
        y_b = torch.stack(y_list, dim=0)

        opt.zero_grad(set_to_none=True)
        losses = fwd(toks_b, y_b)
        losses.sum().backward()
        opt.step()
        if step < t_steps:
            sched.step()

        if step in ckpt_set:
            _dump_step(step)
        if step % log_every == 0 or step == t_steps:
            dt = time.perf_counter() - t0
            loss_str = ", ".join(f"s{s}={losses[i].item():.3f}" for i, s in enumerate(seeds))
            print(f"  [Train {opt_type:5s} wd={wd:.2f}] step {step:6d}/{t_steps} ({dt:6.1f}s) | {loss_str}", flush=True)

    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    for rn in run_names:
        saved_records[rn]["train_time_s"] = round(elapsed, 2)
        saved_records[rn]["batched_seeds"] = seeds
    return saved_records


def update_manifest(ckpt_dir: Path, new_runs: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    manifest_path = ckpt_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"metadata": meta, "runs": {}, "init_equality_checks": {}}
    manifest["metadata"].update(meta)
    manifest["runs"].update(new_runs)

    # Programmatically verify bit-for-bit initial weight equality between Adam and AdamW for every seed!
    adam_runs = {r["seed"]: r for r in manifest["runs"].values() if r["opt_type"] == "adam"}
    adamw_runs = {r["seed"]: r for r in manifest["runs"].values() if r["opt_type"] == "adamw"}
    for s in sorted(set(adam_runs) & set(adamw_runs)):
        if "0" in adam_runs[s]["checkpoints"] and "0" in adamw_runs[s]["checkpoints"]:
            p_a = ckpt_dir / adam_runs[s]["checkpoints"]["0"]["path"]
            p_w = ckpt_dir / adamw_runs[s]["checkpoints"]["0"]["path"]
            sd_a = load_checkpoint_safetensors(p_a)
            sd_w = load_checkpoint_safetensors(p_w)
            max_diff = max(float((sd_a[k] - sd_w[k]).abs().max().item()) for k in PARAM_KEYS)
            sha_eq = adam_runs[s]["checkpoints"]["0"]["sha256"] == adamw_runs[s]["checkpoints"]["0"]["sha256"]
            manifest["init_equality_checks"][f"seed_{s}"] = {
                "max_abs_init_diff_adam_vs_adamw": max_diff,
                "sha256_identical": sha_eq,
                "verified_zero": (max_diff == 0.0 and sha_eq),
            }
            assert max_diff == 0.0 and sha_eq, f"Seed {s} Adam vs AdamW init mismatch: {max_diff}!"

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


# =============================================================================
# 6. Hoogland et al. GP-Smoothed LLC-Derivative Stage-Boundary Detector
# =============================================================================

def detect_gp_stage_boundaries(steps: list[int], llc_mean: list[float]) -> dict[str, Any]:
    """
    Applies Hoogland et al.'s GP-smoothed LLC-derivative stage-boundary procedure
    (`icl/notebooks/main-figures.ipynb` cell 51 & App. B.1):
      - Fit RBF + WhiteKernel GP to `(log(step + 1), llc_mean)` on `step >= 100` (or all positive steps).
      - Differentiate GP mean w.r.t. `log(t + 1)` on a dense 1,000-point log grid.
      - Locate derivative sign-change extrema (peak & post-peak trough) and saddle plateaus
        (local minima of `|d lambda / d log t| < 10`).
    """
    s_arr = np.array(steps, dtype=np.float64)
    y_arr = np.array(llc_mean, dtype=np.float64)
    mask = s_arr >= 100.0 if (s_arr >= 100.0).sum() >= 5 else (s_arr > 0)
    s_fit, y_fit = s_arr[mask], y_arr[mask]
    if len(s_fit) < 4:
        # Smoke test fallback with very few points
        pk_i = int(np.argmax(y_arr))
        post = y_arr[pk_i:]
        tr_i = pk_i + int(np.argmin(post))
        return {
            "method": "fallback_few_points",
            "peak_step_gp": float(s_arr[pk_i]),
            "peak_llc_gp": float(y_arr[pk_i]),
            "trough_step_gp": float(s_arr[tr_i]),
            "trough_llc_gp": float(y_arr[tr_i]),
            "peak_step_grid": int(s_arr[pk_i]),
            "trough_step_grid": int(s_arr[tr_i]),
            "zero_crossings": [],
            "plateaus": [],
        }

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, WhiteKernel

    x = np.log(s_fit + 1.0).reshape(-1, 1)
    kernel = (
        C(100.0, (1e-2, 1e5)) * RBF(1.5, (0.5, 10.0))
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-4, 50.0))
    )
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, random_state=0, normalize_y=True)
    gp.fit(x, y_fit)

    xg = np.linspace(float(x.min()), float(x.max()), 1000).reshape(-1, 1)
    yg, yg_std = gp.predict(xg, return_std=True)
    dy = np.gradient(yg, xg.ravel())
    sg = np.exp(xg.ravel()) - 1.0

    crossings = []
    for i in range(len(dy) - 1):
        if dy[i] >= 0 and dy[i + 1] < 0:
            crossings.append({"type": "max", "step": float(sg[i]), "llc_gp": float(yg[i])})
        elif dy[i] <= 0 and dy[i + 1] > 0:
            crossings.append({"type": "min", "step": float(sg[i]), "llc_gp": float(yg[i])})

    abs_dy = np.abs(dy)
    plateaus = []
    for i in range(1, len(dy) - 1):
        if abs_dy[i] < abs_dy[i - 1] and abs_dy[i] < abs_dy[i + 1] and abs_dy[i] < 10.0:
            plateaus.append({"step": float(sg[i]), "llc_gp": float(yg[i]), "d_dlogt": float(dy[i])})

    # Global peak & post-peak trough from GP curve
    pk_idx = int(np.argmax(yg))
    peak_step_gp = float(sg[pk_idx])
    peak_llc_gp = float(yg[pk_idx])
    post_yg = yg[pk_idx:]
    tr_idx = pk_idx + int(np.argmin(post_yg))
    trough_step_gp = float(sg[tr_idx])
    trough_llc_gp = float(yg[tr_idx])

    # Snap to nearest evaluated checkpoint steps
    peak_step_grid = int(min(steps, key=lambda st: abs(st - peak_step_gp)))
    post_steps = [st for st in steps if st > peak_step_grid] or [steps[-1]]
    trough_step_grid = int(min(post_steps, key=lambda st: abs(st - trough_step_gp)))

    return {
        "method": "GP_RBF_WhiteKernel_d_dlogt",
        "kernel_fitted": str(gp.kernel_),
        "peak_step_gp": peak_step_gp,
        "peak_llc_gp": peak_llc_gp,
        "trough_step_gp": trough_step_gp,
        "trough_llc_gp": trough_llc_gp,
        "peak_step_grid": peak_step_grid,
        "trough_step_grid": trough_step_grid,
        "zero_crossings": crossings,
        "plateaus": plateaus,
        "gp_grid_steps": [float(v) for v in sg[::10]],
        "gp_grid_mean": [float(v) for v in yg[::10]],
        "gp_grid_deriv": [float(v) for v in dy[::10]],
    }


def compute_stage_decompositions(
    seed_runs: list[dict],
    stage_pairs: list[tuple[str, int, int]],
    bal_key: str = "sgld_orbit",
) -> dict[str, Any]:
    out = {}
    for sname, t1, t2 in stage_pairs:
        d_tot_list, d_func_list, d_fib_list, share_list = [], [], [], []
        per_seed = []
        for sr in seed_runs:
            by_step = {c["step"]: c for c in sr["checkpoints"]}
            if t1 not in by_step or t2 not in by_step:
                continue
            c1, c2 = by_step[t1], by_step[t2]
            r1, r2 = c1["sgld_raw"]["mean"], c2["sgld_raw"]["mean"]
            b1, b2 = c1[bal_key]["mean"], c2[bal_key]["mean"]
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
            return float(np.nanmean(a)), float(np.nanstd(a, ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0

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
# 7. Shardable Task Builder & Evaluator (Parts 1–6)
# =============================================================================

def build_eval_task_list(
    manifest: dict[str, Any],
    peak_step: int,
    trough_step: int,
    is_smoke: bool,
    eval_seeds: set[int] | None = None,
    parts: str = "all",
) -> list[dict[str, Any]]:
    """Build deterministic ordered list of all evaluation units across Parts 1–6 so
    `--shard k/N` partitions them cleanly with zero overlap."""
    tasks: list[dict[str, Any]] = []
    runs = {
        rn: r for rn, r in manifest["runs"].items()
        if (eval_seeds is None or int(r["seed"]) in eval_seeds)
    }

    want_traj = parts in ("all", "traj")
    want_deep = parts in ("all", "deep")

    # Part 1 (Adam) & Part 2 (AdamW): every checkpoint RAW + ORBIT (+ ORBIT+ at peak/trough)
    if want_traj:
        for rn in sorted(runs.keys()):
            r = runs[rn]
            steps_sorted = sorted(int(st) for st in r["checkpoints"].keys())
            for st in steps_sorted:
                tasks.append({
                    "task_id": f"traj__{rn}__step_{st}",
                    "kind": "trajectory_ckpt",
                    "run_name": rn,
                    "opt_type": r["opt_type"],
                    "wd": r["wd"],
                    "seed": r["seed"],
                    "step": st,
                    "ckpt_rel": r["checkpoints"][str(st)]["path"],
                    "also_orbit_plus": (st in (peak_step, trough_step)),
                })

    if not want_deep:
        return tasks

    # Find Adam seed 0 run name (fallback to lowest Adam seed)
    adam_rns = sorted(
        [(r["seed"], rn) for rn, r in runs.items() if r["opt_type"] == "adam"]
    )
    assert adam_rns, "No Adam runs found in manifest!"
    seed0_rn = adam_rns[0][1]

    # Part 3: Per-channel exact symmetry ablation at peak & trough (Adam seed 0)
    ablation_channels = [
        "RAW", "LN_mean_proj", "QK", "VO", "MLP",
        "LNf_unembed_all5", "LN1_mlp", "LN0_attn",
        "LN_all_unit_gamma", "ORBIT", "ORBIT+",
    ]
    for st_label, st in [("peak", peak_step), ("trough", trough_step)]:
        if str(st) in runs[seed0_rn]["checkpoints"]:
            for ch in ablation_channels:
                tasks.append({
                    "task_id": f"ablation__{st_label}_{st}__{ch}",
                    "kind": "ablation_channel",
                    "run_name": seed0_rn,
                    "seed": runs[seed0_rn]["seed"],
                    "milestone": st_label,
                    "step": st,
                    "channel": ch,
                    "ckpt_rel": runs[seed0_rn]["checkpoints"][str(st)]["path"],
                })

    # Part 4: Hyperparameter controls (gamma sweep 3.3/13.3/40.0 + eps=1e-4 + full-batch-gradient SGLD)
    # Run at BOTH peak and trough (RAW and ORBIT) on Adam seed 0!
    sens_specs = [
        ("g3.3_e3e-4",   0.0003,  3.3, 66.7, None),
        ("g13.3_e3e-4",  0.0003, 13.3, 66.7, None),
        ("g40.0_e3e-4",  0.0003, 40.0, 66.7, None),
        ("g13.3_e1e-4",  0.0001, 13.3, 66.7, None),
        ("g13.3_FB1024", 0.0003, 13.3, 66.7, 256 if is_smoke else 1024),
    ]
    for st_label, st in [("peak", peak_step), ("trough", trough_step)]:
        if str(st) in runs[seed0_rn]["checkpoints"]:
            for spec_name, eps_s, gam_s, nb_s, fb_s in sens_specs:
                tasks.append({
                    "task_id": f"control__{st_label}_{st}__{spec_name}",
                    "kind": "sgld_control",
                    "run_name": seed0_rn,
                    "seed": runs[seed0_rn]["seed"],
                    "milestone": st_label,
                    "step": st,
                    "spec_name": spec_name,
                    "eps": eps_s, "gamma": gam_s, "nbeta": nb_s, "full_batch_n": fb_s,
                    "ckpt_rel": runs[seed0_rn]["checkpoints"][str(st)]["path"],
                })

    # Part 5: Remedies across ALL Adam seeds at peak & trough (RAW vs ORBIT for all 4 estimators!)
    remedy_estimators = ["vanilla", "rmsprop_psgld", "param_rms_covariant", "orbit_covariant"]
    for s_val, rn in adam_rns:
        for st_label, st in [("peak", peak_step), ("trough", trough_step)]:
            if str(st) in runs[rn]["checkpoints"]:
                for est in remedy_estimators:
                    tasks.append({
                        "task_id": f"remedy__{rn}__{st_label}_{st}__{est}",
                        "kind": "remedy_eval",
                        "run_name": rn,
                        "seed": s_val,
                        "milestone": st_label,
                        "step": st,
                        "estimator": est,
                        "ckpt_rel": runs[rn]["checkpoints"][str(st)]["path"],
                    })

    # Part 6: Exact Gauss-Newton Spectrum (Full + W_QKV + LN_0 + W_fc1 + LN_1) at peak & trough,
    # RAW vs ORBIT, across ALL Adam seeds!
    for s_val, rn in adam_rns:
        for st_label, st in [("peak", peak_step), ("trough", trough_step)]:
            if str(st) in runs[rn]["checkpoints"]:
                tasks.append({
                    "task_id": f"gn__{rn}__{st_label}_{st}",
                    "kind": "gauss_newton",
                    "run_name": rn,
                    "seed": s_val,
                    "milestone": st_label,
                    "step": st,
                    "ckpt_rel": runs[rn]["checkpoints"][str(st)]["path"],
                })

    return tasks


def execute_eval_shard(
    manifest: dict[str, Any],
    ckpt_dir: Path,
    shard_idx: int,
    n_shards: int,
    peak_step: int,
    trough_step: int,
    *,
    n_chains: int,
    n_burn: int,
    n_draws: int,
    m_batch: int,
    n_pool: int,
    pool_seed: int,
    device: str,
    use_compile: bool,
    is_smoke: bool,
    checkpoint_path: Path | None = None,
    eval_seeds: set[int] | None = None,
    parts: str = "all",
) -> dict[str, Any]:
    t_start = time.perf_counter()
    all_tasks = build_eval_task_list(manifest, peak_step, trough_step, is_smoke=is_smoke, eval_seeds=eval_seeds, parts=parts)
    my_tasks = [t for i, t in enumerate(all_tasks) if i % n_shards == shard_idx]
    results: dict[str, Any] = {}
    if checkpoint_path is not None and checkpoint_path.exists():
        prev = json.loads(checkpoint_path.read_text())
        results.update(prev.get("results", {}))
        print(f"[Shard {shard_idx}/{n_shards}] Resumed {len(results)} completed tasks from {checkpoint_path}", flush=True)
    print(
        f"[Shard {shard_idx}/{n_shards}] Executing {len(my_tasks)}/{len(all_tasks)} tasks on {device} "
        f"(chains={n_chains}, burn={n_burn}, draws={n_draws}, m={m_batch}, pool={n_pool:,})",
        flush=True,
    )

    toks_pool, y_pool = generate_sgld_pool(pool_seed, n_pool, device=device)

    # Fixed held-out sets (512 fresh sequences for f64/f32 canonicalization + Gauss-Newton; 2048 for metrics)
    sigma = math.sqrt(0.125)
    g_chk = torch.Generator(device="cpu").manual_seed(7777)
    w_chk = torch.randn(512, 4, generator=g_chk, dtype=torch.float64)
    x_check64 = torch.randn(512, 8, 4, generator=g_chk, dtype=torch.float64)
    y_check64 = torch.bmm(x_check64, w_chk.unsqueeze(-1)) + torch.randn(512, 8, 1, generator=g_chk, dtype=torch.float64) * sigma

    g_ev = torch.Generator(device="cpu").manual_seed(999)
    w_ev = torch.randn(2048, 4, generator=g_ev)
    x_ev = torch.randn(2048, 8, 4, generator=g_ev)
    e_ev = torch.randn(2048, 8, 1, generator=g_ev) * sigma
    y_ev = torch.bmm(x_ev, w_ev.unsqueeze(-1)) + e_ev

    eps_def, gamma_def, nbeta_def = 0.0003, 13.3, 66.7

    def _build_shard_obj() -> dict[str, Any]:
        return {
            "shard_idx": shard_idx,
            "n_shards": n_shards,
            "peak_step": peak_step,
            "trough_step": trough_step,
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda" else platform.machine(),
            "sgld_config": {
                "n_chains": n_chains,
                "n_burn": n_burn,
                "n_draws": n_draws,
                "m_batch": m_batch,
                "n_pool": n_pool,
                "pool_seed": pool_seed,
                "eps": eps_def,
                "gamma": gamma_def,
                "nbeta": nbeta_def,
            },
            "elapsed_s": round(time.perf_counter() - t_start, 2),
            "results": results,
        }

    for idx_t, task in enumerate(my_tasks):
        tid = task["task_id"]
        if tid in results:
            continue
        kind = task["kind"]
        sd_raw = load_checkpoint_safetensors(ckpt_dir / task["ckpt_rel"])

        if kind == "trajectory_ckpt":
            can_orb = canonicalize_state_dict(sd_raw, x_check64, y_check64, channel="ORBIT")
            metrics = evaluate_checkpoint_metrics(sd_raw, x_ev, y_ev, w_ev, e_ev, device=device)
            init_loss = compute_pool_init_loss(sd_raw, toks_pool, y_pool, device=device)
            variants = [sd_raw, can_orb["sd"]]
            can_op = None
            if task["also_orbit_plus"]:
                can_op = canonicalize_state_dict(sd_raw, x_check64, y_check64, channel="ORBIT+")
                variants.append(can_op["sd"])
            sglds = run_sgld_multi_variant(
                variants, toks_pool, y_pool, init_loss,
                estimator="vanilla", n_chains=n_chains, n_burn=n_burn, n_draws=n_draws,
                eps=eps_def, gamma=gamma_def, nbeta=nbeta_def, m_batch=m_batch,
                seed=42, device=device, use_compile=use_compile,
            )
            sgld_raw, sgld_orb = sglds[0], sglds[1]
            rec: dict[str, Any] = {
                "task": task,
                "step": task["step"],
                "init_loss_pool": init_loss,
                "norm_raw": can_orb["norm_raw"],
                "norm_orbit": can_orb["norm_can"],
                "max_delta_y_f64_orbit": can_orb["max_delta_y_f64"],
                "max_delta_y_f32_orbit": can_orb["max_delta_y_f32"],
                "max_delta_all5_f64_orbit": can_orb["max_delta_all5_f64"],
                "max_delta_all5_f32_orbit": can_orb["max_delta_all5_f32"],
                "diffeomorphism_cert_orbit": can_orb["diffeomorphism_cert"],
                "sgld_raw": sgld_raw,
                "sgld_orbit": sgld_orb,
                "excess_raw_minus_orbit": float(sgld_raw["mean"] - sgld_orb["mean"]),
                "metrics": metrics,
            }
            if can_op is not None:
                rec["orbit_plus"] = {
                    "norm_can": can_op["norm_can"],
                    "max_delta_y_f64": can_op["max_delta_y_f64"],
                    "max_delta_y_f32": can_op["max_delta_y_f32"],
                    "max_delta_all5_f64": can_op["max_delta_all5_f64"],
                    "max_delta_all5_f32": can_op["max_delta_all5_f32"],
                    "sgld": sglds[2],
                }
            results[tid] = rec
            print(
                f"  [{idx_t+1:3d}/{len(my_tasks)}] {tid:36s} | MSE={metrics['mse_id']:.4f} | "
                f"RAW={sgld_raw['mean']:6.2f}±{sgld_raw['se']:4.2f} | ORBIT={sgld_orb['mean']:6.2f}±{sgld_orb['se']:4.2f} | "
                f"dy_f64={can_orb['max_delta_y_f64']:.1e} dy_f32={can_orb['max_delta_y_f32']:.1e}",
                flush=True,
            )

        elif kind == "ablation_channel":
            ch = task["channel"]
            can = canonicalize_state_dict(sd_raw, x_check64, y_check64, channel=ch)
            init_loss = compute_pool_init_loss(sd_raw, toks_pool, y_pool, device=device)
            sgld = run_sgld_batched(
                can["sd"], toks_pool, y_pool, init_loss,
                estimator="vanilla", n_chains=n_chains, n_burn=n_burn, n_draws=n_draws,
                eps=eps_def, gamma=gamma_def, nbeta=nbeta_def, m_batch=m_batch,
                seed=42, device=device, use_compile=use_compile,
            )
            results[tid] = {
                "task": task,
                "sgld": sgld,
                "norm": can["norm_can"],
                "max_delta_y_f64": can["max_delta_y_f64"],
                "max_delta_y_f32": can["max_delta_y_f32"],
                "max_delta_all5_f64": can["max_delta_all5_f64"],
                "max_delta_all5_f32": can["max_delta_all5_f32"],
                "diffeomorphism_cert": can["diffeomorphism_cert"],
            }
            print(
                f"  [{idx_t+1:3d}/{len(my_tasks)}] {tid:36s} | LLC={sgld['mean']:6.2f}±{sgld['se']:4.2f} | "
                f"||w||={can['norm_can']:5.1f} | dy_f32={can['max_delta_y_f32']:.1e}",
                flush=True,
            )

        elif kind == "sgld_control":
            can_orb = canonicalize_state_dict(sd_raw, x_check64, y_check64, channel="ORBIT")
            init_loss = compute_pool_init_loss(sd_raw, toks_pool, y_pool, device=device)
            sg_r, sg_o = run_sgld_multi_variant(
                [sd_raw, can_orb["sd"]], toks_pool, y_pool, init_loss,
                estimator="vanilla", n_chains=n_chains, n_burn=n_burn, n_draws=n_draws,
                eps=task["eps"], gamma=task["gamma"], nbeta=task["nbeta"],
                m_batch=m_batch, full_batch_n=task["full_batch_n"],
                seed=100, want_windows=True, device=device, use_compile=use_compile,
            )
            results[tid] = {
                "task": task,
                "raw": sg_r,
                "orbit": sg_o,
                "raw_minus_orbit": float(sg_r["mean"] - sg_o["mean"]),
            }
            print(
                f"  [{idx_t+1:3d}/{len(my_tasks)}] {tid:36s} | RAW={sg_r['mean']:6.2f}±{sg_r['se']:4.2f} (d={sg_r['dist']:5.1f}) | "
                f"ORBIT={sg_o['mean']:6.2f}±{sg_o['se']:4.2f} (d={sg_o['dist']:5.1f})",
                flush=True,
            )

        elif kind == "remedy_eval":
            est = task["estimator"]
            can_orb = canonicalize_state_dict(sd_raw, x_check64, y_check64, channel="ORBIT")
            init_loss = compute_pool_init_loss(sd_raw, toks_pool, y_pool, device=device)
            sg_r, sg_o = run_sgld_multi_variant(
                [sd_raw, can_orb["sd"]], toks_pool, y_pool, init_loss,
                estimator=est, n_chains=n_chains, n_burn=n_burn, n_draws=n_draws,
                eps=eps_def, gamma=gamma_def, nbeta=nbeta_def, m_batch=m_batch,
                seed=200, want_windows=True, device=device, use_compile=use_compile,
            )
            results[tid] = {
                "task": task,
                "raw": sg_r,
                "orbit": sg_o,
                "raw_minus_orbit": float(sg_r["mean"] - sg_o["mean"]),
            }
            print(
                f"  [{idx_t+1:3d}/{len(my_tasks)}] {tid:36s} | RAW={sg_r['mean']:6.2f}±{sg_r['se']:4.2f} | "
                f"ORBIT={sg_o['mean']:6.2f}±{sg_o['se']:4.2f} | R-O={sg_r['mean']-sg_o['mean']:+6.2f}",
                flush=True,
            )

        elif kind == "gauss_newton":
            can_orb = canonicalize_state_dict(sd_raw, x_check64, y_check64, channel="ORBIT")
            gn_raw = compute_gauss_newton_spectrum(
                sd_raw, x_check64.float(), y_check64.float(),
                gamma=gamma_def, nbeta=nbeta_def, device=device,
            )
            gn_orb = compute_gauss_newton_spectrum(
                can_orb["sd"], x_check64.float(), y_check64.float(),
                gamma=gamma_def, nbeta=nbeta_def, device=device,
            )
            results[tid] = {
                "task": task,
                "raw": gn_raw,
                "orbit": gn_orb,
            }
            fr = gn_raw["blocks"]["full_model"]
            fo = gn_orb["blocks"]["full_model"]
            qr = gn_raw["blocks"]["W_QKV"]
            qo = gn_orb["blocks"]["W_QKV"]
            print(
                f"  [{idx_t+1:3d}/{len(my_tasks)}] {tid:36s} | Full GN_ED: RAW={fr['laplace_ed_g13p3']:6.2f} ORBIT={fo['laplace_ed_g13p3']:6.2f} | "
                f"W_QKV >tau: {qr['n_gt_tau']}->{qo['n_gt_tau']}",
                flush=True,
            )

        if checkpoint_path is not None and ((idx_t + 1) % 5 == 0 or (idx_t + 1) == len(my_tasks)):
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_path.write_text(json.dumps(_build_shard_obj(), indent=2))

    return _build_shard_obj()


# =============================================================================
# 8. Merge Shards & Assemble Final Analysis JSON
# =============================================================================

def merge_shards_to_payload(
    manifest: dict[str, Any],
    shard_payloads: list[dict[str, Any]],
    hoogland_ref: dict[str, Any] | None,
    upstream_parity: dict[str, Any] | None,
) -> dict[str, Any]:
    merged_res: dict[str, Any] = {}
    for sp in shard_payloads:
        merged_res.update(sp["results"])

    peak_step = shard_payloads[0]["peak_step"]
    trough_step = shard_payloads[0]["trough_step"]
    sgld_cfg = shard_payloads[0]["sgld_config"]

    # Group Part 1 (Adam) & Part 2 (AdamW) runs
    by_run: dict[str, list[dict[str, Any]]] = {}
    for rec in merged_res.values():
        if rec["task"]["kind"] == "trajectory_ckpt":
            rn = rec["task"]["run_name"]
            by_run.setdefault(rn, []).append(rec)

    adam_runs, adamw_runs = [], []
    for rn, recs in sorted(by_run.items(), key=lambda kv: manifest["runs"][kv[0]]["seed"]):
        recs_sorted = sorted(recs, key=lambda r: r["step"])
        r_meta = manifest["runs"][rn]
        entry = {
            "run_name": rn,
            "seed": r_meta["seed"],
            "wd": r_meta["wd"],
            "train_time_s": r_meta.get("train_time_s"),
            "checkpoints": recs_sorted,
        }
        if r_meta["opt_type"] == "adam":
            adam_runs.append(entry)
        else:
            adamw_runs.append(entry)

    # GP stage-boundary detection directly on our raw Adam LLC mean curve!
    adam_steps = [c["step"] for c in adam_runs[0]["checkpoints"]]
    raw_mat = np.array([[c["sgld_raw"]["mean"] for c in r["checkpoints"]] for r in adam_runs], dtype=np.float64)
    orb_mat = np.array([[c["sgld_orbit"]["mean"] for c in r["checkpoints"]] for r in adam_runs], dtype=np.float64)
    raw_mean = raw_mat.mean(axis=0).tolist()
    gp_ours = detect_gp_stage_boundaries(adam_steps, raw_mean)

    # Quantitative comparison against Hoogland et al.'s released 5-seed `llcs.pt`
    hoogland_comparison: dict[str, Any] | None = None
    if hoogland_ref is not None:
        h_steps = hoogland_ref["steps"]
        h_llc = hoogland_ref["llc_mean"]
        gp_hoog = detect_gp_stage_boundaries(h_steps, h_llc)
        # Align our checkpoints to nearest Hoogland steps (only meaningful when T >= 100k)
        matched_ours, matched_hoog, matched_steps = [], [], []
        for s_i, y_i in zip(adam_steps, raw_mean):
            ns = min(h_steps, key=lambda x: abs(x - s_i))
            if abs(ns - s_i) <= max(50, 0.05 * max(1, s_i)):
                matched_steps.append(s_i)
                matched_ours.append(y_i)
                matched_hoog.append(h_llc[h_steps.index(ns)])
        from scipy.stats import pearsonr
        corr_r = float(pearsonr(matched_ours, matched_hoog)[0]) if len(matched_ours) >= 3 else None
        mae = float(np.mean(np.abs(np.array(matched_ours) - np.array(matched_hoog)))) if matched_ours else None
        hoogland_comparison = {
            "reference_gp_stages": gp_hoog,
            "n_aligned_steps": len(matched_steps),
            "pearson_r_raw_vs_hoogland": corr_r,
            "mae_raw_vs_hoogland": mae,
            "hoogland_peak_step": int(h_steps[int(np.argmax(h_llc))]),
            "hoogland_peak_llc": float(np.max(h_llc)),
            "hoogland_trough_step": int(h_steps[int(np.argmax(h_llc)) + int(np.argmin(h_llc[int(np.argmax(h_llc)):]))]),
            "hoogland_trough_llc": float(np.min(h_llc[int(np.argmax(h_llc)):])),
        }

    # Build stage pairs from GP-detected boundaries + exact empirical peak/trough milestone pair
    lr1_end = min(adam_steps, key=lambda s: abs(s - 984)) if max(adam_steps) >= 2000 else adam_steps[min(1, len(adam_steps) - 1)]
    t_pk = peak_step if peak_step in adam_steps else gp_ours["peak_step_grid"]
    t_tr = trough_step if trough_step in adam_steps else gp_ours["trough_step_grid"]
    t_end = adam_steps[-1]
    stage_pairs = [
        ("LR1_0_to_984", adam_steps[0], lr1_end),
        ("LR2_984_to_peak", lr1_end, t_pk),
        ("LR34_peak_to_trough", t_pk, t_tr),
    ]
    if t_end > t_tr:
        stage_pairs.append(("LR5_trough_to_end", t_tr, t_end))
    if gp_ours["peak_step_grid"] != t_pk:
        stage_pairs.append(("LR34_gp_peak_to_trough", gp_ours["peak_step_grid"], gp_ours["trough_step_grid"]))

    adam_stages = compute_stage_decompositions(adam_runs, stage_pairs, bal_key="sgld_orbit")
    # For AdamW, snap stage pairs to available AdamW steps
    adamw_stages = {}
    if adamw_runs:
        aw_steps = [c["step"] for c in adamw_runs[0]["checkpoints"]]
        aw_pairs = [(nm, min(aw_steps, key=lambda x: abs(x - a)), min(aw_steps, key=lambda x: abs(x - b))) for nm, a, b in stage_pairs]
        adamw_stages = compute_stage_decompositions(adamw_runs, aw_pairs, bal_key="sgld_orbit")

    # Assemble Part 3 (Per-channel ablation)
    part3: dict[str, dict[str, Any]] = {}
    for rec in merged_res.values():
        if rec["task"]["kind"] == "ablation_channel":
            mk = f"{rec['task']['milestone']}_step_{rec['task']['step']}"
            part3.setdefault(mk, {})[rec["task"]["channel"]] = rec
    for mk, ch_map in part3.items():
        if "RAW" in ch_map:
            raw_m = ch_map["RAW"]["sgld"]["mean"]
            for ch, rec in ch_map.items():
                rec["d_vs_raw"] = float(rec["sgld"]["mean"] - raw_m)

    # Assemble Part 4 (SGLD controls & gamma sweep peak->trough comparison)
    part4: dict[str, dict[str, Any]] = {}
    for rec in merged_res.values():
        if rec["task"]["kind"] == "sgld_control":
            mk = rec["task"]["milestone"]
            part4.setdefault(mk, {})[rec["task"]["spec_name"]] = rec
    part4_deltas: dict[str, Any] = {}
    if "peak" in part4 and "trough" in part4:
        for spec_name in sorted(set(part4["peak"]) & set(part4["trough"])):
            pk, tr = part4["peak"][spec_name], part4["trough"][spec_name]
            part4_deltas[spec_name] = {
                "raw_peak": pk["raw"]["mean"],
                "raw_trough": tr["raw"]["mean"],
                "raw_delta_peak_to_trough": float(tr["raw"]["mean"] - pk["raw"]["mean"]),
                "orbit_peak": pk["orbit"]["mean"],
                "orbit_trough": tr["orbit"]["mean"],
                "orbit_delta_peak_to_trough": float(tr["orbit"]["mean"] - pk["orbit"]["mean"]),
                "raw_dist_trough": tr["raw"]["dist"],
                "orbit_dist_trough": tr["orbit"]["dist"],
            }

    # Assemble Part 5 (Remedies across seeds, peak -> trough, RAW vs ORBIT)
    part5_raw: dict[str, list[dict[str, Any]]] = {}
    for rec in merged_res.values():
        if rec["task"]["kind"] == "remedy_eval":
            part5_raw.setdefault(rec["task"]["estimator"], []).append(rec)
    part5_summary: dict[str, Any] = {}
    for est, recs in part5_raw.items():
        by_seed: dict[int, dict[str, Any]] = {}
        for r in recs:
            by_seed.setdefault(r["task"]["seed"], {})[r["task"]["milestone"]] = r
        per_seed_list = []
        for s_val in sorted(by_seed):
            if "peak" in by_seed[s_val] and "trough" in by_seed[s_val]:
                pk, tr = by_seed[s_val]["peak"], by_seed[s_val]["trough"]
                per_seed_list.append({
                    "seed": s_val,
                    "raw_peak": pk["raw"]["mean"],
                    "raw_trough": tr["raw"]["mean"],
                    "raw_delta": float(tr["raw"]["mean"] - pk["raw"]["mean"]),
                    "orbit_peak": pk["orbit"]["mean"],
                    "orbit_trough": tr["orbit"]["mean"],
                    "orbit_delta": float(tr["orbit"]["mean"] - pk["orbit"]["mean"]),
                    "gap_peak_raw_minus_orbit": pk["raw_minus_orbit"],
                    "gap_trough_raw_minus_orbit": tr["raw_minus_orbit"],
                    "raw_trough_hdiff": tr["raw"]["hdiff"],
                    "orbit_trough_hdiff": tr["orbit"]["hdiff"],
                })

        def _agg(field: str) -> tuple[float, float]:
            arr = np.array([x[field] for x in per_seed_list], dtype=np.float64)
            return float(np.nanmean(arr)), (float(np.nanstd(arr, ddof=1) / math.sqrt(len(arr))) if len(arr) > 1 else 0.0)

        part5_summary[est] = {
            "per_seed": per_seed_list,
            "raw_peak_mean_se": list(_agg("raw_peak")),
            "raw_trough_mean_se": list(_agg("raw_trough")),
            "raw_delta_mean_se": list(_agg("raw_delta")),
            "orbit_peak_mean_se": list(_agg("orbit_peak")),
            "orbit_trough_mean_se": list(_agg("orbit_trough")),
            "orbit_delta_mean_se": list(_agg("orbit_delta")),
            "gap_trough_mean_se": list(_agg("gap_trough_raw_minus_orbit")),
        }

    # Assemble Part 6 (Exact Gauss-Newton spectra across seeds)
    part6: dict[str, dict[str, Any]] = {}
    for rec in merged_res.values():
        if rec["task"]["kind"] == "gauss_newton":
            key = f"seed_{rec['task']['seed']}"
            part6.setdefault(key, {})[rec["task"]["milestone"]] = rec

    # Global worst-case float64 and post-float32 canonicalization discrepancies
    all_traj = [r for run in (adam_runs + adamw_runs) for r in run["checkpoints"]]
    worst_f64 = max(r["max_delta_y_f64_orbit"] for r in all_traj)
    worst_f32 = max(r["max_delta_y_f32_orbit"] for r in all_traj)
    worst_all5_f64 = max(r["max_delta_all5_f64_orbit"] for r in all_traj)
    worst_all5_f32 = max(r["max_delta_all5_f32_orbit"] for r in all_traj)

    return {
        "metadata": {
            "script": "experiments/devinterp_fiber_decomposition.py",
            "reference_paper": "Hoogland et al. (2024, TMLR, arXiv:2402.02364)",
            "upstream_icl_commit": UPSTREAM_ICL_COMMIT,
            "recipe_resolution": (
                "Follows paper Table 4 & App. F.2 (K=8, D=4, sigma^2=0.125, 51,717 params, "
                "Adam wd=0, OneCycleLR max_lr=0.003 peak at T/2, B=256) rather than "
                "sweeps/regression/training-runs/L2H4Minf.yaml (which sets max_examples=16 "
                "and omits optimizer/scheduler, falling back to config.py SGD wd=1e-4), "
                "because analysis/L2H4Minf/llcs.pt has 8 token positions matching K=8."
            ),
            "model_config": {
                "name": "L2H4Minf",
                "task_size_D": 4,
                "max_examples_K": 8,
                "noise_variance": 0.125,
                "num_layers_L": 2,
                "num_heads_H": 4,
                "embed_size": 64,
                "mlp_size": 64,
                "n_params": sum(p.numel() for p in make_model().parameters()),
            },
            "sgld_config": sgld_cfg,
            "eval_milestones": {"peak_step": peak_step, "trough_step": trough_step},
            "canonicalization_global_worst_512seqs": {
                "ORBIT_max_delta_y_f64": worst_f64,
                "ORBIT_max_delta_y_f32": worst_f32,
                "ORBIT_max_delta_all5_f64": worst_all5_f64,
                "ORBIT_max_delta_all5_f32": worst_all5_f32,
            },
            "checkpoint_manifest_init_checks": manifest.get("init_equality_checks", {}),
            "upstream_parity_verification": upstream_parity,
            "shard_runtimes_s": [sp["elapsed_s"] for sp in shard_payloads],
            "devices": list({f"{sp['device']}:{sp['gpu_name']}" for sp in shard_payloads}),
        },
        "hoogland_reference": hoogland_ref,
        "hoogland_comparison": hoogland_comparison,
        "part1_adam_wd0": {
            "runs": adam_runs,
            "gp_stage_boundaries": gp_ours,
            "stage_decompositions": adam_stages,
        },
        "part2_adamw": {
            "runs": adamw_runs,
            "stage_decompositions": adamw_stages,
        },
        "part3_channel_ablation": part3,
        "part4_sgld_controls": {
            "milestones": part4,
            "peak_to_trough_deltas": part4_deltas,
        },
        "part5_remedies": part5_summary,
        "part6_gauss_newton": part6,
    }


# =============================================================================
# 9. Publication-Quality 8-Panel Figure (`--replot` reads JSON only)
# =============================================================================

def plot_results(data: dict[str, Any], png_path: Path = DEFAULT_PNG_PATH) -> None:
    plt.rcParams.update({
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.0,
        "legend.fontsize": 7.4,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "figure.dpi": 150,
    })

    fig, axes = plt.subplots(2, 4, figsize=(22.4, 10.4))
    fig.subplots_adjust(hspace=0.38, wspace=0.46, top=0.90, bottom=0.08, left=0.045, right=0.955)

    adam_runs = data["part1_adam_wd0"]["runs"]
    adamw_runs = data["part2_adamw"]["runs"]
    n_seeds = len(adam_runs)
    steps_a = [c["step"] for c in adam_runs[0]["checkpoints"]]
    t_max = max(steps_a)
    x_floor = 100.0 if t_max >= 5000 else 10.0
    x_a = np.array([max(float(s), x_floor) for s in steps_a], dtype=float)

    def stack_curve(runs, key_fn):
        mat = np.array([[key_fn(c) for c in r["checkpoints"]] for r in runs], dtype=float)
        mean = mat.mean(axis=0)
        se = mat.std(axis=0, ddof=1) / math.sqrt(len(runs)) if len(runs) > 1 else np.zeros_like(mean)
        return mean, se

    raw_m, raw_se = stack_curve(adam_runs, lambda c: c["sgld_raw"]["mean"])
    orb_m, orb_se = stack_curve(adam_runs, lambda c: c["sgld_orbit"]["mean"])
    mse_m, _ = stack_curve(adam_runs, lambda c: c["metrics"]["mse_id"])

    # --- Panel A: Faithful 500k Raw vs ORBIT-Balanced LLC + Hoogland 5-seed overlay ---
    ax = axes[0, 0]
    ev_m = data["metadata"]["eval_milestones"]
    t_pk = ev_m["peak_step"]
    t_tr = ev_m["trough_step"]
    ax.axvline(max(t_pk, x_floor), color="#7f8c8d", ls=":", lw=1.1, alpha=0.8)
    ax.axvline(max(t_tr, x_floor), color="#7f8c8d", ls=":", lw=1.1, alpha=0.8)
    if t_tr > t_pk:
        ax.axvspan(max(t_pk, x_floor), t_tr, color="#fde8e8", alpha=0.50)

    href = data.get("hoogland_reference")
    if href is not None and t_max >= 100_000:
        hs = np.array([max(float(s), x_floor) for s in href["steps"]])
        hm = np.array(href["llc_mean"])
        hse = np.array(href["llc_se"])
        ax.plot(hs, hm, "-", color="#7f8c8d", lw=1.6, alpha=0.85, label="Hoogland llcs.pt (5 seeds)")
        ax.fill_between(hs, hm - hse, hm + hse, color="#7f8c8d", alpha=0.18)

    ax.errorbar(x_a, raw_m, yerr=raw_se, fmt="-o", color="#c0392b", lw=2.0, ms=3.8, capsize=2.2, label=f"Ours Raw ({n_seeds} seeds)")
    ax.errorbar(x_a, orb_m, yerr=orb_se, fmt="-s", color="#1f618d", lw=2.0, ms=3.8, capsize=2.2, label="Ours ORBIT-Balanced")
    ax.set_xscale("log")
    if t_max >= 100_000:
        ax.set_xticks([100, 1000, 10000, 46003, 126262, 363636])
        ax.set_xticklabels(["0", "1k", "10k", "46k", "126k", "364k"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel(r"Estimated LLC $\hat\lambda$ ($\gamma=13.3, n\beta=66.7$)")
    ax.set_title(f"A. Faithful 500k Raw vs. ORBIT LLC ({n_seeds} seeds)")
    ax.set_ylim(-15, 190)
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(x_a, mse_m, "--", color="#444444", lw=1.3, alpha=0.75, label="ID MSE")
    ax2.set_ylabel("ID MSE", color="#444444")
    ax2.tick_params(axis="y", labelcolor="#444444")
    ax2.set_ylim(1.5, 4.7)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", frameon=True, framealpha=0.92)

    # --- Panel B: Stagewise Delta hat_lambda decomposition ---
    ax = axes[0, 1]
    stages = data["part1_adam_wd0"]["stage_decompositions"]
    s_keys = [k for k in stages.keys() if k != "LR34_gp_peak_to_trough"]
    def _fmt_k(step_val: int) -> str:
        if step_val == 0:
            return "0"
        if t_max >= 5000:
            return f"{round(step_val / 1000.0)}k"
        return str(step_val)
    s_labels = [f"{k.split('_')[0].replace('LR34', 'LR3–LR4')}\n({_fmt_k(stages[k]['t1'])}→{_fmt_k(stages[k]['t2'])})" for k in s_keys]
    idx = np.arange(len(s_keys))
    w = 0.26
    ax.axhline(0, color="black", lw=0.8)
    ax.bar(idx - w, [stages[k]["d_total_mean"] for k in s_keys], w, yerr=[stages[k]["d_total_se"] for k in s_keys], capsize=2.5, color="#34495e", label=r"$\Delta\hat\lambda_{\mathrm{total}}$ (Raw)")
    ax.bar(idx,     [stages[k]["d_func_mean"]  for k in s_keys], w, yerr=[stages[k]["d_func_se"]  for k in s_keys], capsize=2.5, color="#1f618d", label=r"$\Delta\hat\lambda_{\mathrm{func}}$ (ORBIT)")
    ax.bar(idx + w, [stages[k]["d_fiber_mean"] for k in s_keys], w, yerr=[stages[k]["d_fiber_se"] for k in s_keys], capsize=2.5, color="#e67e22", label=r"$\Delta\hat\lambda_{\mathrm{fiber}}$ (Fiber slide)")
    ax.set_xticks(idx)
    ax.set_xticklabels(s_labels)
    ax.set_ylabel(r"Stagewise $\Delta\hat\lambda$ (mean ± SE)")
    ax.set_title(r"B. Stagewise $\Delta\hat\lambda_{\mathrm{total}} = \Delta\hat\lambda_{\mathrm{func}} + \Delta\hat\lambda_{\mathrm{fiber}}$")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True)
    if "LR34_peak_to_trough" in stages and t_max >= 100_000:
        sh = stages["LR34_peak_to_trough"]["fiber_share_pct_mean"]
        sh_se = stages["LR34_peak_to_trough"]["fiber_share_pct_se"]
        ax.annotate(
            f"LR3–LR4 fiber share:\n{sh:.1f}% ± {sh_se:.1f}%",
            xy=(2 + w, stages["LR34_peak_to_trough"]["d_fiber_mean"] * 0.5),
            xytext=(2.1, 42),
            ha="center",
            fontsize=8.0,
            fontweight="bold",
            color="#a04000",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef5e7", edgecolor="#e67e22", lw=0.9),
            arrowprops=dict(arrowstyle="->", color="#a04000", lw=1.1),
        )

    # --- Panel C: Per-channel exact symmetry ablation at Trough ---
    ax = axes[0, 2]
    trough_keys = [k for k in data["part3_channel_ablation"] if k.startswith("trough")]
    if trough_keys:
        abl = data["part3_channel_ablation"][trough_keys[0]]
        ch_order = [
            "LN_mean_proj", "LNf_unembed_all5", "MLP", "QK", "VO",
            "LN1_mlp", "LN0_attn", "LN_all_unit_gamma", "ORBIT", "ORBIT+",
        ]
        ch_order = [c for c in ch_order if c in abl]
        lbl_map = {
            "LN_mean_proj": "Pre-LN $1_d$ proj",
            "LNf_unembed_all5": "LN$_f$ ↔ W$_U$ (all 5)",
            "MLP": "MLP ReLU scale",
            "QK": "Attn Q–K GL(16)",
            "VO": "Attn V–O GL(16)",
            "LN1_mlp": "Pre-MLP LN$_1$ ↔ FC1",
            "LN0_attn": "Pre-Attn LN$_0$ ↔ QKV",
            "LN_all_unit_gamma": "Unit-$\\gamma$ LN (ctrl)",
            "ORBIT": "ORBIT (pure group)",
            "ORBIT+": "ORBIT+ (extra moves)",
        }
        raw_v = abl["RAW"]["sgld"]["mean"]
        raw_se_a = abl["RAW"]["sgld"]["se"]
        vals = [abl[c]["sgld"]["mean"] - raw_v for c in ch_order]
        errs = [math.hypot(abl[c]["sgld"]["se"], raw_se_a) for c in ch_order]
        cols = []
        for c in ch_order:
            if c == "LN0_attn": cols.append("#c0392b")
            elif c == "LN1_mlp": cols.append("#f39c12")
            elif c == "ORBIT": cols.append("#1f618d")
            elif c == "ORBIT+": cols.append("#16a085")
            elif c == "LN_all_unit_gamma": cols.append("#7d3c98")
            else: cols.append("#95a5a6")
        yp = np.arange(len(ch_order))
        ax.axvline(0, color="black", lw=0.8)
        ax.barh(yp, vals, xerr=errs, color=cols, capsize=2.2, alpha=0.92)
        ax.set_yticks(yp)
        ax.set_yticklabels([lbl_map[c] for c in ch_order])
        ax.set_xlabel(r"$\hat\lambda_{\mathrm{channel}} - \hat\lambda_{\mathrm{raw}}$ at trough")
        ax.set_title(f"C. Symmetry Ablation (Seed 0, step {t_tr:,})")
        ax.grid(True, axis="x", alpha=0.25)

    # --- Panel D: Mechanistic LN0 <-> QKV fiber sliding vs genuine LNf collapse ---
    ax = axes[0, 3]
    ln00_m, ln00_se = stack_curve(adam_runs, lambda c: c["metrics"]["ln_stats"]["blk0_ln0"]["mean_abs_gamma"])
    lnf_m, lnf_se = stack_curve(adam_runs, lambda c: c["metrics"]["ln_stats"]["lnf_row0"]["mean_abs_gamma"])
    qkv0_m, qkv0_se = stack_curve(adam_runs, lambda c: c["metrics"]["ln_stats"]["blk0_ln0"]["downstream_w_norm"])
    lnf_eff_m, _ = stack_curve(adam_runs, lambda c: c["metrics"]["ln_stats"]["lnf_row0"]["frac_eff_lt_005"])
    ax.errorbar(x_a, ln00_m, yerr=ln00_se, fmt="-o", color="#c0392b", lw=1.9, ms=3.8, label=r"Blk0 LN$_0$ mean $|\gamma|$")
    ax.errorbar(x_a, lnf_m, yerr=lnf_se, fmt="-^", color="#27ae60", lw=1.9, ms=3.8, label=r"Final LN$_f$ mean $|\gamma|$")
    ax.plot(x_a, lnf_eff_m, ":^", color="#1e8449", lw=1.5, label=r"LN$_f$ frac $|\gamma W_{U,0}|<0.05$")
    ax.set_xscale("log")
    if t_max >= 100_000:
        ax.set_xticks([100, 1000, 10000, 46003, 126262, 363636])
        ax.set_xticklabels(["0", "1k", "10k", "46k", "126k", "364k"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel(r"Mean $|\gamma|$ / collapsed fraction")
    ax.set_title(r"D. LN$_0$↔QKV Sliding vs. LN$_f$ Collapse")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.errorbar(x_a, qkv0_m, yerr=qkv0_se, fmt="--s", color="#2980b9", lw=1.9, ms=3.8, label=r"$\|W_{\mathrm{QKV},0}\|_F$")
    ax2.set_ylabel(r"$\|W_{\mathrm{QKV},0}\|_F$", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center left", frameon=True, framealpha=0.92)

    # --- Panel E: Matched-Init AdamW Intervention (wd = 0.08) ---
    ax = axes[1, 0]
    ax.errorbar(x_a, raw_m, yerr=raw_se, fmt="-o", color="#c0392b", lw=1.9, ms=3.8, alpha=0.85, label="Adam (wd=0) Raw")
    ax.errorbar(x_a, orb_m, yerr=orb_se, fmt="--o", color="#e67e22", lw=1.7, ms=3.5, alpha=0.85, label="Adam (wd=0) ORBIT")
    if adamw_runs:
        steps_w = [c["step"] for c in adamw_runs[0]["checkpoints"]]
        x_w = np.array([max(float(s), x_floor) for s in steps_w], dtype=float)
        wr_m, wr_se = stack_curve(adamw_runs, lambda c: c["sgld_raw"]["mean"])
        wo_m, wo_se = stack_curve(adamw_runs, lambda c: c["sgld_orbit"]["mean"])
        ax.errorbar(x_w, wr_m, yerr=wr_se, fmt="-s", color="#1f618d", lw=2.0, ms=4.2, label="AdamW (wd=0.08) Raw")
        ax.errorbar(x_w, wo_m, yerr=wo_se, fmt="--s", color="#16a085", lw=1.7, ms=3.8, label="AdamW (wd=0.08) ORBIT")
    ax.set_xscale("log")
    if t_max >= 100_000:
        ax.set_xticks([100, 1000, 10000, 46003, 126262, 363636])
        ax.set_xticklabels(["0", "1k", "10k", "46k", "126k", "364k"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlabel("Training step (log scale)")
    ax.set_ylabel(r"Estimated LLC $\hat\lambda$ (mean ± SE)")
    ax.set_ylim(-15, 190)
    ax.set_title("E. Matched-Init AdamW Intervention (wd = 0.08)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    # --- Panel F: Measured Exact Gauss-Newton Spectrum (Trough: RAW vs ORBIT) ---
    ax = axes[1, 1]
    gn_seeds = sorted(data.get("part6_gauss_newton", {}).keys())
    if gn_seeds and "trough" in data["part6_gauss_newton"][gn_seeds[0]]:
        gnt = data["part6_gauss_newton"][gn_seeds[0]]["trough"]
        tau = gnt["raw"]["threshold_gamma_over_nbeta"]
        ev_fr = np.array(gnt["raw"]["blocks"]["full_model"]["top_512_eigs"])
        ev_fo = np.array(gnt["orbit"]["blocks"]["full_model"]["top_512_eigs"])
        ev_qr = np.array(gnt["raw"]["blocks"]["W_QKV"]["top_512_eigs"])
        ev_qo = np.array(gnt["orbit"]["blocks"]["W_QKV"]["top_512_eigs"])
        rk = np.arange(1, len(ev_fr) + 1)
        ax.semilogy(rk, np.maximum(ev_fr, 1e-6), "-", color="#c0392b", lw=2.0, label=f"Full RAW (ED={gnt['raw']['blocks']['full_model']['laplace_ed_g13p3']:.1f})")
        ax.semilogy(rk, np.maximum(ev_fo, 1e-6), "-", color="#1f618d", lw=2.0, label=f"Full ORBIT (ED={gnt['orbit']['blocks']['full_model']['laplace_ed_g13p3']:.1f})")
        ax.semilogy(rk, np.maximum(ev_qr, 1e-6), "--", color="#e67e22", lw=1.6, label=f"W_QKV RAW (>τ: {gnt['raw']['blocks']['W_QKV']['n_gt_tau']})")
        ax.semilogy(rk, np.maximum(ev_qo, 1e-6), "--", color="#16a085", lw=1.6, label=f"W_QKV ORBIT (>τ: {gnt['orbit']['blocks']['W_QKV']['n_gt_tau']})")
        ax.axhline(tau, color="black", ls=":", lw=1.3, label=fr"$\gamma/(n\beta)={tau:.3f}$")
        ax.set_xlabel("Eigenvalue rank $k$ (top 512 of 4,096)")
        ax.set_ylabel(r"Exact Gauss–Newton eigenvalue $\mu_k$")
        ax.set_title("F. Exact Gauss–Newton Spectrum at Trough")
        ax.set_ylim(1e-4, max(float(ev_fo[0]), float(ev_fr[0])) * 2.5)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", frameon=True, framealpha=0.92)

    # --- Panel G: Remedy Comparison (Peak -> Trough Delta hat_lambda, RAW vs ORBIT) ---
    ax = axes[1, 2]
    rem = data.get("part5_remedies", {})
    est_order = [e for e in ["vanilla", "rmsprop_psgld", "param_rms_covariant", "orbit_covariant"] if e in rem]
    est_labels = {
        "vanilla": "Vanilla\nSGLD",
        "rmsprop_psgld": "RMSProp\npSGLD",
        "param_rms_covariant": "Param-RMS\nCovariant",
        "orbit_covariant": "Orbit-Cov.\nSGLD",
    }
    if est_order:
        x_e = np.arange(len(est_order))
        w_e = 0.34
        r_del = [rem[e]["raw_delta_mean_se"][0] for e in est_order]
        r_err = [rem[e]["raw_delta_mean_se"][1] for e in est_order]
        o_del = [rem[e]["orbit_delta_mean_se"][0] for e in est_order]
        o_err = [rem[e]["orbit_delta_mean_se"][1] for e in est_order]
        ax.axhline(0, color="black", lw=0.8)
        ax.bar(x_e - w_e / 2, r_del, w_e, yerr=r_err, capsize=2.5, color="#c0392b", alpha=0.9, label="RAW (Peak→Trough)")
        ax.bar(x_e + w_e / 2, o_del, w_e, yerr=o_err, capsize=2.5, color="#1f618d", alpha=0.9, label="ORBIT (Peak→Trough)")
        ax.set_xticks(x_e)
        ax.set_xticklabels([est_labels[e] for e in est_order])
        ax.set_ylabel(r"$\Delta\hat\lambda$ (Trough $-$ Peak)")
        ax.set_title("G. Reparametrization-Aware Remedies")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="lower right", frameon=True, framealpha=0.92)

    # --- Panel H: Finite-gamma sweep & Full-Batch-Gradient SGLD control ---
    ax = axes[1, 3]
    p4_m = data.get("part4_sgld_controls", {}).get("milestones", {})
    if "peak" in p4_m and "trough" in p4_m:
        gammas = [3.3, 13.3, 40.0]
        g_keys = ["g3.3_e3e-4", "g13.3_e3e-4", "g40.0_e3e-4"]
        r_tr = [p4_m["trough"][k]["raw"]["mean"] for k in g_keys]
        r_tr_e = [p4_m["trough"][k]["raw"]["se"] for k in g_keys]
        o_tr = [p4_m["trough"][k]["orbit"]["mean"] for k in g_keys]
        o_tr_e = [p4_m["trough"][k]["orbit"]["se"] for k in g_keys]
        r_pk = [p4_m["peak"][k]["raw"]["mean"] for k in g_keys]
        o_pk = [p4_m["peak"][k]["orbit"]["mean"] for k in g_keys]
        ax.errorbar(gammas, r_tr, yerr=r_tr_e, fmt="-o", color="#c0392b", lw=2.0, ms=5.5, capsize=2.5, label="Trough RAW")
        ax.errorbar(gammas, o_tr, yerr=o_tr_e, fmt="-s", color="#1f618d", lw=2.0, ms=5.5, capsize=2.5, label="Trough ORBIT")
        ax.plot(gammas, r_pk, ":o", color="#e74c3c", lw=1.4, ms=4.0, alpha=0.75, label="Peak RAW")
        ax.plot(gammas, o_pk, ":s", color="#5dade2", lw=1.4, ms=4.0, alpha=0.75, label="Peak ORBIT")
        if "g13.3_FB1024" in p4_m["trough"]:
            fb_r = p4_m["trough"]["g13.3_FB1024"]["raw"]["mean"]
            fb_o = p4_m["trough"]["g13.3_FB1024"]["orbit"]["mean"]
            ax.scatter([15.2], [fb_r], marker="D", s=44, color="#922b21", zorder=6, label="Trough Full-Batch-Grad RAW")
            ax.scatter([15.2], [fb_o], marker="D", s=44, color="#154360", zorder=6, label="Trough Full-Batch-Grad ORBIT")
        ax.set_xscale("log")
        ax.set_xticks(gammas)
        ax.set_xticklabels(["3.3", "13.3\n(default)", "40.0"])
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlim(2.6, 50.0)
        ax.set_xlabel(r"Localization $\gamma$")
        ax.set_ylabel(r"Estimated LLC $\hat\lambda$")
        ax.set_title(r"H. Finite-$\gamma$ Sweep & Full-Batch-Grad Control")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower left", fontsize=6.9, frameon=True, framealpha=0.92)

    fig.suptitle(
        "Is That LLC Drop Development or Fiber Sliding? Faithful Decomposition, Gauss–Newton Spectra & Remedies",
        fontsize=12.5,
        fontweight="bold",
        y=0.965,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved 8-panel figure to {png_path}", flush=True)


# =============================================================================
# 10. Main CLI Orchestrator
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track A: Faithful 500k DevInterp Fiber Decomposition + Remedies")
    p.add_argument("--smoke", action="store_true", help="Fast end-to-end verification of all components")
    p.add_argument("--train-only", action="store_true", help="Train models and save .safetensors checkpoints + manifest")
    p.add_argument("--from-checkpoints", "--eval-only", dest="eval_only", action="store_true", help="Evaluate saved checkpoints")
    p.add_argument("--merge-shards", action="store_true", help="Merge shard JSONs into final JSON and plot")
    p.add_argument("--replot", action="store_true", help="Redraw figure directly from saved JSON")

    p.add_argument("--seeds", type=str, default="0,1,2", help="Comma-separated seeds (e.g. 0,1,2 or 0,1,2,3,4)")
    p.add_argument("--optimizers", type=str, default="adam,adamw", help="Comma-separated optimizers (adam,adamw)")
    p.add_argument("--steps", type=int, default=500_000, help="Training steps T (default: 500000)")
    p.add_argument("--adamw-wd", type=float, default=0.08, help="AdamW weight decay (default: 0.08)")

    p.add_argument("--shard", type=str, default="0/1", help="Shard spec k/N for --from-checkpoints (e.g. 0/4)")
    p.add_argument("--eval-seeds", type=str, default="", help="Optional comma-separated seed filter for --from-checkpoints")
    p.add_argument("--parts", type=str, default="all", choices=["all", "traj", "deep"], help="Which parts to evaluate")
    p.add_argument("--n-shards", type=int, default=1, help="Total shards to merge in --merge-shards")
    p.add_argument("--shard-glob", type=str, default="", help="Optional glob pattern inside devinterp_shards/ for --merge-shards")
    p.add_argument("--peak-step", type=int, default=46003, help="Peak milestone step for ablation/controls/remedies/GN")
    p.add_argument("--trough-step", type=int, default=363636, help="Trough milestone step for ablation/controls/remedies/GN")

    p.add_argument("--n-chains", type=int, default=10, help="SGLD chains (Hoogland Table 2 default: 10)")
    p.add_argument("--n-burn", type=int, default=1000, help="SGLD burn-in steps (default: 1000)")
    p.add_argument("--n-draws", type=int, default=4000, help="SGLD draw steps (default: 4000)")
    p.add_argument("--m-batch", type=int, default=1024, help="SGLD minibatch size (default: 1024)")
    p.add_argument("--n-pool", type=int, default=1 << 20, help="SGLD fixed pool size (default: 2^20)")
    p.add_argument("--pool-seed", type=int, default=12345, help="Deterministic SGLD pool seed")

    p.add_argument("--out-dir", type=Path, default=None, help="Override output root directory (useful for --smoke)")
    p.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR, help="Checkpoint directory")
    p.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH, help="Final output JSON path")
    p.add_argument("--png-path", type=Path, default=DEFAULT_PNG_PATH, help="Final output PNG path")
    p.add_argument("--hoogland-ref", type=Path, default=HOOGLAND_REF_PATH, help="Hoogland 5-seed reference JSON")
    p.add_argument("--upstream-parity-json", type=Path, default=None, help="Optional upstream parity JSON to embed")

    p.add_argument("--device", type=str, default="auto", help="cuda | mps | cpu | auto")
    p.add_argument("--tf32", action="store_true", help="Enable TF32 matmul acceleration on Ampere+ GPUs")
    p.add_argument("--no-compile", action="store_true", help="Disable torch.compile on CUDA")
    return p.parse_args()


def _resolve_device(dev_arg: str) -> str:
    if dev_arg != "auto":
        return dev_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    args = parse_args()

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        args.ckpt_dir = args.out_dir / "devinterp_checkpoints"
        args.json_path = args.out_dir / "devinterp_fiber_decomposition.json"
        args.png_path = args.out_dir / "devinterp_fiber_decomposition.png"

    if args.replot:
        data = json.loads(args.json_path.read_text())
        plot_results(data, png_path=args.png_path)
        return

    device = _resolve_device(args.device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
    use_compile = (device == "cuda") and (not args.no_compile)

    # Fast end-to-end --smoke mode
    if args.smoke:
        assert args.out_dir is not None, "Always pass --out-dir with --smoke to keep repo results clean!"
        print(f"=== SMOKE TEST MODE on {device} (out_dir={args.out_dir}) ===", flush=True)
        smoke_steps = [0, 50, 150, 250, 400]
        seeds = [0, 1]
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        for opt_t, wd_v in [("adam", 0.0), ("adamw", args.adamw_wd)]:
            recs = train_batched_runs(
                opt_t, wd_v, seeds, 400, smoke_steps, args.ckpt_dir,
                device=device, use_compile=use_compile,
            )
            manifest = update_manifest(args.ckpt_dir, recs, {"mode": "smoke", "steps": 400, "tf32": args.tf32})
        print("Init equality checks:", json.dumps(manifest["init_equality_checks"], indent=2), flush=True)

        # Verify BatchedDTransformerBMM parity vs DTransformer in f32 and f64
        sd0 = load_checkpoint_safetensors(args.ckpt_dir / "adam_wd0.00_seed0" / "step_000400.safetensors")
        for dt_name, dt in [("float32", torch.float32), ("float64", torch.float64)]:
            m_ref = make_model(dtype=dt, device=device)
            m_ref.load_state_dict({k: v.to(device=device, dtype=dt) for k, v in sd0.items()})
            bm_chk = BatchedDTransformerBMM([sd0, sd0], dtype=dt, device=device)
            toks_t = torch.randn(16, 16, 5, device=device, dtype=dt)
            y1 = m_ref.token_sequence_transformer(toks_t)
            y2 = bm_chk.forward_toks(toks_t.unsqueeze(0).expand(2, -1, -1, -1))
            diff = float((y2 - y1.unsqueeze(0)).abs().max().item())
            print(f"BatchedDTransformerBMM parity ({dt_name}): max|diff| = {diff:.3e}", flush=True)
            assert diff < (1e-5 if dt == torch.float32 else 1e-13)

        # Verify orbit_covariant_sgld exact equivariance along an artificial extreme LN0 rescaling (10x!)
        m_art = make_model(dtype=torch.float64, device="cpu")
        m_art.load_state_dict({k: v.double() for k, v in sd0.items()})
        with torch.no_grad():
            for blk in m_art.token_sequence_transformer.blocks:
                blk.layer_norms[0].weight.mul_(0.15)
                blk.layer_norms[0].bias.mul_(0.15)
                blk.attention.attention.weight.div_(0.15)
        sd_skewed = {k: v.float().cpu() for k, v in m_art.state_dict().items()}
        toks_sp, y_sp = generate_sgld_pool(12345, 1 << 14, device=device)
        il0 = compute_pool_init_loss(sd0, toks_sp, y_sp, device=device)
        il_sk = compute_pool_init_loss(sd_skewed, toks_sp, y_sp, device=device)
        can_sk = canonicalize_state_dict(sd_skewed, torch.randn(64, 8, 4, dtype=torch.float64), torch.randn(64, 8, 1, dtype=torch.float64), channel="ORBIT")
        van_sk = run_sgld_batched(sd_skewed, toks_sp, y_sp, il_sk, estimator="vanilla", n_chains=4, n_burn=50, n_draws=150, m_batch=256, seed=77, device=device, use_compile=use_compile)
        van_or = run_sgld_batched(can_sk["sd"], toks_sp, y_sp, il_sk, estimator="vanilla", n_chains=4, n_burn=50, n_draws=150, m_batch=256, seed=77, device=device, use_compile=use_compile)
        cov_sk = run_sgld_batched(sd_skewed, toks_sp, y_sp, il_sk, estimator="orbit_covariant", n_chains=4, n_burn=50, n_draws=150, m_batch=256, seed=77, device=device, use_compile=use_compile)
        cov_or = run_sgld_batched(can_sk["sd"], toks_sp, y_sp, il_sk, estimator="orbit_covariant", n_chains=4, n_burn=50, n_draws=150, m_batch=256, seed=77, device=device, use_compile=use_compile)
        print(
            f"10x-skewed LN0 equivariance check | Vanilla(RAW={van_sk['mean']:.2f} vs ORBIT={van_or['mean']:.2f}) | "
            f"OrbitCov(RAW={cov_sk['mean']:.2f} vs ORBIT={cov_or['mean']:.2f})",
            flush=True,
        )

        # Test sharding (2 shards) + merge + plot!
        shards = []
        for s_i in range(2):
            sp = execute_eval_shard(
                manifest, args.ckpt_dir, s_i, 2, peak_step=150, trough_step=400,
                n_chains=4, n_burn=50, n_draws=150, m_batch=256, n_pool=1 << 14,
                pool_seed=12345, device=device, use_compile=use_compile, is_smoke=True,
            )
            (args.out_dir / f"shard_{s_i}_of_2.json").write_text(json.dumps(sp, indent=2))
            shards.append(sp)

        href = json.loads(args.hoogland_ref.read_text()) if args.hoogland_ref.exists() else None
        up_p = json.loads(args.upstream_parity_json.read_text()) if (args.upstream_parity_json and args.upstream_parity_json.exists()) else None
        payload = merge_shards_to_payload(manifest, shards, href, up_p)
        args.json_path.write_text(json.dumps(payload, indent=2))
        plot_results(payload, png_path=args.png_path)
        print(f"=== SMOKE TEST PASSED! JSON={args.json_path} PNG={args.png_path} ===", flush=True)
        return

    if args.train_only:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
        opts = [x.strip().lower() for x in args.optimizers.split(",") if x.strip()]
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)
        for opt_t in opts:
            wd_v = 0.0 if opt_t == "adam" else args.adamw_wd
            ck_steps = ADAM_CKPT_STEPS_500K if (opt_t == "adam" and args.steps == 500_000) else (
                ADAMW_CKPT_STEPS_500K if (opt_t == "adamw" and args.steps == 500_000)
                else sorted(set([0, args.steps // 10, args.steps // 4, args.steps // 2, args.steps]))
            )
            recs = train_batched_runs(
                opt_t, wd_v, seeds, args.steps, ck_steps, args.ckpt_dir,
                device=device, use_compile=use_compile,
            )
            manifest = update_manifest(args.ckpt_dir, recs, {"steps": args.steps, "tf32": args.tf32})
        print(f"Training complete. Manifest updated at {args.ckpt_dir / 'manifest.json'}", flush=True)
        return

    if args.eval_only:
        manifest = json.loads((args.ckpt_dir / "manifest.json").read_text())
        s_idx_str, n_sh_str = args.shard.split("/")
        s_idx, n_sh = int(s_idx_str), int(n_sh_str)
        eval_seeds = {int(x) for x in args.eval_seeds.split(",") if x.strip()} if args.eval_seeds.strip() else None
        shard_dir = args.json_path.parent / "devinterp_shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        tag = f"_{args.parts}" if args.parts != "all" else ""
        if eval_seeds is not None:
            tag += "_s" + "".join(str(s) for s in sorted(eval_seeds))
        sp_path = shard_dir / f"shard_{s_idx}_of_{n_sh}{tag}.json"
        sp = execute_eval_shard(
            manifest, args.ckpt_dir, s_idx, n_sh,
            peak_step=args.peak_step, trough_step=args.trough_step,
            n_chains=args.n_chains, n_burn=args.n_burn, n_draws=args.n_draws,
            m_batch=args.m_batch, n_pool=args.n_pool, pool_seed=args.pool_seed,
            device=device, use_compile=use_compile, is_smoke=False,
            checkpoint_path=sp_path, eval_seeds=eval_seeds, parts=args.parts,
        )
        sp_path.write_text(json.dumps(sp, indent=2))
        print(f"Saved shard result to {sp_path}", flush=True)
        return

    if args.merge_shards:
        manifest = json.loads((args.ckpt_dir / "manifest.json").read_text())
        shard_dir = args.json_path.parent / "devinterp_shards"
        if args.shard_glob:
            shard_paths = sorted(shard_dir.glob(args.shard_glob))
            shards = [json.loads(p.read_text()) for p in shard_paths]
        else:
            shards = [json.loads((shard_dir / f"shard_{i}_of_{args.n_shards}.json").read_text()) for i in range(args.n_shards)]
        assert shards, f"No shard files found in {shard_dir}"
        href = json.loads(args.hoogland_ref.read_text()) if args.hoogland_ref.exists() else None
        up_p = json.loads(args.upstream_parity_json.read_text()) if (args.upstream_parity_json and args.upstream_parity_json.exists()) else None
        payload = merge_shards_to_payload(manifest, shards, href, up_p)
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(payload, indent=2))
        print(f"Merged {len(shards)} shards into {args.json_path}", flush=True)
        plot_results(payload, png_path=args.png_path)
        return

    raise SystemExit("Specify one of --smoke, --train-only, --from-checkpoints, --merge-shards, or --replot.")


if __name__ == "__main__":
    main()
