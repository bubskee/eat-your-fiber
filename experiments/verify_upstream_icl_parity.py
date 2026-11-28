"""
Standalone upstream parity verification for `experiments/devinterp_fiber_decomposition.py`.

Loads `timaeus-research/icl`'s `InContextRegressionTransformer` (pinned upstream commit
`5baf73025d279abd2d56c98e13bf5663608709c0`), copies exact weights into our replica,
and verifies `max|Delta y|` across 512 fresh held-out sequences in both float64 and float32,
for both the regression readout (`ys_pred`) and all 5 `DTransformer` token output channels.

Usage:
  python experiments/verify_upstream_icl_parity.py --icl-repo /path/to/timaeus-research/icl
  python experiments/verify_upstream_icl_parity.py --icl-repo /path/to/icl --json-out parity.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
import types
from pathlib import Path

import torch

EXPECTED_UPSTREAM_COMMIT = "5baf73025d279abd2d56c98e13bf5663608709c0"


def _load_upstream_classes(icl_repo: Path):
    """Load `DTransformer` and `InContextRegressionTransformer` directly from `icl_repo`
    without needing `icl` installed or its heavy plotting dependencies."""
    dtrans_path = icl_repo / "src" / "icl" / "regression" / "dtransformer.py"
    model_path = icl_repo / "src" / "icl" / "regression" / "model.py"
    if not dtrans_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Could not find upstream files under {icl_repo}/src/icl/regression/{{dtransformer,model}}.py"
        )

    # Create lightweight namespace packages so `from icl.regression.dtransformer import DTransformer` resolves cleanly
    for mod_name in ("icl", "icl.regression"):
        if mod_name not in sys.modules:
            pkg = types.ModuleType(mod_name)
            pkg.__path__ = []
            sys.modules[mod_name] = pkg

    spec_dt = importlib.util.spec_from_file_location("icl.regression.dtransformer", dtrans_path)
    mod_dt = importlib.util.module_from_spec(spec_dt)
    sys.modules["icl.regression.dtransformer"] = mod_dt
    assert spec_dt.loader is not None
    spec_dt.loader.exec_module(mod_dt)

    spec_m = importlib.util.spec_from_file_location("icl.regression.model", model_path)
    mod_m = importlib.util.module_from_spec(spec_m)
    sys.modules["icl.regression.model"] = mod_m
    assert spec_m.loader is not None
    spec_m.loader.exec_module(mod_m)

    return mod_m.InContextRegressionTransformer, mod_m.to_token_sequence


def verify_upstream_parity(icl_repo: Path, n_seqs: int = 512, seed: int = 2026) -> dict:
    # Import our local replica
    exp_dir = Path(__file__).resolve().parent
    if str(exp_dir) not in sys.path:
        sys.path.insert(0, str(exp_dir))
    from devinterp_fiber_decomposition import make_model, to_token_sequence as replica_to_toks

    UpstreamModel, upstream_to_toks = _load_upstream_classes(icl_repo)

    # Read git commit SHA if available
    commit_sha = "unknown"
    try:
        commit_sha = subprocess.check_output(
            ["git", "-C", str(icl_repo), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        pass

    g = torch.Generator().manual_seed(seed)
    sigma = math.sqrt(0.125)
    w = torch.randn(n_seqs, 4, generator=g, dtype=torch.float64)
    x64 = torch.randn(n_seqs, 8, 4, generator=g, dtype=torch.float64)
    y64 = torch.bmm(x64, w.unsqueeze(-1)) + torch.randn(n_seqs, 8, 1, generator=g, dtype=torch.float64) * sigma

    # Build upstream model with deterministic weights, copy exact state_dict to replica
    torch.manual_seed(seed + 1)
    m_up_f32 = UpstreamModel(
        task_size=4,
        max_examples=8,
        embed_size=64,
        mlp_size=64,
        num_heads=4,
        num_layers=2,
        device="cpu",
        layer_norm=True,
        include_output=True,
    )
    m_rep_f32 = make_model(dtype=torch.float32, device="cpu")
    sd_up = m_up_f32.state_dict()
    sd_rep = m_rep_f32.state_dict()
    assert list(sd_up.keys()) == list(sd_rep.keys()), (
        f"State dict key mismatch: {set(sd_up.keys()) ^ set(sd_rep.keys())}"
    )
    n_params_up = sum(p.numel() for p in m_up_f32.parameters())
    n_params_rep = sum(p.numel() for p in m_rep_f32.parameters())
    assert n_params_up == n_params_rep == 51717, f"Param count mismatch: {n_params_up} vs {n_params_rep}"
    m_rep_f32.load_state_dict(sd_up)

    # Evaluate in float32
    x32, y32 = x64.float(), y64.float()
    with torch.no_grad():
        tok_diff_f32 = float((upstream_to_toks(x32, y32) - replica_to_toks(x32, y32)).abs().max().item())
        out_up_f32 = m_up_f32(x32, y32)
        out_rep_f32 = m_rep_f32(x32, y32)
        max_dy_f32 = float((out_up_f32 - out_rep_f32).abs().max().item())

        all_up_f32 = m_up_f32.token_sequence_transformer(upstream_to_toks(x32, y32))
        all_rep_f32 = m_rep_f32.token_sequence_transformer(replica_to_toks(x32, y32))
        max_dall_f32 = float((all_up_f32 - all_rep_f32).abs().max().item())

    # Evaluate in float64
    m_up_f64 = m_up_f32.double()
    m_rep_f64 = make_model(dtype=torch.float64, device="cpu")
    m_rep_f64.load_state_dict({k: v.double() for k, v in sd_up.items()})
    with torch.no_grad():
        tok_diff_f64 = float((upstream_to_toks(x64, y64) - replica_to_toks(x64, y64)).abs().max().item())
        out_up_f64 = m_up_f64(x64, y64)
        out_rep_f64 = m_rep_f64(x64, y64)
        max_dy_f64 = float((out_up_f64 - out_rep_f64).abs().max().item())

        all_up_f64 = m_up_f64.token_sequence_transformer(upstream_to_toks(x64, y64))
        all_rep_f64 = m_rep_f64.token_sequence_transformer(replica_to_toks(x64, y64))
        max_dall_f64 = float((all_up_f64 - all_rep_f64).abs().max().item())

    return {
        "upstream_repo": "github.com/timaeus-research/icl",
        "expected_commit_sha": EXPECTED_UPSTREAM_COMMIT,
        "verified_commit_sha": commit_sha,
        "commit_matches_pinned": commit_sha == EXPECTED_UPSTREAM_COMMIT,
        "n_params": n_params_up,
        "state_dict_keys_identical": True,
        "n_sequences": n_seqs,
        "seed": seed,
        "tokenization_max_diff_f32": tok_diff_f32,
        "tokenization_max_diff_f64": tok_diff_f64,
        "max_delta_y_f32": max_dy_f32,
        "max_delta_y_f64": max_dy_f64,
        "max_delta_all_5ch_f32": max_dall_f32,
        "max_delta_all_5ch_f64": max_dall_f64,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify replica parity against timaeus-research/icl.")
    parser.add_argument("--icl-repo", type=Path, required=True, help="Path to local clone of timaeus-research/icl")
    parser.add_argument("--n-seqs", type=int, default=512, help="Number of held-out sequences (default: 512)")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed (default: 2026)")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    res = verify_upstream_parity(args.icl_repo, n_seqs=args.n_seqs, seed=args.seed)
    print(json.dumps(res, indent=2))
    assert res["max_delta_y_f64"] < 1e-13, f"Float64 parity failed: {res['max_delta_y_f64']}"
    assert res["max_delta_y_f32"] < 1e-6, f"Float32 parity failed: {res['max_delta_y_f32']}"
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
