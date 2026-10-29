#!/usr/bin/env python3
"""Multi-seed GPT4MTS–Environment under Time-MMD official L=96,H=48."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_npf_conditions import confirm_one
from analyze_npf_gating import domain_gate_params, eval_with_params, load_paired_with_feats
from npf_timemmd_official import DOMAIN_LH, SCALE_SEEDS, run_rel

NPF = Path(__file__).resolve().parents[1]
DS = "Environment"
MEMBER = "GPT4MTS"
SL, PL, _ = DOMAIN_LH[DS]
DATA_CSV = NPF / "third_party/TaTS/data" / f"{DS}.csv"


def paired_ci(d: np.ndarray, n_boot: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = []
    n = len(d)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(float(d[idx].mean()))
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def main() -> None:
    cfg = NPF / "outputs/aggregate/gating_config.json"
    q_star = float(json.loads(cfg.read_text())["q_star"]) if cfg.exists() else 0.60
    explore_path = NPF / "outputs" / run_rel("TaTS", DS, 2025) / "condition_explore.json"
    explore = json.loads(explore_path.read_text()) if explore_path.exists() else None

    util_rows = []
    cond_rows = {}
    gate_rows = []

    for seed in SCALE_SEEDS:
        out = NPF / "outputs" / run_rel(MEMBER, DS, seed)
        paired = out / "paired_test.csv"
        if not paired.exists():
            print("skip missing", paired)
            continue
        import pandas as pd

        df = pd.read_csv(paired)
        d = df["D"].to_numpy(dtype=float)
        lo, hi = paired_ci(d, seed=seed)
        util_rows.append(
            {
                "seed": seed,
                "mean_D": float(d.mean()),
                "ci95": [lo, hi],
                "r_neg": float((d < 0).mean()),
                "n": int(len(d)),
            }
        )

        if explore is not None:
            report = confirm_one(explore, paired, DATA_CSV, SL, PL, MEMBER, DS, min_ratio=0.5)
            (out / "condition_confirm.json").write_text(json.dumps(report, indent=2) + "\n")
            for r in report["results"]:
                feat = r["feature"]
                cond_rows.setdefault(feat, {})[seed] = r.get("confirm_delta")

        params = domain_gate_params(DATA_CSV, SL, PL, q_star)
        gdf = load_paired_with_feats(paired, DATA_CSV, SL, PL)
        m = eval_with_params(gdf, params, seed=seed)
        gate_rows.append(
            {
                "seed": seed,
                "mse_g": m["mse_g"],
                "mse_f0": m["mse_f0"],
                "mse_f1": m["mse_f1"],
                "delta_f1_minus_g": m["delta_f1_minus_g"],
                "beats_f1": m["beats_f1"],
                "close_text_rate": m["close_text_rate"],
            }
        )
        print(
            f"seed{seed}: meanD={d.mean():+.4g} r-={(d<0).mean():.3f} "
            f"Δ(f1-g)={m['delta_f1_minus_g']:+.4g}"
        )

    report = {
        "protocol": "Time-MMD official",
        "member": MEMBER,
        "dataset": DS,
        "seq_len": SL,
        "pred_len": PL,
        "q_star": q_star,
        "utility": util_rows,
        "conditions": cond_rows,
        "gating": gate_rows,
    }
    agg = NPF / "outputs/aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "scale_seeds_official.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"# Multi-seed {MEMBER}–{DS} (L={SL}, H={PL})",
        "",
        "## Utility",
        "| seed | mean D | 95% CI | r- |",
        "|-----:|-------:|--------|---:|",
    ]
    for r in util_rows:
        lines.append(
            f"| {r['seed']} | {r['mean_D']:+.4f} | [{r['ci95'][0]:+.4f},{r['ci95'][1]:+.4f}] | {r['r_neg']:.3f} |"
        )
    lines += ["", "## Gating (frozen q*)", "| seed | MSE(g) | Δ(f1−g) | g<f1 |", "|-----:|-------:|--------:|:----:|"]
    for r in gate_rows:
        lines.append(
            f"| {r['seed']} | {r['mse_g']:.4f} | {r['delta_f1_minus_g']:+.4f} | "
            f"{'Y' if r['beats_f1'] else 'N'} |"
        )
    if cond_rows:
        lines += ["", "## Condition Δ_X (confirm)", "| feature | " + " | ".join(str(s) for s in SCALE_SEEDS) + " |"]
        lines.append("|---------|" + "|".join(["------:"] * len(SCALE_SEEDS)) + "|")
        for feat, by_seed in cond_rows.items():
            vals = " | ".join(
                f"{by_seed[s]:+.4f}" if s in by_seed and by_seed[s] is not None else "NA"
                for s in SCALE_SEEDS
            )
            lines.append(f"| {feat} | {vals} |")
    (agg / "scale_seeds_official.md").write_text("\n".join(lines) + "\n")
    print("wrote", agg / "scale_seeds_official.md")


if __name__ == "__main__":
    main()
