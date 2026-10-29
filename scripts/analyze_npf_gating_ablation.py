#!/usr/bin/env python3
"""Ablation for six-proxy tertile hard-label gate.

- full_any / full_maj / full_all on 6 starred proxies
- trio_any / trio_maj / trio_all on IQR+|b|+Vol
- single high-tertile close for each of the 6 proxies
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_npf_gating import (
    CLOSE_RULES,
    FEATS_3,
    RISK_FEATS,
    apply_gate,
    build_features_train,
    domain_gate_params,
    eval_with_params,
    fit_tertile_cuts,
    high_mask,
    load_paired_with_feats,
    metrics,
    random_gate,
    train_indices,
)


def domain_params_single(data_csv: Path, seq_len: int, pred_len: int, feat: str) -> dict:
    n_rows = len(pd.read_csv(data_csv))
    idx = train_indices(n_rows, seq_len, pred_len)
    train_feats = build_features_train(data_csv, idx, seq_len)
    cuts = fit_tertile_cuts(train_feats, feats=(feat,))
    close_train = high_mask(train_feats, cuts, feat)
    return {
        "mode": "tertile_single",
        "close_rule": "single",
        "feats": [feat],
        "single_feat": feat,
        "tertile_cuts": cuts,
        "n_train_windows": int(len(train_feats)),
        "train_close_rate": float(close_train.mean()),
    }


def eval_single(df, params: dict, feat: str, seed: int = 2025) -> dict:
    close = high_mask(df, params["tertile_cuts"], feat)
    mse0 = df["mse_f0"].to_numpy(dtype=float)
    mse1 = df["mse_f1"].to_numpy(dtype=float)
    m = metrics(apply_gate(mse0, mse1, close), mse0, mse1, close)
    m["close_rule"] = "single"
    m["single_feat"] = feat
    m["random"] = random_gate(mse0, mse1, m["close_text_rate"], seed=seed)
    return m


def units_official() -> list[tuple]:
    from npf_timemmd_official import DOMAIN_LH, run_rel

    specs = [
        ("calibrate", "GPT4MTS", "Environment"),
        ("confirm", "TaTS", "Environment"),
        ("confirm", "CFA-naive", "Environment"),
        ("confirm", "CFA-naive", "Energy"),
        ("confirm", "CFA-naive", "Health"),
        ("confirm", "CFA-naive", "Climate"),
        ("extra", "TaTS", "Energy"),
        ("extra", "GPT4MTS", "Energy"),
        ("extra", "TaTS", "Health"),
        ("extra", "GPT4MTS", "Health"),
        ("extra", "TaTS", "Climate"),
        ("extra", "GPT4MTS", "Climate"),
        ("extra", "TaTS", "Health_AFR"),
        ("extra", "GPT4MTS", "Health_AFR"),
        ("extra", "CFA-naive", "Health_AFR"),
    ]
    out = []
    for role, member, dataset in specs:
        sl, pl, _ = DOMAIN_LH[dataset]
        rel = run_rel(member, dataset)
        folder = rel.split("/")[0]
        run = rel.split("/")[-1]
        src = "cfa" if member == "CFA-naive" else "tats"
        out.append((role, folder, member, dataset, run, src, sl, pl))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    ap.add_argument("--tats-data-root", default=str(Path(__file__).resolve().parents[1] / "third_party/TaTS/data"))
    ap.add_argument("--cfa-data-root", default=str(Path(__file__).resolve().parents[1] / "third_party/cfa/data"))
    args = ap.parse_args()

    root = Path(args.outputs_root)
    tats = Path(args.tats_data_root)
    cfa = Path(args.cfa_data_root)

    cfg_path = root / "aggregate" / "gating_config.json"
    rule_star = "maj"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        rule_star = cfg.get("close_rule") or "maj"

    variants: dict[str, tuple] = {}
    for rule in CLOSE_RULES:
        variants[f"full6_{rule}"] = ("full", rule, RISK_FEATS)
        variants[f"trio3_{rule}"] = ("full", rule, FEATS_3)
    for feat in RISK_FEATS:
        short = {
            "hist_std": "std",
            "hist_iqr": "iqr",
            "pred_ease_ar1": "ar1",
            "hist_abs_slope": "slope",
            "acf1": "acf1",
            "vol_change": "vol",
        }[feat]
        variants[f"single_{short}"] = ("single", feat, (feat,))

    feature_ablation = []
    for role, folder, member, dataset, run, src, sl, pl in units_official():
        run_dir = root / folder / dataset / run
        paired = run_dir / "paired_test.csv"
        if not paired.exists():
            print("skip", paired)
            continue
        data_csv = cfa / dataset / f"{dataset}_NPF.csv" if src == "cfa" else tats / f"{dataset}.csv"
        df = load_paired_with_feats(paired, data_csv, sl, pl)
        unit = f"{member}/{dataset}"
        mean_D = float(df["D"].mean())

        for vname, (kind, arg, feats) in variants.items():
            if kind == "full":
                params = domain_gate_params(data_csv, sl, pl, close_rule=arg, feats=feats)
                m = eval_with_params(df, params, seed=2025)
            else:
                params = domain_params_single(data_csv, sl, pl, arg)
                m = eval_single(df, params, arg)
            feature_ablation.append(
                {
                    "variant": vname,
                    "unit": unit,
                    "role": role,
                    "mean_D": mean_D,
                    "n": len(df),
                    "mse_g": m["mse_g"],
                    "mse_f0": m["mse_f0"],
                    "mse_f1": m["mse_f1"],
                    "delta_f1_minus_g": m["delta_f1_minus_g"],
                    "close_text_rate": m["close_text_rate"],
                    "beats_f1": m["beats_f1"],
                    "not_worse_f0": m["not_worse_f0"],
                    "vs_random": float(m["random"]["mse_g"] - m["mse_g"]),
                }
            )
        print("done", unit)

    all_units = {r["unit"] for r in feature_ablation}
    neg_units = {r["unit"] for r in feature_ablation if r["mean_D"] < 0}

    def summarize(unit_set: set[str]) -> dict:
        out = {}
        for vname in variants:
            rows = [r for r in feature_ablation if r["variant"] == vname and r["unit"] in unit_set]
            if not rows:
                continue
            out[vname] = {
                "n_units": len(rows),
                "n_beats_f1": sum(1 for r in rows if r["beats_f1"]),
                "mean_delta_f1_minus_g": float(np.mean([r["delta_f1_minus_g"] for r in rows])),
                "mean_close_rate": float(np.mean([r["close_text_rate"] for r in rows])),
                "mean_vs_random": float(np.mean([r["vs_random"] for r in rows])),
            }
        return out

    summary_all = summarize(all_units)
    summary_neg = summarize(neg_units)
    report = {
        "close_rule_frozen": rule_star,
        "protocol": "tertile_hard_label_six_starred",
        "risk_feats": list(RISK_FEATS),
        "feats_3": list(FEATS_3),
        "variants": {k: list(v[2]) if v[0] == "full" else [v[1]] for k, v in variants.items()},
        "feature_ablation": feature_ablation,
        "summary_all_15_by_variant": summary_all,
        "summary_meanD_neg_by_variant": summary_neg,
    }
    agg = root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "gating_ablation.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Tertile hard-label ablation (6 starred)",
        "",
        f"Frozen close_rule (config)={rule_star}",
        "",
        "## All 15",
        "",
        "| variant | beats_f1 | mean Δ(f1−g) | mean close | vs random |",
        "|---------|----------|--------------|------------|-----------|",
    ]
    for vname, s in summary_all.items():
        lines.append(
            f"| {vname} | {s['n_beats_f1']}/{s['n_units']} | "
            f"{s['mean_delta_f1_minus_g']:+.4g} | {s['mean_close_rate']:.3f} | "
            f"{s['mean_vs_random']:+.4g} |"
        )
    lines += ["", "## D̄<0", ""]
    lines.append("| variant | beats_f1 | mean Δ | vs random |")
    lines.append("|---------|----------|---------|-----------|")
    for vname, s in summary_neg.items():
        lines.append(
            f"| {vname} | {s['n_beats_f1']}/{s['n_units']} | "
            f"{s['mean_delta_f1_minus_g']:+.4g} | {s['mean_vs_random']:+.4g} |"
        )
    (agg / "gating_ablation.md").write_text("\n".join(lines) + "\n")
    print("=== ablation summary all15 ===")
    for v, s in summary_all.items():
        print(v, s)
    print("wrote", agg / "gating_ablation.md")


if __name__ == "__main__":
    main()
