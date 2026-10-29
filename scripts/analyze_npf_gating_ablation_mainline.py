#!/usr/bin/env python3
"""Mainline structure ablation (top-venue 4-row table).

g*              : six starred + high tertile + ρ=any
Wrong feature   : Shock only + high tertile + any
Wrong polarity  : six starred + low tertile + any
Wrong windows   : permute g* close mask (same #closed windows)

Main-read: frozen held-out D̄<0 units (excl. calibrate GPT4MTS/Environment).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_npf_gating as g
from analyze_npf_gating_official import DOMAIN_LH, MEMBERS, data_csv, run_dir

SEED = 2025
NPF = Path(__file__).resolve().parents[1]
AGG = NPF / "outputs/aggregate"
SHOCK = "shock_score"

MAIN_READ_UNITS = (
    "TaTS/Environment",
    "CFA-naive/Environment",
    "GPT4MTS/Energy",
    "CFA-naive/Energy",
    "GPT4MTS/Health",
    "CFA-naive/Health",
    "CFA-naive/Health_AFR",
    "TaTS/Climate",
    "CFA-naive/Climate",
)


def low_mask(df, cuts, feat: str) -> np.ndarray:
    return df[feat].to_numpy(dtype=float) <= cuts[feat]["q1"]


def close_any(flags: np.ndarray) -> np.ndarray:
    return flags.sum(axis=1) >= 1


def eval_close(mse0, mse1, close, seed: int) -> dict:
    mse_g = g.apply_gate(mse0, mse1, close)
    m = g.metrics(mse_g, mse0, mse1, close)
    rnd = g.random_gate(mse0, mse1, m["close_text_rate"], seed=seed)
    m["delta_rnd"] = float(rnd["mse_g"] - m["mse_g"])
    return m


def permute_close_mask(close: np.ndarray, seed: int) -> np.ndarray:
    """Wrong windows: same #closed, randomly reassigned windows."""
    rng = np.random.default_rng(seed)
    out = close.copy().astype(bool)
    return rng.permutation(out)


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n_units": 0}
    return {
        "n_units": len(rows),
        "n_beats_f1": int(sum(1 for r in rows if r["beats_f1"])),
        "mean_delta_f1": float(np.mean([r["delta_f1_minus_g"] for r in rows])),
        "mean_delta_rnd": float(np.mean([r["delta_rnd"] for r in rows])),
        "mean_close_rate": float(np.mean([r["close_text_rate"] for r in rows])),
        "mean_mse_g": float(np.mean([r["mse_g"] for r in rows])),
        "mean_mse_f1": float(np.mean([r["mse_f1"] for r in rows])),
        "units": [r["unit"] for r in rows],
    }


def check_pass(star: dict, shock: dict, flip: dict, wrong_win: dict) -> dict:
    d_star, rnd_star = star["mean_delta_f1"], star["mean_delta_rnd"]
    out = {
        "wrong_feature": {
            "pass": bool(
                shock["mean_delta_f1"] < d_star
                and shock["mean_delta_rnd"] < rnd_star
            ),
            "delta": shock["mean_delta_f1"],
            "delta_rnd": shock["mean_delta_rnd"],
            "star_delta": d_star,
            "star_delta_rnd": rnd_star,
        },
        "wrong_polarity": {
            "pass": bool(flip["mean_delta_f1"] < d_star),
            "delta": flip["mean_delta_f1"],
            "star_delta": d_star,
        },
        "wrong_windows": {
            "pass": bool(
                wrong_win["mean_delta_f1"] < d_star
                and (
                    wrong_win["mean_delta_rnd"] <= 0.5 * rnd_star
                    if rnd_star > 0
                    else wrong_win["mean_delta_rnd"] <= rnd_star
                )
            ),
            "delta": wrong_win["mean_delta_f1"],
            "delta_rnd": wrong_win["mean_delta_rnd"],
            "star_delta": d_star,
            "star_delta_rnd": rnd_star,
        },
    }
    out["all_pass"] = bool(
        out["wrong_feature"]["pass"]
        and out["wrong_polarity"]["pass"]
        and out["wrong_windows"]["pass"]
    )
    return out


def main() -> None:
    root = NPF / "outputs"
    rows: list[dict] = []

    for dataset, (sl, pl) in DOMAIN_LH.items():
        for member in MEMBERS:
            unit = f"{member}/{dataset}"
            role = (
                "calibrate"
                if (member == "GPT4MTS" and dataset == "Environment")
                else "eval"
            )
            paired = run_dir(root, member, dataset, sl, pl) / "paired_test.csv"
            csv = data_csv(member, dataset)
            if not paired.exists() or not csv.exists():
                print("skip", unit)
                continue

            feats_all = tuple(list(g.RISK_FEATS) + [SHOCK])
            params = g.domain_gate_params(
                csv, sl, pl, close_rule="any", feats=feats_all
            )
            cuts = params["tertile_cuts"]
            df = g.load_paired_with_feats(paired, csv, sl, pl)
            if SHOCK not in df.columns:
                print("missing shock", unit)
                continue

            mse0 = df["mse_f0"].to_numpy(dtype=float)
            mse1 = df["mse_f1"].to_numpy(dtype=float)
            mean_d = float(df["D"].mean())
            base = {
                "unit": unit,
                "role": role,
                "mean_D": mean_d,
                "main_read": unit in MAIN_READ_UNITS,
                "seq_len": sl,
                "pred_len": pl,
            }

            # g*
            flags_high = np.stack(
                [g.high_mask(df, cuts, f) for f in g.RISK_FEATS], axis=1
            )
            close_star = close_any(flags_high)
            m = eval_close(mse0, mse1, close_star, SEED)
            rows.append({**base, "variant": "g_star", **m})

            # wrong feature
            close_shock = g.high_mask(df, cuts, SHOCK)
            m = eval_close(mse0, mse1, close_shock, SEED)
            rows.append({**base, "variant": "wrong_feature", **m})

            # wrong polarity
            flags_low = np.stack(
                [low_mask(df, cuts, f) for f in g.RISK_FEATS], axis=1
            )
            close_flip = close_any(flags_low)
            m = eval_close(mse0, mse1, close_flip, SEED)
            rows.append({**base, "variant": "wrong_polarity", **m})

            # wrong windows: permute close mask
            close_perm = permute_close_mask(close_star, seed=SEED)
            assert int(close_perm.sum()) == int(close_star.sum())
            m = eval_close(mse0, mse1, close_perm, SEED)
            rows.append({**base, "variant": "wrong_windows", **m})

            print(
                f"ok {unit} n={len(df)} "
                f"Δ*={rows[-4]['delta_f1_minus_g']:+.4f} "
                f"main_read={unit in MAIN_READ_UNITS}"
            )

    by_v = {}
    for v in ("g_star", "wrong_feature", "wrong_polarity", "wrong_windows"):
        vrows = [r for r in rows if r["variant"] == v]
        by_v[v] = {
            "main_read": summarize([r for r in vrows if r["main_read"]]),
            "all15": summarize(vrows),
        }

    passes = check_pass(
        by_v["g_star"]["main_read"],
        by_v["wrong_feature"]["main_read"],
        by_v["wrong_polarity"]["main_read"],
        by_v["wrong_windows"]["main_read"],
    )

    report = {
        "protocol": "mainline 4-row structure ablation",
        "design_doc": "docs/消融实验设计_主线版.md",
        "close_rule": "any",
        "feats": list(g.RISK_FEATS),
        "main_read_units": list(MAIN_READ_UNITS),
        "wrong_windows_definition": "permute g* close mask; same n_closed",
        "by_variant": by_v,
        "pass_criteria": passes,
        "rows": rows,
    }
    AGG.mkdir(parents=True, exist_ok=True)
    (AGG / "gating_ablation_mainline.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    labels = {
        "g_star": "g* (history gate)",
        "wrong_feature": "Wrong feature (Shock)",
        "wrong_polarity": "Wrong polarity (low tertile)",
        "wrong_windows": "Wrong windows (permute close)",
    }
    lines = [
        "# Mainline structure ablation (4-row)",
        "",
        "Design: `docs/消融实验设计_主线版.md`. "
        "Main-read: 9 held-out D̄<0 units (excl. calibrate).",
        "",
        "## Pass criteria (main-read)",
        "",
        f"- wrong_feature: **{'PASS' if passes['wrong_feature']['pass'] else 'FAIL'}**",
        f"- wrong_polarity: **{'PASS' if passes['wrong_polarity']['pass'] else 'FAIL'}**",
        f"- wrong_windows: **{'PASS' if passes['wrong_windows']['pass'] else 'FAIL'}**",
        f"- all: **{'PASS' if passes['all_pass'] else 'FAIL'}**",
        "",
        "## Main-read table",
        "",
        "| Method | beats f1 | mean Δ(f1−g) | mean Δrnd | close% |",
        "|--------|----------|--------------|-----------|--------|",
    ]
    for v in ("g_star", "wrong_feature", "wrong_polarity", "wrong_windows"):
        s = by_v[v]["main_read"]
        lines.append(
            f"| {labels[v]} | {s['n_beats_f1']}/{s['n_units']} | "
            f"{s['mean_delta_f1']:+.5f} | {s['mean_delta_rnd']:+.5f} | "
            f"{100*s['mean_close_rate']:.1f}% |"
        )

    lines += [
        "",
        "## Per-unit main-read (Δ / Δrnd)",
        "",
        "| unit | Δ g* | Δrnd g* | Δ feat | Δrnd feat | Δ pol | Δ win | Δrnd win |",
        "|------|------|---------|--------|-----------|-------|-------|----------|",
    ]
    for u in MAIN_READ_UNITS:
        def get(v):
            for r in rows:
                if r["unit"] == u and r["variant"] == v:
                    return r
            return None

        a, b, c, d = (
            get("g_star"),
            get("wrong_feature"),
            get("wrong_polarity"),
            get("wrong_windows"),
        )
        lines.append(
            f"| {u} | {a['delta_f1_minus_g']:+.4f} | {a['delta_rnd']:+.4f} | "
            f"{b['delta_f1_minus_g']:+.4f} | {b['delta_rnd']:+.4f} | "
            f"{c['delta_f1_minus_g']:+.4f} | {d['delta_f1_minus_g']:+.4f} | "
            f"{d['delta_rnd']:+.4f} |"
        )

    lines += [
        "",
        "## All15 appendix",
        "",
        "| Method | beats f1 | mean Δ | mean Δrnd | close% |",
        "|--------|----------|--------|-----------|--------|",
    ]
    for v in ("g_star", "wrong_feature", "wrong_polarity", "wrong_windows"):
        s = by_v[v]["all15"]
        lines.append(
            f"| {labels[v]} | {s['n_beats_f1']}/{s['n_units']} | "
            f"{s['mean_delta_f1']:+.5f} | {s['mean_delta_rnd']:+.5f} | "
            f"{100*s['mean_close_rate']:.1f}% |"
        )

    md = "\n".join(lines) + "\n"
    (AGG / "gating_ablation_mainline.md").write_text(md)
    print(md)
    print("wrote", AGG / "gating_ablation_mainline.md")


if __name__ == "__main__":
    main()
