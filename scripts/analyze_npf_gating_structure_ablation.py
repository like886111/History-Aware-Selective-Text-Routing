#!/usr/bin/env python3
"""B3 structure ablation for tertile hard-label gate (A3 revised protocol).

Main system: six starred proxies + train high-tertile + ρ=any.
Variants: LOO, single_*, shock_only, flip-all, random-polarity (M=10),
shuffle-matched (primary), shuffle-raw (appendix).
Main-read: frozen held-out D̄<0 units (excludes calibrate GPT4MTS/Environment).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_npf_gating as g
from analyze_npf_gating_official import DOMAIN_LH, MEMBERS, data_csv, run_dir

SEED = 2025
NPF = Path(__file__).resolve().parents[1]
AGG = NPF / "outputs/aggregate"
SHOCK_FEAT = "shock_score"
SHORT = {
    "hist_std": "std",
    "hist_iqr": "iqr",
    "pred_ease_ar1": "ar1",
    "hist_abs_slope": "slope",
    "acf1": "acf1",
    "vol_change": "vol",
    "shock_score": "shock",
}

# Frozen main-read set (held-out & mean_D<0; excludes calibrate unit).
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


def low_mask(df: pd.DataFrame, cuts: dict, feat: str) -> np.ndarray:
    return df[feat].to_numpy(dtype=float) <= cuts[feat]["q1"]


def flags_from_polarity(
    df: pd.DataFrame, cuts: dict, feats: tuple[str, ...], polarity: dict[str, str]
) -> np.ndarray:
    cols = []
    for f in feats:
        pol = polarity[f]
        if pol == "high":
            cols.append(g.high_mask(df, cuts, f))
        elif pol == "low":
            cols.append(low_mask(df, cuts, f))
        else:
            raise ValueError(pol)
    return np.stack(cols, axis=1)


def close_any(flags: np.ndarray) -> np.ndarray:
    return flags.sum(axis=1) >= 1


def eval_close(mse0: np.ndarray, mse1: np.ndarray, close: np.ndarray, seed: int) -> dict:
    mse_g = g.apply_gate(mse0, mse1, close)
    m = g.metrics(mse_g, mse0, mse1, close)
    rnd = g.random_gate(mse0, mse1, m["close_text_rate"], seed=seed)
    m["delta_rnd"] = float(rnd["mse_g"] - m["mse_g"])
    m["random_mse_g"] = float(rnd["mse_g"])
    return m


def match_close_count(close: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    """Force exactly n_target True entries (A3 shuffle-matched)."""
    out = close.copy().astype(bool)
    n = int(out.sum())
    rng = np.random.default_rng(seed)
    idx_on = np.flatnonzero(out)
    idx_off = np.flatnonzero(~out)
    if n > n_target:
        drop = rng.choice(idx_on, size=n - n_target, replace=False)
        out[drop] = False
    elif n < n_target:
        need = n_target - n
        if need > len(idx_off):
            need = len(idx_off)
        add = rng.choice(idx_off, size=need, replace=False)
        out[add] = True
    return out


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


def check_pass(main_sum: dict, by_variant: dict) -> dict:
    full = by_variant["full6_any"]["main_read"]
    d_full = full["mean_delta_f1"]
    rnd_full = full["mean_delta_rnd"]
    beats_full = full["n_beats_f1"]
    abs_d = abs(d_full)

    out = {}

    # LOO
    loo_ok = True
    loo_detail = []
    for name, pack in by_variant.items():
        if not name.startswith("loo_drop_"):
            continue
        s = pack["main_read"]
        if abs_d < 1e-4:
            ok = s["n_beats_f1"] <= beats_full  # "not worse" → full >= loo beats
            # A3: full6 beats_f1 >= worst LOO; each LOO beats not required higher
            ok = s["n_beats_f1"] <= beats_full
        else:
            ok = (d_full - s["mean_delta_f1"]) >= -0.1 * abs_d
        loo_detail.append({"variant": name, "ok": bool(ok), "mean_delta": s["mean_delta_f1"], "beats": s["n_beats_f1"]})
        loo_ok = loo_ok and ok
    worst_loo_beats = min((d["beats"] for d in loo_detail), default=0)
    loo_ok = loo_ok and (beats_full >= worst_loo_beats)
    out["leave_one_out"] = {"pass": bool(loo_ok), "detail": loo_detail, "full_beats": beats_full}

    # shock
    sh = by_variant["shock_only"]["main_read"]
    out["shock_only"] = {
        "pass": bool(
            sh["mean_delta_f1"] < d_full and sh["mean_delta_rnd"] < rnd_full
        ),
        "mean_delta": sh["mean_delta_f1"],
        "mean_delta_rnd": sh["mean_delta_rnd"],
        "full_delta": d_full,
        "full_delta_rnd": rnd_full,
    }

    # flip
    fl = by_variant["flip_all"]["main_read"]
    out["flip_all"] = {
        "pass": bool(fl["mean_delta_f1"] < d_full),
        "mean_delta": fl["mean_delta_f1"],
        "full_delta": d_full,
    }

    # random polarity
    rp = by_variant["random_polarity"]["main_read"]
    out["random_polarity"] = {
        "pass": bool(d_full > rp["mean_delta_f1"]),
        "full_delta": d_full,
        "rand_mean_delta": rp["mean_delta_f1"],
        "rand_std_delta": rp.get("std_delta_f1"),
        "rand_mean_close": rp["mean_close_rate"],
        "full_close": full["mean_close_rate"],
    }

    # shuffle matched
    sm = by_variant["shuffle_matched"]["main_read"]
    rnd_ok = abs(sm["mean_delta_rnd"]) < 0.5 * abs(rnd_full) if abs(rnd_full) > 1e-12 else abs(sm["mean_delta_rnd"]) < 1e-4
    out["shuffle_matched"] = {
        "pass": bool(rnd_ok and sm["mean_delta_f1"] < d_full),
        "mean_delta": sm["mean_delta_f1"],
        "mean_delta_rnd": sm["mean_delta_rnd"],
        "full_delta": d_full,
        "full_delta_rnd": rnd_full,
        "rnd_criterion_ok": bool(rnd_ok),
    }

    out["all_critical_pass"] = bool(
        out["leave_one_out"]["pass"]
        and out["shock_only"]["pass"]
        and out["flip_all"]["pass"]
        and out["random_polarity"]["pass"]
        and out["shuffle_matched"]["pass"]
    )
    return out


def eval_unit_variants(df: pd.DataFrame, cuts: dict, unit: str, role: str, mean_d: float) -> list[dict]:
    mse0 = df["mse_f0"].to_numpy(dtype=float)
    mse1 = df["mse_f1"].to_numpy(dtype=float)
    feats6 = g.RISK_FEATS
    rows = []

    def add(variant: str, close: np.ndarray, extra: dict | None = None):
        m = eval_close(mse0, mse1, close, seed=SEED)
        rec = {
            "variant": variant,
            "unit": unit,
            "role": role,
            "mean_D": mean_d,
            "main_read": unit in MAIN_READ_UNITS,
            **m,
        }
        if extra:
            rec.update(extra)
        rows.append(rec)

    # full6
    flags_high = np.stack([g.high_mask(df, cuts, f) for f in feats6], axis=1)
    close_main = close_any(flags_high)
    n_main = int(close_main.sum())
    add("full6_any", close_main)

    # LOO
    for i, drop in enumerate(feats6):
        keep = tuple(f for f in feats6 if f != drop)
        flags = np.stack([g.high_mask(df, cuts, f) for f in keep], axis=1)
        add(f"loo_drop_{SHORT[drop]}", close_any(flags), {"dropped": drop})

    # singles
    for f in feats6:
        add(f"single_{SHORT[f]}", g.high_mask(df, cuts, f), {"feats": [f]})

    # shock only
    if SHOCK_FEAT not in cuts:
        # fit shock cut from same train cuts dict may lack shock — use domain params with shock
        pass
    add("shock_only", g.high_mask(df, cuts, SHOCK_FEAT), {"feats": [SHOCK_FEAT]})

    # flip-all
    flags_low = np.stack([low_mask(df, cuts, f) for f in feats6], axis=1)
    add("flip_all", close_any(flags_low))

    # random polarity M=10
    rp_deltas = []
    rp_closes = []
    rp_rnds = []
    rp_mse_g = []
    for m_i, seed in enumerate(range(SEED, SEED + 10)):
        rng = np.random.default_rng(seed)
        pol = {f: ("high" if rng.random() < 0.5 else "low") for f in feats6}
        flags = flags_from_polarity(df, cuts, feats6, pol)
        close = close_any(flags)
        m = eval_close(mse0, mse1, close, seed=seed)
        rp_deltas.append(m["delta_f1_minus_g"])
        rp_closes.append(m["close_text_rate"])
        rp_rnds.append(m["delta_rnd"])
        rp_mse_g.append(m["mse_g"])
        add(
            f"random_polarity_m{m_i}",
            close,
            {"polarity_seed": seed, "polarity": pol},
        )
    # per-unit aggregate over M (for main-read means)
    rows.append(
        {
            "variant": "random_polarity",
            "unit": unit,
            "role": role,
            "mean_D": mean_d,
            "main_read": unit in MAIN_READ_UNITS,
            "n": int(len(mse0)),
            "mse_f0": float(mse0.mean()),
            "mse_f1": float(mse1.mean()),
            "mse_g": float(np.mean(rp_mse_g)),
            "delta_f1_minus_g": float(np.mean(rp_deltas)),
            "delta_rnd": float(np.mean(rp_rnds)),
            "close_text_rate": float(np.mean(rp_closes)),
            "beats_f1": bool(np.mean(rp_deltas) > 0),
            "is_aggregate_random_polarity": True,
            "std_delta_f1_over_M": float(np.std(rp_deltas)),
            "n_runs": 10,
        }
    )

    # shuffle-raw: permute each column of flags_high
    rng = np.random.default_rng(SEED)
    flags_shuf = flags_high.copy()
    for j in range(flags_shuf.shape[1]):
        flags_shuf[:, j] = rng.permutation(flags_shuf[:, j])
    close_raw = close_any(flags_shuf)
    add("shuffle_raw", close_raw)

    # shuffle-matched
    close_matched = match_close_count(close_raw, n_main, seed=SEED + 7)
    add("shuffle_matched", close_matched, {"n_close_target": n_main})

    return rows


def main() -> None:
    root = NPF / "outputs"
    all_rows: list[dict] = []

    for dataset, (sl, pl) in DOMAIN_LH.items():
        for member in MEMBERS:
            unit = f"{member}/{dataset}"
            role = "calibrate" if (member == "GPT4MTS" and dataset == "Environment") else "eval"
            rd = run_dir(root, member, dataset, sl, pl)
            paired = rd / "paired_test.csv"
            csv = data_csv(member, dataset)
            if not paired.exists() or not csv.exists():
                print("skip", unit)
                continue
            # cuts include shock for negative control
            feats_all = tuple(list(g.RISK_FEATS) + [SHOCK_FEAT])
            params = g.domain_gate_params(csv, sl, pl, close_rule="any", feats=feats_all)
            df = g.load_paired_with_feats(paired, csv, sl, pl)
            # ensure shock column present (build_features should include it)
            if SHOCK_FEAT not in df.columns:
                print("missing shock in feats", unit)
                continue
            mean_d = float(df["D"].mean())
            unit_rows = eval_unit_variants(df, params["tertile_cuts"], unit, role, mean_d)
            all_rows.extend(unit_rows)
            print(f"ok {unit} n={len(df)} main_read={unit in MAIN_READ_UNITS}")

    # group by variant
    variants = sorted({r["variant"] for r in all_rows})
    by_variant: dict = {}
    for v in variants:
        vrows = [r for r in all_rows if r["variant"] == v]
        # skip per-run random_polarity_m* from top summary except keep in raw
        main_rows = [r for r in vrows if r.get("main_read")]
        by_variant[v] = {
            "all15": summarize(vrows),
            "main_read": summarize(main_rows),
        }
        if v == "random_polarity":
            # std across units of the per-unit mean-of-M deltas already in rows
            mr = main_rows
            by_variant[v]["main_read"]["std_delta_f1"] = float(
                np.std([r["delta_f1_minus_g"] for r in mr])
            ) if mr else None

    # For random_polarity pass check, also average the m0..m9 on main_read
    rp_runs = [v for v in variants if v.startswith("random_polarity_m")]
    if rp_runs:
        # recompute main_read mean delta as mean over units of (mean over M)
        # already stored in aggregate random_polarity rows
        pass

    # pass criteria only on key variants
    key = {
        "full6_any": by_variant["full6_any"],
        "shock_only": by_variant["shock_only"],
        "flip_all": by_variant["flip_all"],
        "random_polarity": by_variant["random_polarity"],
        "shuffle_matched": by_variant["shuffle_matched"],
    }
    for name, pack in by_variant.items():
        if name.startswith("loo_drop_"):
            key[name] = pack
    passes = check_pass({}, key)

    report = {
        "protocol": "A3 structure ablation revised",
        "close_rule": "any",
        "feats": list(g.RISK_FEATS),
        "main_read_units": list(MAIN_READ_UNITS),
        "n_row_records": len(all_rows),
        "by_variant": {
            k: v
            for k, v in by_variant.items()
            if not k.startswith("random_polarity_m")
        },
        "pass_criteria": passes,
        "rows": all_rows,
    }
    AGG.mkdir(parents=True, exist_ok=True)
    (AGG / "gating_structure_ablation.json").write_text(json.dumps(report, indent=2) + "\n")

    # markdown
    lines = [
        "# Structure ablation (A3) — tertile hard gate",
        "",
        f"Main system: six starred + high tertile + ρ=any. "
        f"Main-read: {len(MAIN_READ_UNITS)} held-out D̄<0 units (excl. calibrate).",
        "",
        "## Pass criteria (main-read)",
        "",
        f"- leave-one-out: **{'PASS' if passes['leave_one_out']['pass'] else 'FAIL'}**",
        f"- shock_only: **{'PASS' if passes['shock_only']['pass'] else 'FAIL'}**",
        f"- flip_all: **{'PASS' if passes['flip_all']['pass'] else 'FAIL'}**",
        f"- random_polarity: **{'PASS' if passes['random_polarity']['pass'] else 'FAIL'}**",
        f"- shuffle_matched: **{'PASS' if passes['shuffle_matched']['pass'] else 'FAIL'}**",
        f"- all critical: **{'PASS' if passes['all_critical_pass'] else 'FAIL'}**",
        "",
        "## Main-read summary",
        "",
        "| variant | beats_f1 | mean Δ(f1−g) | mean Δrnd | mean close |",
        "|---------|----------|--------------|-----------|------------|",
    ]
    show = [
        "full6_any",
        *[f"loo_drop_{s}" for s in SHORT.values() if s != "shock"],
        *[f"single_{s}" for s in SHORT.values() if s != "shock"],
        "shock_only",
        "flip_all",
        "random_polarity",
        "shuffle_matched",
        "shuffle_raw",
    ]
    for v in show:
        if v not in by_variant:
            continue
        s = by_variant[v]["main_read"]
        if s["n_units"] == 0:
            continue
        lines.append(
            f"| {v} | {s['n_beats_f1']}/{s['n_units']} | "
            f"{s['mean_delta_f1']:+.5f} | {s['mean_delta_rnd']:+.5f} | "
            f"{s['mean_close_rate']:.3f} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- `random_polarity` row = per-unit mean over M=10, then mean over main-read units.",
        "- `shuffle_matched` forces same #closed windows as full6_any per unit.",
        "- single_* stronger than full6 does not change the main method (pre-registered).",
        "",
    ]
    (AGG / "gating_structure_ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", AGG / "gating_structure_ablation.md")


if __name__ == "__main__":
    main()
