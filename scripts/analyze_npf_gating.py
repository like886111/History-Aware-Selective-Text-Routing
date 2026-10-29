#!/usr/bin/env python3
"""Tertile hard-label gate on six starred harmful proxies (§4.3).

Proxies (high tertile = more harmful):
  hist_std, hist_iqr, pred_ease_ar1, hist_abs_slope, acf1, vol_change
Per domain, tertile cutoffs are fit on train windows only.
Close rules (frozen on GPT4MTS–Environment):
  any — n_high >= 1
  maj — n_high >= 4  (strict majority of 6)
  all — n_high == 6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_npf_conditions import build_features

# Six starred regimes from §4.3 (Shock excluded).
RISK_FEATS = (
    "hist_std",
    "hist_iqr",
    "pred_ease_ar1",
    "hist_abs_slope",
    "acf1",
    "vol_change",
)
# Family-representative subset for 6-vs-3 ablation.
FEATS_3 = ("hist_iqr", "hist_abs_slope", "vol_change")
CLOSE_RULES = ("any", "maj", "all")


def maj_threshold(n_feats: int) -> int:
    """Strict majority: > n/2, i.e. >= floor(n/2)+1; for n=6 → 4."""
    return n_feats // 2 + 1


def train_indices(n_rows: int, seq_len: int, pred_len: int) -> np.ndarray:
    num_train = int(n_rows * 0.7)
    n = num_train - seq_len - pred_len + 1
    if n <= 0:
        raise RuntimeError(f"no train windows: n_rows={n_rows} sl={seq_len} pl={pred_len}")
    return np.arange(n, dtype=int)


def load_paired_with_feats(
    paired_csv: Path,
    data_csv: Path,
    seq_len: int,
    pred_len: int,
) -> pd.DataFrame:
    paired = pd.read_csv(paired_csv)
    feats = build_features(data_csv, paired["index"].to_numpy(), seq_len, pred_len)
    return paired.merge(feats, on="index", how="inner")


def fit_z(df: pd.DataFrame, feats: tuple[str, ...] | None = None) -> dict:
    stats = {}
    for f in feats or RISK_FEATS:
        x = df[f].to_numpy(dtype=float)
        stats[f] = {"mean": float(np.nanmean(x)), "std": float(np.nanstd(x) + 1e-12)}
    return stats


def fit_tertile_cuts(df: pd.DataFrame, feats: tuple[str, ...] = RISK_FEATS) -> dict:
    cuts = {}
    for f in feats:
        x = df[f].to_numpy(dtype=float)
        q1, q2 = np.quantile(x[~np.isnan(x)], [1.0 / 3.0, 2.0 / 3.0])
        cuts[f] = {"q1": float(q1), "q2": float(q2)}
    return cuts


def high_mask(df: pd.DataFrame, cuts: dict, feat: str) -> np.ndarray:
    return df[feat].to_numpy(dtype=float) >= cuts[feat]["q2"]


def close_from_tertiles(
    df: pd.DataFrame,
    cuts: dict,
    rule: str,
    feats: tuple[str, ...] = RISK_FEATS,
) -> np.ndarray:
    flags = np.stack([high_mask(df, cuts, f) for f in feats], axis=1)
    n_high = flags.sum(axis=1)
    if rule == "any":
        return n_high >= 1
    if rule == "maj":
        return n_high >= maj_threshold(len(feats))
    if rule == "all":
        return n_high >= len(feats)
    raise ValueError(f"unknown close rule: {rule}")


def risk_scores(df: pd.DataFrame, z_stats: dict) -> np.ndarray:
    r = np.zeros(len(df), dtype=float)
    for f in RISK_FEATS:
        if f not in z_stats:
            continue
        mu, sd = z_stats[f]["mean"], z_stats[f]["std"]
        r += (df[f].to_numpy(dtype=float) - mu) / sd
    return r


def domain_gate_params(
    data_csv: Path,
    seq_len: int,
    pred_len: int,
    close_rule: str = "maj",
    q_star: float | None = None,
    feats: tuple[str, ...] | None = None,
) -> dict:
    del q_star
    feats = tuple(feats) if feats is not None else RISK_FEATS
    if close_rule not in CLOSE_RULES:
        raise ValueError(close_rule)
    n_rows = len(pd.read_csv(data_csv))
    idx = train_indices(n_rows, seq_len, pred_len)
    train_feats = build_features_train(data_csv, idx, seq_len)
    cuts = fit_tertile_cuts(train_feats, feats=feats)
    close_train = close_from_tertiles(train_feats, cuts, close_rule, feats=feats)
    return {
        "mode": "tertile_hard",
        "close_rule": close_rule,
        "feats": list(feats),
        "maj_k": maj_threshold(len(feats)),
        "tertile_cuts": cuts,
        "n_train_windows": int(len(train_feats)),
        "train_close_rate": float(close_train.mean()),
        "data_csv": str(data_csv),
        "q_star": None,
        "z_stats": fit_z(train_feats, feats),
        "tau": None,
    }


def build_features_train(data_csv: Path, indices: np.ndarray, seq_len: int) -> pd.DataFrame:
    from analyze_npf_conditions import (
        acf1,
        is_contentful,
        pred_ease_ar1,
        shock_score,
        vol_change,
    )

    df = pd.read_csv(data_csv)
    ot = df["OT"].astype(float).to_numpy()
    text_col = None
    for c in ("fact", "Final_Search_4", "Final_Search_3", "Final_Output"):
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise KeyError(f"no text column in {data_csv}")
    fact = df[text_col].fillna("").astype(str).to_numpy()
    dates = df["date"].astype(str).to_numpy()
    rows = []
    for idx in indices:
        abs_begin = int(idx)
        abs_end = abs_begin + seq_len
        hist = ot[abs_begin:abs_end]
        if len(hist) < seq_len:
            continue
        texts = fact[abs_begin:abs_end]
        x = np.arange(seq_len, dtype=float)
        slope = float(np.polyfit(x, hist, 1)[0]) if seq_len >= 2 else 0.0
        content = np.array([is_contentful(t) for t in texts], dtype=bool)
        text_lens = np.array([len(t.strip()) for t in texts], dtype=float)
        rows.append(
            {
                "index": int(idx),
                "hist_std": float(hist.std()),
                "hist_iqr": float(np.quantile(hist, 0.75) - np.quantile(hist, 0.25)),
                "hist_abs_slope": abs(slope),
                "pred_ease_ar1": pred_ease_ar1(hist),
                "acf1": acf1(hist),
                "shock_score": shock_score(hist),
                "vol_change": vol_change(hist),
                "text_nonempty_rate": float(content.mean()),
                "text_mean_len": float(text_lens.mean()),
                "date_hist_start": dates[abs_begin],
            }
        )
    return pd.DataFrame(rows)


def apply_gate(mse0: np.ndarray, mse1: np.ndarray, close_text: np.ndarray) -> np.ndarray:
    return np.where(close_text, mse0, mse1)


def metrics(mse_g: np.ndarray, mse0: np.ndarray, mse1: np.ndarray, close_text: np.ndarray) -> dict:
    return {
        "n": int(len(mse_g)),
        "mse_g": float(mse_g.mean()),
        "mse_f0": float(mse0.mean()),
        "mse_f1": float(mse1.mean()),
        "mse_oracle": float(np.minimum(mse0, mse1).mean()),
        "delta_f1_minus_g": float(mse1.mean() - mse_g.mean()),
        "delta_f0_minus_g": float(mse0.mean() - mse_g.mean()),
        "close_text_rate": float(close_text.mean()),
        "beats_f1": bool(mse_g.mean() < mse1.mean()),
        "not_worse_f0": bool(mse_g.mean() <= mse0.mean()),
    }


def random_gate(mse0: np.ndarray, mse1: np.ndarray, rate: float, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    close = rng.random(len(mse0)) < rate
    m = metrics(apply_gate(mse0, mse1, close), mse0, mse1, close)
    m["kind"] = "random_same_rate"
    return m


def eval_with_params(df: pd.DataFrame, params: dict, seed: int = 0) -> dict:
    rule = params.get("close_rule", "maj")
    cuts = params["tertile_cuts"]
    feats = tuple(params.get("feats") or RISK_FEATS)
    close = close_from_tertiles(df, cuts, rule, feats=feats)
    mse0 = df["mse_f0"].to_numpy(dtype=float)
    mse1 = df["mse_f1"].to_numpy(dtype=float)
    mse_g = apply_gate(mse0, mse1, close)
    m = metrics(mse_g, mse0, mse1, close)
    m["kind"] = f"tertile_hard_{rule}"
    m["close_rule"] = rule
    m["feats"] = list(feats)
    m["maj_k"] = maj_threshold(len(feats))
    m["n_train_windows"] = params.get("n_train_windows")
    m["train_close_rate"] = params.get("train_close_rate")
    for f in feats:
        m[f"high_rate_{f}"] = float(high_mask(df, cuts, f).mean())
    m["random"] = random_gate(mse0, mse1, m["close_text_rate"], seed=seed)
    return m


def calibrate_close_rule(
    df_cal: pd.DataFrame,
    data_csv_cal: Path,
    seq_len: int,
    pred_len: int,
    feats: tuple[str, ...] | None = None,
) -> tuple[dict, list[dict]]:
    feats = tuple(feats) if feats is not None else RISK_FEATS
    candidates = []
    for rule in CLOSE_RULES:
        params = domain_gate_params(data_csv_cal, seq_len, pred_len, close_rule=rule, feats=feats)
        m = eval_with_params(df_cal, params)
        m["quantile"] = None
        m["close_rule"] = rule
        candidates.append(m)

    prefer = [c for c in candidates if c["not_worse_f0"] and c["beats_f1"]]
    if not prefer:
        prefer = [c for c in candidates if c["beats_f1"]]
    if not prefer:
        prefer = candidates
    best = max(prefer, key=lambda c: c["delta_f1_minus_g"])
    rule_star = str(best["close_rule"])

    config = {
        "gate_feature": "tertile_hard_label_six_starred",
        "risk_feats": list(feats),
        "close_rule": rule_star,
        "maj_k": maj_threshold(len(feats)),
        "q_star": None,
        "kind": best["kind"],
        "calibrate_unit": "GPT4MTS/Environment",
        "note": (
            "Close text when starred proxies fall in the train-defined high tertile. "
            "Rule frozen on development; per-domain tertile cuts from train only."
        ),
        "selected_on_calibrate": {
            "mse_g": best["mse_g"],
            "mse_f0": best["mse_f0"],
            "mse_f1": best["mse_f1"],
            "delta_f1_minus_g": best["delta_f1_minus_g"],
            "close_text_rate": best["close_text_rate"],
            "not_worse_f0": best["not_worse_f0"],
            "close_rule": rule_star,
        },
    }
    return config, candidates


def calibrate_q(df_cal, data_csv_cal, seq_len, pred_len):
    return calibrate_close_rule(df_cal, data_csv_cal, seq_len, pred_len)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs-root", default=str(Path(__file__).resolve().parents[1] / "outputs"))
    ap.add_argument(
        "--tats-data-root",
        default=str(Path(__file__).resolve().parents[1] / "third_party/TaTS/data"),
    )
    ap.add_argument("--seq-len", type=int, default=96)
    ap.add_argument("--pred-len", type=int, default=48)
    args = ap.parse_args()

    root = Path(args.outputs_root)
    tats = Path(args.tats_data_root)
    cal_dir = root / "gpt4mts/Environment/NPF_Environment_GPT4MTS_seed2025_sl96_pl48"
    cal_csv = tats / "Environment.csv"
    df_cal = load_paired_with_feats(cal_dir / "paired_test.csv", cal_csv, args.seq_len, args.pred_len)
    config, candidates = calibrate_close_rule(df_cal, cal_csv, args.seq_len, args.pred_len)
    print("selected rule", config["close_rule"], "feats", config["risk_feats"])
    for c in candidates:
        print(
            c["close_rule"],
            f"Δ={c['delta_f1_minus_g']:+.4g}",
            f"close={c['close_text_rate']:.3f}",
            f"≤f0={c['not_worse_f0']}",
        )
    agg = root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "gating_config.json").write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
