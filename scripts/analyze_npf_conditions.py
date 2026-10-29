"""Expert condition discovery (H1–H6) with D_rel primary stratification.

Explore: Environment × TaTS → C_hat by observed harm magnitude (discovery-first).
Pre-registered H* directions are motivational guesses only; promotion does not require
matching them. Confirm: held-out units; pass = same observed harm sign as explore
(magnitude is reported but not required).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6

NA_PAT = re.compile(
    r"(no information available|no relevant information|^NA\b|^\s*$)",
    re.IGNORECASE,
)

# coarse lexicon for H3 (English Time-MMD fact text)
UP_WORDS = re.compile(
    r"\b(increase|increased|increasing|rise|rising|rose|up|higher|grow|growth|"
    r"surge|improve|improved|gain|gains|bullish|hotter|warmer)\b",
    re.I,
)
DOWN_WORDS = re.compile(
    r"\b(decrease|decreased|decreasing|fall|falling|fell|down|lower|drop|"
    r"decline|declining|shrink|worse|worsen|cooler|colder|bearish)\b",
    re.I,
)

# Motivational guesses (H*): used only for reporting, not for promotion.
H_SPEC: dict[str, list[tuple[str, str]]] = {
    "H1": [("pred_ease_ar1", "low_more_harm"), ("acf1", "high_more_harm"), ("hist_abs_slope", "low_more_harm")],
    "H2": [("hist_std", "high_more_harm"), ("hist_iqr", "high_more_harm"), ("shock_score", "high_more_harm"), ("vol_change", "high_more_harm")],
    "H3": [("lex_conflict", "high_more_harm")],
    "H4": [],  # structure: near vs far; handled separately
    "H5": [("text_nonempty_rate", "low_more_harm"), ("text_mean_len", "low_more_harm")],
    "H6": [("mse_f0", "high_more_harm"), ("prior_vs_hist_end", "high_more_harm")],  # diagnostic only
}

# Deployable pool for discovery-first promotion (pre-forecast history proxies).
DEPLOYABLE_FEATS = (
    "pred_ease_ar1",
    "acf1",
    "hist_abs_slope",
    "hist_std",
    "hist_iqr",
    "shock_score",
    "vol_change",
)
MIN_PROMOTE_ABS_DELTA = 0.01  # drop near-zero separations
# No hard quota: promote every deployable proxy with |Δ_X| ≥ floor.


def is_contentful(s: str) -> bool:
    s = (s or "").strip()
    if not s or s.lower() in {"nan", "na"}:
        return False
    if NA_PAT.search(s):
        return False
    return True


def text_polarity(texts: np.ndarray) -> float:
    """+1 up / -1 down / 0 neutral from window texts."""
    blob = " ".join(str(t) for t in texts if is_contentful(str(t)))
    if not blob:
        return 0.0
    u = len(UP_WORDS.findall(blob))
    d = len(DOWN_WORDS.findall(blob))
    if u == d:
        return 0.0
    return 1.0 if u > d else -1.0


def test_border(n_rows: int, seq_len: int) -> tuple[int, int]:
    num_test = int(n_rows * 0.2)
    border1 = n_rows - num_test - seq_len
    return border1, n_rows


def pred_ease_ar1(hist: np.ndarray) -> float:
    """MSE of 1-step persistence / AR(1) on last quarter of window (lower = easier)."""
    if hist.size < 6:
        return float("nan")
    # fit AR(1) on first 3/4, score on last 1/4
    cut = max(3, int(hist.size * 0.75))
    y = hist[1:cut]
    x = hist[: cut - 1]
    if x.std() < 1e-12:
        a, b = 0.0, float(x.mean())
    else:
        a = float(np.cov(x, y, ddof=0)[0, 1] / (x.var() + 1e-12))
        b = float(y.mean() - a * x.mean())
    errs = []
    for t in range(cut, hist.size):
        pred = a * hist[t - 1] + b
        errs.append((hist[t] - pred) ** 2)
    return float(np.mean(errs)) if errs else float("nan")


def acf1(hist: np.ndarray) -> float:
    if hist.size < 3 or hist.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(hist[:-1], hist[1:])[0, 1])


def shock_score(hist: np.ndarray) -> float:
    med = float(np.median(hist))
    mad = float(np.median(np.abs(hist - med))) + 1e-6
    return float(np.max(np.abs(hist - med) / mad))


def vol_change(hist: np.ndarray) -> float:
    mid = hist.size // 2
    if mid < 2:
        return 1.0
    a = float(hist[:mid].std()) + 1e-6
    b = float(hist[mid:].std()) + 1e-6
    return b / a


def build_features(
    data_csv: Path,
    indices: np.ndarray,
    seq_len: int,
    pred_len: int,
    text_col: str | None = None,
) -> pd.DataFrame:
    del pred_len  # reserved for horizon alignment docs
    df = pd.read_csv(data_csv)
    ot = df["OT"].astype(float).to_numpy()
    if text_col is None:
        for c in ("fact", "Final_Search_4", "Final_Search_3", "Final_Output"):
            if c in df.columns:
                text_col = c
                break
        if text_col is None:
            raise KeyError(f"no text column in {data_csv}; cols={list(df.columns)}")
    fact = df[text_col].fillna("").astype(str).to_numpy()
    dates = df["date"].astype(str).to_numpy()
    n = len(df)
    border1, _ = test_border(n, seq_len)
    has_prior = "prior_history_avg" in df.columns
    prior = df["prior_history_avg"].astype(float).to_numpy() if has_prior else None

    rows = []
    for idx in indices:
        s_begin = int(idx)
        abs_begin = border1 + s_begin
        abs_end = abs_begin + seq_len
        hist = ot[abs_begin:abs_end]
        texts = fact[abs_begin:abs_end]
        if len(hist) < seq_len:
            continue
        x = np.arange(seq_len, dtype=float)
        slope = float(np.polyfit(x, hist, 1)[0]) if seq_len >= 2 else 0.0
        content = np.array([is_contentful(t) for t in texts], dtype=bool)
        text_lens = np.array([len(t.strip()) for t in texts], dtype=float)
        pol = text_polarity(texts)
        slope_sign = 0.0 if abs(slope) < 1e-12 else (1.0 if slope > 0 else -1.0)
        lex_conflict = float(pol != 0 and slope_sign != 0 and pol != slope_sign)
        row = {
            "index": int(idx),
            "abs_begin": int(abs_begin),
            "date_hist_start": dates[abs_begin],
            "date_hist_end": dates[abs_end - 1],
            "hist_mean": float(hist.mean()),
            "hist_std": float(hist.std()),
            "hist_iqr": float(np.quantile(hist, 0.75) - np.quantile(hist, 0.25)),
            "hist_abs_slope": abs(slope),
            "hist_slope": slope,
            "pred_ease_ar1": pred_ease_ar1(hist),
            "acf1": acf1(hist),
            "shock_score": shock_score(hist),
            "vol_change": vol_change(hist),
            "lex_conflict": lex_conflict,
            "text_polarity": pol,
            "text_nonempty_rate": float(content.mean()),
            "text_any_content": bool(content.any()),
            "text_mean_len": float(text_lens.mean()),
            "text_max_len": float(text_lens.max()),
            "fact_last": texts[-1][:240],
        }
        if prior is not None:
            # prior vs last hist point as cheap quality proxy when future y unavailable here
            row["prior_vs_hist_end"] = float((prior[abs_end - 1] - hist[-1]) ** 2)
        rows.append(row)
    return pd.DataFrame(rows)


def attach_d_rel(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["D_rel"] = df["D"] / (df["mse_f0"] + EPS)
    dh_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"D_h\d+", c)],
        key=lambda c: int(c[3:]),
    )
    if dh_cols:
        dh = df[dh_cols].to_numpy(dtype=float)
        q = max(1, dh.shape[1] // 4)
        df["D_near"] = dh[:, :q].mean(axis=1)
        df["D_far"] = dh[:, -q:].mean(axis=1)
        # approximate D_rel by horizon using mse_f0 scale (same denom)
        denom = (df["mse_f0"].to_numpy() + EPS)[:, None]
        dhr = dh / denom
        df["D_rel_near"] = dhr[:, :q].mean(axis=1)
        df["D_rel_far"] = dhr[:, -q:].mean(axis=1)
        # middle half (exclude near/far quarters) for table Mid column
        if dh.shape[1] > 2 * q:
            df["D_mid"] = dh[:, q:-q].mean(axis=1)
            df["D_rel_mid"] = dhr[:, q:-q].mean(axis=1)
    return df


def tercile_delta(df: pd.DataFrame, feat: str, dcol: str = "D_rel") -> dict | None:
    x = df[feat].to_numpy(dtype=float)
    d = df[dcol].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(d)
    x, d = x[mask], d[mask]
    if x.size < 20:
        return None
    if np.unique(x).size < 3:
        med = np.median(x)
        low, high = d[x <= med], d[x > med]
        if low.size == 0 or high.size == 0:
            return None
        return {
            "feature": feat,
            "dcol": dcol,
            "split": "median",
            "n_low": int(low.size),
            "n_high": int(high.size),
            "mean_low": float(low.mean()),
            "mean_high": float(high.mean()),
            "neg_rate_low": float((low < 0).mean()),
            "neg_rate_high": float((high < 0).mean()),
            "delta_mean_low_minus_high": float(low.mean() - high.mean()),
            "corr_feat_d": float(np.corrcoef(x, d)[0, 1]) if x.std() > 0 else None,
        }
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    low, mid, high = d[x <= q1], d[(x > q1) & (x <= q2)], d[x > q2]
    return {
        "feature": feat,
        "dcol": dcol,
        "split": "tercile",
        "n_low": int(low.size),
        "n_mid": int(mid.size),
        "n_high": int(high.size),
        "mean_low": float(low.mean()),
        "mean_mid": float(mid.mean()),
        "mean_high": float(high.mean()),
        "neg_rate_low": float((low < 0).mean()),
        "neg_rate_mid": float((mid < 0).mean()),
        "neg_rate_high": float((high < 0).mean()),
        "delta_mean_low_minus_high": float(low.mean() - high.mean()),
        "corr_feat_d": float(np.corrcoef(x, d)[0, 1]) if x.std() > 0 else None,
    }


def direction_from_delta(delta: float) -> str:
    # delta = mean(low) - mean(high); negative => low more harmful
    return "low_more_harm" if delta < 0 else "high_more_harm"


def _feat_to_hypothesis(feat: str) -> str:
    for hid, specs in H_SPEC.items():
        for f, _ in specs:
            if f == feat:
                return hid
    return "H?"


def propose_c_hat(strata: list[dict], horizon: dict | None) -> list[dict]:
    """Discovery-first: promote all deployable proxies with |Δ_X| ≥ floor.

    Pre-registered directions are recorded for transparency but do not gate promotion.
    Near-zero gaps are excluded from the confirm shortlist.
    """
    by_feat = {s["feature"]: s for s in strata if s is not None}
    declared_map = {f: d for specs in H_SPEC.values() for f, d in specs}
    out: list[dict] = []
    scored: list[tuple] = []

    for feat in DEPLOYABLE_FEATS:
        s = by_feat.get(feat)
        if s is None:
            continue
        delta = float(s["delta_mean_low_minus_high"])
        if abs(delta) < MIN_PROMOTE_ABS_DELTA:
            out.append(
                {
                    "hypothesis": _feat_to_hypothesis(feat),
                    "feature": feat,
                    "declared_direction": declared_map.get(feat),
                    "observed_direction": direction_from_delta(delta),
                    "direction_match": direction_from_delta(delta) == declared_map.get(feat),
                    "direction": direction_from_delta(delta),
                    "explore_delta_mean_D_rel_low_minus_high": delta,
                    "explore_mean_D_rel_low": s.get("mean_low"),
                    "explore_mean_D_rel_high": s.get("mean_high"),
                    "deployable": True,
                    "status": "weak_not_promoted",
                    "note": f"|Δ_X|<{MIN_PROMOTE_ABS_DELTA}",
                }
            )
            continue
        observed = direction_from_delta(delta)
        declared = declared_map.get(feat)
        scored.append((abs(delta), feat, declared, observed, delta, s))

    scored.sort(key=lambda t: -t[0])
    for abs_d, feat, declared, observed, delta, s in scored:
        out.append(
            {
                "hypothesis": _feat_to_hypothesis(feat),
                "feature": feat,
                "declared_direction": declared,
                "observed_direction": observed,
                "direction_match": observed == declared if declared else None,
                "direction": observed,
                "explore_delta_mean_D_rel_low_minus_high": float(delta),
                "explore_mean_D_rel_low": s.get("mean_low"),
                "explore_mean_D_rel_high": s.get("mean_high"),
                "deployable": True,
                "status": "C_hat_explore_only_needs_confirm",
            }
        )

    # H6 diagnostics (never promoted to confirm shortlist)
    for feat, declared in H_SPEC.get("H6", []):
        s = by_feat.get(feat)
        if s is None:
            continue
        delta = float(s["delta_mean_low_minus_high"])
        observed = direction_from_delta(delta)
        out.append(
            {
                "hypothesis": "H6",
                "feature": feat,
                "declared_direction": declared,
                "observed_direction": observed,
                "direction_match": observed == declared,
                "direction": observed,
                "explore_delta_mean_D_rel_low_minus_high": delta,
                "explore_mean_D_rel_low": s.get("mean_low"),
                "explore_mean_D_rel_high": s.get("mean_high"),
                "deployable": False,
                "status": "diagnostic_only",
            }
        )

    # H4 structure (report observed near/far harm; not in top-3 feature quota)
    if horizon is not None and "mean_D_rel_far" in horizon:
        far = horizon["mean_D_rel_far"]
        near = horizon["mean_D_rel_near"]
        observed = "far_more_harm" if far < near else "near_more_harm"
        out.append(
            {
                "hypothesis": "H4",
                "feature": "horizon_far_vs_near",
                "declared_direction": "far_more_harm",
                "observed_direction": observed,
                "direction_match": observed == "far_more_harm",
                "direction": observed,
                "explore_delta_mean_D_rel_near_minus_far": float(near - far),
                "explore_mean_D_rel_near": float(near),
                "explore_mean_D_rel_far": float(far),
                "deployable": False,
                "status": "structure_report_only",
                "note": "horizon comparison is not a pre-forecast window feature for routing",
            }
        )

    return out


def run_explore(args: argparse.Namespace) -> dict:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(args.paired_csv)
    feats = build_features(
        Path(args.data_csv),
        paired["index"].to_numpy(),
        args.seq_len,
        args.pred_len,
    )
    df = attach_d_rel(paired.merge(feats, on="index", how="inner"))

    horizon = None
    if "D_rel_near" in df.columns:
        horizon = {
            "mean_D_by_h": [
                float(df[c].mean())
                for c in sorted(
                    [c for c in df.columns if re.fullmatch(r"D_h\d+", c)],
                    key=lambda c: int(c[3:]),
                )
            ],
            "mean_D_near_quarter": float(df["D_near"].mean()),
            "mean_D_far_quarter": float(df["D_far"].mean()),
            "mean_D_rel_near": float(df["D_rel_near"].mean()),
            "mean_D_rel_far": float(df["D_rel_far"].mean()),
            "delta_D_rel_near_minus_far": float(
                df["D_rel_near"].mean() - df["D_rel_far"].mean()
            ),
        }
        if "D_rel_mid" in df.columns:
            horizon["mean_D_mid_half"] = float(df["D_mid"].mean())
            horizon["mean_D_rel_mid"] = float(df["D_rel_mid"].mean())

    feat_pool = [
        "pred_ease_ar1",
        "acf1",
        "hist_abs_slope",
        "hist_std",
        "hist_iqr",
        "shock_score",
        "vol_change",
        "lex_conflict",
        "text_nonempty_rate",
        "text_mean_len",
        "mse_f0",
        "prior_vs_hist_end",
    ]
    strata_rel, strata_d = [], []
    for f in feat_pool:
        if f in df.columns:
            strata_rel.append(tercile_delta(df, f, "D_rel"))
            strata_d.append(tercile_delta(df, f, "D"))

    # scale-illusion flags: strong on D but vanishes on D_rel
    illusion = []
    for sr, sd in zip(strata_rel, strata_d):
        if sr is None or sd is None:
            continue
        d_abs = abs(sd["delta_mean_low_minus_high"])
        r_abs = abs(sr["delta_mean_low_minus_high"])
        if d_abs > 1e-6 and r_abs < 0.25 * d_abs:
            illusion.append(
                {
                    "feature": sr["feature"],
                    "delta_D": sd["delta_mean_low_minus_high"],
                    "delta_D_rel": sr["delta_mean_low_minus_high"],
                    "flag": "possible_scale_illusion",
                }
            )

    c_all = propose_c_hat([s for s in strata_rel if s is not None], horizon)
    c_hat_confirm = [
        c for c in c_all if c.get("status") == "C_hat_explore_only_needs_confirm"
    ]
    # discovery-first order: largest |Δ| first
    c_hat_confirm = sorted(
        c_hat_confirm,
        key=lambda c: -abs(
            c.get("explore_delta_mean_D_rel_low_minus_high")
            or c.get("explore_delta_mean_D_rel_near_minus_far")
            or 0.0
        ),
    )
    c_hat_diag = [c for c in c_all if c.get("status") == "diagnostic_only"]

    report = {
        "role": "explore",
        "protocol": "04_condition_discovery_first_harm",
        "member": args.member,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": args.seed,
        "n": int(len(df)),
        "mean_D": float(df["D"].mean()),
        "mean_D_rel": float(df["D_rel"].mean()),
        "neg_rate": float((df["D"] < 0).mean()),
        "neg_rate_D_rel": float((df["D_rel"] < 0).mean()),
        "horizon": horizon,
        "strata_D_rel": strata_rel,
        "strata_D": strata_d,
        "scale_illusion_flags": illusion,
        "C_hat_all": c_all,
        "C_hat": c_hat_confirm,
        "C_hat_diagnostic": c_hat_diag,
        "next_step": "Confirm C_hat on ≥1 other member and ≥1 other domain (L1-pass units).",
    }
    df.to_csv(out / "paired_test_with_features.csv", index=False)
    path = out / "condition_explore.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        f"# Condition explore (expert H*) — {args.member}/{args.dataset}",
        "",
        f"n={len(df)} mean_D={df['D'].mean():.6g} mean_D_rel={df['D_rel'].mean():.6g} "
        f"neg_rate={(df['D']<0).mean():.4f}",
        "",
        "## C_hat (for confirm)",
        "",
    ]
    for c in c_hat_confirm:
        lines.append(
            f"- **{c['hypothesis']}/{c['feature']}**: {c['direction']} "
            f"ΔD_rel(low−high)={c.get('explore_delta_mean_D_rel_low_minus_high', c.get('explore_delta_mean_D_rel_near_minus_far')):.4g}"
        )
    if c_hat_diag:
        lines += ["", "## Diagnostic (not deployable)", ""]
        for c in c_hat_diag:
            lines.append(f"- {c['feature']}: {c['direction']}")
    if horizon:
        lines += [
            "",
            "## H4 horizon",
            f"- mean_D_rel near={horizon['mean_D_rel_near']:.4g}"
            + (
                f" mid={horizon['mean_D_rel_mid']:.4g}"
                if "mean_D_rel_mid" in horizon
                else ""
            )
            + f" far={horizon['mean_D_rel_far']:.4g}",
        ]
    lines += ["", "_Explore only — do not claim until confirm._", ""]
    (out / "condition_notes.md").write_text("\n".join(lines) + "\n")

    for p in Path(args.output_dir).resolve().parents:
        if p.name == "outputs":
            agg = p / "aggregate"
            agg.mkdir(parents=True, exist_ok=True)
            (agg / "condition_explore.json").write_text(json.dumps(report, indent=2) + "\n")
            break

    print("=== NPF condition explore (expert) ===")
    print(f"n={len(df)} mean_D={df['D'].mean():.6g} mean_D_rel={df['D_rel'].mean():.6g}")
    for c in c_hat_confirm:
        print(
            "C_hat:",
            c["hypothesis"],
            c["feature"],
            c["direction"],
            c.get("explore_delta_mean_D_rel_low_minus_high")
            or c.get("explore_delta_mean_D_rel_near_minus_far"),
        )
    for c in c_all:
        if c.get("status") == "rewritten_not_promoted":
            print("REWRITE:", c["hypothesis"], c.get("note"))
    print("wrote", path)
    return report


def confirm_one(
    explore: dict,
    paired_csv: Path,
    data_csv: Path,
    seq_len: int,
    pred_len: int,
    member: str,
    dataset: str,
    min_ratio: float,
) -> dict:
    c_hats = explore.get("C_hat", [])
    paired = pd.read_csv(paired_csv)
    feats = build_features(data_csv, paired["index"].to_numpy(), seq_len, pred_len)
    df = attach_d_rel(paired.merge(feats, on="index", how="inner"))

    results = []
    for c in c_hats:
        feat = c["feature"]
        if feat == "horizon_far_vs_near":
            if "D_rel_near" not in df.columns:
                results.append({**c, "confirm_status": "missing_feature"})
                continue
            near = float(df["D_rel_near"].mean())
            far = float(df["D_rel_far"].mean())
            delta = near - far  # >0 means far more negative
            exp_delta = c.get("explore_delta_mean_D_rel_near_minus_far")
            # explore stored near-far; same sign means far still more harm if both positive
            same_sign = exp_delta is not None and (delta * exp_delta) > 0
            strong = exp_delta is not None and abs(delta) >= min_ratio * abs(exp_delta)
            # also record whether far is more negative if direction is far_more_harm
            dir_ok = (far < near) if c.get("direction") == "far_more_harm" else (near < far)
            # Pass = directional replication only (same sign as explore).
            passed = bool(same_sign and dir_ok)
            results.append(
                {
                    **c,
                    "confirm_delta": delta,
                    "confirm_mean_D_rel_near": near,
                    "confirm_mean_D_rel_far": far,
                    "same_sign": same_sign,
                    "strong_enough": strong,
                    "confirm_status": "pass" if passed else "fail",
                    "confirm_member": member,
                    "confirm_dataset": dataset,
                }
            )
            continue

        if feat not in df.columns and feat != "mse_f0":
            results.append({**c, "confirm_status": "missing_feature"})
            continue
        strata = tercile_delta(df, feat, "D_rel")
        delta = None if strata is None else strata["delta_mean_low_minus_high"]
        exp_delta = c.get("explore_delta_mean_D_rel_low_minus_high")
        same_sign = delta is not None and exp_delta is not None and (delta * exp_delta) > 0
        strong = (
            delta is not None
            and exp_delta is not None
            and abs(delta) >= min_ratio * abs(exp_delta)
        )
        # Pass = directional replication only (same sign as explore).
        passed = bool(same_sign)
        results.append(
            {
                **c,
                "confirm_delta": delta,
                "same_sign": same_sign,
                "strong_enough": strong,
                "confirm_status": "pass" if passed else "fail",
                "confirm_member": member,
                "confirm_dataset": dataset,
                "strata": strata,
            }
        )

    return {
        "role": "confirm",
        "member": member,
        "dataset": dataset,
        "n": int(len(df)),
        "mean_D": float(df["D"].mean()),
        "mean_D_rel": float(df["D_rel"].mean()),
        "l1_mean_D_neg": bool(df["D"].mean() < 0),
        "l1_pass": bool(len(df) >= 200 and df["D"].mean() < 0),
        "results": results,
        "passed": [r for r in results if r.get("confirm_status") == "pass"],
    }


def run_confirm(args: argparse.Namespace) -> dict:
    explore = json.loads(Path(args.explore_json).read_text())
    report = confirm_one(
        explore,
        Path(args.paired_csv),
        Path(args.data_csv),
        args.seq_len,
        args.pred_len,
        args.member,
        args.dataset,
        args.min_ratio,
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "condition_confirm.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("=== NPF condition confirm ===")
    for r in report["results"]:
        print(r["feature"], r["confirm_status"], r.get("confirm_delta"))
    print("wrote", path)
    return report


def run_matrix(args: argparse.Namespace) -> None:
    """Explore TaTS×Environment then confirm on all other L1-pass units."""
    root = Path(args.outputs_root)
    tats_data = Path(args.tats_data_root)
    cfa_data = Path(args.cfa_data_root)

    explore_dir = (
        root
        / "tats"
        / "Environment"
        / "NPF_Environment_iTransformer_seed2025_sl24_pl48"
    )
    explore_args = argparse.Namespace(
        paired_csv=str(explore_dir / "paired_test.csv"),
        data_csv=str(tats_data / "Environment.csv"),
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        output_dir=str(explore_dir),
        member="TaTS",
        dataset="Environment",
        backbone="iTransformer",
        seed=2025,
    )
    explore = run_explore(explore_args)
    explore_json = explore_dir / "condition_explore.json"

    # L1-pass units excluding explore itself
    units = [
        ("tats", "TaTS", "Energy", f"NPF_Energy_iTransformer_seed2025_sl24_pl48", "tats"),
        ("gpt4mts", "GPT4MTS", "Environment", f"NPF_Environment_GPT4MTS_seed2025_sl24_pl48", "tats"),
        ("gpt4mts", "GPT4MTS", "Health", f"NPF_Health_GPT4MTS_seed2025_sl24_pl48", "tats"),
        ("naive_cfa", "CFA-naive", "Health", f"NPF_Health_iTransformer_seed2025_sl24_pl48", "cfa"),
        ("tats", "TaTS", "Health", f"NPF_Health_iTransformer_seed2025_sl24_pl48", "tats"),  # L1 fail diagnostic
        ("naive_cfa", "CFA-naive", "Environment", f"NPF_Environment_iTransformer_seed2025_sl24_pl48", "cfa"),
        ("naive_cfa", "CFA-naive", "Energy", f"NPF_Energy_iTransformer_seed2025_sl24_pl48", "cfa"),
        ("gpt4mts", "GPT4MTS", "Energy", f"NPF_Energy_GPT4MTS_seed2025_sl24_pl48", "tats"),
        ("gpt4mts", "GPT4MTS", "Climate", f"NPF_Climate_GPT4MTS_seed2025_sl24_pl48", "tats"),
        ("tats", "TaTS", "SocialGood", f"NPF_SocialGood_iTransformer_seed2025_sl24_pl48", "tats"),
        ("gpt4mts", "GPT4MTS", "SocialGood", f"NPF_SocialGood_GPT4MTS_seed2025_sl24_pl48", "tats"),
        ("naive_cfa", "CFA-naive", "SocialGood", f"NPF_SocialGood_iTransformer_seed2025_sl24_pl48", "cfa"),
    ]

    confirms = []
    for folder, member, dataset, run, src in units:
        run_dir = root / folder / dataset / run
        paired = run_dir / "paired_test.csv"
        if not paired.exists():
            print("skip missing", paired)
            continue
        data_csv = (
            cfa_data / dataset / f"{dataset}_NPF.csv"
            if src == "cfa"
            else tats_data / f"{dataset}.csv"
        )
        report = confirm_one(
            explore,
            paired,
            data_csv,
            args.seq_len,
            args.pred_len,
            member,
            dataset,
            args.min_ratio,
        )
        report["explore_json"] = str(explore_json)
        report["run_dir"] = str(run_dir)
        (run_dir / "condition_confirm.json").write_text(json.dumps(report, indent=2) + "\n")
        confirms.append(report)
        print(
            f"confirm {member}/{dataset}: "
            + ", ".join(f"{r['feature']}={r['confirm_status']}" for r in report["results"])
        )

    # paradigm-level: only L1-pass confirms (n≥200 & mean_D<0) count
    l1_confirms = [c for c in confirms if c.get("l1_pass")]
    feat_pass: dict[str, list[str]] = {}
    for c in l1_confirms:
        for r in c.get("passed", []):
            feat_pass.setdefault(r["feature"], []).append(f"{c['member']}/{c['dataset']}")

    paradigm = []
    for feat, units_ok in feat_pass.items():
        # explore unit (TaTS×Environment) defines C_hat; confirm needs ≥1 other member
        # and ≥1 held-out domain or non-TaTS member (protocol §2/§4)
        all_units = ["TaTS/Environment"] + units_ok
        mems = {u.split("/")[0] for u in all_units}
        non_explore = [u for u in units_ok if u != "TaTS/Environment"]
        cross_member = len(mems) >= 2
        cross_split = any(u.split("/")[1] != "Environment" for u in units_ok) or any(
            u.split("/")[0] != "TaTS" for u in units_ok
        )
        paradigm.append(
            {
                "feature": feat,
                "passed_units_L1": units_ok,
                "n_members_incl_explore": len(mems),
                "paradigm_claim_ok": bool(cross_member and len(non_explore) >= 1 and cross_split),
            }
        )

    summary = {
        "explore": {
            "member": "TaTS",
            "dataset": "Environment",
            "C_hat": explore.get("C_hat"),
            "mean_D": explore.get("mean_D"),
            "mean_D_rel": explore.get("mean_D_rel"),
        },
        "confirms": [
            {
                "member": c["member"],
                "dataset": c["dataset"],
                "l1": c["l1_pass"],
                "n": c["n"],
                "mean_D": c["mean_D"],
                "passed": [r["feature"] for r in c.get("passed", [])],
                "results": [
                    {
                        "feature": r["feature"],
                        "status": r["confirm_status"],
                        "delta": r.get("confirm_delta"),
                    }
                    for r in c.get("results", [])
                ],
            }
            for c in confirms
        ],
        "paradigm_conditions": paradigm,
        "claimable": [p for p in paradigm if p["paradigm_claim_ok"]],
    }
    agg = root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "condition_matrix.json").write_text(json.dumps(summary, indent=2) + "\n")

    md = ["# Condition matrix (expert H*)", "", "## Explore C_hat (TaTS×Environment)", ""]
    for c in explore.get("C_hat", []):
        md.append(
            f"- {c['hypothesis']}/{c['feature']}: {c['direction']} "
            f"Δ={c.get('explore_delta_mean_D_rel_low_minus_high', c.get('explore_delta_mean_D_rel_near_minus_far'))}"
        )
    md += ["", "## Confirm (only L1 n≥200 & mean_D<0 count for claims)", ""]
    for c in summary["confirms"]:
        flag = "L1" if c["l1"] else "no-L1"
        md.append(
            f"- **{c['member']}/{c['dataset']}** ({flag}, n={c['n']}, mean_D={c['mean_D']:.4g}): "
            f"pass={c['passed'] or '∅'}"
        )
    md += ["", "## Claimable paradigm conditions", ""]
    if summary["claimable"]:
        for p in summary["claimable"]:
            md.append(f"- **{p['feature']}** on {p['passed_units_L1']}")
    else:
        md.append("_None passed paradigm bar (≥2 members × held-out L1 unit)._")
    (agg / "condition_matrix.md").write_text("\n".join(md) + "\n")
    print("=== matrix summary ===")
    print("claimable:", summary["claimable"])
    print("wrote", agg / "condition_matrix.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("explore")
    e.add_argument("--paired-csv", required=True)
    e.add_argument("--data-csv", required=True)
    e.add_argument("--seq-len", type=int, required=True)
    e.add_argument("--pred-len", type=int, required=True)
    e.add_argument("--output-dir", required=True)
    e.add_argument("--member", default="TaTS")
    e.add_argument("--dataset", default="Environment")
    e.add_argument("--backbone", default="iTransformer")
    e.add_argument("--seed", type=int, default=2025)

    c = sub.add_parser("confirm")
    c.add_argument("--explore-json", required=True)
    c.add_argument("--paired-csv", required=True)
    c.add_argument("--data-csv", required=True)
    c.add_argument("--seq-len", type=int, required=True)
    c.add_argument("--pred-len", type=int, required=True)
    c.add_argument("--output-dir", required=True)
    c.add_argument("--member", required=True)
    c.add_argument("--dataset", required=True)
    c.add_argument("--min-ratio", type=float, default=0.5)

    m = sub.add_parser("matrix")
    m.add_argument(
        "--outputs-root",
        default=str(Path(__file__).resolve().parents[1] / "outputs"),
    )
    m.add_argument(
        "--tats-data-root",
        default=str(Path(__file__).resolve().parents[1] / "third_party/TaTS/data"),
    )
    m.add_argument(
        "--cfa-data-root",
        default=str(Path(__file__).resolve().parents[1] / "third_party/cfa/data"),
    )
    m.add_argument("--seq-len", type=int, default=24)
    m.add_argument("--pred-len", type=int, default=48)
    m.add_argument("--min-ratio", type=float, default=0.5)

    args = ap.parse_args()
    if args.cmd == "explore":
        run_explore(args)
    elif args.cmd == "confirm":
        run_confirm(args)
    else:
        run_matrix(args)


if __name__ == "__main__":
    main()
