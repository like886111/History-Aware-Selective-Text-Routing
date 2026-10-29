"""Phenomenon-driven pilot: worst-D cases + feature–D stratification → C_hat candidates.

Explore-only: does NOT claim conditions; writes condition_explore.json for later confirm.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


NA_PAT = re.compile(
    r"(no information available|no relevant information|^NA\b|^\s*$)",
    re.IGNORECASE,
)


def is_contentful(s: str) -> bool:
    s = (s or "").strip()
    if not s or s.lower() in {"nan", "na"}:
        return False
    if NA_PAT.search(s):
        return False
    return True


def test_border(n_rows: int, seq_len: int) -> tuple[int, int]:
    """Match Dataset_Custom borders for flag=test."""
    num_train = int(n_rows * 0.7)
    num_test = int(n_rows * 0.2)
    border1 = n_rows - num_test - seq_len
    border2 = n_rows
    return border1, border2


def build_features(
    data_csv: Path,
    indices: np.ndarray,
    seq_len: int,
    pred_len: int,
    text_col: str | None = None,
) -> pd.DataFrame:
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

    rows = []
    for idx in indices:
        s_begin = int(idx)  # within test slice
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
        rows.append(
            {
                "index": int(idx),
                "abs_begin": int(abs_begin),
                "date_hist_start": dates[abs_begin],
                "date_hist_end": dates[abs_end - 1],
                "hist_mean": float(hist.mean()),
                "hist_std": float(hist.std()),
                "hist_iqr": float(np.quantile(hist, 0.75) - np.quantile(hist, 0.25)),
                "hist_abs_slope": abs(slope),
                "hist_slope": slope,
                "text_nonempty_rate": float(content.mean()),
                "text_any_content": bool(content.any()),
                "text_mean_len": float(text_lens.mean()),
                "text_max_len": float(text_lens.max()),
                "fact_last": texts[-1][:240],
            }
        )
    return pd.DataFrame(rows)


def tercile_delta(df: pd.DataFrame, feat: str, dcol: str = "D") -> dict | None:
    x = df[feat].to_numpy(dtype=float)
    d = df[dcol].to_numpy(dtype=float)
    if np.unique(x).size < 3:
        # fallback binary split
        med = np.median(x)
        low, high = d[x <= med], d[x > med]
        if low.size == 0 or high.size == 0:
            return None
        return {
            "feature": feat,
            "split": "median",
            "n_low": int(low.size),
            "n_high": int(high.size),
            "mean_D_low": float(low.mean()),
            "mean_D_high": float(high.mean()),
            "neg_rate_low": float((low < 0).mean()),
            "neg_rate_high": float((high < 0).mean()),
            "delta_mean_D_low_minus_high": float(low.mean() - high.mean()),
            "corr_feat_D": float(np.corrcoef(x, d)[0, 1]) if x.std() > 0 else None,
        }
    q1, q2 = np.quantile(x, [1 / 3, 2 / 3])
    low, mid, high = d[x <= q1], d[(x > q1) & (x <= q2)], d[x > q2]
    return {
        "feature": feat,
        "split": "tercile",
        "n_low": int(low.size),
        "n_mid": int(mid.size),
        "n_high": int(high.size),
        "mean_D_low": float(low.mean()),
        "mean_D_mid": float(mid.mean()),
        "mean_D_high": float(high.mean()),
        "neg_rate_low": float((low < 0).mean()),
        "neg_rate_mid": float((mid < 0).mean()),
        "neg_rate_high": float((high < 0).mean()),
        # negative delta => low-feat subset more harmful (mean_D more negative)
        "delta_mean_D_low_minus_high": float(low.mean() - high.mean()),
        "corr_feat_D": float(np.corrcoef(x, d)[0, 1]) if x.std() > 0 else None,
    }


def propose_c_hat(strata: list[dict], top_k: int = 3) -> list[dict]:
    """Heuristic: largest |delta|; phrase direction from sign."""
    scored = []
    for s in strata:
        if s is None:
            continue
        delta = s["delta_mean_D_low_minus_high"]
        scored.append((abs(delta), s))
    scored.sort(key=lambda t: t[0], reverse=True)
    out = []
    for _, s in scored[:top_k]:
        feat = s["feature"]
        delta = s["delta_mean_D_low_minus_high"]
        if delta < 0:
            # low feat → more negative mean_D → more harm
            hyp = f"lower {feat} associated with more negative mean_D (explore)"
            direction = "low_more_harm"
        else:
            hyp = f"higher {feat} associated with more negative mean_D (explore)"
            direction = "high_more_harm"
        out.append(
            {
                "feature": feat,
                "hypothesis": hyp,
                "direction": direction,
                "explore_delta_mean_D_low_minus_high": delta,
                "status": "C_hat_explore_only_needs_confirm",
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--paired-csv", required=True)
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--pred-len", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--member", default="TaTS")
    ap.add_argument("--dataset", default="Environment")
    ap.add_argument("--backbone", default="iTransformer")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--worst-k", type=int, default=30)
    ap.add_argument("--best-k", type=int, default=10)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(args.paired_csv)
    feats = build_features(
        Path(args.data_csv),
        paired["index"].to_numpy(),
        args.seq_len,
        args.pred_len,
    )
    df = paired.merge(feats, on="index", how="inner")

    # horizon summary from D_h* columns if present
    dh_cols = [c for c in df.columns if c.startswith("D_h")]
    horizon = None
    if dh_cols:
        dh = df[dh_cols].to_numpy(dtype=float)
        near = dh[:, : max(1, dh.shape[1] // 4)].mean(axis=1)
        far = dh[:, -max(1, dh.shape[1] // 4) :].mean(axis=1)
        df["D_near"] = near
        df["D_far"] = far
        horizon = {
            "mean_D_by_h": dh.mean(axis=0).tolist(),
            "neg_rate_by_h": (dh < 0).mean(axis=0).tolist(),
            "mean_D_near_quarter": float(near.mean()),
            "mean_D_far_quarter": float(far.mean()),
            "delta_near_minus_far": float(near.mean() - far.mean()),
        }

    feat_names = [
        "hist_std",
        "hist_abs_slope",
        "hist_iqr",
        "hist_mean",
        "text_nonempty_rate",
        "text_mean_len",
        "mse_f0",
    ]
    strata = []
    for f in feat_names:
        if f in df.columns:
            strata.append(tercile_delta(df, f))

    # binary text content
    if "text_any_content" in df.columns:
        a = df.loc[df["text_any_content"], "D"].to_numpy()
        b = df.loc[~df["text_any_content"], "D"].to_numpy()
        strata.append(
            {
                "feature": "text_any_content",
                "split": "binary",
                "n_true": int(a.size),
                "n_false": int(b.size),
                "mean_D_true": float(a.mean()) if a.size else None,
                "mean_D_false": float(b.mean()) if b.size else None,
                "neg_rate_true": float((a < 0).mean()) if a.size else None,
                "neg_rate_false": float((b < 0).mean()) if b.size else None,
                "delta_mean_D_low_minus_high": float(b.mean() - a.mean())
                if a.size and b.size
                else 0.0,
                "note": "low=no content, high=has content; delta=mean_D(no)-mean_D(yes)",
            }
        )

    c_hat = propose_c_hat([s for s in strata if s is not None], top_k=3)

    worst = df.nsmallest(args.worst_k, "D")
    best = df.nlargest(args.best_k, "D")
    case_cols = [
        "index",
        "D",
        "mse_f0",
        "mse_f1",
        "hist_std",
        "hist_abs_slope",
        "text_nonempty_rate",
        "text_mean_len",
        "date_hist_start",
        "date_hist_end",
        "fact_last",
    ]
    worst[case_cols].to_csv(out / "worst_D_cases.csv", index=False)
    best[case_cols].to_csv(out / "best_D_cases.csv", index=False)

    # short markdown for human skim
    lines = [
        f"# Phenomenon pilot — {args.member} / {args.dataset} / {args.backbone} / seed{args.seed}",
        "",
        f"n={len(df)}  mean_D={df['D'].mean():.6g}  neg_rate={(df['D']<0).mean():.4f}",
        "",
        "## Worst-D snippets (top)",
        "",
    ]
    for _, r in worst.head(8).iterrows():
        lines.append(
            f"- idx={int(r['index'])} D={r['D']:.4g} std={r['hist_std']:.3g} "
            f"text_rate={r['text_nonempty_rate']:.2f} | {r['fact_last'][:120]}"
        )
    lines += ["", "## C_hat (explore only)", ""]
    for c in c_hat:
        lines.append(f"- **{c['feature']}**: {c['hypothesis']} (Δ={c['explore_delta_mean_D_low_minus_high']:.4g})")
    lines += ["", "_Do not claim until confirm split passes._", ""]
    (out / "phenomenon_notes.md").write_text("\n".join(lines) + "\n")

    report = {
        "role": "explore",
        "member": args.member,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": args.seed,
        "n": int(len(df)),
        "mean_D": float(df["D"].mean()),
        "neg_rate": float((df["D"] < 0).mean()),
        "horizon": horizon,
        "strata": strata,
        "C_hat": c_hat,
        "artifacts": {
            "worst_D_cases": str(out / "worst_D_cases.csv"),
            "best_D_cases": str(out / "best_D_cases.csv"),
            "phenomenon_notes": str(out / "phenomenon_notes.md"),
            "paired_with_features": str(out / "paired_test_with_features.csv"),
        },
        "next_step": "Confirm each C_hat on held-out domain/seed and ≥1 other member; do not claim yet.",
    }
    df.to_csv(out / "paired_test_with_features.csv", index=False)
    explore_path = out / "condition_explore.json"
    explore_path.write_text(json.dumps(report, indent=2) + "\n")

    # also copy to aggregate for visibility
    agg = Path(args.output_dir).resolve().parents[2] / "aggregate"
    # outputs/tats/Environment/RUN/... -> parents[2]=outputs
    # safer: NPF outputs/aggregate
    for p in Path(args.output_dir).resolve().parents:
        if p.name == "outputs":
            agg = p / "aggregate"
            break
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "condition_explore.json").write_text(json.dumps(report, indent=2) + "\n")

    print("=== NPF phenomenon explore ===")
    print(f"n={len(df)} mean_D={df['D'].mean():.6g} neg_rate={(df['D']<0).mean():.4f}")
    for c in c_hat:
        print("C_hat:", c["feature"], c["direction"], c["explore_delta_mean_D_low_minus_high"])
    print("wrote", explore_path)
    print("wrote", out / "phenomenon_notes.md")


if __name__ == "__main__":
    main()
