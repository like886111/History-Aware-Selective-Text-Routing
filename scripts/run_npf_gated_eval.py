"""Inference-side hard gate eval: blend f0/f1 dumps by frozen q* + per-domain train z/τ.

Works for TaTS / CFA-naive / GPT4MTS (any member with paired dumps + data csv).
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
    build_features,
    domain_gate_params,
    metrics,
    random_gate,
    risk_scores,
)
from analyze_npf_l1 import load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0-dump", required=True)
    ap.add_argument("--f1-dump", required=True)
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--config", required=True, help="gating_config.json with q_star")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", default="")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--pred-len", type=int, default=48)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    q_star = float(cfg["q_star"])
    params = domain_gate_params(
        Path(args.data_csv), args.seq_len, args.pred_len, q_star
    )

    y0, t0, i0 = load_split(Path(args.f0_dump), args.split)
    y1, t1, i1 = load_split(Path(args.f1_dump), args.split)
    map1 = {int(ix): j for j, ix in enumerate(i1)}
    keep0, keep1 = [], []
    for j, ix in enumerate(i0):
        k = map1.get(int(ix))
        if k is not None:
            keep0.append(j)
            keep1.append(k)
    y0, t0, i0 = y0[keep0], t0[keep0], i0[keep0]
    y1, t1 = y1[keep1], t1[keep1]

    feats = build_features(Path(args.data_csv), i0, args.seq_len, args.pred_len)
    # align feats to i0 order
    feat_map = {int(r["index"]): r for _, r in feats.iterrows()}
    rows = [feat_map[int(ix)] for ix in i0 if int(ix) in feat_map]
    if len(rows) != len(i0):
        # drop missing
        mask = np.array([int(ix) in feat_map for ix in i0])
        y0, t0, i0, y1, t1 = y0[mask], t0[mask], i0[mask], y1[mask], t1[mask]
        rows = [feat_map[int(ix)] for ix in i0]
    feat_df = pd.DataFrame(rows)

    r = risk_scores(feat_df, params["z_stats"])
    close = r >= params["tau"]  # True → use f0

    # blend predictions
    yg = np.where(close[:, None], y0, y1)
    mse0 = ((y0 - t0) ** 2).mean(axis=1)
    mse1 = ((y1 - t1) ** 2).mean(axis=1)
    mseg = ((yg - t0) ** 2).mean(axis=1)
    m = metrics(mseg, mse0, mse1, close)
    m["random"] = random_gate(mse0, mse1, m["close_text_rate"], seed=args.seed)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dump_g = out / "dumps" / "gated" / args.split
    dump_g.mkdir(parents=True, exist_ok=True)
    np.save(dump_g / "pred.npy", yg)
    np.save(dump_g / "true.npy", t0)
    np.save(dump_g / "index.npy", i0)
    np.save(dump_g / "close_text.npy", close.astype(np.uint8))

    paired = pd.DataFrame(
        {
            "index": i0,
            "mse_f0": mse0,
            "mse_f1": mse1,
            "mse_g": mseg,
            "close_text": close.astype(int),
            "risk_score": r,
            "D": mse0 - mse1,
            "D_g": mse0 - mseg,  # >0 means g better than f0
        }
    )
    paired.to_csv(out / "paired_gated_test.csv", index=False)

    report = {
        "role": "gated_eval",
        "member": args.member,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "q_star": q_star,
        "domain_params": params,
        "metrics": m,
        "success_vs_f1": m["beats_f1"],
        "success_vs_f0": m["not_worse_f0"],
        "artifacts": {
            "paired_gated_test": str(out / "paired_gated_test.csv"),
            "gated_dump": str(dump_g),
        },
    }
    path = out / "gating_eval.json"
    path.write_text(json.dumps(report, indent=2) + "\n")

    print("=== NPF gated eval ===")
    print(f"{args.member}/{args.dataset} q*={q_star:.2f} tau={params['tau']:.4g}")
    print(
        f"mse_g={m['mse_g']:.6g} f0={m['mse_f0']:.6g} f1={m['mse_f1']:.6g} "
        f"oracle={m['mse_oracle']:.6g} close={m['close_text_rate']:.3f} "
        f"Δ(f1−g)={m['delta_f1_minus_g']:+.4g} Δ(f0−g)={m['delta_f0_minus_g']:+.4g}"
    )
    print("wrote", path)


if __name__ == "__main__":
    main()
