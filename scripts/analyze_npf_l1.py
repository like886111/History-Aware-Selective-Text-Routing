"""NPF L1: build paired_test.csv and existence report from f0/f1 dumps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_split(dump_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = dump_dir / split
    pred = np.load(d / "pred.npy")
    true = np.load(d / "true.npy")
    index = np.load(d / "index.npy")
    if pred.ndim == 3:
        pred = pred[..., 0]
        true = true[..., 0]
    return pred.astype(np.float64), true.astype(np.float64), index.astype(np.int64)


def summarize(d: np.ndarray) -> dict:
    if d.size == 0:
        return {"n": 0, "neg_rate": None, "mean_D": None, "median_D": None}
    return {
        "n": int(d.size),
        "neg_rate": float((d < 0).mean()),
        "mean_D": float(d.mean()),
        "median_D": float(np.median(d)),
        "p10_D": float(np.quantile(d, 0.1)),
        "p90_D": float(np.quantile(d, 0.9)),
        "std_D": float(d.std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--f0-dump", required=True)
    ap.add_argument("--f1-dump", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--member", default="TaTS")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--pred-len", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument(
        "--neg-rate-thresh",
        type=float,
        default=0.35,
        help="descriptive only; unit_pass uses mean_D < 0",
    )
    args = ap.parse_args()

    f0_dir = Path(args.f0_dump)
    f1_dir = Path(args.f1_dump)
    y0, t0, i0 = load_split(f0_dir, args.split)
    y1, t1, i1 = load_split(f1_dir, args.split)

    # Align on shared indices (order-preserving intersection)
    map1 = {int(ix): j for j, ix in enumerate(i1)}
    keep0, keep1 = [], []
    for j, ix in enumerate(i0):
        k = map1.get(int(ix))
        if k is not None:
            keep0.append(j)
            keep1.append(k)
    if not keep0:
        raise RuntimeError("no overlapping indices between f0 and f1 dumps")
    y0, t0, i0 = y0[keep0], t0[keep0], i0[keep0]
    y1, t1, i1 = y1[keep1], t1[keep1], i1[keep1]

    max_true_diff = float(np.nanmax(np.abs(t0 - t1)))
    if max_true_diff > 1e-5:
        raise RuntimeError(f"true mismatch after index align: max_diff={max_true_diff}")

    mse0_h = (y0 - t0) ** 2
    mse1_h = (y1 - t1) ** 2
    mse0 = mse0_h.mean(axis=1)
    mse1 = mse1_h.mean(axis=1)
    d = mse0 - mse1
    d_h = mse0_h - mse1_h

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    paired = pd.DataFrame(
        {
            "index": i0,
            "mse_f0": mse0,
            "mse_f1": mse1,
            "D": d,
        }
    )
    for h in range(d_h.shape[1]):
        paired[f"D_h{h}"] = d_h[:, h]
    paired_path = out / f"paired_{args.split}.csv"
    paired.to_csv(paired_path, index=False)

    stats = summarize(d)
    # Unit pass for "stable harm": mean_D < 0 (neg_rate is descriptive only).
    l1_pass = bool(stats["n"] >= 200 and stats["mean_D"] is not None and stats["mean_D"] < 0)
    report = {
        "member": args.member,
        "dataset": args.dataset,
        "backbone": args.backbone,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "split": args.split,
        "f0_dump": str(f0_dir),
        "f1_dump": str(f1_dir),
        "definition": {
            "D": "MSE(f0)-MSE(f1); D<0 => text harmful on sample",
            "claim": "L1 existence for phenomenon-driven condition discovery",
        },

        "aggregate_mse_f0": float(mse0.mean()),
        "aggregate_mse_f1": float(mse1.mean()),
        "L1": stats,
        "L1_criteria": {
            "n_ge_200": bool(stats["n"] >= 200),
            "mean_D_neg": bool(stats["mean_D"] is not None and stats["mean_D"] < 0),
            "neg_rate_descriptive": stats["neg_rate"],
            "neg_rate_thresh_unused_for_pass": args.neg_rate_thresh,
            "pass": l1_pass,
        },
        "neg_rate_by_h": (d_h < 0).mean(axis=0).tolist(),
        "mean_D_by_h": d_h.mean(axis=0).tolist(),
        "paired_csv": str(paired_path),
    }
    report_path = out / "l1_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("=== NPF L1 ===")
    print(
        f"{args.member} {args.dataset} {args.backbone} seed{args.seed} "
        f"n={stats['n']} neg_rate={stats['neg_rate']:.4f} mean_D={stats['mean_D']:.6g} "
        f"L1_pass={l1_pass}"
    )
    print("wrote", paired_path)
    print("wrote", report_path)


if __name__ == "__main__":
    main()
