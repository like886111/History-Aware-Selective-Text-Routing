"""Confirm C_hat on a held-out paired_test.csv using the same effect as explore.

The effect is the tercile gap in relative fusion utility,
D_rel = D / (mse_f0 + eps). Raw D is stored only as a performance summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_npf_conditions import confirm_one


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore-json", required=True)
    ap.add_argument("--paired-csv", required=True)
    ap.add_argument("--data-csv", required=True)
    ap.add_argument("--seq-len", type=int, required=True)
    ap.add_argument("--pred-len", type=int, required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--member", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument(
        "--min-ratio",
        type=float,
        default=0.5,
        help="|confirm delta| >= ratio * |explore delta|",
    )
    args = ap.parse_args()

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
    report["explore_json"] = str(args.explore_json)
    report["effect"] = "D_rel"
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "condition_confirm.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("=== NPF condition confirm (D_rel) ===")
    for r in report["results"]:
        print(r["feature"], r["confirm_status"], r.get("confirm_delta"))
    print("wrote", path)


if __name__ == "__main__":
    main()
