#!/usr/bin/env python3
"""Confirm GPT4MTS–Environment C_hat on all official-path member×domain units."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_npf_conditions import confirm_one
from npf_timemmd_official import DOMAIN_LH, MEMBERS, run_rel

NPF = Path(__file__).resolve().parents[1]

# Primary exploration unit (aligned with Sec. 4.2 / paper protocol).
EXPLORE_MEMBER = "GPT4MTS"
EXPLORE_DATASET = "Environment"


def data_csv(member: str, dataset: str) -> Path:
    if member == "CFA-naive":
        return NPF / "third_party/cfa/data" / dataset / f"{dataset}_NPF.csv"
    return NPF / "third_party/TaTS/data" / f"{dataset}.csv"


def main() -> None:
    seed = 2025
    env_sl, env_pl, _ = DOMAIN_LH[EXPLORE_DATASET]
    explore_path = (
        NPF
        / "outputs"
        / run_rel(EXPLORE_MEMBER, EXPLORE_DATASET, seed)
        / "condition_explore.json"
    )
    if not explore_path.exists():
        raise SystemExit(f"missing explore: {explore_path}")
    explore = json.loads(explore_path.read_text())

    rows = []
    for dataset, (sl, pl, _) in DOMAIN_LH.items():
        for member in MEMBERS:
            rel = run_rel(member, dataset, seed)
            out = NPF / "outputs" / rel
            paired = out / "paired_test.csv"
            csv = data_csv(member, dataset)
            if not paired.exists() or not csv.exists():
                print("skip", rel)
                continue
            report = confirm_one(
                explore, paired, csv, sl, pl, member, dataset, min_ratio=0.5
            )
            report["explore_json"] = str(explore_path)
            report["effect"] = "D_rel"
            report["seq_len"] = sl
            report["pred_len"] = pl
            (out / "condition_confirm.json").write_text(json.dumps(report, indent=2) + "\n")
            n_pass = sum(1 for r in report["results"] if r.get("confirm_status") == "pass")
            n_same = sum(1 for r in report["results"] if r.get("same_sign"))
            rows.append(
                {
                    "member": member,
                    "dataset": dataset,
                    "seq_len": sl,
                    "pred_len": pl,
                    "n_pass": n_pass,
                    "n_same_sign": n_same,
                    "n_features": len(report["results"]),
                    "results": report["results"],
                }
            )
            print(
                f"{member}/{dataset} L={sl} H={pl}: "
                f"pass={n_pass}/{len(report['results'])} same_sign={n_same}/{len(report['results'])}"
            )

    agg = NPF / "outputs/aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "condition_confirm_official.json").write_text(
        json.dumps({"explore": str(explore_path), "units": rows}, indent=2) + "\n"
    )
    lines = [
        "# Condition confirm — Time-MMD official (L,H)",
        "",
        f"Explore: {EXPLORE_MEMBER}/{EXPLORE_DATASET} L={env_sl} H={env_pl}",
        "",
        "| unit | L | H | same_direction |",
        "|------|---|---|---------------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['member']}/{r['dataset']} | {r['seq_len']} | {r['pred_len']} | "
            f"{r['n_same_direction']}/{r['n_features']} |"
        )
    (agg / "condition_confirm_official.md").write_text("\n".join(lines) + "\n")
    print("wrote", agg / "condition_confirm_official.md")


if __name__ == "__main__":
    main()
