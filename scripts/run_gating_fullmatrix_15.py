#!/usr/bin/env python3
"""Run the complete 15-unit official gating suite and write paper-ready aggregates.

Steps:
  1) calibrate + evaluate all 15 units (IQR+|b|+Vol)
  2) feature ablation + q-sweep on all 15
  3) write gating_fullmatrix_15.{json,md} with stratified summaries
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

NPF = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
AGG = NPF / "outputs" / "aggregate"


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(NPF))


def summarize_official() -> dict:
    rep = json.loads((AGG / "gating_official.json").read_text())
    rows = rep["results"]
    assert len(rows) == 15, len(rows)

    def pack(subset: list[dict], label: str) -> dict:
        if not subset:
            return {"label": label, "n": 0}
        return {
            "label": label,
            "n": len(subset),
            "n_beats_f1": sum(1 for r in subset if r["beats_f1"]),
            "mean_mse_f0": float(np.mean([r["mse_f0"] for r in subset])),
            "mean_mse_f1": float(np.mean([r["mse_f1"] for r in subset])),
            "mean_mse_g": float(np.mean([r["mse_g"] for r in subset])),
            "mean_delta_f1_g": float(np.mean([r["delta_f1_minus_g"] for r in subset])),
            "mean_gain_vs_random": float(
                np.mean([r["random"]["mse_g"] - r["mse_g"] for r in subset])
            ),
            "units": [r["unit"] for r in subset],
        }

    all_rows = rows
    neg = [r for r in rows if r["mean_D"] < 0]
    nonneg = [r for r in rows if r["mean_D"] >= 0]
    held = [r for r in rows if r["role"] != "calibrate"]
    held_neg = [r for r in held if r["mean_D"] < 0]

    out = {
        "protocol": rep["protocol"],
        "risk_feats": rep["risk_feats"],
        "q_star": rep["config"]["q_star"],
        "calibrate_unit": rep["config"].get("calibrate_unit"),
        "calibrate_selected": rep["config"].get("selected_on_calibrate"),
        "n_units": 15,
        "strata": {
            "all_15": pack(all_rows, "all_15"),
            "meanD_neg_10": pack(neg, "meanD_neg"),
            "meanD_nonneg_5": pack(nonneg, "meanD_nonneg"),
            "held_out_14": pack(held, "held_out"),
            "held_out_neg": pack(held_neg, "held_out_neg"),
        },
        "units": [
            {
                "unit": r["unit"],
                "member": r["member"],
                "dataset": r["dataset"],
                "role": r["role"],
                "seq_len": r["seq_len"],
                "pred_len": r["pred_len"],
                "n": r["n"],
                "mean_D": r["mean_D"],
                "mse_f0": r["mse_f0"],
                "mse_f1": r["mse_f1"],
                "mse_g": r["mse_g"],
                "delta_f1_g": r["delta_f1_minus_g"],
                "close_rate": r["close_text_rate"],
                "mse_random": r["random"]["mse_g"],
                "gain_vs_random": float(r["random"]["mse_g"] - r["mse_g"]),
                "beats_f1": r["beats_f1"],
            }
            for r in rows
        ],
    }
    return out


def summarize_ablation() -> dict:
    abl = json.loads((AGG / "gating_ablation.json").read_text())
    off = {u["unit"]: u for u in summarize_official()["units"]}
    by_var: dict[str, list] = {}
    for row in abl["feature_ablation"]:
        by_var.setdefault(row["variant"], []).append(row)

    summary = {}
    for v, rows in by_var.items():
        neg = [r for r in rows if off[r["unit"]]["mean_D"] < 0]
        summary[v] = {
            "n": len(rows),
            "n_beats_f1": sum(1 for r in rows if r["beats_f1"]),
            "mean_delta_15": float(np.mean([r["delta_f1_minus_g"] for r in rows])),
            "mean_delta_neg10": float(np.mean([r["delta_f1_minus_g"] for r in neg])) if neg else None,
            "n_beats_neg10": sum(1 for r in neg if r["beats_f1"]),
        }

    # q sweep: mean delta over 15 at each q
    q_rows = abl.get("q_sweep", [])
    q_sum: dict[float, list] = {}
    for r in q_rows:
        q_sum.setdefault(float(r["q"]), []).append(r)
    q_summary = {}
    for q, rows in sorted(q_sum.items()):
        # one row per unit
        by_u: dict[str, dict] = {}
        for r in rows:
            by_u[r["unit"]] = r
        rows15 = list(by_u.values())
        neg = [r for r in rows15 if off[r["unit"]]["mean_D"] < 0]
        q_summary[str(q)] = {
            "n": len(rows15),
            "mean_delta_15": float(np.mean([r["delta_f1_minus_g"] for r in rows15])),
            "n_beats_15": sum(1 for r in rows15 if r["beats_f1"]),
            "mean_delta_neg10": float(np.mean([r["delta_f1_minus_g"] for r in neg])),
            "n_beats_neg10": sum(1 for r in neg if r["beats_f1"]),
        }
    return {"variants": summary, "q_sweep": q_summary, "q_star_default": abl["q_star_default"]}


def write_md(full: dict, abl: dict) -> None:
    lines = [
        "# Gating full-matrix suite (15 member–domain units)",
        "",
        f"Risk feats: `{full['risk_feats']}`",
        f"Calibrate: **{full['calibrate_unit']}**; **q\\*={full['q_star']:.2f}**",
        f"Protocol: {full['protocol']}",
        "",
        "## Strata (unconditional history gate)",
        "",
        "| stratum | n | beats f1 | mean Δ(f1−g) | mean gain vs random |",
        "|---------|---|----------|--------------|---------------------|",
    ]
    for key in ("all_15", "meanD_neg_10", "meanD_nonneg_5", "held_out_14", "held_out_neg"):
        s = full["strata"][key]
        if s["n"] == 0:
            continue
        lines.append(
            f"| {s['label']} | {s['n']} | {s['n_beats_f1']}/{s['n']} | "
            f"{s['mean_delta_f1_g']:+.4g} | {s['mean_gain_vs_random']:+.4g} |"
        )

    lines += [
        "",
        "## Unit table",
        "",
        "| unit | role | meanD | mse_f0 | mse_f1 | mse_g | Δ(f1−g) | close% | mse_rand | gain_rand |",
        "|------|------|-------|--------|--------|-------|---------|--------|----------|-----------|",
    ]
    for u in full["units"]:
        lines.append(
            f"| {u['unit']} | {u['role']} | {u['mean_D']:+.4g} | "
            f"{u['mse_f0']:.6g} | {u['mse_f1']:.6g} | {u['mse_g']:.6g} | "
            f"{u['delta_f1_g']:+.4g} | {100*u['close_rate']:.1f}% | "
            f"{u['mse_random']:.6g} | {u['gain_vs_random']:+.4g} |"
        )

    lines += [
        "",
        "## Feature ablation (mean over 15 / over meanD<0)",
        "",
        "| variant | beats 15 | mean Δ 15 | beats neg10 | mean Δ neg10 |",
        "|---------|----------|-----------|-------------|--------------|",
    ]
    for v, s in abl["variants"].items():
        lines.append(
            f"| {v} | {s['n_beats_f1']}/{s['n']} | {s['mean_delta_15']:+.4g} | "
            f"{s['n_beats_neg10']}/10 | {s['mean_delta_neg10']:+.4g} |"
        )

    lines += [
        "",
        "## q-sweep (full risk; mean over 15)",
        "",
        "| q | beats 15 | mean Δ 15 | beats neg10 | mean Δ neg10 |",
        "|---|----------|-----------|-------------|--------------|",
    ]
    for q, s in abl["q_sweep"].items():
        lines.append(
            f"| {q} | {s['n_beats_15']}/{s['n']} | {s['mean_delta_15']:+.4g} | "
            f"{s['n_beats_neg10']}/10 | {s['mean_delta_neg10']:+.4g} |"
        )

    (AGG / "gating_fullmatrix_15.md").write_text("\n".join(lines) + "\n")
    print("wrote", AGG / "gating_fullmatrix_15.md")


def main() -> None:
    py = sys.executable
    run([py, str(SCRIPTS / "analyze_npf_gating_official.py")])
    run([py, str(SCRIPTS / "analyze_npf_gating_ablation.py"), "--protocol", "official"])

    full = summarize_official()
    # need ablation after official for q_star; re-read official inside summarize_ablation
    abl = summarize_ablation()
    report = {"fullmatrix": full, "ablation": abl}
    AGG.mkdir(parents=True, exist_ok=True)
    (AGG / "gating_fullmatrix_15.json").write_text(json.dumps(report, indent=2) + "\n")
    write_md(full, abl)
    print("wrote", AGG / "gating_fullmatrix_15.json")


if __name__ == "__main__":
    main()
