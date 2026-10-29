"""Phase B3: confirm C_hat + hard gate on new seed/horizon dumps; write aggregate md."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NPF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(NPF / "scripts"))
from analyze_npf_gating import domain_gate_params, eval_with_params, load_paired_with_feats  # noqa: E402

EXPLORE = (
    NPF
    / "outputs/tats/Environment/NPF_Environment_iTransformer_seed2025_sl24_pl48/condition_explore.json"
)
PY = sys.executable
Q_STAR = 0.6

# (member, dataset, run_dir relative to outputs, data_csv, seq, pred, seed)
UNITS = []


def add(member, dataset, rel, data_csv, seed, seq=24, pred=48):
    UNITS.append((member, dataset, rel, data_csv, seed, seq, pred))


# multi-seed extras
add("GPT4MTS", "Climate", "gpt4mts/Climate/NPF_Climate_GPT4MTS_seed2024_sl24_pl48", "TaTS/Climate.csv", 2024)
add("GPT4MTS", "Climate", "gpt4mts/Climate/NPF_Climate_GPT4MTS_seed2026_sl24_pl48", "TaTS/Climate.csv", 2026)
add("GPT4MTS", "Climate", "gpt4mts/Climate/NPF_Climate_GPT4MTS_seed2027_sl24_pl48", "TaTS/Climate.csv", 2027)
add("GPT4MTS", "Health", "gpt4mts/Health/NPF_Health_GPT4MTS_seed2024_sl24_pl48", "TaTS/Health.csv", 2024)
add("GPT4MTS", "Health", "gpt4mts/Health/NPF_Health_GPT4MTS_seed2026_sl24_pl48", "TaTS/Health.csv", 2026)
add("GPT4MTS", "Health", "gpt4mts/Health/NPF_Health_GPT4MTS_seed2027_sl24_pl48", "TaTS/Health.csv", 2027)
add("TaTS", "Energy", "tats/Energy/NPF_Energy_iTransformer_seed2024_sl24_pl48", "TaTS/Energy.csv", 2024)
add("TaTS", "Energy", "tats/Energy/NPF_Energy_iTransformer_seed2026_sl24_pl48", "TaTS/Energy.csv", 2026)
add("TaTS", "Energy", "tats/Energy/NPF_Energy_iTransformer_seed2027_sl24_pl48", "TaTS/Energy.csv", 2027)
add("CFA-naive", "Health", "naive_cfa/Health/NPF_Health_iTransformer_seed2024_sl24_pl48", "cfa/Health/Health_NPF.csv", 2024)
add("CFA-naive", "Health", "naive_cfa/Health/NPF_Health_iTransformer_seed2026_sl24_pl48", "cfa/Health/Health_NPF.csv", 2026)
add("CFA-naive", "Health", "naive_cfa/Health/NPF_Health_iTransformer_seed2027_sl24_pl48", "cfa/Health/Health_NPF.csv", 2027)
add("GPT4MTS", "Environment", "gpt4mts/Environment/NPF_Environment_GPT4MTS_seed2024_sl24_pl48", "TaTS/Environment.csv", 2024)
add("GPT4MTS", "Environment", "gpt4mts/Environment/NPF_Environment_GPT4MTS_seed2026_sl24_pl48", "TaTS/Environment.csv", 2026)
add("GPT4MTS", "Environment", "gpt4mts/Environment/NPF_Environment_GPT4MTS_seed2027_sl24_pl48", "TaTS/Environment.csv", 2027)
add("TaTS", "Environment", "tats/Environment/NPF_Environment_iTransformer_seed2024_sl24_pl48", "TaTS/Environment.csv", 2024)
add("TaTS", "Environment", "tats/Environment/NPF_Environment_iTransformer_seed2026_sl24_pl48", "TaTS/Environment.csv", 2026)
add("TaTS", "Environment", "tats/Environment/NPF_Environment_iTransformer_seed2027_sl24_pl48", "TaTS/Environment.csv", 2027)
# horizons
for pl in (24, 96):
    add("TaTS", "Energy", f"tats/Energy/NPF_Energy_iTransformer_seed2025_sl24_pl{pl}", "TaTS/Energy.csv", 2025, 24, pl)
    add("GPT4MTS", "Climate", f"gpt4mts/Climate/NPF_Climate_GPT4MTS_seed2025_sl24_pl{pl}", "TaTS/Climate.csv", 2025, 24, pl)
    add("TaTS", "Environment", f"tats/Environment/NPF_Environment_iTransformer_seed2025_sl24_pl{pl}", "TaTS/Environment.csv", 2025, 24, pl)


def data_path(tag: str) -> Path:
    if tag.startswith("TaTS/"):
        return NPF / "third_party/TaTS/data" / tag.split("/", 1)[1]
    if tag.startswith("cfa/"):
        return NPF / "third_party/cfa/data" / tag.split("/", 1)[1]
    raise ValueError(tag)


def main() -> None:
    confirm_rows = []
    gate_rows = []
    for member, dataset, rel, dtag, seed, seq, pred in UNITS:
        run_dir = NPF / "outputs" / rel
        paired = run_dir / "paired_test.csv"
        if not paired.exists():
            continue
        csv = data_path(dtag)
        # confirm
        if EXPLORE.exists():
            subprocess.call(
                [
                    PY,
                    str(NPF / "scripts/analyze_npf_confirm.py"),
                    "--explore-json",
                    str(EXPLORE),
                    "--paired-csv",
                    str(paired),
                    "--data-csv",
                    str(csv),
                    "--seq-len",
                    str(seq),
                    "--pred-len",
                    str(pred),
                    "--output-dir",
                    str(run_dir),
                    "--member",
                    member,
                    "--dataset",
                    dataset,
                ]
            )
            cj = run_dir / "condition_confirm.json"
            if cj.exists():
                conf = json.loads(cj.read_text())
                confirm_rows.append(
                    {
                        "member": member,
                        "dataset": dataset,
                        "seed": seed,
                        "pred_len": pred,
                        "confirm": conf,
                    }
                )
        # hard gate
        try:
            df = load_paired_with_feats(paired, csv, seq, pred)
            params = domain_gate_params(csv, seq, pred, Q_STAR)
            m = eval_with_params(df, params, seed=seed)
            gate_rows.append(
                {
                    "member": member,
                    "dataset": dataset,
                    "seed": seed,
                    "pred_len": pred,
                    **{k: m[k] for k in ("mse_g", "mse_f0", "mse_f1", "delta_f1_minus_g", "close_text_rate", "beats_f1", "not_worse_f0")},
                    "vs_random": m["mse_g"] - m["random"]["mse_g"] if "random" in m else None,
                }
            )
        except Exception as e:
            gate_rows.append(
                {
                    "member": member,
                    "dataset": dataset,
                    "seed": seed,
                    "pred_len": pred,
                    "error": str(e),
                }
            )

    agg = NPF / "outputs/aggregate"
    agg.mkdir(parents=True, exist_ok=True)

    # flatten confirm for md
    lines = ["# Condition confirm on scale units", "", "Explore locked: Env×TaTS×seed2025", ""]
    lines.append("| unit | seed | pl | hist_std | acf1 | shock |")
    lines.append("|------|-----:|---:|:--------:|:----:|:-----:|")
    for r in confirm_rows:
        c = r["confirm"]
        # support both list and dict layouts
        feats = {}
        if isinstance(c, dict):
            if "results" in c:
                for item in c["results"]:
                    feats[item.get("feature") or item.get("name")] = item.get("pass") or item.get("status")
            elif "features" in c:
                feats = c["features"]
            else:
                for k, v in c.items():
                    if isinstance(v, dict) and ("pass" in v or "status" in v):
                        feats[k] = v.get("pass", v.get("status"))
        def cell(name):
            # from results list
            for item in (c.get("results") or []):
                if item.get("feature") == name:
                    return item.get("confirm_status") or item.get("status") or "—"
            v = feats.get(name)
            if v is True or v == "pass":
                return "pass"
            if v is False or v == "fail":
                return "fail"
            return str(v)[:8] if v is not None else "—"

        lines.append(
            f"| {r['member']}/{r['dataset']} | {r['seed']} | {r['pred_len']} | "
            f"{cell('hist_std')} | {cell('acf1')} | {cell('shock_score')} |"
        )
    (agg / "condition_scale_confirm.md").write_text("\n".join(lines) + "\n")
    (agg / "condition_scale_confirm.json").write_text(json.dumps(confirm_rows, indent=2) + "\n")

    glines = [
        "# Hard gate on scale units (q*=0.6 frozen)",
        "",
        "| unit | seed | pl | mse_g | mse_f0 | mse_f1 | Δ(f1−g) | close% | <f1 | ≤f0 |",
        "|------|-----:|---:|------:|-------:|-------:|--------:|-------:|:--:|:--:|",
    ]
    for r in gate_rows:
        if "error" in r:
            glines.append(f"| {r['member']}/{r['dataset']} | {r['seed']} | {r['pred_len']} | ERROR {r['error'][:40]} |")
            continue
        glines.append(
            f"| {r['member']}/{r['dataset']} | {r['seed']} | {r['pred_len']} | "
            f"{r['mse_g']:.6g} | {r['mse_f0']:.6g} | {r['mse_f1']:.6g} | "
            f"{r['delta_f1_minus_g']:+.4g} | {100*r['close_text_rate']:.1f}% | "
            f"{'Y' if r['beats_f1'] else 'N'} | {'Y' if r['not_worse_f0'] else 'N'} |"
        )
    (agg / "gating_scale.md").write_text("\n".join(glines) + "\n")
    (agg / "gating_scale.json").write_text(json.dumps(gate_rows, indent=2) + "\n")
    print("wrote", agg / "condition_scale_confirm.md")
    print("wrote", agg / "gating_scale.md")


if __name__ == "__main__":
    main()
