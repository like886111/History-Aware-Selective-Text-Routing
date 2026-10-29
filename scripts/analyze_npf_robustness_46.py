#!/usr/bin/env python3
"""§4.6 robustness analyses under frozen S + high-tertile + ρ=any.

R1: boundary units with mean_D>0 from gating_official.json
R2: multi-horizon on Dev GPT4MTS–Environment and CFA-naive Health/Energy
R3: multi-seed on Dev (+ CFA-naive–Energy when paired exists)

Does not re-select S or ρ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_npf_gating import (  # noqa: E402
    domain_gate_params,
    eval_with_params,
    load_paired_with_feats,
)
from npf_timemmd_official import DOMAIN_LH, SCALE_SEEDS  # noqa: E402

NPF = Path(__file__).resolve().parents[1]
AGG = NPF / "outputs" / "aggregate"
DATA = {
    "Environment": NPF / "third_party/TaTS/data/Environment.csv",
    "Energy": NPF / "third_party/TaTS/data/Energy.csv",
    "Health": NPF / "third_party/TaTS/data/Health.csv",
}

# Official H grids by domain (Time-MMD)
H_GRID = {
    "Environment": (48, 96, 192, 336),
    "Energy": (12, 24, 36, 48),
    "Health": (12, 24, 36, 48),
}

ROUTING_FEATS = tuple(
    json.loads((AGG / "gating_config.json").read_text())["risk_feats"]
)


def paired_path(member: str, dataset: str, seed: int, pl: int) -> Path:
    sl, _, _ = DOMAIN_LH[dataset]
    if member == "GPT4MTS":
        return (
            NPF
            / f"outputs/gpt4mts/{dataset}/NPF_{dataset}_GPT4MTS_seed{seed}_sl{sl}_pl{pl}/paired_test.csv"
        )
    if member == "CFA-naive":
        return (
            NPF
            / f"outputs/naive_cfa/{dataset}/NPF_{dataset}_iTransformer_seed{seed}_sl{sl}_pl{pl}/paired_test.csv"
        )
    raise ValueError(member)


def mean_D_and_gate(paired: Path, dataset: str, pl: int, seed: int) -> dict:
    sl, _, _ = DOMAIN_LH[dataset]
    data_csv = DATA[dataset]
    dfp = pd.read_csv(paired)
    d = dfp["D"].to_numpy(dtype=float)
    params = domain_gate_params(
        data_csv, sl, pl, close_rule="any", feats=ROUTING_FEATS
    )
    gdf = load_paired_with_feats(paired, data_csv, sl, pl)
    m = eval_with_params(gdf, params, seed=seed)
    return {
        "n": int(len(d)),
        "mean_D": float(d.mean()),
        "r_neg": float((d < 0).mean()),
        "mse_f0": float(m["mse_f0"]),
        "mse_f1": float(m["mse_f1"]),
        "mse_g": float(m["mse_g"]),
        "delta_f1_minus_g": float(m["delta_f1_minus_g"]),
        "beats_f1": bool(m["beats_f1"]),
        "close_text_rate": float(m["close_text_rate"]),
        "sign_D": "neg" if d.mean() < 0 else ("pos" if d.mean() > 0 else "zero"),
        "sign_delta": (
            "pos"
            if m["delta_f1_minus_g"] > 1e-12
            else ("neg" if m["delta_f1_minus_g"] < -1e-12 else "zero")
        ),
    }


def try_delta_X(paired: Path, dataset: str, pl: int) -> dict[str, float] | None:
    """Per-proxy Δ_X = mean D_rel(low tertile) - mean D_rel(high tertile)."""
    try:
        sl, _, _ = DOMAIN_LH[dataset]
        gdf = load_paired_with_feats(paired, DATA[dataset], sl, pl)
        if "mse_f0" in gdf.columns and "mse_f1" in gdf.columns:
            d_use = (
                (gdf["mse_f0"] - gdf["mse_f1"]) / gdf["mse_f0"].clip(lower=1e-12)
            ).to_numpy(dtype=float)
        else:
            d_use = gdf["D"].to_numpy(dtype=float)
        out: dict[str, float] = {}
        for feat in ROUTING_FEATS:
            if feat not in gdf.columns:
                out[feat] = float("nan")
                continue
            x = gdf[feat].to_numpy(dtype=float)
            q1, q2 = np.nanquantile(x, [1.0 / 3.0, 2.0 / 3.0])
            low = d_use[x <= q1]
            high = d_use[x >= q2]
            out[feat] = (
                float(np.nanmean(low) - np.nanmean(high))
                if len(low) and len(high)
                else float("nan")
            )
        return out
    except Exception as e:  # noqa: BLE001
        print("delta_X failed", paired, e)
        return None


def r1_boundary() -> dict:
    gat = json.loads((AGG / "gating_official.json").read_text())
    rows = []
    for r in gat["results"]:
        md = r.get("mean_D")
        if md is None or md <= 0:
            continue
        dr = r["random"]["mse_g"] - r["mse_g"]
        rows.append(
            {
                "unit": r["unit"],
                "member": r["member"],
                "dataset": r["dataset"],
                "mean_D": float(md),
                "mse_f1": float(r["mse_f1"]),
                "mse_g": float(r["mse_g"]),
                "delta_f1_minus_g": float(r["delta_f1_minus_g"]),
                "delta_rnd": float(dr),
                "sign_delta": "neg" if r["delta_f1_minus_g"] < 0 else "nonneg",
            }
        )
    return {"protocol": "R1_boundary_meanD_pos", "n": len(rows), "rows": rows}


def r3_seeds() -> dict:
    units = [
        ("GPT4MTS", "Environment", "Dev", list(SCALE_SEEDS)),
        ("CFA-naive", "Energy", "Test", list(SCALE_SEEDS)),
    ]
    out_units = []
    for member, dataset, role, seeds in units:
        sl, pl, _ = DOMAIN_LH[dataset]
        seed_rows = []
        for seed in seeds:
            paired = paired_path(member, dataset, seed, pl)
            if not paired.exists():
                seed_rows.append({"seed": seed, "status": "missing", "path": str(paired)})
                continue
            metrics = mean_D_and_gate(paired, dataset, pl, seed)
            dx = try_delta_X(paired, dataset, pl)
            n_pos = None
            if dx:
                n_pos = sum(1 for f in ROUTING_FEATS if dx.get(f, 0) > 0)
            seed_rows.append(
                {
                    "seed": seed,
                    "status": "ok",
                    **metrics,
                    "delta_X": dx,
                    "n_routing_delta_X_pos": n_pos,
                    "n_routing": len(ROUTING_FEATS),
                }
            )
            print(
                f"R3 {member}/{dataset} seed{seed}: "
                f"meanD={metrics['mean_D']:+.4f} Δ={metrics['delta_f1_minus_g']:+.4f} "
                f"ΔX+={n_pos}/{len(ROUTING_FEATS)}"
            )
        out_units.append(
            {
                "unit": f"{member}/{dataset}",
                "role": role,
                "seq_len": sl,
                "pred_len": pl,
                "seeds": seed_rows,
            }
        )
    return {
        "protocol": "R3_multiseed_frozen_S_any",
        "routing_feats": list(ROUTING_FEATS),
        "units": out_units,
    }


def r2_horizons() -> dict:
    specs = [
        ("GPT4MTS", "Environment", "Dev"),
        ("CFA-naive", "Health", "Test"),
        ("CFA-naive", "Energy", "Test"),
    ]
    units = []
    for member, dataset, role in specs:
        sl, primary_pl, _ = DOMAIN_LH[dataset]
        h_rows = []
        for pl in H_GRID[dataset]:
            paired = paired_path(member, dataset, 2025, pl)
            if not paired.exists():
                h_rows.append(
                    {
                        "H": pl,
                        "primary": pl == primary_pl,
                        "status": "missing",
                        "path": str(paired),
                    }
                )
                continue
            metrics = mean_D_and_gate(paired, dataset, pl, 2025)
            h_rows.append(
                {
                    "H": pl,
                    "primary": pl == primary_pl,
                    "status": "ok",
                    **metrics,
                }
            )
            print(
                f"R2 {member}/{dataset} H={pl}: "
                f"meanD={metrics['mean_D']:+.4f} Δ={metrics['delta_f1_minus_g']:+.4f}"
            )
        units.append(
            {
                "unit": f"{member}/{dataset}",
                "role": role,
                "seq_len": sl,
                "horizons": h_rows,
            }
        )
    return {
        "protocol": "R2_multihorizon_frozen_S_any",
        "routing_feats": list(ROUTING_FEATS),
        "seed": 2025,
        "units": units,
    }


def write_md(r1: dict, r2: dict, r3: dict) -> str:
    lines = [
        "# §4.6 Robustness archive (frozen S, high tertile, ρ=any)",
        "",
        "## R1 Boundary (mean_D > 0)",
        "",
        "| unit | mean_D | Δ(f1−g) | Δrnd |",
        "|------|-------:|--------:|-----:|",
    ]
    for r in r1["rows"]:
        lines.append(
            f"| {r['unit']} | {r['mean_D']:+.4f} | {r['delta_f1_minus_g']:+.4f} | {r['delta_rnd']:+.4f} |"
        )
    lines += ["", "## R2 Multi-horizon (seed 2025)", ""]
    for u in r2["units"]:
        lines += [
            f"### {u['unit']} ({u['role']}, L={u['seq_len']})",
            "",
            "| H | status | mean_D | Δ(f1−g) | sign_D | sign_Δ | primary |",
            "|--:|:------:|-------:|--------:|:------:|:------:|:-------:|",
        ]
        for h in u["horizons"]:
            if h["status"] != "ok":
                lines.append(
                    f"| {h['H']} | missing | — | — | — | — | {str(h.get('primary', False))} |"
                )
            else:
                lines.append(
                    f"| {h['H']} | ok | {h['mean_D']:+.4f} | {h['delta_f1_minus_g']:+.4f} | "
                    f"{h['sign_D']} | {h['sign_delta']} | {str(h['primary'])} |"
                )
        lines.append("")
    lines += ["## R3 Multi-seed", ""]
    for u in r3["units"]:
        lines += [
            f"### {u['unit']} ({u['role']})",
            "",
            "| seed | status | mean_D | Δ(f1−g) | ΔX+ / |S| | sign_D | sign_Δ |",
            "|-----:|:------:|-------:|--------:|--------:|:------:|:------:|",
        ]
        for s in u["seeds"]:
            if s["status"] != "ok":
                lines.append(f"| {s['seed']} | missing | — | — | — | — | — |")
            else:
                lines.append(
                    f"| {s['seed']} | ok | {s['mean_D']:+.4f} | {s['delta_f1_minus_g']:+.4f} | "
                    f"{s['n_routing_delta_X_pos']}/{s['n_routing']} | {s['sign_D']} | {s['sign_delta']} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    AGG.mkdir(parents=True, exist_ok=True)
    r1 = r1_boundary()
    r2 = r2_horizons()
    r3 = r3_seeds()
    report = {
        "routing_feats": list(ROUTING_FEATS),
        "close_rule": "any",
        "R1": r1,
        "R2": r2,
        "R3": r3,
    }
    (AGG / "robustness_46.json").write_text(json.dumps(report, indent=2) + "\n")
    md = write_md(r1, r2, r3)
    (AGG / "robustness_46.md").write_text(md)
    print("wrote", AGG / "robustness_46.json")
    print(md)


if __name__ == "__main__":
    main()
