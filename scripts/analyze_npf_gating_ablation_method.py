#!/usr/bin/env python3
"""Method ablation for history gate (reviewer design).

Variants (each changes one method knob):
  g_star     : six starred + high tertile + ρ=any
  low_tertile: six starred + low tertile + ρ=any   (ablate M2)
  rho_maj    : six starred + high tertile + ρ=maj  (ablate M3)
  rho_all    : six starred + high tertile + ρ=all  (ablate M3)
  family3    : {IQR,|b|,Vol} + high tertile + ρ=any (ablate M1)

Δrnd uses each variant's own close rate (no n* matching).
Primary read: all 15 member--domain units (official matrix).
Secondary: former 9-unit D̄<0 subset (excl. calibrate) kept for reference.
Design: docs/消融实验设计_审稿人视角.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_npf_gating as g
from analyze_npf_gating_official import DOMAIN_LH, MEMBERS, data_csv, run_dir

SEED = 2025
NPF = Path(__file__).resolve().parents[1]
AGG = NPF / "outputs/aggregate"

# Legacy negative-fusion subset (reference only; primary = all15).
NEG_FUSION_UNITS = (
    "TaTS/Environment",
    "CFA-naive/Environment",
    "GPT4MTS/Energy",
    "CFA-naive/Energy",
    "GPT4MTS/Health",
    "CFA-naive/Health",
    "CFA-naive/Health_AFR",
    "TaTS/Climate",
    "CFA-naive/Climate",
)

VARIANTS = ("g_star", "low_tertile", "rho_maj", "rho_all", "family3")
PRIMARY = "all15"


def low_mask(df, cuts, feat: str) -> np.ndarray:
    return df[feat].to_numpy(dtype=float) <= cuts[feat]["q1"]


def close_low_any(df, cuts, feats: tuple[str, ...]) -> np.ndarray:
    flags = np.stack([low_mask(df, cuts, f) for f in feats], axis=1)
    return flags.sum(axis=1) >= 1


def eval_close(mse0, mse1, close, seed: int) -> dict:
    mse_g = g.apply_gate(mse0, mse1, close)
    m = g.metrics(mse_g, mse0, mse1, close)
    rnd = g.random_gate(mse0, mse1, m["close_text_rate"], seed=seed)
    m["delta_rnd"] = float(rnd["mse_g"] - m["mse_g"])
    m["mse_f0"] = float(np.mean(mse0))
    return m


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n_units": 0}
    d_rnd = [r["delta_rnd"] for r in rows]
    d_f1 = [r["delta_f1_minus_g"] for r in rows]
    return {
        "n_units": len(rows),
        "n_beats_f1": int(sum(1 for r in rows if r["beats_f1"])),
        "n_delta_rnd_pos": int(sum(1 for x in d_rnd if x > 0)),
        "mean_delta_f1": float(np.mean(d_f1)),
        "median_delta_f1": float(np.median(d_f1)),
        "mean_delta_rnd": float(np.mean(d_rnd)),
        "median_delta_rnd": float(np.median(d_rnd)),
        "mean_close_rate": float(np.mean([r["close_text_rate"] for r in rows])),
        "median_close_rate": float(np.median([r["close_text_rate"] for r in rows])),
        "mean_mse_g": float(np.mean([r["mse_g"] for r in rows])),
        "mean_mse_f1": float(np.mean([r["mse_f1"] for r in rows])),
        "mean_mse_f0": float(np.mean([r["mse_f0"] for r in rows])),
        "units": [r["unit"] for r in rows],
    }


def check_pass(by_v: dict, key: str = PRIMARY) -> dict:
    star = by_v["g_star"][key]
    flip = by_v["low_tertile"][key]
    maj = by_v["rho_maj"][key]
    allr = by_v["rho_all"][key]
    fam = by_v["family3"][key]
    n = star["n_units"]

    g_ok = bool(
        star["median_delta_rnd"] > 0 and star["n_delta_rnd_pos"] >= max(6, (n + 1) // 2)
    )
    flip_ok = bool(
        flip["median_delta_rnd"] < star["median_delta_rnd"]
        and flip["n_delta_rnd_pos"] < star["n_delta_rnd_pos"]
    )
    rho_ok = bool(
        maj["median_delta_f1"] > 0
        or maj["median_delta_rnd"] > 0
        or allr["median_delta_f1"] > 0
        or allr["median_delta_rnd"] > 0
    )
    fam_note = {
        "median_delta_rnd": fam["median_delta_rnd"],
        "n_delta_rnd_pos": fam["n_delta_rnd_pos"],
        "retains_selection": bool(fam["median_delta_rnd"] > 0),
        "close_to_star": bool(
            abs(fam["median_delta_rnd"] - star["median_delta_rnd"])
            <= 0.5 * max(abs(star["median_delta_rnd"]), 1e-12)
        ),
    }
    return {
        "primary_key": key,
        "n_units": n,
        "g_star_selection": {"pass": g_ok, **{k: star[k] for k in (
            "median_delta_rnd", "n_delta_rnd_pos", "median_delta_f1"
        )}},
        "polarity_m2": {
            "pass": flip_ok,
            "med_rnd_flip": flip["median_delta_rnd"],
            "med_rnd_star": star["median_delta_rnd"],
            "s_flip": flip["n_delta_rnd_pos"],
            "s_star": star["n_delta_rnd_pos"],
        },
        "rho_m3": {
            "pass": rho_ok,
            "maj_med_delta_f1": maj["median_delta_f1"],
            "maj_med_delta_rnd": maj["median_delta_rnd"],
            "all_med_delta_f1": allr["median_delta_f1"],
            "all_med_delta_rnd": allr["median_delta_rnd"],
        },
        "family3_m1": fam_note,
        "all_core_pass": bool(g_ok and flip_ok),
    }


def main() -> None:
    root = NPF / "outputs"
    rows: list[dict] = []
    feats6 = tuple(g.RISK_FEATS)
    feats3 = tuple(g.FEATS_3)

    for dataset, (sl, pl) in DOMAIN_LH.items():
        for member in MEMBERS:
            unit = f"{member}/{dataset}"
            role = (
                "calibrate"
                if (member == "GPT4MTS" and dataset == "Environment")
                else "eval"
            )
            paired = run_dir(root, member, dataset, sl, pl) / "paired_test.csv"
            csv = data_csv(member, dataset)
            if not paired.exists() or not csv.exists():
                print("skip", unit)
                continue

            params = g.domain_gate_params(
                csv, sl, pl, close_rule="any", feats=feats6
            )
            cuts = params["tertile_cuts"]
            # Family-3 needs its own train cuts on the same features (subset of cuts ok)
            df = g.load_paired_with_feats(paired, csv, sl, pl)
            mse0 = df["mse_f0"].to_numpy(dtype=float)
            mse1 = df["mse_f1"].to_numpy(dtype=float)
            mean_d = float(df["D"].mean())
            base = {
                "unit": unit,
                "role": role,
                "mean_D": mean_d,
                "neg_fusion_subset": unit in NEG_FUSION_UNITS,
                "seq_len": sl,
                "pred_len": pl,
            }

            close_star = g.close_from_tertiles(df, cuts, "any", feats=feats6)
            rows.append({**base, "variant": "g_star", **eval_close(mse0, mse1, close_star, SEED)})

            close_flip = close_low_any(df, cuts, feats6)
            rows.append({**base, "variant": "low_tertile", **eval_close(mse0, mse1, close_flip, SEED)})

            close_maj = g.close_from_tertiles(df, cuts, "maj", feats=feats6)
            rows.append({**base, "variant": "rho_maj", **eval_close(mse0, mse1, close_maj, SEED)})

            close_all = g.close_from_tertiles(df, cuts, "all", feats=feats6)
            rows.append({**base, "variant": "rho_all", **eval_close(mse0, mse1, close_all, SEED)})

            close_f3 = g.close_from_tertiles(df, cuts, "any", feats=feats3)
            rows.append({**base, "variant": "family3", **eval_close(mse0, mse1, close_f3, SEED)})

            last = [r for r in rows if r["unit"] == unit]
            print(
                f"ok {unit} n={len(df)} "
                f"Δ*={last[0]['delta_f1_minus_g']:+.4f} "
                f"rnd*={last[0]['delta_rnd']:+.4f}"
            )

    by_v = {}
    for v in VARIANTS:
        vrows = [r for r in rows if r["variant"] == v]
        by_v[v] = {
            "all15": summarize(vrows),
            "neg_fusion9": summarize([r for r in vrows if r["neg_fusion_subset"]]),
        }

    passes = check_pass(by_v, key=PRIMARY)
    unit_order = [r["unit"] for r in rows if r["variant"] == "g_star"]
    report = {
        "protocol": "method ablation M1/M2/M3; primary=all15",
        "design_doc": "docs/消融实验设计_审稿人视角.md",
        "delta_rnd_definition": "matched to each variant's own close rate; no n* matching",
        "primary_read": PRIMARY,
        "feats6": list(feats6),
        "feats3": list(feats3),
        "neg_fusion_units": list(NEG_FUSION_UNITS),
        "seed": SEED,
        "by_variant": by_v,
        "pass_criteria": passes,
        "rows": rows,
    }
    AGG.mkdir(parents=True, exist_ok=True)
    (AGG / "gating_ablation_method.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )

    labels = {
        "g_star": "Full \(g^*\) (6, high, any)",
        "low_tertile": "Low-tertile (M2)",
        "rho_maj": "ρ=maj (M3)",
        "rho_all": "ρ=all (M3)",
        "family3": "Family-3 (M1)",
    }
    nprim = passes["n_units"]
    lines = [
        "# Method ablation (M1/M2/M3)",
        "",
        "Design: `docs/消融实验设计_审稿人视角.md`.",
        f"Primary read: **all {nprim}** member--domain units.",
        "Reference: 9 \(\\overline{D}<0\) units (excl. calibrate) in `neg_fusion9`.",
        "Δrnd: each variant's own close rate (no n* match).",
        "",
        f"## Pass / readouts (primary={PRIMARY})",
        "",
        f"- g* selection: **{'PASS' if passes['g_star_selection']['pass'] else 'FAIL'}** "
        f"(med Δrnd={passes['g_star_selection']['median_delta_rnd']:+.5f}, "
        f"s={passes['g_star_selection']['n_delta_rnd_pos']}/{nprim})",
        f"- polarity M2: **{'PASS' if passes['polarity_m2']['pass'] else 'FAIL'}** "
        f"(med flip={passes['polarity_m2']['med_rnd_flip']:+.5f} vs "
        f"star={passes['polarity_m2']['med_rnd_star']:+.5f}; "
        f"s={passes['polarity_m2']['s_flip']} vs {passes['polarity_m2']['s_star']})",
        f"- rho M3: **{'PASS' if passes['rho_m3']['pass'] else 'FAIL'}** "
        f"(maj med Δ/Δrnd={passes['rho_m3']['maj_med_delta_f1']:+.5f}/"
        f"{passes['rho_m3']['maj_med_delta_rnd']:+.5f}; "
        f"all={passes['rho_m3']['all_med_delta_f1']:+.5f}/"
        f"{passes['rho_m3']['all_med_delta_rnd']:+.5f})",
        f"- family3 M1: retains_selection="
        f"{passes['family3_m1']['retains_selection']}, "
        f"close_to_star={passes['family3_m1']['close_to_star']} "
        f"(med Δrnd={passes['family3_m1']['median_delta_rnd']:+.5f}, "
        f"s={passes['family3_m1']['n_delta_rnd_pos']}/{nprim})",
        f"- core (g* + polarity): **{'PASS' if passes['all_core_pass'] else 'FAIL'}**",
        "",
        "## Primary table (all15)",
        "",
        "| Variant | beats f1 | med Δ(f1−g) | med Δrnd | #Δrnd>0 | close% |",
        "|---------|----------|-------------|----------|---------|--------|",
    ]
    for v in VARIANTS:
        s = by_v[v]["all15"]
        lines.append(
            f"| {labels[v]} | {s['n_beats_f1']}/{s['n_units']} | "
            f"{s['median_delta_f1']:+.5f} | {s['median_delta_rnd']:+.5f} | "
            f"{s['n_delta_rnd_pos']}/{s['n_units']} | "
            f"{100*s['median_close_rate']:.1f}% |"
        )

    lines += [
        "",
        "## Primary means (secondary)",
        "",
        "| Variant | mean Δ | mean Δrnd | mean close% |",
        "|---------|--------|-----------|-------------|",
    ]
    for v in VARIANTS:
        s = by_v[v]["all15"]
        lines.append(
            f"| {labels[v]} | {s['mean_delta_f1']:+.5f} | "
            f"{s['mean_delta_rnd']:+.5f} | {100*s['mean_close_rate']:.1f}% |"
        )

    lines += [
        "",
        "## Reference: neg_fusion9 medians",
        "",
        "| Variant | med Δ | med Δrnd | #Δrnd>0 | close% |",
        "|---------|-------|----------|---------|--------|",
    ]
    for v in VARIANTS:
        s = by_v[v]["neg_fusion9"]
        lines.append(
            f"| {labels[v]} | {s['median_delta_f1']:+.5f} | "
            f"{s['median_delta_rnd']:+.5f} | "
            f"{s['n_delta_rnd_pos']}/{s['n_units']} | "
            f"{100*s['median_close_rate']:.1f}% |"
        )

    lines += [
        "",
        "## Per-unit (all15)",
        "",
        "| unit | Δ g* | Δrnd g* | Δ flip | Δrnd flip | Δ maj | Δrnd maj | Δ all | close%* |",
        "|------|------|---------|--------|-----------|-------|----------|-------|---------|",
    ]
    for u in unit_order:
        def get(v, u=u):
            for r in rows:
                if r["unit"] == u and r["variant"] == v:
                    return r
            raise KeyError(u, v)

        a, b, c, d = (get(v) for v in ("g_star", "low_tertile", "rho_maj", "rho_all"))
        lines.append(
            f"| {u} | {a['delta_f1_minus_g']:+.4f} | {a['delta_rnd']:+.4f} | "
            f"{b['delta_f1_minus_g']:+.4f} | {b['delta_rnd']:+.4f} | "
            f"{c['delta_f1_minus_g']:+.4f} | {c['delta_rnd']:+.4f} | "
            f"{d['delta_f1_minus_g']:+.4f} | "
            f"{100*a['close_text_rate']:.1f}% |"
        )

    md = "\n".join(lines) + "\n"
    (AGG / "gating_ablation_method.md").write_text(md)
    status = [
        "# Method ablation status (primary=all15)",
        "",
        f"- core (g* selection + polarity M2): **{'PASS' if passes['all_core_pass'] else 'FAIL'}**",
        f"- rho M3 weak check: **{'PASS' if passes['rho_m3']['pass'] else 'FAIL'}**",
        f"- family3 retains_selection: **{passes['family3_m1']['retains_selection']}**",
        "",
        "See `outputs/aggregate/gating_ablation_method.md`.",
        "",
    ]
    (AGG / "gating_ablation_method_status.md").write_text("\n".join(status))
    print(md)
    print("wrote", AGG / "gating_ablation_method.md")


if __name__ == "__main__":
    main()
