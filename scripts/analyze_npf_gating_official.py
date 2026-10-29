#!/usr/bin/env python3
"""Six-proxy tertile hard-label gate under Time-MMD official (L,H).

Close text when starred proxies (Std, IQR, AR1-err, |b|, acf1, Vol) are in
the train-defined high tertile. Calibrate close_rule ∈ {any, maj, all}
on GPT4MTS–Environment. Also evaluate all three rules on every unit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_npf_gating as g

SEED = 2025
NPF = Path(__file__).resolve().parents[1]

DOMAIN_LH = {
    "Environment": (96, 48),
    "Energy": (36, 24),
    "Health": (36, 24),
    "Health_AFR": (36, 24),
    "Climate": (8, 8),
}
MEMBERS = ("TaTS", "GPT4MTS", "CFA-naive")
CAL_MEMBER, CAL_DATASET = "GPT4MTS", "Environment"


def run_dir(root: Path, member: str, dataset: str, sl: int, pl: int) -> Path:
    if member == "TaTS":
        return root / f"tats/{dataset}/NPF_{dataset}_iTransformer_seed{SEED}_sl{sl}_pl{pl}"
    if member == "GPT4MTS":
        return root / f"gpt4mts/{dataset}/NPF_{dataset}_GPT4MTS_seed{SEED}_sl{sl}_pl{pl}"
    if member == "CFA-naive":
        return root / f"naive_cfa/{dataset}/NPF_{dataset}_iTransformer_seed{SEED}_sl{sl}_pl{pl}"
    raise ValueError(member)


def data_csv(member: str, dataset: str) -> Path:
    if member == "CFA-naive":
        return NPF / "third_party/cfa/data" / dataset / f"{dataset}_NPF.csv"
    return NPF / "third_party/TaTS/data" / f"{dataset}.csv"


def pack(rows: list[dict]) -> dict:
    if not rows:
        return {"n_units": 0}
    return {
        "n_units": len(rows),
        "n_beats_f1": sum(1 for r in rows if r["beats_f1"]),
        "mean_mse_g": float(np.mean([r["mse_g"] for r in rows])),
        "mean_mse_f0": float(np.mean([r["mse_f0"] for r in rows])),
        "mean_mse_f1": float(np.mean([r["mse_f1"] for r in rows])),
        "mean_delta_f1": float(np.mean([r["delta_f1_minus_g"] for r in rows])),
        "mean_close_rate": float(np.mean([r["close_text_rate"] for r in rows])),
        "mean_gain_vs_random": float(
            np.mean([r["random"]["mse_g"] - r["mse_g"] for r in rows])
        ),
        "units": [r["unit"] for r in rows],
    }


def eval_matrix(root: Path, rule: str, feats: tuple[str, ...]) -> tuple[list[dict], dict]:
    results = []
    domain_params = {}
    for dataset, (sl, pl) in DOMAIN_LH.items():
        for member in MEMBERS:
            role = (
                "calibrate"
                if (member == CAL_MEMBER and dataset == CAL_DATASET)
                else "eval"
            )
            rd = run_dir(root, member, dataset, sl, pl)
            paired = rd / "paired_test.csv"
            csv = data_csv(member, dataset)
            if not paired.exists() or not csv.exists():
                print("skip missing", member, dataset)
                continue
            params = g.domain_gate_params(csv, sl, pl, close_rule=rule, feats=feats)
            key = f"{member}/{dataset}"
            domain_params[key] = {
                "close_rule": params["close_rule"],
                "feats": params["feats"],
                "maj_k": params["maj_k"],
                "tertile_cuts": params["tertile_cuts"],
                "train_close_rate": params["train_close_rate"],
                "n_train_windows": params["n_train_windows"],
                "seq_len": sl,
                "pred_len": pl,
            }
            df = g.load_paired_with_feats(paired, csv, sl, pl)
            mean_D = float(df["D"].mean())
            m = g.eval_with_params(df, params, seed=SEED)
            m.update(
                role=role,
                member=member,
                dataset=dataset,
                unit=key,
                seq_len=sl,
                pred_len=pl,
                mean_D=mean_D,
                r_neg=float((df["D"] < 0).mean()),
                n=int(len(df)),
            )
            results.append(m)
            print(
                f"[{len(feats)}|{rule}] {key} "
                f"Δ={m['delta_f1_minus_g']:+.4g} close={m['close_text_rate']:.3f} "
                f"meanD={mean_D:+.4g}"
            )
    return results, domain_params


def summarize(results: list[dict]) -> dict:
    neg = [r for r in results if r["mean_D"] < 0]
    nonneg = [r for r in results if r["mean_D"] >= 0]
    return {
        "all_15": pack(results),
        "meanD_neg": pack(neg),
        "meanD_nonneg": pack(nonneg),
    }


def main() -> None:
    root = NPF / "outputs"
    cal_sl, cal_pl = DOMAIN_LH[CAL_DATASET]
    cal_dir = run_dir(root, CAL_MEMBER, CAL_DATASET, cal_sl, cal_pl)
    cal_csv = data_csv(CAL_MEMBER, CAL_DATASET)
    if not (cal_dir / "paired_test.csv").exists():
        raise SystemExit(f"missing calibrate paired: {cal_dir / 'paired_test.csv'}")

    df_cal = g.load_paired_with_feats(cal_dir / "paired_test.csv", cal_csv, cal_sl, cal_pl)
    config, candidates = g.calibrate_close_rule(
        df_cal, cal_csv, cal_sl, cal_pl, feats=g.RISK_FEATS
    )
    config["calibrate_unit"] = f"{CAL_MEMBER}/{CAL_DATASET}"
    rule = str(config["close_rule"])
    print(
        "calibrated close_rule",
        rule,
        "maj_k",
        config["maj_k"],
        "feats",
        g.RISK_FEATS,
        "on",
        config["calibrate_unit"],
    )
    for c in candidates:
        mark = " ←" if c["close_rule"] == rule else ""
        print(
            f"  cal {c['close_rule']}: Δ={c['delta_f1_minus_g']:+.4g} "
            f"close={c['close_text_rate']:.3f} ≤f0={c['not_worse_f0']}{mark}"
        )

    # Main: frozen ρ* with 6 feats; also store all 3 rules on 6 feats + 3-feat mirror.
    by_rule_6 = {}
    for r in g.CLOSE_RULES:
        res, dparams = eval_matrix(root, r, g.RISK_FEATS)
        by_rule_6[r] = {"results": res, "summary": summarize(res), "domain_params": dparams}

    by_rule_3 = {}
    for r in g.CLOSE_RULES:
        res, dparams = eval_matrix(root, r, g.FEATS_3)
        by_rule_3[r] = {"results": res, "summary": summarize(res), "domain_params": dparams}

    main_block = by_rule_6[rule]
    results = main_block["results"]
    domain_params = main_block["domain_params"]

    # Per-run JSON for frozen rule
    for m in results:
        rd = run_dir(root, m["member"], m["dataset"], m["seq_len"], m["pred_len"])
        (rd / "gating_offline_official.json").write_text(
            json.dumps(
                {
                    "close_rule": rule,
                    "risk_feats": list(g.RISK_FEATS),
                    **{k: v for k, v in m.items() if k != "random"},
                    "random": m["random"],
                },
                indent=2,
            )
            + "\n"
        )

    agg = root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": (
            "Time-MMD official L/H; tertile hard-label on six starred proxies; "
            "GPT4MTS-Env calibrate close_rule; report all rules + 3-feat contrast"
        ),
        "risk_feats": list(g.RISK_FEATS),
        "feats_3": list(g.FEATS_3),
        "close_rule": rule,
        "maj_k": g.maj_threshold(len(g.RISK_FEATS)),
        "domain_LH": DOMAIN_LH,
        "config": config,
        "calibrate_candidates": [
            {k: c[k] for k in (
                "close_rule", "mse_g", "mse_f0", "mse_f1",
                "delta_f1_minus_g", "close_text_rate", "beats_f1", "not_worse_f0",
            )}
            for c in candidates
        ],
        "results": results,
        "summary": main_block["summary"],
        "by_rule_6": {r: by_rule_6[r]["summary"] for r in g.CLOSE_RULES},
        "by_rule_3": {r: by_rule_3[r]["summary"] for r in g.CLOSE_RULES},
        "results_by_rule_6": {
            r: [
                {
                    "unit": x["unit"],
                    "mean_D": x["mean_D"],
                    "mse_f1": x["mse_f1"],
                    "mse_g": x["mse_g"],
                    "delta_f1_minus_g": x["delta_f1_minus_g"],
                    "close_text_rate": x["close_text_rate"],
                    "vs_random": x["random"]["mse_g"] - x["mse_g"],
                    "beats_f1": x["beats_f1"],
                }
                for x in by_rule_6[r]["results"]
            ]
            for r in g.CLOSE_RULES
        },
        "domain_params": domain_params,
    }
    (agg / "gating_official.json").write_text(json.dumps(report, indent=2) + "\n")
    (agg / "gating_fullmatrix_15.json").write_text(json.dumps(report, indent=2) + "\n")
    (agg / "gating_config.json").write_text(
        json.dumps(
            {
                "close_rule": rule,
                "maj_k": report["maj_k"],
                "q_star": None,
                "risk_feats": list(g.RISK_FEATS),
                "feats_3": list(g.FEATS_3),
                "protocol": report["protocol"],
                "calibrate_unit": f"{CAL_MEMBER}/{CAL_DATASET}",
                "seq_len": cal_sl,
                "pred_len": cal_pl,
            },
            indent=2,
        )
        + "\n"
    )

    def fmt_sum(s: dict) -> str:
        if not s or s.get("n_units", 0) == 0:
            return "n/a"
        return (
            f"beats={s['n_beats_f1']}/{s['n_units']} "
            f"meanΔ={s['mean_delta_f1']:+.4g} "
            f"close={s['mean_close_rate']:.3f} "
            f"vs_rand={s['mean_gain_vs_random']:+.4g}"
        )

    lines = [
        "# NPF tertile hard-label gate — six starred proxies, official 15 units",
        "",
        f"Calibrate: {CAL_MEMBER}/{CAL_DATASET} L={cal_sl}, H={cal_pl}",
        f"Feats(6): {list(g.RISK_FEATS)}; maj_k={report['maj_k']}; frozen ρ*=**{rule}**",
        f"Feats(3) contrast: {list(g.FEATS_3)}",
        "",
        "## Calibrate candidates (6 feats)",
    ]
    for c in candidates:
        mark = " ← selected" if c["close_rule"] == rule else ""
        lines.append(
            f"- {c['close_rule']}: mse_g={c['mse_g']:.6g} "
            f"Δ={c['delta_f1_minus_g']:+.4g} close={c['close_text_rate']:.3f} "
            f"≤f0={c['not_worse_f0']}{mark}"
        )
    lines += ["", "## Summary by rule (6 feats)", ""]
    for r in g.CLOSE_RULES:
        s = by_rule_6[r]["summary"]
        tag = " ← frozen" if r == rule else ""
        lines.append(f"- **{r}** all15: {fmt_sum(s['all_15'])}{tag}")
        lines.append(f"  D̄<0: {fmt_sum(s['meanD_neg'])}")
        lines.append(f"  D̄≥0: {fmt_sum(s['meanD_nonneg'])}")
    lines += ["", "## Summary by rule (3 feats contrast)", ""]
    for r in g.CLOSE_RULES:
        s = by_rule_3[r]["summary"]
        lines.append(f"- **{r}** all15: {fmt_sum(s['all_15'])}")
        lines.append(f"  D̄<0: {fmt_sum(s['meanD_neg'])}")

    lines += [
        "",
        f"## Per-unit table (frozen ρ*={rule}, 6 feats)",
        "",
        "| unit | role | meanD | mse_f1 | mse_g | Δ(f1−g) | close% | Δrnd |",
        "|------|------|-------|--------|-------|---------|--------|------|",
    ]
    for r in results:
        rnd = r["random"]["mse_g"] - r["mse_g"]
        lines.append(
            f"| {r['unit']} | {r['role']} | {r['mean_D']:+.4g} | "
            f"{r['mse_f1']:.6g} | {r['mse_g']:.6g} | {r['delta_f1_minus_g']:+.4g} | "
            f"{100*r['close_text_rate']:.1f}% | {rnd:+.4g} |"
        )
    (agg / "gating_official.md").write_text("\n".join(lines) + "\n")
    (agg / "gating_fullmatrix_15.md").write_text("\n".join(lines) + "\n")
    print("wrote", agg / "gating_official.md")
    print("FROZEN", rule, "all15", fmt_sum(main_block["summary"]["all_15"]))


if __name__ == "__main__":
    main()
