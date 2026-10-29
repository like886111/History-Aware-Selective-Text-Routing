#!/usr/bin/env python3
"""Official (L,H) gating under revised evaluation protocol (KBS review response).

Changes vs v1:
1. Full member×domain matrix (positive, neutral, and negative mean_D).
2. Primary metrics: mitigation Δ(f1−g), regret Δ(g−f0), not_worse_f0.
3. Safe deployment: enable history gate on test iff vali mean_D < 0;
   otherwise fall back to always-on f1 (no suppression).
4. Text-side control router: close text when lookback text emptiness is high
   (matched close-rate via train quantile), vs history router.
5. Stratified summaries by sign(mean_D).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_npf_gating as g
from analyze_npf_global_rate import find_dump_roots, load_aligned_split
from npf_timemmd_official import DOMAIN_LH, MEMBERS

SEED = 2025
NPF = Path(__file__).resolve().parents[1]


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


def vali_mean_D(run: Path) -> float | None:
    roots = find_dump_roots(run)
    if roots is None:
        return None
    f0, f1 = roots
    val = load_aligned_split(f0, f1, "vali")
    if val is None:
        return None
    # D = mse0 - mse1; negative => text hurts on average
    return float((val["mse0"] - val["mse1"]).mean())


def text_side_gate(df, q_star: float, seed: int = SEED) -> dict:
    """Close text when lookback text is emptier (higher emptiness risk).

    Score = -text_nonempty_rate in the lookback (or 0 if column missing).
    Threshold = train-style quantile on the test distribution proxy via
    empirical q_star of the risk on the same windows (fair matched capacity).
    """
    if "text_nonempty_rate" not in df.columns:
        # fallback: no text signal → never close
        close = np.zeros(len(df), dtype=bool)
    else:
        emptiness = 1.0 - df["text_nonempty_rate"].to_numpy(dtype=float)
        tau = float(np.quantile(emptiness, q_star))
        close = emptiness >= tau
    mse0 = df["mse_f0"].to_numpy(dtype=float)
    mse1 = df["mse_f1"].to_numpy(dtype=float)
    m = g.metrics(g.apply_gate(mse0, mse1, close), mse0, mse1, close)
    m["kind"] = "text_side_emptiness"
    m["random"] = g.random_gate(mse0, mse1, m["close_text_rate"], seed=seed)
    return m


def enrich(m: dict, df, mean_D: float, vali_D: float | None) -> dict:
    regret_f0 = float(m["mse_g"] - m["mse_f0"])  # >0 means worse than f0
    mitigation = float(m["mse_f1"] - m["mse_g"])  # >0 means better than f1
    stratum = "negative" if mean_D < 0 else ("positive" if mean_D > 0 else "zero")
    # Safe policy: if vali suggests text not harmful, keep f1
    if vali_D is None:
        enable = mean_D < 0  # fallback (diagnostic only; prefer vali)
        enable_src = "fallback_test_meanD"
    else:
        enable = vali_D < 0
        enable_src = "vali_meanD"
    if enable:
        mse_safe = m["mse_g"]
        close_safe = m["close_text_rate"]
    else:
        mse_safe = m["mse_f1"]
        close_safe = 0.0
    out = {
        **m,
        "mean_D": mean_D,
        "vali_mean_D": vali_D,
        "stratum": stratum,
        "regret_f0": regret_f0,
        "mitigation_f1": mitigation,
        "gate_enabled_by_vali": bool(enable),
        "enable_source": enable_src,
        "mse_safe": float(mse_safe),
        "close_rate_safe": float(close_safe),
        "safe_beats_f1": bool(mse_safe < m["mse_f1"]),
        "safe_not_worse_f0": bool(mse_safe <= m["mse_f0"]),
        "r_neg": float((df["D"] < 0).mean()),
    }
    return out


def summarize(rows: list[dict], label: str) -> dict:
    if not rows:
        return {"label": label, "n_units": 0}
    return {
        "label": label,
        "n_units": len(rows),
        "n_beats_f1": sum(1 for r in rows if r["beats_f1"]),
        "n_not_worse_f0": sum(1 for r in rows if r["not_worse_f0"]),
        "n_safe_beats_f1": sum(1 for r in rows if r["safe_beats_f1"]),
        "n_safe_not_worse_f0": sum(1 for r in rows if r["safe_not_worse_f0"]),
        "mean_mitigation_f1": float(np.mean([r["mitigation_f1"] for r in rows])),
        "mean_regret_f0": float(np.mean([r["regret_f0"] for r in rows])),
        "mean_mse_g": float(np.mean([r["mse_g"] for r in rows])),
        "mean_mse_safe": float(np.mean([r["mse_safe"] for r in rows])),
        "mean_mse_f0": float(np.mean([r["mse_f0"] for r in rows])),
        "mean_mse_f1": float(np.mean([r["mse_f1"] for r in rows])),
        "units": [r["unit"] for r in rows],
    }


def main() -> None:
    root = NPF / "outputs"
    cal_sl, cal_pl, _ = DOMAIN_LH["Environment"]
    cal_dir = run_dir(root, "TaTS", "Environment", cal_sl, cal_pl)
    cal_csv = data_csv("TaTS", "Environment")
    df_cal = g.load_paired_with_feats(cal_dir / "paired_test.csv", cal_csv, cal_sl, cal_pl)
    config, candidates = g.calibrate_q(df_cal, cal_csv, cal_sl, cal_pl)
    q_star = float(config["q_star"])

    results = []
    text_results = []
    for dataset, (sl, pl, _) in DOMAIN_LH.items():
        for member in MEMBERS:
            rd = run_dir(root, member, dataset, sl, pl)
            paired = rd / "paired_test.csv"
            csv = data_csv(member, dataset)
            if not paired.exists() or not csv.exists():
                print("skip", member, dataset)
                continue
            params = g.domain_gate_params(csv, sl, pl, q_star)
            df = g.load_paired_with_feats(paired, csv, sl, pl)
            mean_D = float(df["D"].mean())
            vD = vali_mean_D(rd)
            hist = g.eval_with_params(df, params, seed=SEED)
            row = enrich(hist, df, mean_D, vD)
            role = "calibrate" if (member == "TaTS" and dataset == "Environment") else "eval"
            row.update(
                role=role,
                member=member,
                dataset=dataset,
                unit=f"{member}/{dataset}",
                seq_len=sl,
                pred_len=pl,
                n=int(len(df)),
            )
            results.append(row)
            (rd / "gating_v2_official.json").write_text(
                json.dumps({"q_star": q_star, **{k: v for k, v in row.items() if k != "random"}}, indent=2)
                + "\n"
            )

            tgate = text_side_gate(df, q_star, seed=SEED)
            trow = enrich(tgate, df, mean_D, vD)
            trow.update(unit=row["unit"], member=member, dataset=dataset, role=role)
            text_results.append(trow)
            print(
                f"{row['unit']}: stratum={row['stratum']} enable={row['gate_enabled_by_vali']} "
                f"Δf1={row['mitigation_f1']:+.4g} regret_f0={row['regret_f0']:+.4g} "
                f"safe_mse={row['mse_safe']:.6g}"
            )

    held = [r for r in results if r["role"] != "calibrate"]
    neg = [r for r in held if r["stratum"] == "negative"]
    pos = [r for r in held if r["stratum"] == "positive"]
    summaries = {
        "all_held_out": summarize(held, "all_held_out"),
        "held_negative_meanD": summarize(neg, "held_negative_meanD"),
        "held_positive_meanD": summarize(pos, "held_positive_meanD"),
    }

    # History vs text-side on held-out negatives
    hist_vs_text = []
    for h, t in zip(results, text_results):
        if h["role"] == "calibrate":
            continue
        hist_vs_text.append(
            {
                "unit": h["unit"],
                "stratum": h["stratum"],
                "mse_hist": h["mse_g"],
                "mse_text": t["mse_g"],
                "hist_beats_text": bool(h["mse_g"] < t["mse_g"]),
                "hist_mitigation": h["mitigation_f1"],
                "text_mitigation": t["mitigation_f1"],
            }
        )

    agg = root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol": "Time-MMD official L/H + v2 evaluation (full matrix, safe deploy, text-side)",
        "q_star": q_star,
        "config": {k: v for k, v in config.items() if k != "domain_params"},
        "calibrate_candidates": candidates,
        "results": [{k: v for k, v in r.items() if k != "random"} for r in results],
        "text_side_results": [{k: v for k, v in r.items() if k != "random"} for r in text_results],
        "summaries": summaries,
        "hist_vs_text_side": hist_vs_text,
    }
    (agg / "gating_v2_official.json").write_text(json.dumps(report, indent=2) + "\n")
    (agg / "gating_config.json").write_text(
        json.dumps(
            {
                "q_star": q_star,
                "protocol": "v2 official",
                "calibrate_unit": "TaTS/Environment",
                "seq_len": cal_sl,
                "pred_len": cal_pl,
                "safe_policy": "enable gate on test iff vali mean_D < 0; else always f1",
            },
            indent=2,
        )
        + "\n"
    )

    lines = [
        "# Gating v2 — full matrix + safe deploy + text-side",
        "",
        f"q*={q_star:.2f} (TaTS/Environment). Safe: enable iff vali mean_D<0.",
        "",
        "## Summaries",
    ]
    for s in summaries.values():
        lines.append(
            f"- **{s['label']}**: n={s['n_units']}, "
            f"g<f1={s.get('n_beats_f1',0)}/{s['n_units']}, "
            f"g≤f0={s.get('n_not_worse_f0',0)}/{s['n_units']}, "
            f"safe<f1={s.get('n_safe_beats_f1',0)}/{s['n_units']}, "
            f"mean mitig={s.get('mean_mitigation_f1')}, mean regret_f0={s.get('mean_regret_f0')}"
        )
    lines += [
        "",
        "| unit | stratum | valiD | enable | mse_f0 | mse_f1 | mse_g | mse_safe | mitig | regret_f0 | close% |",
        "|------|---------|------:|:------:|-------:|-------:|------:|---------:|------:|----------:|-------:|",
    ]
    for r in results:
        vd = "NA" if r["vali_mean_D"] is None else f"{r['vali_mean_D']:+.4g}"
        lines.append(
            f"| {r['unit']} | {r['stratum']} | {vd} | "
            f"{'Y' if r['gate_enabled_by_vali'] else 'N'} | "
            f"{r['mse_f0']:.6g} | {r['mse_f1']:.6g} | {r['mse_g']:.6g} | {r['mse_safe']:.6g} | "
            f"{r['mitigation_f1']:+.4g} | {r['regret_f0']:+.4g} | {100*r['close_text_rate']:.1f}% |"
        )
    lines += [
        "",
        "## History vs text-side (held-out)",
        "| unit | stratum | mse_hist | mse_text | hist<text |",
        "|------|---------|---------:|---------:|:---------:|",
    ]
    for r in hist_vs_text:
        lines.append(
            f"| {r['unit']} | {r['stratum']} | {r['mse_hist']:.6g} | {r['mse_text']:.6g} | "
            f"{'Y' if r['hist_beats_text'] else 'N'} |"
        )
    (agg / "gating_v2_official.md").write_text("\n".join(lines) + "\n")
    print("wrote", agg / "gating_v2_official.md")


if __name__ == "__main__":
    main()
