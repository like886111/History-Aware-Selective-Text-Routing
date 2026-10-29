"""Aggregate L1 across pred_len for Phase B2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NPF = Path(__file__).resolve().parents[1]
OUT = NPF / "outputs" / "aggregate"

UNITS = [
    ("TaTS", "Environment", "tats/Environment/NPF_Environment_iTransformer_seed2025_sl24_pl{pl}"),
    ("TaTS", "Energy", "tats/Energy/NPF_Energy_iTransformer_seed2025_sl24_pl{pl}"),
    ("GPT4MTS", "Climate", "gpt4mts/Climate/NPF_Climate_GPT4MTS_seed2025_sl24_pl{pl}"),
]


def bootstrap_ci(d: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(B)
    for b in range(B):
        means[b] = d[rng.integers(0, n, n)].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def load_unit(rel: str, pl: int) -> dict | None:
    root = NPF / "outputs" / rel.format(pl=pl)
    p = root / "l1_report.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    paired = root / "paired_test.csv"
    ci_low = ci_high = None
    if paired.exists():
        import pandas as pd

        arr = pd.read_csv(paired)["D"].to_numpy(dtype=float)
        ci_low, ci_high = bootstrap_ci(arr)
    L = d["L1"]
    return {
        "member": d["member"],
        "dataset": d["dataset"],
        "pred_len": pl,
        "n": L["n"],
        "mean_D": L["mean_D"],
        "neg_rate": L["neg_rate"],
        "ci_low": ci_low,
        "ci_high": ci_high,
        "L1_pass": bool(d.get("L1_criteria", {}).get("pass", False)),
        "mse_f0": d.get("aggregate_mse_f0"),
        "mse_f1": d.get("aggregate_mse_f1"),
        "status": "ok",
    }


def main() -> None:
    rows = []
    for member, dataset, tmpl in UNITS:
        for pl in (24, 48, 96):
            r = load_unit(tmpl, pl)
            if r is None:
                rows.append({"member": member, "dataset": dataset, "pred_len": pl, "status": "missing"})
            else:
                rows.append(r)

    # success: >=1 main-evidence unit keeps L1 or same-sign mean_D on pl!=48
    main_ok = False
    for r in rows:
        if r.get("status") != "ok":
            continue
        if r["dataset"] == "Climate" and r["member"] == "GPT4MTS" and r["pred_len"] != 48:
            if r["L1_pass"] or (r["mean_D"] is not None and r["mean_D"] < 0):
                main_ok = True
        if r["dataset"] == "Energy" and r["member"] == "TaTS" and r["pred_len"] != 48:
            if r["L1_pass"] or (r["mean_D"] is not None and r["mean_D"] < 0):
                main_ok = True

    report = {"rows": rows, "success_horizon_same_sign": main_ok}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scale_horizons.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Scale: multi-horizon L1",
        "",
        f"Success (same-sign on pl≠48 for Energy or Climate): {main_ok}",
        "",
        "| member | dataset | pl | n | mean_D | CI | neg_rate | L1 |",
        "|--------|---------|---:|--:|-------:|----|---------:|:--:|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['member']} | {r['dataset']} | {r['pred_len']} | — | — | missing | — | — |")
            continue
        ci = "—"
        if r.get("ci_low") is not None:
            ci = f"[{r['ci_low']:+.4g}, {r['ci_high']:+.4g}]"
        lines.append(
            f"| {r['member']} | {r['dataset']} | {r['pred_len']} | {r['n']} | "
            f"{r['mean_D']:+.4g} | {ci} | {r['neg_rate']:.3f} | {'Y' if r['L1_pass'] else 'N'} |"
        )
    (OUT / "scale_horizons.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
