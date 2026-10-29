"""Aggregate L1 reports across seeds for Phase B1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NPF = Path(__file__).resolve().parents[1]
OUT = NPF / "outputs" / "aggregate"


def bootstrap_ci(d: np.ndarray, B: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(d)
    means = np.empty(B)
    for b in range(B):
        means[b] = d[rng.integers(0, n, n)].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))

# (member, dataset, path_tmpl with {seed}, baseline_seed, extra_seeds)
UNITS = [
    ("TaTS", "Environment", "tats/Environment/NPF_Environment_iTransformer_seed{seed}_sl24_pl48", 2025, [2024, 2026, 2027]),
    ("TaTS", "Energy", "tats/Energy/NPF_Energy_iTransformer_seed{seed}_sl24_pl48", 2025, [2024, 2026, 2027]),
    ("GPT4MTS", "Environment", "gpt4mts/Environment/NPF_Environment_GPT4MTS_seed{seed}_sl24_pl48", 2025, [2024, 2026, 2027]),
    ("GPT4MTS", "Health", "gpt4mts/Health/NPF_Health_GPT4MTS_seed{seed}_sl24_pl48", 2025, [2024, 2026, 2027]),
    ("CFA-naive", "Health", "naive_cfa/Health/NPF_Health_iTransformer_seed{seed}_sl24_pl48", 2025, [2024, 2026, 2027]),
    ("GPT4MTS", "Climate", "gpt4mts/Climate/NPF_Climate_GPT4MTS_seed{seed}_sl24_pl48", 2025, [2024, 2026, 2027]),
]


def load_l1(rel: str, seed: int) -> dict | None:
    root = NPF / "outputs" / rel.format(seed=seed)
    p = root / "l1_report.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    d["_root"] = str(root)
    return d


def main() -> None:
    rows = []
    for member, dataset, tmpl, base, extras in UNITS:
        for seed in [base] + extras:
            d = load_l1(tmpl, seed)
            if d is None:
                rows.append(
                    {
                        "member": member,
                        "dataset": dataset,
                        "seed": seed,
                        "status": "missing",
                    }
                )
                continue
            L = d.get("L1", {})
            n = L.get("n", d.get("n"))
            mean_d = L.get("mean_D")
            ci_low = ci_high = None
            paired = Path(d["_root"]) / "paired_test.csv"
            if paired.exists():
                import pandas as pd

                arr = pd.read_csv(paired)["D"].to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_ci(arr)
            l1 = bool(d.get("L1_criteria", {}).get("pass", False))
            if not l1 and mean_d is not None and n is not None:
                l1 = bool(n >= 200 and mean_d < 0)
            strong = bool(l1 and ci_high is not None and ci_high < 0)
            rows.append(
                {
                    "member": member,
                    "dataset": dataset,
                    "seed": seed,
                    "status": "ok",
                    "n": n,
                    "mean_D": mean_d,
                    "neg_rate": L.get("neg_rate"),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "L1_pass": l1,
                    "L1_CI_strict": strong,
                    "mse_f0": d.get("aggregate_mse_f0"),
                    "mse_f1": d.get("aggregate_mse_f1"),
                    "rel": tmpl.format(seed=seed),
                }
            )

    kept = []
    for member, dataset, tmpl, base, extras in UNITS:
        base_row = next(
            (
                r
                for r in rows
                if r.get("member") == member
                and r.get("dataset") == dataset
                and r.get("seed") == base
                and r.get("status") == "ok"
            ),
            None,
        )
        if not base_row or not base_row.get("L1_pass"):
            continue
        for seed in extras:
            er = next(
                (
                    r
                    for r in rows
                    if r.get("member") == member
                    and r.get("dataset") == dataset
                    and r.get("seed") == seed
                    and r.get("status") == "ok"
                ),
                None,
            )
            if er and er.get("L1_pass"):
                kept.append(f"{member}/{dataset}@seed{seed}")

    report = {
        "rows": rows,
        "extra_seed_L1_kept": kept,
        "success_ge2": len(set(k.split("@")[0] for k in kept)) >= 2 or len(kept) >= 2,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "scale_seeds.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Scale: multi-seed L1",
        "",
        f"Extra-seed L1 kept: {kept}",
        f"Success (≥2): {report['success_ge2']}",
        "",
        "| member | dataset | seed | n | mean_D | CI | neg_rate | L1 | L1+CI<0 |",
        "|--------|---------|-----:|--:|-------:|----|---------:|:--:|:-------:|",
    ]
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r['member']} | {r['dataset']} | {r['seed']} | — | — | missing | — | — | — |")
            continue
        ci = "—"
        if r.get("ci_low") is not None and r.get("ci_high") is not None:
            ci = f"[{r['ci_low']:+.4g}, {r['ci_high']:+.4g}]"
        lines.append(
            f"| {r['member']} | {r['dataset']} | {r['seed']} | {r['n']} | "
            f"{r['mean_D']:+.4g} | {ci} | {r['neg_rate']:.3f} | "
            f"{'Y' if r['L1_pass'] else 'N'} | {'Y' if r['L1_CI_strict'] else 'N'} |"
        )
    (OUT / "scale_seeds.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", OUT / "scale_seeds.md")


if __name__ == "__main__":
    main()
