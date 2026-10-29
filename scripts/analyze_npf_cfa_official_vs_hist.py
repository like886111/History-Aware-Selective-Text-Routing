"""Compare CFA official fusion modes vs history hard gate on CFA-naive units.

Official modes (gating/film/cfa) are end-to-end trained selective fusion
on the same backbone/data/seed as CFA-naive f0/f1 — not post-hoc routers
on the naive pair. History gate g routes between existing f0/f1 dumps
using the frozen tertile hard-label rule (six starred proxies, ρ=any).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_npf_gating import (  # noqa: E402
    RISK_FEATS,
    domain_gate_params,
    eval_with_params,
    load_paired_with_feats,
)

NPF = Path(__file__).resolve().parents[1]
MODES = ("gating", "film", "cfa")
DEFAULT_DATASETS = ("Health", "Environment", "Energy", "Climate", "SocialGood")


def default_close_rule() -> str:
    cfg = NPF / "outputs/aggregate/gating_config.json"
    if cfg.exists():
        rule = json.loads(cfg.read_text()).get("close_rule")
        if rule:
            return str(rule)
    return "any"


def dump_mse(dump_test: Path) -> float:
    pred = np.load(dump_test / "pred.npy")
    true = np.load(dump_test / "true.npy")
    if pred.ndim == 3:
        pred = pred[..., 0]
        true = true[..., 0]
    return float(((pred - true) ** 2).mean())


def eval_one(
    dataset: str,
    seed: int,
    seq_len: int,
    pred_len: int,
    close_rule: str,
) -> dict | None:
    tag = f"NPF_{dataset}_iTransformer_seed{seed}_sl{seq_len}_pl{pred_len}"
    naive = NPF / "outputs/naive_cfa" / dataset / tag
    official = NPF / "outputs/cfa_official" / dataset / tag
    data_csv = NPF / "third_party/cfa/data" / dataset / f"{dataset}_NPF.csv"
    paired = naive / "paired_test.csv"

    if not paired.exists() or not official.exists() or not data_csv.exists():
        return None
    for mode in MODES:
        if not (official / "dumps" / f"official_{mode}" / "test" / "pred.npy").exists():
            return None

    params = domain_gate_params(
        data_csv, seq_len, pred_len, close_rule=close_rule, feats=RISK_FEATS
    )
    df = load_paired_with_feats(paired, data_csv, seq_len, pred_len)
    hist = eval_with_params(df, params, seed=2025)

    row = {
        "dataset": dataset,
        "seed": seed,
        "n_test": int(len(df)),
        "close_rule": close_rule,
        "mse_f0": hist["mse_f0"],
        "mse_f1": hist["mse_f1"],
        "mse_hist_g": hist["mse_g"],
        "delta_f1_minus_g": hist["delta_f1_minus_g"],
        "hist_close_rate": hist["close_text_rate"],
        "beats_f1": hist["beats_f1"],
        "delta_rnd": float(hist["random"]["mse_g"] - hist["mse_g"]),
    }
    for mode in MODES:
        m = dump_mse(official / "dumps" / f"official_{mode}" / "test")
        row[f"mse_{mode}"] = m
        row[f"hist_beats_{mode}"] = bool(np.isfinite(m) and hist["mse_g"] < m)
        row[f"delta_{mode}_minus_hist"] = float(m - hist["mse_g"]) if np.isfinite(m) else None
    vals = [row[f"mse_{m}"] for m in MODES]
    if not all(np.isfinite(v) for v in vals + [row["mse_f0"], row["mse_f1"], row["mse_hist_g"]]):
        return None
    row["mse_best_official"] = float(min(vals))
    row["hist_beats_best_official"] = bool(hist["mse_g"] < row["mse_best_official"])
    return row


def write_reports(rows: list[dict], close_rule: str) -> None:
    agg = NPF / "outputs/aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    note = (
        "Official CFA modes are end-to-end trained selective fusion; "
        "history g is a post-hoc hard router on naive unimodal vs first-concat "
        f"(six starred tertile labels, ρ={close_rule}). "
        "Same backbone (iTransformer), seed, (L,H) per domain."
    )
    report = {
        "protocol_note": note,
        "close_rule": close_rule,
        "q_star": None,
        "feats": list(RISK_FEATS),
        "units": rows,
    }
    # Keep previous file as deprecated snapshot if present
    old = agg / "cfa_official_vs_hist_multi.json"
    if old.exists():
        prev = json.loads(old.read_text())
        if prev.get("q_star") is not None or prev.get("close_rule") is None:
            (agg / "cfa_official_vs_hist_multi_DEPRECATED_qstar.json").write_text(
                json.dumps(prev, indent=2) + "\n"
            )

    (agg / "cfa_official_vs_hist_multi.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# CFA official modes vs history gate (multi-domain)",
        "",
        note,
        "",
        "| Domain | MSE(f0) | MSE(f1) | MSE(g) | Δ(f1−g) | close% | gating | film | cfa | g<best official |",
        "|--------|--------:|--------:|-------:|--------:|-------:|-------:|-----:|----:|:---------------:|",
    ]
    for r in rows:
        best = r["mse_best_official"]
        lines.append(
            f"| {r['dataset']} | {r['mse_f0']:.4f} | {r['mse_f1']:.4f} | "
            f"{r['mse_hist_g']:.4f} | {r['delta_f1_minus_g']:+.4f} | "
            f"{100*r['hist_close_rate']:.1f}% | "
            f"{r['mse_gating']:.4f} | {r['mse_film']:.4f} | "
            f"{r['mse_cfa']:.4f} | {'Y' if r['mse_hist_g'] < best else 'N'} |"
        )
    if rows:
        n = len(rows)
        mean = {
            k: sum(r[k] for r in rows) / n
            for k in (
                "mse_f0",
                "mse_f1",
                "mse_hist_g",
                "delta_f1_minus_g",
                "hist_close_rate",
                "mse_gating",
                "mse_film",
                "mse_cfa",
                "mse_best_official",
            )
        }
        lines.append(
            f"| mean | {mean['mse_f0']:.4f} | {mean['mse_f1']:.4f} | {mean['mse_hist_g']:.4f} | "
            f"{mean['delta_f1_minus_g']:+.4f} | {100*mean['hist_close_rate']:.1f}% | "
            f"{mean['mse_gating']:.4f} | {mean['mse_film']:.4f} | {mean['mse_cfa']:.4f} | "
            f"{'Y' if mean['mse_hist_g'] < mean['mse_best_official'] else 'N'} |"
        )
        beats = sum(1 for r in rows if r["hist_beats_best_official"])
        beats_f1 = sum(1 for r in rows if r["beats_f1"])
        lines += [
            "",
            f"History \(g\) beats \(f_1\) on **{beats_f1}/{n}** domains; "
            f"beats best official mode on **{beats}/{n}** domains.",
        ]
    (agg / "cfa_official_vs_hist_multi.md").write_text("\n".join(lines) + "\n")

    for r in rows:
        ds = r["dataset"].lower()
        single = {"protocol_note": note, "close_rule": close_rule, "unit": r}
        (agg / f"cfa_official_vs_hist_{ds}.json").write_text(json.dumps(single, indent=2) + "\n")

    print((agg / "cfa_official_vs_hist_multi.md").read_text())
    print("wrote", agg / "cfa_official_vs_hist_multi.md")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument("--dataset", default=None, help="single-domain shorthand")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--pred-len", type=int, default=48)
    ap.add_argument("--close-rule", default=None, help="default: gating_config.json or any")
    ap.add_argument(
        "--protocol",
        choices=("legacy", "official"),
        default="official",
        help="official = per-domain Time-MMD (L,H)",
    )
    args = ap.parse_args()
    close_rule = args.close_rule or default_close_rule()

    if args.protocol == "official":
        from npf_timemmd_official import CFA_OFFICIAL_DATASETS, DOMAIN_LH

        datasets = [args.dataset] if args.dataset else list(CFA_OFFICIAL_DATASETS)
        rows = []
        for ds in datasets:
            sl, pl, _ = DOMAIN_LH[ds]
            row = eval_one(ds, args.seed, sl, pl, close_rule)
            if row is None:
                print(f"skip incomplete: {ds} sl={sl} pl={pl}")
                continue
            row["seq_len"] = sl
            row["pred_len"] = pl
            rows.append(row)
            print(
                f"{ds} L={sl} H={pl}: g={row['mse_hist_g']:.4f} "
                f"Δ={row['delta_f1_minus_g']:+.4f} close={row['hist_close_rate']:.3f} "
                f"gating={row['mse_gating']:.4f} film={row['mse_film']:.4f} cfa={row['mse_cfa']:.4f}"
            )
    else:
        datasets = [args.dataset] if args.dataset else args.datasets
        rows = []
        for ds in datasets:
            row = eval_one(ds, args.seed, args.seq_len, args.pred_len, close_rule)
            if row is None:
                print(f"skip incomplete: {ds}")
                continue
            rows.append(row)
            print(
                f"{ds}: g={row['mse_hist_g']:.4f} "
                f"gating={row['mse_gating']:.4f} film={row['mse_film']:.4f} cfa={row['mse_cfa']:.4f}"
            )
    if not rows:
        raise SystemExit("no complete domains")
    write_reports(rows, close_rule)


if __name__ == "__main__":
    main()
