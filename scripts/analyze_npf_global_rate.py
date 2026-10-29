"""Global suppression-rate baseline vs history hard gate.

Global soft rate α*: on a calibration validation split, choose
  α* = argmin_α mean_t ||(1-α) y1 + α y0 - y||^2
and freeze it (same α for every test window / domain).

Global hard rate ρ*: on the same validation split, set
  ρ* = fraction of windows with mse_f0 < mse_f1
and apply random hard suppression at ρ* on test (no history features).

History gate uses frozen q* risk routing (window-selective; features from RISK_FEATS).
"""
from __future__ import annotations

import argparse
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
    random_gate,
)
from analyze_npf_soft_gate import align_pair, load_split_preds  # noqa: E402

NPF = Path(__file__).resolve().parents[1]


def default_q_star() -> float:
    cfg = NPF / "outputs/aggregate/gating_config.json"
    if cfg.exists():
        return float(json.loads(cfg.read_text())["q_star"])
    return 0.60


def find_dump_roots(run_dir: Path) -> tuple[Path, Path] | None:
    """Return (f0_root, f1_root) directories that contain split subdirs."""
    dumps = run_dir / "dumps"
    if not dumps.exists():
        return None
    # Prefer explicit f0_*/f1_* layout (GPT4MTS / CFA)
    for f0_name, f1_name in (
        ("f0_numeric", "f1_text"),
        ("f0_unimodal", "f1_first_concat"),
    ):
        a, b = dumps / f0_name, dumps / f1_name
        if a.exists() and b.exists():
            return a, b
    # TaTS-style nested run folders
    f0s = sorted(dumps.glob("*_f0_*"))
    f1s = sorted(dumps.glob("*_f1_*"))
    if f0s and f1s:
        return f0s[0], f1s[0]
    return None


def split_mse(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    return ((y - t) ** 2).mean(axis=1)


def soft_mix_mse(y0: np.ndarray, y1: np.ndarray, true: np.ndarray, alpha: float) -> float:
    yg = (1.0 - alpha) * y1 + alpha * y0
    return float(((yg - true) ** 2).mean())


def tune_alpha(y0: np.ndarray, y1: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    grid = np.linspace(0.0, 1.0, 21)
    scores = [soft_mix_mse(y0, y1, true, float(a)) for a in grid]
    i = int(np.argmin(scores))
    return float(grid[i]), float(scores[i])


def load_aligned_split(f0_root: Path, f1_root: Path, split: str):
    if not (f0_root / split / "pred.npy").exists():
        return None
    if not (f1_root / split / "pred.npy").exists():
        return None
    y0, t0, i0 = load_split_preds(f0_root, split)
    y1, t1, i1 = load_split_preds(f1_root, split)
    y0, t0, i0, y1, t1 = align_pair(y0, t0, i0, y1, t1, i1)
    # use f0 truth as reference (aligned)
    return {
        "y0": y0,
        "y1": y1,
        "true": t0,
        "index": i0,
        "mse0": split_mse(y0, t0),
        "mse1": split_mse(y1, t0),
    }


UNITS_LEGACY = [
    # (role, member, domain, run_rel, data_tag, sl, pl)
    ("calibrate", "TaTS", "Environment", "tats/Environment/NPF_Environment_iTransformer_seed2025_sl24_pl48", "TaTS/Environment.csv", 24, 48),
    ("held-out", "TaTS", "Energy", "tats/Energy/NPF_Energy_iTransformer_seed2025_sl24_pl48", "TaTS/Energy.csv", 24, 48),
    ("held-out", "GPT4MTS", "Environment", "gpt4mts/Environment/NPF_Environment_GPT4MTS_seed2025_sl24_pl48", "TaTS/Environment.csv", 24, 48),
    ("held-out", "GPT4MTS", "Climate", "gpt4mts/Climate/NPF_Climate_GPT4MTS_seed2025_sl24_pl48", "TaTS/Climate.csv", 24, 48),
    ("held-out", "CFA-naive", "Climate", "naive_cfa/Climate/NPF_Climate_iTransformer_seed2025_sl24_pl48", "cfa/Climate/Climate_NPF.csv", 24, 48),
]


def units_official() -> list[tuple]:
    from npf_timemmd_official import DOMAIN_LH, data_tag, run_rel

    specs = [
        ("calibrate", "TaTS", "Environment"),
        ("held-out", "TaTS", "Energy"),
        ("held-out", "GPT4MTS", "Environment"),
        ("held-out", "GPT4MTS", "Climate"),
        ("held-out", "CFA-naive", "Climate"),
        ("held-out", "GPT4MTS", "Health"),
        ("held-out", "CFA-naive", "Health"),
        ("held-out", "CFA-naive", "Energy"),
        ("held-out", "TaTS", "Climate"),
        ("held-out", "CFA-naive", "Environment"),
    ]
    out = []
    for role, member, domain in specs:
        sl, pl, _ = DOMAIN_LH[domain]
        out.append((role, member, domain, run_rel(member, domain), data_tag(member, domain), sl, pl))
    return out


def data_csv(tag: str) -> Path:
    if tag.startswith("TaTS/"):
        return NPF / "third_party/TaTS/data" / tag.split("/", 1)[1]
    if tag.startswith("cfa/"):
        return NPF / "third_party/cfa/data" / tag.split("/", 1)[1]
    raise ValueError(tag)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--pred-len", type=int, default=48)
    ap.add_argument("--q-star", type=float, default=None)
    ap.add_argument("--random-seeds", type=int, default=20)
    ap.add_argument("--protocol", choices=("legacy", "official"), default="official")
    args = ap.parse_args()
    if args.q_star is None:
        args.q_star = default_q_star()

    units = units_official() if args.protocol == "official" else UNITS_LEGACY

    # --- calibrate global rates on TaTS Environment validation ---
    cal_run = NPF / "outputs" / units[0][3]
    roots = find_dump_roots(cal_run)
    if roots is None:
        raise SystemExit(f"missing dumps for calibration unit {cal_run}")
    f0_c, f1_c = roots
    val = load_aligned_split(f0_c, f1_c, "vali")
    if val is None:
        raise SystemExit("calibration unit missing vali dumps")

    alpha_star, alpha_val_mse = tune_alpha(val["y0"], val["y1"], val["true"])
    rho_star = float((val["mse0"] < val["mse1"]).mean())
    # expected MSE under random hard at rho on val (closed form)
    rho_val_exp = float(rho_star * val["mse0"].mean() + (1.0 - rho_star) * val["mse1"].mean())

    cal_info = {
        "calibrate_unit": "TaTS/Environment",
        "split": "vali",
        "protocol": args.protocol,
        "alpha_star": alpha_star,
        "alpha_vali_mse": alpha_val_mse,
        "rho_star_oracle_frac": rho_star,
        "rho_vali_expected_mse": rho_val_exp,
        "vali_mse_f0": float(val["mse0"].mean()),
        "vali_mse_f1": float(val["mse1"].mean()),
        "n_vali": int(len(val["mse0"])),
    }

    rows = []
    for role, member, domain, rel, dtag, sl, pl in units:
        run = NPF / "outputs" / rel
        paired = run / "paired_test.csv"
        roots = find_dump_roots(run)
        if not paired.exists() or roots is None:
            print("skip", rel)
            continue
        f0_r, f1_r = roots
        test = load_aligned_split(f0_r, f1_r, "test")
        if test is None:
            print("skip no test preds", rel)
            continue

        # history gate
        params = domain_gate_params(data_csv(dtag), sl, pl, args.q_star)
        df = load_paired_with_feats(paired, data_csv(dtag), sl, pl)
        hist = eval_with_params(df, params, seed=2025)

        # global soft α*
        mse_global_soft = soft_mix_mse(test["y0"], test["y1"], test["true"], alpha_star)

        # global hard ρ*: average random over seeds
        mse0, mse1 = test["mse0"], test["mse1"]
        rand_mses = []
        for s in range(args.random_seeds):
            m = random_gate(mse0, mse1, rho_star, seed=10_000 + s)
            rand_mses.append(m["mse_g"])
        mse_global_hard = float(np.mean(rand_mses))
        mse_global_hard_std = float(np.std(rand_mses))

        # rate-matched random to history close rate (existing baseline)
        matched = hist["random"]["mse_g"]

        row = {
            "role": role,
            "member": member,
            "domain": domain,
            "seq_len": sl,
            "pred_len": pl,
            "n_test": int(len(mse0)),
            "mse_f0": float(mse0.mean()),
            "mse_f1": float(mse1.mean()),
            "mse_hist_g": hist["mse_g"],
            "hist_close_rate": hist["close_text_rate"],
            "mse_global_soft": mse_global_soft,
            "alpha_star": alpha_star,
            "mse_global_hard_rho": mse_global_hard,
            "mse_global_hard_rho_std": mse_global_hard_std,
            "rho_star": rho_star,
            "mse_random_match_hist": matched,
            "delta_f1_minus_hist": float(mse1.mean() - hist["mse_g"]),
            "delta_global_soft_minus_hist": float(mse_global_soft - hist["mse_g"]),
            "delta_global_hard_minus_hist": float(mse_global_hard - hist["mse_g"]),
            "delta_random_match_minus_hist": float(matched - hist["mse_g"]),
            "hist_beats_f1": bool(hist["mse_g"] < mse1.mean()),
            "hist_beats_global_soft": bool(hist["mse_g"] < mse_global_soft),
            "hist_beats_global_hard": bool(hist["mse_g"] < mse_global_hard),
        }
        rows.append(row)
        print(
            f"{member}/{domain}: hist={row['mse_hist_g']:.4f} "
            f"softα*={mse_global_soft:.4f} hardρ*={mse_global_hard:.4f} "
            f"f1={row['mse_f1']:.4f}"
        )

    report = {"calibration": cal_info, "q_star": args.q_star, "units": rows}
    agg = NPF / "outputs/aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "global_rate_vs_hist.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Global suppression rate vs history hard gate",
        "",
        f"Calibration: TaTS/Environment **vali** → α*={alpha_star:.2f}, "
        f"ρ*={rho_star:.3f} (oracle fraction mse0<mse1).",
        f"History gate: frozen q*={args.q_star}.",
        "",
        "| member | domain | role | mse_f1 | mse_hist | mse_α* | mse_ρ* | mse_rand@hist | hist<α* | hist<ρ* |",
        "|--------|--------|------|-------:|---------:|-------:|-------:|--------------:|:-------:|:-------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['member']} | {r['domain']} | {r['role']} | {r['mse_f1']:.4f} | "
            f"{r['mse_hist_g']:.4f} | {r['mse_global_soft']:.4f} | {r['mse_global_hard_rho']:.4f} | "
            f"{r['mse_random_match_hist']:.4f} | "
            f"{'Y' if r['hist_beats_global_soft'] else 'N'} | "
            f"{'Y' if r['hist_beats_global_hard'] else 'N'} |"
        )
    held = [r for r in rows if r["role"] == "held-out"]
    if held:
        lines += [
            "",
            f"Held-out: hist beats global soft α* in "
            f"**{sum(r['hist_beats_global_soft'] for r in held)}/{len(held)}**; "
            f"beats global hard ρ* in "
            f"**{sum(r['hist_beats_global_hard'] for r in held)}/{len(held)}**.",
        ]
    (agg / "global_rate_vs_hist.md").write_text("\n".join(lines) + "\n")
    print("wrote", agg / "global_rate_vs_hist.md")


if __name__ == "__main__":
    main()
