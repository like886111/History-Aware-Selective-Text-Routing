#!/usr/bin/env python3
"""Strict explore→confirm starring for §4.3.

Explore: GPT4MTS–Environment only (official L=96, H=48, seed 2025).
Promote deployable history proxies with Δ_X = mean(D_rel|low)−mean(D_rel|high) > 0
and |Δ_X| ≥ MIN_PROMOTE_ABS_DELTA (same floor as analyze_npf_conditions).

Confirm: all other official member×domain units; pass = same sign of Δ_X.
Does NOT re-select proxies from the confirm matrix.

Writes:
  outputs/aggregate/star_explore_confirm.{json,md}
  updates gating_config.json proxy_list_policy + risk_feats to explore freeze
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_npf_conditions import (  # noqa: E402
    DEPLOYABLE_FEATS,
    MIN_PROMOTE_ABS_DELTA,
    confirm_one,
    run_explore,
)
from npf_timemmd_official import DOMAIN_LH, MEMBERS, run_rel  # noqa: E402

NPF = Path(__file__).resolve().parents[1]
EXPLORE_MEMBER = "GPT4MTS"
EXPLORE_DATASET = "Environment"
SEED = 2025

# History proxies eligible for routing starring (exclude text lexicons).
HISTORY_DEPLOYABLE = (
    "hist_std",
    "hist_iqr",
    "pred_ease_ar1",
    "hist_abs_slope",
    "acf1",
    "vol_change",
    "shock_score",
)


def data_csv(member: str, dataset: str) -> Path:
    if member == "CFA-naive":
        return NPF / "third_party/cfa/data" / dataset / f"{dataset}_NPF.csv"
    return NPF / "third_party/TaTS/data" / f"{dataset}.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--include-shock",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, allow shock_score in explore promote set when Δ_X>0",
    )
    ap.add_argument("--update-gating-config", action=argparse.BooleanOptionalAction, default=True)
    args = ap.parse_args()

    sl, pl, _ = DOMAIN_LH[EXPLORE_DATASET]
    explore_dir = NPF / "outputs" / run_rel(EXPLORE_MEMBER, EXPLORE_DATASET, SEED)
    paired = explore_dir / "paired_test.csv"
    csv = data_csv(EXPLORE_MEMBER, EXPLORE_DATASET)
    if not paired.exists() or not csv.exists():
        raise SystemExit(f"missing explore paired/data: {paired} {csv}")

    explore_args = argparse.Namespace(
        paired_csv=str(paired),
        data_csv=str(csv),
        seq_len=sl,
        pred_len=pl,
        output_dir=str(explore_dir),
        member=EXPLORE_MEMBER,
        dataset=EXPLORE_DATASET,
        backbone="GPT2",
        seed=SEED,
    )
    explore = run_explore(explore_args)
    explore_json = explore_dir / "condition_explore.json"

    # Build promote set strictly from explore C_hat / strata (history deployable).
    promoted = []
    for c in explore.get("C_hat", []):
        feat = c.get("feature")
        if feat not in HISTORY_DEPLOYABLE:
            continue
        if feat == "shock_score" and not args.include_shock:
            continue
        delta = c.get("explore_delta_mean_D_rel_low_minus_high")
        if delta is None or delta <= 0 or abs(delta) < MIN_PROMOTE_ABS_DELTA:
            continue
        if c.get("observed_direction") != "high_more_harm":
            continue
        promoted.append(
            {
                "feature": feat,
                "explore_delta_X": float(delta),
                "direction": "high_more_harm",
            }
        )
    promoted.sort(key=lambda x: -x["explore_delta_X"])
    star_feats = [p["feature"] for p in promoted]

    # Confirm on all units except explore
    confirm_units = []
    for dataset, (usl, upl, _) in DOMAIN_LH.items():
        for member in MEMBERS:
            if member == EXPLORE_MEMBER and dataset == EXPLORE_DATASET:
                continue
            rel = run_rel(member, dataset, SEED)
            out = NPF / "outputs" / rel
            paired_u = out / "paired_test.csv"
            csv_u = data_csv(member, dataset)
            if not paired_u.exists() or not csv_u.exists():
                print("skip", member, dataset)
                continue
            report = confirm_one(
                explore, paired_u, csv_u, usl, upl, member, dataset, min_ratio=0.5
            )
            # Restrict to starred history feats
            by_feat = {r["feature"]: r for r in report.get("results", [])}
            feat_rows = []
            for feat in star_feats:
                r = by_feat.get(feat)
                if r is None:
                    feat_rows.append(
                        {"feature": feat, "same_sign": None, "confirm_delta": None, "confirm_status": "missing"}
                    )
                else:
                    feat_rows.append(
                        {
                            "feature": feat,
                            "same_sign": bool(r.get("same_sign")),
                            "confirm_delta": r.get("confirm_delta"),
                            "confirm_status": r.get("confirm_status"),
                        }
                    )
            n_same = sum(1 for r in feat_rows if r.get("same_sign") is True)
            confirm_units.append(
                {
                    "unit": f"{member}/{dataset}",
                    "member": member,
                    "dataset": dataset,
                    "seq_len": usl,
                    "pred_len": upl,
                    "mean_D": report.get("mean_D"),
                    "n_same_sign": n_same,
                    "n_star": len(star_feats),
                    "features": feat_rows,
                }
            )
            print(
                f"confirm {member}/{dataset}: same_sign={n_same}/{len(star_feats)}"
            )

    # Per-feature confirm rate among non-explore units
    feat_stats = {}
    for feat in star_feats:
        signs = []
        for u in confirm_units:
            for r in u["features"]:
                if r["feature"] == feat and r.get("same_sign") is not None:
                    signs.append(bool(r["same_sign"]))
        feat_stats[feat] = {
            "n_confirm_units": len(signs),
            "n_same_sign": int(sum(signs)),
            "same_sign_rate": float(sum(signs) / len(signs)) if signs else None,
        }

    agg = NPF / "outputs/aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": "strict_explore_confirm_starring",
        "explore_unit": f"{EXPLORE_MEMBER}/{EXPLORE_DATASET}",
        "seq_len": sl,
        "pred_len": pl,
        "seed": SEED,
        "min_promote_abs_delta": MIN_PROMOTE_ABS_DELTA,
        "include_shock": bool(args.include_shock),
        "explore_json": str(explore_json),
        "promoted": promoted,
        "star_feats": star_feats,
        "confirm_units": confirm_units,
        "feat_confirm_stats": feat_stats,
        "note": (
            "Stars frozen from explore only. Confirm reports same-sign rates; "
            "does not re-select the proxy list."
        ),
    }
    (agg / "star_explore_confirm.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "# Strict explore→confirm starring (§4.3)",
        "",
        f"Explore: **{EXPLORE_MEMBER}/{EXPLORE_DATASET}** (L={sl}, H={pl}, seed={SEED}).",
        f"Promote: history deployable, Δ_X>0, |Δ_X|≥{MIN_PROMOTE_ABS_DELTA}"
        + (", include shock" if args.include_shock else ", exclude shock")
        + ".",
        "",
        "## Promoted on explore (frozen \(\\mathcal{S}\))",
        "",
        "| feature | explore Δ_X | direction |",
        "|---------|------------:|-----------|",
    ]
    for p in promoted:
        lines.append(
            f"| {p['feature']} | {p['explore_delta_X']:+.4f} | {p['direction']} |"
        )
    lines += [
        "",
        "## Confirm same-sign rates (14 held-out units)",
        "",
        "| feature | same_sign | rate |",
        "|---------|----------:|-----:|",
    ]
    for feat in star_feats:
        st = feat_stats[feat]
        lines.append(
            f"| {feat} | {st['n_same_sign']}/{st['n_confirm_units']} | "
            f"{100*st['same_sign_rate']:.1f}% |"
        )
    lines += [
        "",
        "## Per-unit confirm",
        "",
        "| unit | same_sign | mean_D |",
        "|------|----------:|-------:|",
    ]
    for u in confirm_units:
        md = u.get("mean_D")
        md_s = f"{md:+.4f}" if md is not None else "NA"
        lines.append(
            f"| {u['unit']} | {u['n_same_sign']}/{u['n_star']} | {md_s} |"
        )
    (agg / "star_explore_confirm.md").write_text("\n".join(lines) + "\n")

    if args.update_gating_config:
        cfg_path = agg / "gating_config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        # Routing freeze: prefer high_more_harm history feats excluding shock unless include_shock
        route_feats = [f for f in star_feats if f != "shock_score" or args.include_shock]
        # Default paper routing often drops shock; if include_shock keep it in star list but
        # write both star_feats and risk_feats (routing).
        cfg["risk_feats"] = route_feats
        cfg["proxy_list_policy"] = {
            "mode": "strict_explore_confirm",
            "strict_explore_only": True,
            "explore_unit": f"{EXPLORE_MEMBER}/{EXPLORE_DATASET}",
            "star_feats": star_feats,
            "routing_feats": route_feats,
            "include_shock_in_routing": bool(args.include_shock and "shock_score" in star_feats),
            "explore_json": str(explore_json),
            "archive": str(agg / "star_explore_confirm.json"),
            "note": (
                "S frozen on GPT4MTS–Environment only; confirm matrix does not re-select. "
                "Old 11/15 full-matrix starring superseded."
            ),
        }
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
        print("updated", cfg_path)

    status = [
        "# Star explore→confirm status",
        "",
        f"- Explore: {EXPLORE_MEMBER}/{EXPLORE_DATASET}",
        f"- Frozen |S| = {len(star_feats)}: {', '.join(star_feats)}",
        f"- Confirm archive: `outputs/aggregate/star_explore_confirm.md`",
        "",
    ]
    (agg / "star_explore_confirm_status.md").write_text("\n".join(status))
    print("\n".join(lines))
    print("wrote", agg / "star_explore_confirm.md")


if __name__ == "__main__":
    main()
