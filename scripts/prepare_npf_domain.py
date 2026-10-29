"""Build TaTS/CFA NPF CSVs for Climate and Health_AFR from upstream Time-MMD."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MMD = Path(__file__).resolve().parents[2] / "Time-MMD"
NPF = Path(__file__).resolve().parents[1]
TATS_DATA = NPF / "third_party/TaTS/data"
CFA_DATA = NPF / "third_party/cfa/data"


def prior_from_ot(ot: np.ndarray, lookback: int = 12) -> tuple[np.ndarray, np.ndarray]:
    avg = np.zeros(len(ot), dtype=float)
    std = np.zeros(len(ot), dtype=float)
    for i in range(len(ot)):
        lo = max(0, i - lookback)
        w = ot[lo:i]
        if w.size == 0:
            avg[i] = 0.0
            std[i] = 0.0
        else:
            avg[i] = float(np.mean(w))
            std[i] = float(np.std(w))
    return avg, std


def attach_text(num: pd.DataFrame, search: pd.DataFrame) -> pd.DataFrame:
    num = num.copy()
    search = search.copy()
    num["start_date"] = pd.to_datetime(num["start_date"])
    num["end_date"] = pd.to_datetime(num["end_date"])
    search["start_date"] = pd.to_datetime(search["start_date"])
    search["end_date"] = pd.to_datetime(search["end_date"])
    # prefer exact start_date; else asof nearest start (±7d)
    s1 = search.drop_duplicates("start_date", keep="first")
    exact = num.merge(s1[["start_date", "fact", "preds"]], on="start_date", how="left")
    if exact["fact"].notna().mean() >= 0.5:
        out = exact
    else:
        left = num.sort_values("start_date")
        right = s1.sort_values("start_date")
        out = pd.merge_asof(
            left,
            right[["start_date", "fact", "preds"]],
            on="start_date",
            direction="nearest",
            tolerance=pd.Timedelta("7D"),
        )
    out["fact"] = out["fact"].fillna("NA")
    out["preds"] = out["preds"].fillna("NA")
    return out


def build_climate() -> pd.DataFrame:
    num = pd.read_csv(MMD / "numerical/Climate/Climate.csv")
    search = pd.read_csv(MMD / "textual/Climate/Climate_search.csv")
    num["date"] = pd.to_datetime(num["date"])
    num = num.sort_values("date").reset_index(drop=True)
    num["OT"] = pd.to_numeric(num["OT"], errors="coerce")
    df = attach_text(num, search)
    ot = df["OT"].to_numpy(dtype=float)
    avg, std = prior_from_ot(ot)
    out = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "start_date": pd.to_datetime(df["start_date"]).dt.strftime("%Y-%m-%d"),
            "end_date": pd.to_datetime(df["end_date"]).dt.strftime("%Y-%m-%d"),
            "OT": ot,
            "prior_history_avg": avg,
            "prior_history_std": std,
            "fact": df["fact"].astype(str),
            "preds": df["preds"].astype(str),
        }
    )
    return out.dropna(subset=["OT"]).reset_index(drop=True)


def build_health_afr() -> pd.DataFrame:
    num = pd.read_csv(MMD / "numerical/Health_AFR/Health_AFR.csv")
    # columns: date, date.1, end_date, OT
    num = num.rename(columns={"date.1": "start_date"})
    num["date"] = pd.to_datetime(num["date"], errors="coerce")
    num["start_date"] = pd.to_datetime(num["start_date"], errors="coerce")
    num["end_date"] = pd.to_datetime(num["end_date"], errors="coerce")
    num["OT"] = pd.to_numeric(num["OT"], errors="coerce")
    num.loc[~np.isfinite(num["OT"]), "OT"] = np.nan
    # interpolate short gaps then drop remaining
    num = num.sort_values("date").reset_index(drop=True)
    num["OT"] = num["OT"].interpolate(limit=3)
    search = pd.read_csv(MMD / "textual/Health_AFR/Health_AFR_search.csv")
    df = attach_text(num, search)
    ot = df["OT"].to_numpy(dtype=float)
    avg, std = prior_from_ot(ot)
    out = pd.DataFrame(
        {
            "date": df["date"].dt.strftime("%Y-%m-%d"),
            "start_date": pd.to_datetime(df["start_date"]).dt.strftime("%Y-%m-%d"),
            "end_date": pd.to_datetime(df["end_date"]).dt.strftime("%Y-%m-%d"),
            "OT": ot,
            "prior_history_avg": avg,
            "prior_history_std": std,
            "fact": df["fact"].astype(str),
            "preds": df["preds"].astype(str),
        }
    )
    return out.dropna(subset=["OT"]).reset_index(drop=True)


def write_pair(name: str, df: pd.DataFrame) -> None:
    TATS_DATA.mkdir(parents=True, exist_ok=True)
    tats_path = TATS_DATA / f"{name}.csv"
    df.to_csv(tats_path, index=False)

    cfa_dir = CFA_DATA / name
    cfa_dir.mkdir(parents=True, exist_ok=True)
    cfa = pd.DataFrame(
        {
            "date": df["date"],
            "OT": df["OT"],
            "prior_history_avg": df["prior_history_avg"],
            "start_date": df["start_date"],
            "end_date": df["end_date"],
            "Final_Search_4": df["fact"],
        }
    )
    cfa_path = cfa_dir / f"{name}_NPF.csv"
    cfa.to_csv(cfa_path, index=False)

    n = len(df)
    test_n = int(n * 0.2) - 48 + 1
    print(f"{name}: n={n} test_n≈{test_n} fact_nonempty={(df['fact'].str.strip()!='NA').mean():.3f}")
    print("  wrote", tats_path)
    print("  wrote", cfa_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="+", default=["Climate", "Health_AFR"])
    args = ap.parse_args()
    builders = {"Climate": build_climate, "Health_AFR": build_health_afr}
    for d in args.domains:
        if d not in builders:
            raise SystemExit(f"unknown domain {d}; choose from {list(builders)}")
        write_pair(d, builders[d]())


if __name__ == "__main__":
    main()
