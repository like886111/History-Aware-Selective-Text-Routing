"""Shared Time-MMD official (L,H) by frequency for NPF re-runs."""

from __future__ import annotations

DOMAIN_LH: dict[str, tuple[int, int, int]] = {
    # dataset -> (seq_len, pred_len, label_len)
    "Environment": (96, 48, 48),
    "Energy": (36, 24, 18),
    "Health": (36, 24, 18),
    "Health_AFR": (36, 24, 18),
    "Climate": (8, 8, 4),
}

DOMAIN_BATCH: dict[str, int] = {
    "Environment": 16,
    "Energy": 32,
    "Health": 32,
    "Health_AFR": 32,
    "Climate": 32,
}

MEMBERS = ("TaTS", "GPT4MTS", "CFA-naive")

# Paper multi-seed robustness (GPT4MTS–Environment)
SCALE_SEEDS = (2025, 2026, 2027)

# CFA official comparison domains (exclude Health_AFR / SocialGood)
CFA_OFFICIAL_DATASETS = ("Environment", "Energy", "Health", "Climate")


def run_rel(member: str, dataset: str, seed: int = 2025) -> str:
    sl, pl, _ = DOMAIN_LH[dataset]
    if member == "TaTS":
        return f"tats/{dataset}/NPF_{dataset}_iTransformer_seed{seed}_sl{sl}_pl{pl}"
    if member == "GPT4MTS":
        return f"gpt4mts/{dataset}/NPF_{dataset}_GPT4MTS_seed{seed}_sl{sl}_pl{pl}"
    if member == "CFA-naive":
        return f"naive_cfa/{dataset}/NPF_{dataset}_iTransformer_seed{seed}_sl{sl}_pl{pl}"
    raise ValueError(member)


def data_tag(member: str, dataset: str) -> str:
    if member == "CFA-naive":
        return f"cfa/{dataset}/{dataset}_NPF.csv"
    return f"TaTS/{dataset}.csv"
