# History-Aware Selective Text Routing

Code accompanying the paper *History-Aware Selective Text Routing for Mitigating Negative Fusion in Multimodal Time-Series Forecasting*.

This repository contains analysis, routing, and evaluation code for history-only selective text suppression on naive plug-in multimodal forecasters (TaTS, GPT4MTS, CFA-naive), plus vendored forecasting member sources.

## Layout

```text
scripts/       # Paired utility, history characterization, hard gate, ablations, robustness
third_party/   # Vendored member code (TaTS / GPT4MTS / CFA); see third_party/README.md
```

Batch training / orchestration shell scripts (`run_*.sh`) are **not** included.

## Data

Experiments use the public [Time-MMD](https://github.com/AdityaLab/Time-MMD) benchmark. Place or symlink the dataset according to each member’s data loader expectations.

## Third-party members

Upstream repositories (keep their licenses):

| Path | Upstream |
|------|----------|
| `third_party/TaTS` | https://github.com/iDEA-iSAIL-Lab-UIUC/TaTS |
| `third_party/GPT4MTS` | https://github.com/Flora-jia-jfr/GPT4MTS-Prompt-based-Large-Language-Model-for-Multimodal-Time-series-Forecasting |
| `third_party/cfa` | https://github.com/seunghan96/cfa |

Checkpoints and raw data dumps are omitted; train or download weights as needed. CFA can also be fetched with `scripts/fetch_cfa.sh`.

## Core analysis entry points

| Script | Role |
|--------|------|
| `scripts/analyze_npf_phenomenon.py` | Window-level negative fusion diagnostics |
| `scripts/analyze_npf_conditions.py` / `analyze_npf_star_explore_confirm.py` | History proxies; explore–confirm |
| `scripts/analyze_npf_gating.py` / `analyze_npf_gating_official.py` / `analyze_npf_gating_v2_official.py` | History-aware hard routing |
| `scripts/analyze_npf_gating_structure_ablation.py` | Structure ablations |
| `scripts/analyze_npf_robustness_46.py` | Robustness (§4.6-style) |
| `scripts/run_npf_gated_eval.py` | Apply frozen gate to paired dumps |

Exact paths and dump layouts depend on local experiment outputs; adapt roots when reproducing.
