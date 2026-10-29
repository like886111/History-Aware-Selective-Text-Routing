# 本线脚本（现象驱动 → 条件 → 硬门控）

1. `run_npf_tats_pair.sh` — TaTS 成对 f0/f1 + dump + L1 + 现象探索  
2. `run_npf_cfa_pair.sh` — CFA unimodal vs first-concat + L1 + confirm  
3. `run_npf_gpt4mts_pair.py` — GPT4MTS 无/有 soft prompt + L1 + confirm  
4. `analyze_npf_l1.py` / `analyze_npf_conditions.py` / `analyze_npf_gating.py`  
5. **硬门控**：`analyze_npf_gating.py`（冻结 q*）→ `run_npf_gated_eval.py` / `run_npf_tats_gated.sh` / `run_npf_gated_all.sh`

门控：风险分 `z(hist_std)+z(acf1)+z(pred_ease)` ≥ 域内 train \(\tau(q^*)\) → 回退 f0。协议见 `protocol/05_gating_design.md`。
