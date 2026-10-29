"""Learnable soft gate on hist features: y_g = (1-α)y1 + α y0, α=σ(w·φ+b).

Train only on train-split dumps (no test D). Hyperparams frozen from
Environment×TaTS×seed2025 val; other units domain-refit w,b with same structure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_npf_gating import (  # noqa: E402
    RISK_FEATS,
    build_features_train,
    domain_gate_params,
    eval_with_params,
    fit_z,
    load_paired_with_feats,
    metrics,
    random_gate,
    risk_scores,
    train_indices,
)

NPF = Path(__file__).resolve().parents[1]


def load_split_preds(dump_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = dump_dir / split
    pred = np.load(d / "pred.npy")
    true = np.load(d / "true.npy")
    index = np.load(d / "index.npy")
    if pred.ndim == 3:
        pred = pred[..., 0]
        true = true[..., 0]
    return pred.astype(np.float64), true.astype(np.float64), index.astype(np.int64)


def align_pair(y0, t0, i0, y1, t1, i1):
    map1 = {int(x): j for j, x in enumerate(i1)}
    k0, k1 = [], []
    for j, x in enumerate(i0):
        k = map1.get(int(x))
        if k is not None:
            k0.append(j)
            k1.append(k)
    return y0[k0], t0[k0], i0[k0], y1[k1], t1[k1]


def z_transform(df: pd.DataFrame, z_stats: dict) -> np.ndarray:
    cols = []
    for f in RISK_FEATS:
        mu, sd = z_stats[f]["mean"], z_stats[f]["std"]
        cols.append((df[f].to_numpy(dtype=float) - mu) / sd)
    return np.stack(cols, axis=1)


class SoftGate(nn.Module):
    def __init__(self, d_in: int = 3):
        super().__init__()
        self.lin = nn.Linear(d_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lin(x)).squeeze(-1)


def fit_soft_gate(
    phi: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray,
    true: np.ndarray,
    *,
    lr: float = 0.05,
    epochs: int = 400,
    l2: float = 1e-3,
    seed: int = 0,
) -> SoftGate:
    """Minimize mean_t ((1-α)y1+α y0 - true)^2 + l2||w||^2 on train windows."""
    torch.manual_seed(seed)
    device = "cpu"
    model = SoftGate(phi.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    X = torch.tensor(phi, dtype=torch.float32, device=device)
    Y0 = torch.tensor(y0, dtype=torch.float32, device=device)
    Y1 = torch.tensor(y1, dtype=torch.float32, device=device)
    T = torch.tensor(true, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        alpha = model(X).unsqueeze(-1)  # [N,1]
        yg = (1.0 - alpha) * Y1 + alpha * Y0
        loss = ((yg - T) ** 2).mean()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def eval_soft(
    model: SoftGate,
    phi: np.ndarray,
    mse0: np.ndarray,
    mse1: np.ndarray,
    y0: np.ndarray | None = None,
    y1: np.ndarray | None = None,
    true: np.ndarray | None = None,
) -> dict:
    model.eval()
    X = torch.tensor(phi, dtype=torch.float32)
    alpha = model(X).numpy()
    if y0 is not None and y1 is not None and true is not None:
        a = alpha[:, None]
        yg = (1.0 - a) * y1 + a * y0
        mse_g = ((yg - true) ** 2).mean(axis=1)
    else:
        # fallback linear mix of MSEs (upper bound style; prefer pred mix)
        mse_g = alpha * mse0 + (1.0 - alpha) * mse1
    close = alpha >= 0.5  # diagnostic rate
    m = metrics(mse_g, mse0, mse1, close)
    m["kind"] = "soft_gate_logistic"
    m["mean_alpha"] = float(alpha.mean())
    m["alpha_ge_0.5_rate"] = float(close.mean())
    return m, alpha


def resolve_dumps(run_dir: Path, member: str, dataset: str, seed: int, seq_len: int, pred_len: int):
    if member == "TaTS":
        root = run_dir / "dumps"
        f0 = root / f"{dataset}_{seed}_{seq_len}_{pred_len}_f0_numeric"
        f1 = root / f"{dataset}_{seed}_{seq_len}_{pred_len}_f1_text"
        return f0, f1
    if member == "GPT4MTS":
        return run_dir / "dumps/f0_numeric", run_dir / "dumps/f1_text"
    if member == "CFA-naive":
        return run_dir / "dumps/f0_unimodal", run_dir / "dumps/f1_first_concat"
    raise ValueError(member)


def prepare_split(
    f0: Path,
    f1: Path,
    data_csv: Path,
    split: str,
    seq_len: int,
    pred_len: int,
    z_stats: dict | None,
):
    y0, t0, i0 = load_split_preds(f0, split)
    y1, t1, i1 = load_split_preds(f1, split)
    y0, t0, i0, y1, t1 = align_pair(y0, t0, i0, y1, t1, i1)
    if split == "train":
        # train indices are absolute from 0
        feats = build_features_train(data_csv, i0, seq_len)
    else:
        from analyze_npf_conditions import build_features

        feats = build_features(data_csv, i0, seq_len, pred_len)
    df = pd.DataFrame({"index": i0}).merge(feats, on="index", how="inner")
    # realign preds to merged order
    order = {int(x): j for j, x in enumerate(i0)}
    keep = [order[int(ix)] for ix in df["index"].to_numpy()]
    y0, y1, t0 = y0[keep], y1[keep], t0[keep]
    mse0 = ((y0 - t0) ** 2).mean(axis=1)
    mse1 = ((y1 - t0) ** 2).mean(axis=1)
    if z_stats is None:
        z_stats = fit_z(df)
    phi = z_transform(df, z_stats)
    return df, phi, y0, y1, t0, mse0, mse1, z_stats


def calibrate_hyperparams(cal_dir: Path, data_csv: Path, seq_len: int, pred_len: int) -> dict:
    """Pick lr/epochs/l2 on Environment TaTS val (not test)."""
    f0, f1 = resolve_dumps(cal_dir, "TaTS", "Environment", 2025, seq_len, pred_len)
    # z from train
    _, phi_tr, y0_tr, y1_tr, t_tr, _, _, z_stats = prepare_split(
        f0, f1, data_csv, "train", seq_len, pred_len, None
    )
    _, phi_va, y0_va, y1_va, t_va, mse0_va, mse1_va, _ = prepare_split(
        f0, f1, data_csv, "vali", seq_len, pred_len, z_stats
    )
    best = None
    for lr in (0.05, 0.1):
        for l2 in (1e-3, 1e-2):
            for epochs in (300, 500):
                model = fit_soft_gate(phi_tr, y0_tr, y1_tr, t_tr, lr=lr, epochs=epochs, l2=l2)
                m, _ = eval_soft(model, phi_va, mse0_va, mse1_va, y0_va, y1_va, t_va)
                score = m["delta_f1_minus_g"]
                if best is None or score > best["score"]:
                    best = {"lr": lr, "l2": l2, "epochs": epochs, "score": score, "val": m}
    return {
        "lr": best["lr"],
        "l2": best["l2"],
        "epochs": best["epochs"],
        "calibrate_unit": "TaTS/Environment/seed2025",
        "val_metrics": best["val"],
    }


def fit_mode_for(f0: Path, f1: Path) -> str:
    if (f0 / "train/pred.npy").exists() and (f1 / "train/pred.npy").exists():
        return "domain_refit"
    return "transfer_w"


def run_unit(
    role: str,
    member: str,
    dataset: str,
    run_dir: Path,
    data_csv: Path,
    seq_len: int,
    pred_len: int,
    seed: int,
    hp: dict,
    q_star: float = 0.6,
    transfer_state: dict | None = None,
) -> dict:
    f0, f1 = resolve_dumps(run_dir, member, dataset, seed, seq_len, pred_len)
    if not (f0 / "test/pred.npy").exists():
        return {"role": role, "member": member, "dataset": dataset, "seed": seed, "status": "missing_test_dump"}

    mode = fit_mode_for(f0, f1)
    # domain z from train windows (features only; no labels)
    n_rows = len(pd.read_csv(data_csv))
    tr_idx = train_indices(n_rows, seq_len, pred_len)
    z_stats = fit_z(build_features_train(data_csv, tr_idx, seq_len))

    model = SoftGate(len(RISK_FEATS))
    if mode == "domain_refit":
        _, phi_tr, y0_tr, y1_tr, t_tr, _, _, z_stats = prepare_split(
            f0, f1, data_csv, "train", seq_len, pred_len, None
        )
        model = fit_soft_gate(
            phi_tr,
            y0_tr,
            y1_tr,
            t_tr,
            lr=hp["lr"],
            epochs=hp["epochs"],
            l2=hp["l2"],
            seed=seed,
        )
    else:
        if transfer_state is None:
            return {
                "role": role,
                "member": member,
                "dataset": dataset,
                "seed": seed,
                "status": "need_transfer_w",
            }
        with torch.no_grad():
            model.lin.weight.copy_(torch.tensor(transfer_state["w"], dtype=torch.float32).view(1, -1))
            model.lin.bias.copy_(torch.tensor([transfer_state["b"]], dtype=torch.float32))

    # test eval: prefer pred mix; GPT/CFA only have test preds
    y0, t0, i0 = load_split_preds(f0, "test")
    y1, t1, i1 = load_split_preds(f1, "test")
    y0, t0, i0, y1, t1 = align_pair(y0, t0, i0, y1, t1, i1)
    from analyze_npf_conditions import build_features

    feats = build_features(data_csv, i0, seq_len, pred_len)
    df = pd.DataFrame({"index": i0}).merge(feats, on="index", how="inner")
    order = {int(x): j for j, x in enumerate(i0)}
    keep = [order[int(ix)] for ix in df["index"].to_numpy()]
    y0, y1, t0 = y0[keep], y1[keep], t0[keep]
    mse0 = ((y0 - t0) ** 2).mean(axis=1)
    mse1 = ((y1 - t0) ** 2).mean(axis=1)
    phi_te = z_transform(df, z_stats)
    soft_m, alpha = eval_soft(model, phi_te, mse0, mse1, y0, y1, t0)
    soft_m["fit_mode"] = mode
    soft_m["random"] = random_gate(mse0, mse1, soft_m["alpha_ge_0.5_rate"], seed=seed)

    hard_params = domain_gate_params(data_csv, seq_len, pred_len, q_star)
    paired = load_paired_with_feats(run_dir / "paired_test.csv", data_csv, seq_len, pred_len)
    hard_m = eval_with_params(paired, hard_params, seed=seed)

    w = model.lin.weight.detach().cpu().numpy().ravel().tolist()
    b = float(model.lin.bias.detach().cpu().item())
    return {
        "role": role,
        "member": member,
        "dataset": dataset,
        "seed": seed,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "status": "ok",
        "fit_mode": mode,
        "soft": soft_m,
        "hard": hard_m,
        "weights": {"w": w, "b": b, "features": list(RISK_FEATS)},
        "beats_hard": bool(soft_m["mse_g"] < hard_m["mse_g"]),
        "run_dir": str(run_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--pred-len", type=int, default=48)
    ap.add_argument("--q-star", type=float, default=0.6)
    args = ap.parse_args()
    root = NPF / "outputs"
    tats = NPF / "third_party/TaTS/data"
    cfa = NPF / "third_party/cfa/data"

    cal_dir = root / "tats/Environment/NPF_Environment_iTransformer_seed2025_sl24_pl48"
    hp = calibrate_hyperparams(cal_dir, tats / "Environment.csv", args.seq_len, args.pred_len)

    units = [
        ("calibrate", "TaTS", "Environment", cal_dir, tats / "Environment.csv", 2025),
        ("confirm", "TaTS", "Energy", root / "tats/Energy/NPF_Energy_iTransformer_seed2025_sl24_pl48", tats / "Energy.csv", 2025),
        ("confirm", "GPT4MTS", "Environment", root / "gpt4mts/Environment/NPF_Environment_GPT4MTS_seed2025_sl24_pl48", tats / "Environment.csv", 2025),
        ("confirm", "GPT4MTS", "Health", root / "gpt4mts/Health/NPF_Health_GPT4MTS_seed2025_sl24_pl48", tats / "Health.csv", 2025),
        ("confirm", "CFA-naive", "Health", root / "naive_cfa/Health/NPF_Health_iTransformer_seed2025_sl24_pl48", cfa / "Health/Health_NPF.csv", 2025),
        ("extra", "GPT4MTS", "Climate", root / "gpt4mts/Climate/NPF_Climate_GPT4MTS_seed2025_sl24_pl48", tats / "Climate.csv", 2025),
        # seed2024 replicates
        ("seed_rep", "TaTS", "Energy", root / "tats/Energy/NPF_Energy_iTransformer_seed2024_sl24_pl48", tats / "Energy.csv", 2024),
        ("seed_rep", "GPT4MTS", "Climate", root / "gpt4mts/Climate/NPF_Climate_GPT4MTS_seed2024_sl24_pl48", tats / "Climate.csv", 2024),
    ]

    results = []
    transfer_state = None
    for role, member, dataset, run_dir, data_csv, seed in units:
        if not run_dir.exists() or not (run_dir / "paired_test.csv").exists():
            results.append({"role": role, "member": member, "dataset": dataset, "seed": seed, "status": "missing"})
            continue
        print(f"soft-gate {member}/{dataset} seed{seed} ...")
        r = run_unit(
            role,
            member,
            dataset,
            run_dir,
            data_csv,
            args.seq_len,
            args.pred_len,
            seed,
            hp,
            args.q_star,
            transfer_state=transfer_state,
        )
        results.append(r)
        if role == "calibrate" and r.get("status") == "ok":
            transfer_state = {"w": r["weights"]["w"], "b": r["weights"]["b"]}

    confirm = [r for r in results if r.get("role") == "confirm" and r.get("status") == "ok"]
    n_beats_f1 = sum(1 for r in confirm if r["soft"]["beats_f1"])
    mean_soft = float(np.mean([r["soft"]["mse_g"] for r in confirm])) if confirm else None
    mean_hard = float(np.mean([r["hard"]["mse_g"] for r in confirm])) if confirm else None

    report = {
        "hyperparams": hp,
        "q_star_hard": args.q_star,
        "results": results,
        "summary_confirm": {
            "n_units": len(confirm),
            "soft_beats_f1": n_beats_f1,
            "mean_mse_soft": mean_soft,
            "mean_mse_hard": mean_hard,
            "soft_not_worse_hard_mean": bool(mean_soft is not None and mean_hard is not None and mean_soft <= mean_hard),
        },
    }
    agg = root / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)
    (agg / "soft_gate.json").write_text(json.dumps(report, indent=2) + "\n")

    lines = [
        "# Soft gate vs hard gate",
        "",
        f"Hyperparams (val-calibrated on Env×TaTS×2025): lr={hp['lr']} l2={hp['l2']} epochs={hp['epochs']}",
        f"Confirm: soft beats f1 on {n_beats_f1}/{len(confirm)}; mean mse soft={mean_soft} hard={mean_hard}",
        "",
        "| role | unit | seed | mse_soft | mse_hard | mse_f0 | mse_f1 | Δ(f1−soft) | soft<hard | mean_α |",
        "|------|------|-----:|---------:|---------:|-------:|-------:|-----------:|:---------:|-------:|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('role')} | {r.get('member')}/{r.get('dataset')} | {r.get('seed')} | — | — | — | — | — | {r.get('status')} | — |")
            continue
        s, h = r["soft"], r["hard"]
        lines.append(
            f"| {r['role']} | {r['member']}/{r['dataset']} | {r['seed']} | "
            f"{s['mse_g']:.6g} | {h['mse_g']:.6g} | {s['mse_f0']:.6g} | {s['mse_f1']:.6g} | "
            f"{s['delta_f1_minus_g']:+.4g} | {'Y' if r['beats_hard'] else 'N'} | {s['mean_alpha']:.3f} |"
        )
    (agg / "soft_gate.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("wrote", agg / "soft_gate.md")


if __name__ == "__main__":
    main()
