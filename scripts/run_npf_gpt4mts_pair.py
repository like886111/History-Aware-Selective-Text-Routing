"""NPF GPT4MTS pair: f0 (no soft prompt) vs f1 (soft prompt) on Environment."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model, GPT2Tokenizer

NPF_ROOT = Path(__file__).resolve().parents[1]
G4_ROOT = NPF_ROOT / "third_party" / "GPT4MTS"
sys.path.insert(0, str(G4_ROOT))

from models.GPT4MTS import GPT4MTS  # noqa: E402


class EnvNPF(Dataset):
    """Univariate Environment series in GPT4MTS window format."""

    def __init__(self, csv_path: Path, flag: str, seq_len: int, label_len: int, pred_len: int, device: str):
        assert flag in {"train", "val", "test"}
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        df = pd.read_csv(csv_path)
        ot = df["OT"].astype(float).to_numpy().reshape(-1, 1)
        texts = df["fact"].fillna("").astype(str).tolist() if "fact" in df.columns else [""] * len(df)
        texts = [
            re.sub(r"[^a-zA-Z0-9 ]+", "", t) if t.strip() else "Predict the future time step given the past"
            for t in texts
        ]
        n = len(ot)
        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test
        border1s = [0, num_train - seq_len, n - num_test - seq_len]
        border2s = [num_train, num_train + num_vali, n]
        type_map = {"train": 0, "val": 1, "test": 2}
        b1, b2 = border1s[type_map[flag]], border2s[type_map[flag]]

        scaler = StandardScaler()
        scaler.fit(ot[border1s[0] : border2s[0]])
        ot_s = scaler.transform(ot)
        self.data_x = ot_s[b1:b2]
        self.data_y = ot_s[b1:b2]
        self.tot_len = len(self.data_x) - seq_len - pred_len + 1
        if self.tot_len <= 0:
            raise RuntimeError(f"split {flag} too short: {len(self.data_x)}")

        # Use local GPT-2 pooled embeddings (768-d) — BERT not available offline here.
        # Include CSV stem so empty/shuffle interventions do not reuse orig Climate cache.
        emb_cache = csv_path.parent / f".gpt2_pool_emb_{csv_path.stem}_{flag}_n{b2-b1}.npy"
        if emb_cache.exists():
            self.data_s = torch.from_numpy(np.load(emb_cache)).float()
        else:
            print(f"Encoding GPT-2 pooled text for {flag} ({b2-b1} rows)...")
            tok = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            lm = GPT2Model.from_pretrained("gpt2", local_files_only=True).to(device)
            lm.eval()
            split_texts = texts[b1:b2]
            embs = []
            bs = 8
            with torch.no_grad():
                for i in range(0, len(split_texts), bs):
                    batch = split_texts[i : i + bs]
                    inputs = tok(
                        batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=128,
                    ).to(device)
                    out = lm(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1)
                    pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
                    embs.append(pooled.detach().cpu())
            self.data_s = torch.cat(embs, dim=0).float()
            np.save(emb_cache, self.data_s.numpy())
            print("cached", emb_cache)

    def __len__(self):
        return self.tot_len

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = torch.tensor(self.data_x[s_begin:s_end], dtype=torch.float32)
        seq_y = torch.tensor(self.data_y[r_begin:r_end], dtype=torch.float32)
        seq_s = self.data_s[s_begin:s_end]
        # dummy marks
        seq_x_mark = torch.zeros(self.seq_len, 4)
        seq_y_mark = torch.zeros(self.label_len + self.pred_len, 4)
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_s, index


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_one(args, use_text: int, out_dir: Path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tag = "f1_text" if use_text else "f0_numeric"
    print(f"== GPT4MTS {tag} use_text={use_text} on {device} ==")
    set_seed(args.seed)

    train_set = EnvNPF(args.data_csv, "train", args.seq_len, args.label_len, args.pred_len, device)
    val_set = EnvNPF(args.data_csv, "val", args.seq_len, args.label_len, args.pred_len, device)
    test_set = EnvNPF(args.data_csv, "test", args.seq_len, args.label_len, args.pred_len, device)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=False)

    class C:
        pass

    cfg = C()
    cfg.is_gpt = 1
    cfg.revin = True
    cfg.patch_size = 8
    cfg.pretrain = 1
    cfg.stride = 4
    cfg.seq_len = args.seq_len
    cfg.pred_len = args.pred_len
    cfg.d_model = 768
    cfg.gpt_layers = 6
    cfg.freeze = 1

    # Fix GPT2 cache path in model if needed
    model = GPT4MTS(cfg, device)
    # override hf cache path hardcoded in upstream if load failed — already constructed
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()

    ckpt_dir = out_dir / "checkpoints" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_path = ckpt_dir / "checkpoint.pth"
    patience, bad = args.patience, 0

    for epoch in range(args.epochs):
        model.train()
        losses = []
        for batch_x, batch_y, _, _, batch_s, _ in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_s = batch_s.to(device)
            optim.zero_grad()
            out = model(batch_x, 0, batch_s, use_text=use_text)
            out = out[:, -args.pred_len :, :]
            y = batch_y[:, -args.pred_len :, :]
            loss = crit(out, y)
            loss.backward()
            optim.step()
            losses.append(loss.item())
        model.eval()
        vlosses = []
        with torch.no_grad():
            for batch_x, batch_y, _, _, batch_s, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_s = batch_s.to(device)
                out = model(batch_x, 0, batch_s, use_text=use_text)[:, -args.pred_len :, :]
                y = batch_y[:, -args.pred_len :, :]
                vlosses.append(crit(out, y).item())
        tr, va = float(np.mean(losses)), float(np.mean(vlosses))
        print(f"Epoch {epoch+1}/{args.epochs} train={tr:.6f} val={va:.6f}")
        if va < best_val:
            best_val = va
            bad = 0
            torch.save(model.state_dict(), best_path)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping")
                break

    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    def dump_split(loader, split_name: str) -> float:
        preds, trues, indices = [], [], []
        with torch.no_grad():
            for batch_x, batch_y, _, _, batch_s, idx in loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                batch_s = batch_s.to(device)
                out = model(batch_x, 0, batch_s, use_text=use_text)[:, -args.pred_len :, :]
                y = batch_y[:, -args.pred_len :, :]
                preds.append(out.detach().cpu().numpy())
                trues.append(y.detach().cpu().numpy())
                indices.append(idx.numpy() if torch.is_tensor(idx) else np.asarray(idx))
        pred = np.concatenate(preds, 0)
        true = np.concatenate(trues, 0)
        index = np.concatenate(indices, 0)
        dump = out_dir / "dumps" / tag / split_name
        dump.mkdir(parents=True, exist_ok=True)
        np.save(dump / "pred.npy", pred)
        np.save(dump / "true.npy", true)
        np.save(dump / "index.npy", index)
        return float(((pred - true) ** 2).mean())

    # train/vali loaders without drop_last for full coverage
    train_loader_full = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    val_loader_full = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=False)
    mse_tr = dump_split(train_loader_full, "train")
    mse_va = dump_split(val_loader_full, "vali")
    mse = dump_split(test_loader, "test")
    n_test = int(np.load(out_dir / "dumps" / tag / "test" / "pred.npy").shape[0])
    print(f"{tag} test mse={mse:.6f} n={n_test} (train={mse_tr:.6f} val={mse_va:.6f})")
    (out_dir / f"metrics_{tag}.json").write_text(
        json.dumps({"mse": mse, "n": n_test, "mse_train": mse_tr, "mse_val": mse_va}, indent=2) + "\n"
    )
    return out_dir / "dumps" / tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-csv", type=Path, default=NPF_ROOT / "third_party/TaTS/data/Environment.csv")
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--label-len", type=int, default=12)
    ap.add_argument("--pred-len", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    ap.add_argument(
        "--explore-json",
        type=Path,
        default=None,
        help="C_hat explore json to confirm (default: Environment TaTS explore)",
    )
    ap.add_argument("--dataset-name", type=str, default=None)
    args = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    if args.out_dir is None:
        args.out_dir = (
            NPF_ROOT
            / "outputs/gpt4mts/Environment"
            / f"NPF_Environment_GPT4MTS_seed{args.seed}_sl{args.seq_len}_pl{args.pred_len}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_one(args, 0, args.out_dir)
    run_one(args, 1, args.out_dir)

    # L1
    import subprocess

    ds_name = args.dataset_name or args.data_csv.stem
    subprocess.check_call(
        [
            sys.executable,
            str(NPF_ROOT / "scripts/analyze_npf_l1.py"),
            "--f0-dump",
            str(args.out_dir / "dumps/f0_numeric"),
            "--f1-dump",
            str(args.out_dir / "dumps/f1_text"),
            "--split",
            "test",
            "--member",
            "GPT4MTS",
            "--dataset",
            ds_name,
            "--backbone",
            "GPT4MTS",
            "--seed",
            str(args.seed),
            "--seq-len",
            str(args.seq_len),
            "--pred-len",
            str(args.pred_len),
            "--output-dir",
            str(args.out_dir),
        ]
    )

    explore = args.explore_json or (
        NPF_ROOT
        / "outputs/tats/Environment/NPF_Environment_iTransformer_seed2025_sl24_pl48/condition_explore.json"
    )
    if explore.exists():
        rc = subprocess.call(
            [
                sys.executable,
                str(NPF_ROOT / "scripts/analyze_npf_confirm.py"),
                "--explore-json",
                str(explore),
                "--paired-csv",
                str(args.out_dir / "paired_test.csv"),
                "--data-csv",
                str(args.data_csv),
                "--seq-len",
                str(args.seq_len),
                "--pred-len",
                str(args.pred_len),
                "--output-dir",
                str(args.out_dir),
                "--member",
                "GPT4MTS",
                "--dataset",
                ds_name,
            ]
        )
        if rc != 0:
            print("WARN: confirm step failed (L1 artifacts kept); rc=", rc)
    print("Done. Artifacts under", args.out_dir)


if __name__ == "__main__":
    main()
