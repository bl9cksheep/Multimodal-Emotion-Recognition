# src/train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import seed_everything
from .dataset import build_index, MultiModalEmotionDataset
from .model import MultiModalEmotionModel
from .eval import evaluate

import sys
from tqdm.auto import tqdm



PATCHTST_NAME = "/root/project/multi_emo_rec/models/granite-timeseries-patchtst"
CACHE_DIR = "/root/.cache/huggingface"


def collate_fn(batch_list):
    keys = [b["key"] for b in batch_list]

    sids = torch.stack([b["sid"] for b in batch_list], dim=0)      # (B,)
    ecg  = torch.stack([b["ecg"] for b in batch_list], dim=0)      # (B,L1)
    eda  = torch.stack([b["eda"] for b in batch_list], dim=0)      # (B,L2)
    ppg  = torch.stack([b["ppg"] for b in batch_list], dim=0)      # (B,L2)
    y    = torch.stack([b["label"] for b in batch_list], dim=0)    # (B,)

    return {"key": keys, "sid": sids, "ecg": ecg, "eda": eda, "ppg": ppg, "label": y}


def split_by_subject(keys, index, n_splits=5, seed=42):
    rng = np.random.RandomState(seed)
    sids = sorted({index[k]["sid"] for k in keys})
    rng.shuffle(sids)
    folds = np.array_split(sids, n_splits)

    for fold_id in range(n_splits):
        val_sids = set(folds[fold_id].tolist())
        val_keys = [k for k in keys if index[k]["sid"] in val_sids]
        tr_keys  = [k for k in keys if index[k]["sid"] not in val_sids]
        yield fold_id + 1, tr_keys, val_keys


def train_one_fold(
    fold_id: int,
    index,
    train_keys,
    val_keys,
    device,
    out_dir: str,
    batch_size: int = 8,
    epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    lambda_c: float = 0.2,
    temperature: float = 0.1,
    patience: int = 10
):
    os.makedirs(out_dir, exist_ok=True)

    train_ds = MultiModalEmotionDataset(index, train_keys, normalize=True, preload=True)
    val_ds   = MultiModalEmotionDataset(index, val_keys, normalize=True, preload=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2,
                              collate_fn=collate_fn, drop_last=True, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2,
                              collate_fn=collate_fn, pin_memory=True)

    model = MultiModalEmotionModel(
        patchtst_name=PATCHTST_NAME,
        cache_dir=CACHE_DIR,
        emb_dim=256,
        n_classes=4
    ).to(device)

    print("model param device:", next(model.parameters()).device)
    # PatchTST 在 model.encoder.model 里（你的结构是这样）
    print("encoder param device:", next(model.encoder.model.parameters()).device)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay)
    ce  = nn.CrossEntropyLoss()

    best_mf1 = -1.0
    bad = 0
    best_path = os.path.join(out_dir, f"best_fold{fold_id}.pt")

    disable_tqdm = not sys.stdout.isatty()

    epoch_bar = tqdm(range(1, epochs + 1), desc=f"Fold {fold_id} Epoch", disable=disable_tqdm)
    for ep in epoch_bar:
        model.train()
        losses = []

        batch_bar = tqdm(train_loader, desc=f"Fold {fold_id} Ep {ep}", leave=False, disable=disable_tqdm)
        for batch in batch_bar:
            ecg = batch["ecg"].to(device, non_blocking=True)
            eda = batch["eda"].to(device, non_blocking=True)
            ppg = batch["ppg"].to(device, non_blocking=True)
            y   = batch["label"].to(device, non_blocking=True)


            out = model({"ecg": ecg, "eda": eda, "ppg": ppg}, temperature=temperature)
            loss_cls = ce(out["logits"], y)
            loss = loss_cls + lambda_c * out["lc"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()

            losses.append(loss.item())
            batch_bar.set_postfix(loss=f"{loss.item():.4f}")

        metrics = evaluate(model, val_loader, device)
        mf1 = metrics["macro_f1"]
        tr_loss = float(np.mean(losses)) if losses else 0.0

        # 更新 epoch 进度条右侧信息
        epoch_bar.set_postfix(tr_loss=f"{tr_loss:.4f}", val_acc=f"{metrics['acc']:.3f}", val_mF1=f"{mf1:.3f}")

        # 用 tqdm.write 避免冲掉进度条
        tqdm.write(
            f"[Fold {fold_id}] Epoch {ep:03d} | train_loss={tr_loss:.4f} | "
            f"val_acc={metrics['acc']:.4f} | val_mF1={mf1:.4f}"
        )

        if mf1 > best_mf1:
            best_mf1 = mf1
            bad = 0
            torch.save({"model": model.state_dict(), "best_mf1": best_mf1}, best_path)
        else:
            bad += 1
            if bad >= patience:
                tqdm.write(f"[Fold {fold_id}] Early stopping at epoch {ep}. Best mF1={best_mf1:.4f}")
                break

    ckpt = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    final_metrics = evaluate(model, val_loader, device)
    final_metrics["best_path"] = best_path
    return final_metrics


def run_cv(
    data_root: str = "/root/project/multi_emo_rec/data",
    out_dir: str = "/root/project/multi_emo_rec/outputs",
    seed: int = 42,
    n_splits: int = 5
):
    seed_everything(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("==== Device Check ====")
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("selected device:", device)
    if torch.cuda.is_available():
        print("gpu name:", torch.cuda.get_device_name(0))
        print("current_device id:", torch.cuda.current_device())
    print("======================")


    index = build_index(data_root)
    keys = sorted(index.keys())

    all_metrics = []
    for fold_id, tr_keys, va_keys in split_by_subject(keys, index, n_splits=n_splits, seed=seed):
        fold_out = os.path.join(out_dir, f"fold_{fold_id}")
        metrics = train_one_fold(
            fold_id=fold_id,
            index=index,
            train_keys=tr_keys,
            val_keys=va_keys,
            device=device,
            out_dir=fold_out,
            batch_size=8,
            epochs=60,
            lr=1e-3,
            weight_decay=1e-4,
            lambda_c=0.2,
            temperature=0.1,
            patience=10
        )
        print(metrics["report"])
        print("Confusion Matrix:\n", metrics["cm"])
        all_metrics.append(metrics)

    accs = [m["acc"] for m in all_metrics]
    mf1s = [m["macro_f1"] for m in all_metrics]
    print(f"\nSubject-wise CV Summary ({n_splits}-fold): "
          f"acc={np.mean(accs):.4f}±{np.std(accs):.4f}, macroF1={np.mean(mf1s):.4f}±{np.std(mf1s):.4f}")


if __name__ == "__main__":
    run_cv()
