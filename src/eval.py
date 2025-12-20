# src/eval.py
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from .utils import ID2LABEL

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []

    for batch in loader:
        ecg = batch["ecg"].to(device, non_blocking=True).float()
        eda = batch["eda"].to(device, non_blocking=True).float()
        ppg = batch["ppg"].to(device, non_blocking=True).float()
        y   = batch["label"].to(device, non_blocking=True).long()
        out = model({"ecg": ecg, "eda": eda, "ppg": ppg})
        pred = torch.argmax(out["logits"], dim=-1)

        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)

    acc = accuracy_score(ys, ps)
    mf1 = f1_score(ys, ps, average="macro")
    cm  = confusion_matrix(ys, ps)
    report = classification_report(
        ys, ps,
        target_names=[ID2LABEL[i] for i in range(4)],
        digits=4,
        zero_division=0
    )
    return {"acc": acc, "macro_f1": mf1, "cm": cm, "report": report}
