# src/utils.py
import os, re, random
import numpy as np
import torch

# 你定义：1=悲伤，2=平静，3=开心，4=愤怒
DIGIT2CLASS = {1: 0, 2: 1, 3: 2, 4: 3}
ID2LABEL = {0: "sad", 1: "calm", 2: "happy", 3: "angry"}

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_from_filename(path: str):
    """
    解析:
      1_ecg_final_1.csv  -> sid=1, modality=ecg, label_digit=1
    """
    fn = os.path.basename(path)
    m = re.match(r"(\d+)_([a-zA-Z]+)_final_(\d+)\.csv$", fn)
    if m is None:
        raise ValueError(f"Bad filename format: {fn}")
    sid = int(m.group(1))
    modality = m.group(2).lower()
    label_digit = int(m.group(3))
    if label_digit not in DIGIT2CLASS:
        raise ValueError(f"Label digit must be 1~4, got {label_digit} in {fn}")
    return sid, modality, label_digit

def make_sample_key(sid: int, label_digit: int) -> str:
    # 每个人每个情绪是一条样本
    return f"{sid}_L{label_digit}"

def zscore(x: np.ndarray, eps: float = 1e-6):
    mu = float(x.mean())
    sd = float(x.std())
    return (x - mu) / (sd + eps)

def pick_signal_column(df):
    """
    从csv中挑信号列：
    - 排除 timestamp/time/ts
    - 优先选第一个剩余列
    """
    cols = list(df.columns)
    drop = {"timestamp", "time", "ts"}
    cand = [c for c in cols if str(c).lower() not in drop]
    if not cand:
        raise ValueError(f"No valid signal column found. columns={cols}")
    return cand[0]
