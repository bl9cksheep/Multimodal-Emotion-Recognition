# src/dataset.py
import os
import glob
import pandas as pd
import numpy as np
import torch
from typing import Dict, List
from torch.utils.data import Dataset

from .utils import (
    parse_from_filename, make_sample_key, pick_signal_column, zscore, DIGIT2CLASS
)


def build_index(data_root: str) -> Dict[str, Dict]:
    """
    Scan CSV files under data_root/{ecg,eda,ppg} and align three modalities by (sid, label_digit).

    Returns:
      index[sample_key] = {
        "sid": sid,
        "label_digit": label_digit,
        "label": class_id(0~3),
        "ecg": path, "eda": path, "ppg": path
      }
    """
    # Normalize path for cross-platform usage (safe for GitHub)
    data_root = os.path.abspath(os.path.expanduser(data_root))

    temp: Dict[str, Dict] = {}

    for modality in ["ecg", "eda", "ppg"]:
        paths = glob.glob(os.path.join(data_root, modality, "*.csv"))
        for p in paths:
            sid, m, label_digit = parse_from_filename(p)
            key = make_sample_key(sid, label_digit)
            cls = DIGIT2CLASS[label_digit]

            temp.setdefault(key, {})
            temp[key]["sid"] = sid
            temp[key]["label_digit"] = label_digit
            temp[key]["label"] = cls
            temp[key][modality] = p

    # Keep only samples that have all three modalities
    index = {
        k: v for k, v in temp.items()
        if all(mm in v for mm in ["ecg", "eda", "ppg"]) and "label" in v
    }
    return index


class MultiModalEmotionDataset(Dataset):
    def __init__(
        self,
        index: Dict[str, Dict],
        sample_keys: List[str],
        normalize: bool = True,
        ecg_len: int = 15000,   # 250 Hz * 60 s
        other_len: int = 3000,  # 50 Hz * 60 s
        preload: bool = True,   # NEW: whether to preload all data into memory
    ):
        self.index = index
        self.sample_keys = sample_keys
        self.normalize = normalize
        self.ecg_len = ecg_len
        self.other_len = other_len
        self.preload = preload

        # Cache: key -> dict(tensors)
        self.cache: Dict[str, Dict[str, torch.Tensor]] = {}

        if self.preload:
            for k in self.sample_keys:
                rec = self.index[k]
                ecg = self._read_1d_signal(rec["ecg"], self.ecg_len)
                eda = self._read_1d_signal(rec["eda"], self.other_len)
                ppg = self._read_1d_signal(rec["ppg"], self.other_len)

                self.cache[k] = {
                    "key": k,
                    "sid": torch.tensor(int(rec["sid"]), dtype=torch.long),
                    "ecg": torch.from_numpy(ecg).float(),   # (L,)
                    "eda": torch.from_numpy(eda).float(),   # (L,)
                    "ppg": torch.from_numpy(ppg).float(),   # (L,)
                    "label": torch.tensor(int(rec["label"]), dtype=torch.long),
                }

    def _read_1d_signal(self, path: str, target_len: int) -> np.ndarray:
        """
        Read a single-channel signal from a CSV, then:
          - align length by cropping / zero-padding
          - optionally apply z-score normalization

        Returns:
          numpy float32 array with shape (target_len,)
        """
        df = pd.read_csv(path)
        col = pick_signal_column(df)
        x = df[col].to_numpy(dtype=np.float32)

        # Fixed length: crop or pad with zeros
        if len(x) >= target_len:
            x = x[:target_len]
        else:
            x = np.pad(x, (0, target_len - len(x)), mode="constant")

        if self.normalize:
            x = zscore(x)

        return x

    def __len__(self):
        return len(self.sample_keys)

    def __getitem__(self, i: int):
        k = self.sample_keys[i]

        # Preloaded: return directly from memory
        if self.preload:
            return self.cache[k]

        # Otherwise: load on demand from disk (slower)
        rec = self.index[k]
        ecg = self._read_1d_signal(rec["ecg"], self.ecg_len)
        eda = self._read_1d_signal(rec["eda"], self.other_len)
        ppg = self._read_1d_signal(rec["ppg"], self.other_len)

        return {
            "key": k,
            "sid": torch.tensor(int(rec["sid"]), dtype=torch.long),
            "ecg": torch.from_numpy(ecg).float(),
            "eda": torch.from_numpy(eda).float(),
            "ppg": torch.from_numpy(ppg).float(),
            "label": torch.tensor(int(rec["label"]), dtype=torch.long),
        }
