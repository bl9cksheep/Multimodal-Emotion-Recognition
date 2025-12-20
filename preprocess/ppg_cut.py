import os
import re
import glob
import numpy as np
import pandas as pd
from io import StringIO

# ====== Parameters ======
FS = 50
ONE_MIN_SAMPLES = 60 * FS  # 3000 samples per minute

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "ppg")           # e.g., repo/data/ppg
OUT_EDA_DIR = os.path.join(BASE_DIR, "..", "outputs", "eda")     # e.g., repo/outputs/eda
OUT_PPG_DIR = os.path.join(BASE_DIR, "..", "outputs", "ppg")     # e.g., repo/outputs/ppg
OUT_ACC_DIR = os.path.join(BASE_DIR, "..", "outputs", "acc")     # e.g., repo/outputs/acc

# 4 time ranges (seconds). Extract the centered 1-minute window in each range.
RANGES_SEC = [
    (0 * 60 + 50, 4 * 60 + 20),    # 0:50 - 4:20
    (5 * 60 + 10, 6 * 60 + 40),    # 5:10 - 6:40
    (7 * 60 + 24, 9 * 60 + 24),    # 7:24 - 9:24
    (9 * 60 + 40, 10 * 60 + 40),   # 9:40 - 10:40
]

os.makedirs(OUT_EDA_DIR, exist_ok=True)
os.makedirs(OUT_PPG_DIR, exist_ok=True)
os.makedirs(OUT_ACC_DIR, exist_ok=True)

ACC_COLS = ["ax", "ay", "az", "gx", "gy", "gz"]

def numeric_prefix_ppg(path: str) -> int:
    base = os.path.basename(path)
    m = re.match(r"(\d+)_ppg_after\.xlsx$", base, flags=re.IGNORECASE)
    return int(m.group(1)) if m else 10**18

def center_one_minute(start_sec: float, end_sec: float):
    mid = (start_sec + end_sec) / 2.0
    win_start = mid - 30.0
    win_end = win_start + 60.0
    return win_start, win_end

def sec_to_idx(sec: float) -> int:
    return int(round(sec * FS))

def extract_window_pad0_1d(x: np.ndarray, win_start_sec: float) -> np.ndarray:
    i0 = sec_to_idx(win_start_sec)
    i1 = i0 + ONE_MIN_SAMPLES

    n = len(x)
    out = np.zeros(ONE_MIN_SAMPLES, dtype=float)

    src0 = max(i0, 0)
    src1 = min(i1, n)
    if src1 <= src0:
        print(f"[Zero-pad] Window fully out of bounds: [{i0},{i1}), signal length={n}, output all zeros")
        return out

    dst0 = src0 - i0
    dst1 = dst0 + (src1 - src0)
    out[dst0:dst1] = x[src0:src1]

    if i0 < 0 or i1 > n:
        print(f"[Zero-pad] Window out of bounds: [{i0},{i1}), signal length={n}, padded with zeros")

    return out

def extract_window_pad0_2d(X: np.ndarray, win_start_sec: float) -> np.ndarray:
    """X shape: (N, C). Output: (3000, C). Out-of-bounds parts are padded with zeros."""
    i0 = sec_to_idx(win_start_sec)
    i1 = i0 + ONE_MIN_SAMPLES

    n, c = X.shape
    out = np.zeros((ONE_MIN_SAMPLES, c), dtype=float)

    src0 = max(i0, 0)
    src1 = min(i1, n)
    if src1 <= src0:
        print(f"[Zero-pad] Window fully out of bounds: [{i0},{i1}), signal length={n}, output all zeros (2D)")
        return out

    dst0 = src0 - i0
    dst1 = dst0 + (src1 - src0)
    out[dst0:dst1, :] = X[src0:src1, :]

    if i0 < 0 or i1 > n:
        print(f"[Zero-pad] Window out of bounds: [{i0},{i1}), signal length={n}, padded with zeros (2D)")

    return out

def read_onecol_comma_xlsx(xlsx_path: str) -> pd.DataFrame:
    """
    Compatible reading for:
    1) Normal .xlsx files
    2) Fake .xlsx (actually text/CSV but named .xlsx)
    3) .xlsx with only one column, where each row is a comma-separated string
    """
    # First try reading as Excel
    try:
        raw = pd.read_excel(xlsx_path, header=None, engine=None)
        # If there's only one column and each row looks like "timestamp,skin,ax..." (comma-separated text)
        if raw.shape[1] == 1:
            lines = raw.iloc[:, 0].dropna().astype(str).tolist()
            text = "\n".join(lines)
            return pd.read_csv(StringIO(text), sep=",")
        else:
            # Already multi-column; reading again with the first row as header is usually more reasonable
            return pd.read_excel(xlsx_path)
    except Exception:
        # Fall back to plain-text reading (common: file is not an actual Excel file)
        # Important: read as bytes first, then try utf-8/gbk decoding to avoid encoding issues
        with open(xlsx_path, "rb") as f:
            b = f.read()

        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                s = b.decode(enc)
                # Parse directly as CSV
                return pd.read_csv(StringIO(s), sep=",")
            except Exception:
                pass

        # If all attempts fail, raise a clear error
        raise ValueError("Cannot read as Excel or parse as CSV text. The file may be corrupted or not comma-separated.")

def to_numeric_series_keep_len(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"Column not found: {col}. Existing columns: {list(df.columns)}")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

def to_numeric_matrix_keep_len(df: pd.DataFrame, cols) -> np.ndarray:
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Column not found: {c}. Existing columns: {list(df.columns)}")
    mat = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return mat

def main():
    pattern = os.path.join(INPUT_DIR, "*_ppg_after.xlsx")
    files = glob.glob(pattern)
    files = sorted(files, key=numeric_prefix_ppg)

    if not files:
        print("No input files found:", pattern)
        return

    windows = [center_one_minute(a, b) for (a, b) in RANGES_SEC]

    print("The 4 centered 1-minute windows to extract (seconds):")
    for k, (ws, we) in enumerate(windows, 1):
        print(f"  window{k}: {ws:.3f}s -> {we:.3f}s (60s, {ONE_MIN_SAMPLES} samples)")

    for path in files:
        base = os.path.basename(path)
        m = re.match(r"(\d+)_ppg_after\.xlsx$", base, flags=re.IGNORECASE)
        if not m:
            print(f"[Skip] Filename does not match *_ppg_after.xlsx: {base}")
            continue
        idx = m.group(1)

        try:
            df = read_onecol_comma_xlsx(path)
        except Exception as e:
            print(f"[Failed] Read/parse {base}: {e}")
            continue

        try:
            eda = to_numeric_series_keep_len(df, "skin")
            ppg = to_numeric_series_keep_len(df, "spo2")      # Use SpO2 as a proxy for the PPG raw waveform
            acc6 = to_numeric_matrix_keep_len(df, ACC_COLS)   # (N, 6)
        except Exception as e:
            print(f"[Failed] Column extraction failed for {base}: {e}")
            continue

        for k, (ws, we) in enumerate(windows, 1):
            eda_seg = extract_window_pad0_1d(eda, ws)          # (3000,)
            ppg_seg = extract_window_pad0_1d(ppg, ws)          # (3000,)
            acc_seg = extract_window_pad0_2d(acc6, ws)         # (3000, 6)

            out_eda = os.path.join(OUT_EDA_DIR, f"{idx}_eda_final_{k}.csv")
            out_ppg = os.path.join(OUT_PPG_DIR, f"{idx}_ppg_final_{k}.csv")
            out_acc = os.path.join(OUT_ACC_DIR, f"{idx}_acc_final_{k}.csv")

            pd.DataFrame(eda_seg).to_csv(out_eda, index=False, header=False)
            pd.DataFrame(ppg_seg).to_csv(out_ppg, index=False, header=False)
            pd.DataFrame(acc_seg).to_csv(out_acc, index=False, header=False)

        print(f"[Done] {base} -> exported 4 windows each for eda/ppg/acc")

if __name__ == "__main__":
    main()
