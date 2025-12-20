import os
import re
import glob
import numpy as np
import pandas as pd

# ====== Parameters ======
FS = 250
ONE_MIN_SAMPLES = 60 * FS  # 15000 samples per minute at 250 Hz

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "..", "data", "ecg")        # e.g., repo/data/ecg
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "outputs", "ecg")    # e.g., repo/outputs/ecg

# Four time ranges (seconds). We extract the centered 1-minute window for each range.
RANGES_SEC = [
    (0 * 60 + 50, 4 * 60 + 20),    # 0:50 - 4:20
    (5 * 60 + 10, 6 * 60 + 40),    # 5:10 - 6:40
    (7 * 60 + 24, 9 * 60 + 24),    # 7:24 - 9:24
    (9 * 60 + 40, 10 * 60 + 40),   # 9:40 - 10:40
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def numeric_prefix(path: str) -> int:
    """For sorting: extract the leading number from filename, e.g., 12_ecg_after.csv -> 12."""
    base = os.path.basename(path)
    m = re.match(r"(\d+)_ecg_after\.csv$", base, flags=re.IGNORECASE)
    return int(m.group(1)) if m else 10**18

def read_single_column_ecg(csv_path: str) -> np.ndarray:
    """
    Read single-column ECG:
    - Allows the first row to be a header like 'ECG'
    - Returns a pure numeric numpy array
    """
    # Common case: first row is a header (e.g., 'ECG')
    df = pd.read_csv(csv_path)

    # If it is not exactly 1 column (rare), fall back to a more robust read
    if df.shape[1] != 1:
        df = pd.read_csv(csv_path, header=None)

    s = df.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.to_numpy(dtype=float)

def center_one_minute(start_sec: float, end_sec: float):
    """Given a time interval in seconds, return the centered 1-minute window [start, end)."""
    mid = (start_sec + end_sec) / 2.0
    win_start = mid - 30.0
    win_end = win_start + 60.0
    return win_start, win_end

def sec_to_idx(sec: float) -> int:
    """Seconds -> sample index (rounded)."""
    return int(round(sec * FS))

def extract_window(x: np.ndarray, win_start_sec: float) -> np.ndarray:
    """Extract a 1-minute segment (15000 samples). Out-of-bounds parts are padded with zeros."""
    i0 = sec_to_idx(win_start_sec)
    i1 = i0 + ONE_MIN_SAMPLES

    n = len(x)
    out = np.zeros(ONE_MIN_SAMPLES, dtype=float)  # default all zeros

    # Valid overlap in source signal: [src0, src1)
    src0 = max(i0, 0)
    src1 = min(i1, n)

    # If there is no overlap, return all zeros
    if src1 <= src0:
        return out

    # Target write region in output: [dst0, dst1)
    dst0 = src0 - i0
    dst1 = dst0 + (src1 - src0)

    out[dst0:dst1] = x[src0:src1]

    if i0 < 0 or i1 > n:
        print(f"[Zero-pad] Window out of bounds: [{i0},{i1}), signal length={n}, padded with zeros")

    return out

def main():
    pattern = os.path.join(INPUT_DIR, "*_ecg_after.csv")
    files = glob.glob(pattern)
    files = sorted(files, key=numeric_prefix)

    if not files:
        print("No input files found:", pattern)
        return

    # Pre-compute the four centered 1-minute windows (for sanity check)
    windows = [center_one_minute(a, b) for (a, b) in RANGES_SEC]

    print("The 4 centered 1-minute windows to extract (seconds):")
    for k, (ws, we) in enumerate(windows, 1):
        print(f"  window{k}: {ws:.3f}s -> {we:.3f}s (len=60s)")

    for path in files:
        base = os.path.basename(path)
        m = re.match(r"(\d+)_ecg_after\.csv$", base, flags=re.IGNORECASE)
        if not m:
            print(f"[Skip] Filename does not match *_ecg_after.csv: {base}")
            continue
        idx = m.group(1)

        try:
            x = read_single_column_ecg(path)
        except Exception as e:
            print(f"[Failed] Reading {base}: {e}")
            continue

        for k, (ws, we) in enumerate(windows, 1):
            try:
                seg = extract_window(x, ws)
            except Exception as e:
                print(f"[Failed] {base} window{k}: {e}")
                continue

            out_name = f"{idx}_ecg_final_{k}.csv"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            # Save as a single column, no header
            pd.DataFrame(seg).to_csv(out_path, index=False, header=False)

        print(f"[Done] {base} -> {idx}_ecg_final_1~4.csv")

if __name__ == "__main__":
    main()


