import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def detect_peak_change(signal, fs, approx_time, window=5.0):
    """
    Detect an abrupt change point around an approximate time by maximizing
    the first-order difference within a search window.

    Args:
        signal: 1D numpy array
        fs: sampling rate (Hz)
        approx_time: approximate time (seconds after start)
        window: search half-window (±window seconds)

    Returns:
        peak_idx: index of the detected change point in the original signal
    """
    approx_idx = int(approx_time * fs)
    search_start = max(0, approx_idx - int(window * fs))
    search_end = min(len(signal) - 1, approx_idx + int(window * fs))

    # First-order absolute difference in the search region
    diff = np.abs(np.diff(signal[search_start:search_end]))

    peak_rel = int(np.argmax(diff))
    peak_idx = search_start + peak_rel
    return peak_idx

# ====== Paths (project-relative, safe for GitHub) ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input / output paths (adjust to your repo layout if needed)
INPUT_XLSX = os.path.join(BASE_DIR, "..", "data", "mmj_converted_sensor_data_final.xlsx")
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "outputs", "eda_after_impact.csv")
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# -------------------------------------------
# 1) Read data (replace with read_csv if your file is CSV)
# -------------------------------------------
df = pd.read_excel(INPUT_XLSX)

# -------------------------------------------
# 2) Timestamp conversion + time interpolation
# -------------------------------------------
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%H:%M:%S.%f")

# Use timestamp as the index
df = df.set_index("timestamp")

# Sort by time (to avoid out-of-order timestamps)
df = df.sort_index()

# Time-based interpolation (linear in time)
df_interp = df.interpolate(method="time")
# df_interp = df_interp.iloc[:2000]  # optional: truncate for quick testing

approx_time = 13   # approximate change time (seconds)
fs = 50            # sampling rate (Hz)
signals = ["ax", "ay", "az", "gx", "gy", "gz"]
peak_list = []

# -------------------------------------------
# 3) Detect per-axis change points and visualize
# -------------------------------------------
plt.figure(figsize=(18, 10))

for i, col in enumerate(signals):
    sig = df_interp[col].values

    peak_idx = detect_peak_change(sig, fs, approx_time)
    peak_list.append(peak_idx)

    plt.subplot(2, 3, i + 1)
    plt.plot(df_interp.index, sig, label=col.upper())
    plt.scatter(df_interp.index[peak_idx], sig[peak_idx],
                color="red", s=40, label="Detected Change")

    plt.title(f"{col.upper()} (peak @ index {peak_idx})")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# Use the median of detected peaks as a unified change point
peak_list = np.array(peak_list)
global_peak = int(np.median(peak_list))

# -------------------------------------------
# 4) Visualize unified change point across all axes
# -------------------------------------------
plt.figure(figsize=(18, 10))

for i, col in enumerate(signals):
    sig = df_interp[col].values

    plt.subplot(2, 3, i + 1)
    plt.plot(df_interp.index, sig)

    # Mark the unified change point in red
    plt.scatter(df_interp.index[global_peak], sig[global_peak],
                color="red", s=40, label="Unified Change Point")

    plt.title(f"{col.upper()}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# global_peak is the row index in the interpolated dataframe
cut_df = df_interp.iloc[global_peak:].copy()

# -------------------------------------------
# 5) Save cropped data (all columns preserved)
# -------------------------------------------
cut_df.to_csv(OUTPUT_CSV)
print("Saved full data after the change point to:", OUTPUT_CSV)

"""
# -------------------------------------------
# (Optional) Additional plots
# -------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(df_short.index, df_short['skin'])
plt.title("Skin Conductance")
plt.grid(True)
plt.show()

cols_acc = ['ax', 'ay', 'az']
cols_gyro = ['gx', 'gy', 'gz']
all_cols = cols_acc + cols_gyro

plt.figure(figsize=(18, 8))

for i, col in enumerate(all_cols):
    plt.subplot(2, 3, i + 1)
    plt.plot(df_short.index, df_short[col])
    if col in cols_acc:
        plt.title(f"Acceleration {col.upper()}")
    else:
        plt.title(f"Gyroscope {col.upper()}")
    plt.grid(True)

plt.tight_layout()
plt.show()

print("Interpolation complete! Visualization done.")
"""
