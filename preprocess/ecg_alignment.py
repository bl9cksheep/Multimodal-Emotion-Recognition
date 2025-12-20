import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ====== Paths (project-relative, safe for GitHub) ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input / output paths (adjust to your repo layout if needed)
INPUT_CSV = os.path.join(BASE_DIR, "..", "data", "mmj_ecg.csv")         # e.g., repo/data/mmj_ecg.csv
OUTPUT_CSV = os.path.join(BASE_DIR, "..", "outputs", "ecg_after_impact.csv")  # e.g., repo/outputs/ecg_after_impact.csv
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# ====== 1. Read data ======
df = pd.read_csv(INPUT_CSV)
ecg = df.iloc[:, 1].values  # ECG (2nd column)
# ecg = ecg[:250 * 20]       # Keep first 20 seconds (5000 samples at 250 Hz), optional

# ====== 2. Smooth ECG ======
ecg_smooth = savgol_filter(ecg, window_length=31, polyorder=3)

# ====== 3. First-order difference to emphasize abrupt changes ======
diff = np.abs(np.diff(ecg_smooth))

# ====== 4. Adaptive thresholding to detect the impact point ======
k = 6
threshold = np.mean(diff) + k * np.std(diff)
candidate_indices = np.where(diff > threshold)[0]

if len(candidate_indices) > 0:
    impact_point = int(candidate_indices[0])
    print("Detected impact point index:", impact_point)
else:
    raise ValueError("No impact point detected. Try adjusting k.")

# ====== 5. Crop ECG after the impact point ======
ecg_after = ecg[impact_point:]

# ====== 6. Save to CSV (ECG only) ======
pd.DataFrame(ecg_after, columns=["ECG"]).to_csv(OUTPUT_CSV, index=False)
print("Saved ECG after impact to:", OUTPUT_CSV)

# ====== 7. Plot for verification ======
plt.figure(figsize=(12, 4))
plt.plot(ecg, label="ECG")
plt.axvline(impact_point, color="r", linestyle="--", label="Detected Impact")
plt.legend()
plt.title("Detected Impact Point")
plt.xlabel("Sample Index")
plt.show()
