import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# === 1. 读取数据 ===
df = pd.read_csv(r"E:\大四上学习资料\mmj_ecg.csv")
ecg = df.iloc[:, 1].values     # ECG（第二列）
#ecg = ecg[:250*20]             # 取前 20 秒（5000 点）

# === 2. 平滑 ECG ===
ecg_smooth = savgol_filter(ecg, window_length=31, polyorder=3)

# === 3. 一阶差分增强突变 ===
diff = np.abs(np.diff(ecg_smooth))

# === 4. 自适应阈值检测跳变点 ===
k = 6
threshold = np.mean(diff) + k * np.std(diff)
candidate_indices = np.where(diff > threshold)[0]

if len(candidate_indices) > 0:
    impact_point = candidate_indices[0]
    print("突变点位置:", impact_point)
else:
    raise ValueError("未检测到突变，请尝试调整 k 值")

# === 5. 截取突变点之后的数据 ===
ecg_after = ecg[impact_point:]

# === 6. 保存为 CSV（只保存 ECG）===
output_path = r"E:\大四上学习资料\ecg_after_impact.csv"
pd.DataFrame(ecg_after, columns=["ECG"]).to_csv(output_path, index=False)

print("已保存突变后的 ECG 到：", output_path)

# === 7. 画图验证 ===
plt.figure(figsize=(12,4))
plt.plot(ecg, label="ECG")
plt.axvline(impact_point, color='r', linestyle='--', label="Detected Impact")
plt.legend()
plt.title("Detected Impact Point")
plt.xlabel("Sample Index")
plt.show()