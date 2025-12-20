import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def detect_peak_change(signal, fs, approx_time, window=5.0):
    """
    signal: 1D array
    fs: sample rate (Hz)
    approx_time: approximate time (seconds after start)
    window: search window (±window seconds)
    """
    
    approx_idx = int(approx_time * fs)
    search_start = max(0, approx_idx - int(window * fs))
    search_end   = min(len(signal)-1, approx_idx + int(window * fs))
    
    # first-order difference in the search region
    diff = np.abs(np.diff(signal[search_start:search_end]))
    
    peak_rel = np.argmax(diff)
    peak_idx = search_start + peak_rel
    
    return peak_idx

# -------------------------------------------
# 1. 读取数据（如果你有 csv 文件替换成 read_csv）
# -------------------------------------------
df = pd.read_excel(r"E:\大四上学习资料\mmj_converted_sensor_data_final.xlsx")
# 2. 时间戳转换 + 插值
# -------------------------------------------
df['timestamp'] = pd.to_datetime(df['timestamp'], format="%H:%M:%S.%f")

# 以时间戳为索引
df = df.set_index('timestamp')

# 按时间排序（防止乱序）
df = df.sort_index()

# 插值（线性）
df_interp = df.interpolate(method='time')
#df_interp = df_interp.iloc[:2000]

approx_time = 13  # 预估突变时间（秒）
fs = 50          # 采样率（Hz）
signals = ['ax','ay','az','gx','gy','gz']
peak_list = []

plt.figure(figsize=(18, 10))

for i, col in enumerate(signals):
    sig = df_interp[col].values
    
    peak_idx = detect_peak_change(sig, fs, approx_time)
    peak_list.append(peak_idx)

    plt.subplot(2, 3, i+1)
    plt.plot(df_interp.index, sig, label=col.upper())
    plt.scatter(df_interp.index[peak_idx], sig[peak_idx], color='red', s=40, label='Detected Change')
    
    plt.title(f"{col.upper()}  (peak @ index {peak_idx})")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

peak_list = np.array(peak_list)
global_peak = int(np.median(peak_list))

plt.figure(figsize=(18, 10))

for i, col in enumerate(signals):
    sig = df_interp[col].values

    plt.subplot(2, 3, i+1)
    plt.plot(df_interp.index, sig)
    
    # 统一突变点标红
    plt.scatter(df_interp.index[global_peak], sig[global_peak], 
                color='red', s=40, label='Unified Change Point')
    
    plt.title(f"{col.upper()}")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()

# global_peak 是突变点对应的插值数据行号
cut_df = df_interp.iloc[global_peak:].copy()

# 保存 CSV（包含所有列：加速度、角速度、皮肤电、心率等）
save_path = r"E:\大四上学习资料\eda_after_impact.csv"
cut_df.to_csv(save_path)

print("已成功保存突变点后的完整数据到：", save_path)

'''
# -------------------------------------------
# 3. 绘制每项数据
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
    plt.subplot(2, 3, i+1)
    plt.plot(df_short.index, df_short[col])
    if col in cols_acc:
        plt.title(f"Acceleration {col.upper()}")
    else:
        plt.title(f"Gyroscope {col.upper()}")
    plt.grid(True)

plt.tight_layout()
plt.show()

print("插值完成！可视化已完成。")
'''