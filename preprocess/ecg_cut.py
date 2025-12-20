import os
import re
import glob
import numpy as np
import pandas as pd

# ====== 参数 ======
FS = 250
ONE_MIN_SAMPLES = 60 * FS

INPUT_DIR = r"E:\大四上学习资料\对齐\ecg"
OUTPUT_DIR = r"E:\大四上学习资料\final"

# 你的四个时间窗（单位：秒），都会取“中心的一分钟”
RANGES_SEC = [
    (0*60 + 50, 4*60 + 20),   # 0:50 - 4:20
    (5*60 + 10, 6*60 + 40),   # 5:10 - 6:40
    (7*60 + 24, 9*60 + 24),   # 7:24 - 9:24
    (9*60 + 40, 10*60 + 40), # 9:40 - 10:40
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def numeric_prefix(path: str) -> int:
    """用于排序：从文件名提取开头数字，比如 12_ecg_after.csv -> 12"""
    base = os.path.basename(path)
    m = re.match(r"(\d+)_ecg_after\.csv$", base, flags=re.IGNORECASE)
    return int(m.group(1)) if m else 10**18

def read_single_column_ecg(csv_path: str) -> np.ndarray:
    """
    读取单列 ECG：
    - 允许第一行是表头 'ECG'
    - 输出纯数值数组
    """
    # 常规情况：第一行是表头 ECG
    df = pd.read_csv(csv_path)

    # 若读出来不是1列（极少数异常），尝试更鲁棒的方式
    if df.shape[1] != 1:
        df = pd.read_csv(csv_path, header=None)

    s = df.iloc[:, 0]
    s = pd.to_numeric(s, errors="coerce").dropna()
    return s.to_numpy(dtype=float)

def center_one_minute(start_sec: float, end_sec: float):
    """给定区间起止秒，返回中心1分钟的 [start_sec, end_sec)"""
    mid = (start_sec + end_sec) / 2.0
    win_start = mid - 30.0
    win_end = win_start + 60.0
    return win_start, win_end

def sec_to_idx(sec: float) -> int:
    """秒 -> 采样点索引（四舍五入）"""
    return int(round(sec * FS))

def extract_window(x: np.ndarray, win_start_sec: float) -> np.ndarray:
    """从信号中截取 1分钟(15000点)，起点给定为秒；越界时用0补齐"""
    i0 = sec_to_idx(win_start_sec)
    i1 = i0 + ONE_MIN_SAMPLES

    n = len(x)
    out = np.zeros(ONE_MIN_SAMPLES, dtype=float)  # 默认全0

    # 需要从原信号拷贝的有效区间：[src0, src1)
    src0 = max(i0, 0)
    src1 = min(i1, n)

    # 如果完全没有重叠，直接返回全0
    if src1 <= src0:
        return out

    # out 中对应写入位置：[dst0, dst1)
    dst0 = src0 - i0
    dst1 = dst0 + (src1 - src0)

    out[dst0:dst1] = x[src0:src1]

    if i0 < 0 or i1 > n:
        print(f"[补0] 窗口越界：[{i0},{i1})，信号长度={n}，已补零")

    return out


def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*_ecg_after.csv"))
    files = sorted(files, key=numeric_prefix)

    if not files:
        print("未找到输入文件：", os.path.join(INPUT_DIR, "*_ecg_after.csv"))
        return

    # 预计算四个中心一分钟窗口的起点秒（便于检查）
    windows = []
    for (a, b) in RANGES_SEC:
        ws, we = center_one_minute(a, b)
        windows.append((ws, we))

    print("将提取的4个中心一分钟窗口(秒)：")
    for k, (ws, we) in enumerate(windows, 1):
        print(f"  window{k}: {ws:.3f}s -> {we:.3f}s  (len=60s)")

    for path in files:
        base = os.path.basename(path)
        m = re.match(r"(\d+)_ecg_after\.csv$", base, flags=re.IGNORECASE)
        if not m:
            print(f"[跳过] 文件名不符合 *_ecg_after.csv：{base}")
            continue
        idx = m.group(1)

        try:
            x = read_single_column_ecg(path)
        except Exception as e:
            print(f"[失败] 读取 {base}：{e}")
            continue

        for k, (ws, we) in enumerate(windows, 1):
            try:
                seg = extract_window(x, ws)
            except Exception as e:
                print(f"[失败] {base} window{k}：{e}")
                continue

            out_name = f"{idx}_ecg_final_{k}.csv"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            # 保存为单列，无表头
            pd.DataFrame(seg).to_csv(out_path, index=False, header=False)

        print(f"[完成] {base} -> {idx}_ecg_final_1~4.csv")

if __name__ == "__main__":
    main()
