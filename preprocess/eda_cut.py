import os
import re
import glob
import numpy as np
import pandas as pd
from io import StringIO

# ====== 参数 ======
FS = 50
ONE_MIN_SAMPLES = 60 * FS  # 3000

INPUT_DIR = r"E:\大四上学习资料\对齐\ppg"
OUT_EDA_DIR = r"E:\大四上学习资料\final\eda"
OUT_PPG_DIR = r"E:\大四上学习资料\final\ppg"
OUT_ACC_DIR = r"E:\大四上学习资料\final\acc"

# 4个时间窗（单位：秒），取中心1分钟
RANGES_SEC = [
    (0*60 + 50, 4*60 + 20),   # 0:50 - 4:20
    (5*60 + 10, 6*60 + 40),   # 5:10 - 6:40
    (7*60 + 24, 9*60 + 24),   # 7:24 - 9:24
    (9*60 + 40, 10*60 + 40),  # 9:40 - 10:40
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
        print(f"[补0] 窗口完全越界：[{i0},{i1})，信号长度={n}，输出全0")
        return out

    dst0 = src0 - i0
    dst1 = dst0 + (src1 - src0)
    out[dst0:dst1] = x[src0:src1]

    if i0 < 0 or i1 > n:
        print(f"[补0] 窗口越界：[{i0},{i1})，信号长度={n}，已补零")

    return out

def extract_window_pad0_2d(X: np.ndarray, win_start_sec: float) -> np.ndarray:
    """X shape: (N, C)，输出 (3000, C)，越界补0"""
    i0 = sec_to_idx(win_start_sec)
    i1 = i0 + ONE_MIN_SAMPLES

    n, c = X.shape
    out = np.zeros((ONE_MIN_SAMPLES, c), dtype=float)

    src0 = max(i0, 0)
    src1 = min(i1, n)
    if src1 <= src0:
        print(f"[补0] 窗口完全越界：[{i0},{i1})，信号长度={n}，输出全0(2D)")
        return out

    dst0 = src0 - i0
    dst1 = dst0 + (src1 - src0)
    out[dst0:dst1, :] = X[src0:src1, :]

    if i0 < 0 or i1 > n:
        print(f"[补0] 窗口越界：[{i0},{i1})，信号长度={n}，已补零(2D)")

    return out

def read_onecol_comma_xlsx(xlsx_path: str) -> pd.DataFrame:
    """
    兼容读取：
    1) 正常xlsx
    2) 伪xlsx（实际是文本/CSV但后缀是xlsx）
    3) xlsx里只有一列，每行是逗号分隔字符串
    """
    # 先尝试按Excel读取
    try:
        raw = pd.read_excel(xlsx_path, header=None, engine=None)
        # 如果只有一列，且每行像 "timestamp,skin,ax..." 这种逗号分隔文本
        if raw.shape[1] == 1:
            lines = raw.iloc[:, 0].dropna().astype(str).tolist()
            text = "\n".join(lines)
            return pd.read_csv(StringIO(text), sep=",")
        else:
            # 已经是多列结构，尝试用第一行当表头再读一次更合理
            return pd.read_excel(xlsx_path)
    except Exception:
        # 按纯文本读取（常见：文件根本不是excel）
        # 重要：用二进制读，再尝试utf-8/gbk解码，避免编码问题
        with open(xlsx_path, "rb") as f:
            b = f.read()

        for enc in ("utf-8-sig", "utf-8", "gbk"):
            try:
                s = b.decode(enc)
                # 直接按CSV解析
                return pd.read_csv(StringIO(s), sep=",")
            except Exception:
                pass

        # 都失败就抛出明确错误
        raise ValueError("文件既无法按Excel读取，也无法按文本CSV解析。可能文件损坏或不是逗号分隔格式。")

def to_numeric_series_keep_len(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns:
        raise KeyError(f"列不存在：{col}，实际列为：{list(df.columns)}")
    return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

def to_numeric_matrix_keep_len(df: pd.DataFrame, cols) -> np.ndarray:
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"列不存在：{c}，实际列为：{list(df.columns)}")
    mat = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return mat

def main():
    files = glob.glob(os.path.join(INPUT_DIR, "*_ppg_after.xlsx"))
    files = sorted(files, key=numeric_prefix_ppg)

    if not files:
        print("未找到输入文件：", os.path.join(INPUT_DIR, "*_ppg_after.xlsx"))
        return

    windows = [center_one_minute(a, b) for (a, b) in RANGES_SEC]

    print("将提取的4个中心一分钟窗口(秒)：")
    for k, (ws, we) in enumerate(windows, 1):
        print(f"  window{k}: {ws:.3f}s -> {we:.3f}s (60s, {ONE_MIN_SAMPLES} samples)")

    for path in files:
        base = os.path.basename(path)
        m = re.match(r"(\d+)_ppg_after\.xlsx$", base, flags=re.IGNORECASE)
        if not m:
            print(f"[跳过] 文件名不符合 *_ppg_after.xlsx：{base}")
            continue
        idx = m.group(1)

        try:
            df = read_onecol_comma_xlsx(path)
        except Exception as e:
            print(f"[失败] 读取/解析 {base}：{e}")
            continue

        try:
            eda = to_numeric_series_keep_len(df, "skin")
            ppg = to_numeric_series_keep_len(df, "spo2")   # 用血氧代替ppg原始波形
            acc6 = to_numeric_matrix_keep_len(df, ACC_COLS)  # (N,6)
        except Exception as e:
            print(f"[失败] {base} 取列失败：{e}")
            continue

        for k, (ws, we) in enumerate(windows, 1):
            eda_seg = extract_window_pad0_1d(eda, ws)          # (3000,)
            ppg_seg = extract_window_pad0_1d(ppg, ws)          # (3000,)
            acc_seg = extract_window_pad0_2d(acc6, ws)         # (3000,6)

            out_eda = os.path.join(OUT_EDA_DIR, f"{idx}_eda_final_{k}.csv")
            out_ppg = os.path.join(OUT_PPG_DIR, f"{idx}_ppg_final_{k}.csv")
            out_acc = os.path.join(OUT_ACC_DIR, f"{idx}_acc_final_{k}.csv")

            pd.DataFrame(eda_seg).to_csv(out_eda, index=False, header=False)
            pd.DataFrame(ppg_seg).to_csv(out_ppg, index=False, header=False)
            pd.DataFrame(acc_seg).to_csv(out_acc, index=False, header=False)

        print(f"[完成] {base} -> eda/ppg/acc 各输出 4 个窗口")

if __name__ == "__main__":
    main()
