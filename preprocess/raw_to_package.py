import re
import os

# --- 常量定义 ---
# 文件路径
FILE_PATH = "/Users/dongyang/Desktop/课程专设/数据/SaveWindows2025_11_11_10-16-19.txt"
OUTPUT_FILENAME = "split_sensor_packets_variable_length.txt"

# --- 主程序 ---

def split_and_save_packets_variable(file_path):
    """
    读取文件，将十六进制数据流按 FA...AF 边界拆分成独立数据包（可变长度），并保存到新文件。
    """
    print(f"Reading data from: {file_path}")
    
    try:
        # 使用 'gbk' 编码读取文件
        with open(file_path, 'r', encoding='gbk') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The file was not found at {file_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    # 1. 预处理：创建时间戳映射和连续数据流
    timestamps_map = [] # 存储 (hex_start_index, timestamp)
    full_hex_data = "" # 存储所有行中的纯十六进制数据

    for line in lines:
        match = re.search(r"\[(.*?)\]收←◆(.*)", line)
        if match:
            timestamp = match.group(1).strip()
            # 移除数据中的所有空格并追加到完整数据流
            hex_str = match.group(2).replace(' ', '').strip()
            
            # 记录当前行的 hex_str 在 full_hex_data 中的起始位置
            timestamps_map.append((len(full_hex_data), timestamp))
            full_hex_data += hex_str
    
    # 2. 查找和提取数据包（使用非贪婪匹配）
    processed_packets = []
    # 正则表达式: FA (帧头) + (.*?) (非贪婪匹配所有字符) + AF (帧尾)
    packet_pattern = r"(FA.*?AF)"

    # 使用 finditer 在连续数据流中找到所有匹配项
    for match in re.finditer(packet_pattern, full_hex_data):
        full_packet_hex = match.group(1) # 完整的 FA...AF 字符串
        start_index = match.start()
        
        # 确定关联的时间戳
        # 找到最近的、起始索引小于或等于数据包起始索引的时间戳
        # 默认使用第一个时间戳
        associated_timestamp = timestamps_map[0][1] 
        for ts_start_index, timestamp in timestamps_map:
            if ts_start_index <= start_index:
                associated_timestamp = timestamp
            else:
                # 列表按 start_index 排序，可以提前退出
                break 
        
        # 格式化输出：[时间戳] + 十六进制数据包
        # 计算长度，并每隔 2 个字符加一个空格
        hex_len = len(full_packet_hex)
        spaced_packet = ' '.join(full_packet_hex[i:i+2] for i in range(0, hex_len, 2))
        
        output_line = f"[{associated_timestamp}] {spaced_packet}\n"
        processed_packets.append(output_line)

    if not processed_packets:
        print("Warning: No valid FA...AF data packets were found using variable length matching.")
        return

    # 3. 写入新的文本文件
    output_dir = os.path.dirname(file_path)
    if not output_dir:
        output_dir = "."
        
    output_path = os.path.join(output_dir, OUTPUT_FILENAME)
    
    try:
        # 使用 utf-8 编码写入
        with open(output_path, 'w', encoding='utf-8') as outfile:
            outfile.writelines(processed_packets)
            
        print(f"Successfully split and saved {len(processed_packets)} data packets (variable length).")
        print(f"Output saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: Failed to save data to file at {output_path}. Error: {e}")


# --- 程序执行部分 ---
if __name__ == "__main__":
    split_and_save_packets_variable(FILE_PATH)