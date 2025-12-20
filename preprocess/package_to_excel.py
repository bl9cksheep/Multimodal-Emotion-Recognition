import pandas as pd
import re
import os

# --- 常量定义 ---
# 假设已分包文件路径（请根据实际情况修改 FILE_DIR）
FILE_DIR = "/Users/dongyang/Desktop/课程专设/数据/"
INPUT_FILENAME = "split_sensor_packets_variable_length.txt"
OUTPUT_FILENAME = "converted_sensor_data_final.xlsx"

# 最终确认的完整数据包标准：19 字节 (FA + 17 数据字节 + AF)
EXPECTED_BYTE_COUNT = 19 
EXPECTED_HEX_CHAR_COUNT = EXPECTED_BYTE_COUNT * 2 # 38 个十六进制字符

INPUT_FILE_PATH = os.path.join(FILE_DIR, INPUT_FILENAME)
OUTPUT_FILE_PATH = os.path.join(FILE_DIR, OUTPUT_FILENAME)

# 数据转换参数 (保持不变)
GSR_MAX_ADC = 4095.0
GSR_MAX_VOLTAGE = 3.3
ACC_MAX_ADC = 32768.0
ACC_MAX_RANGE = 2.0 
GYRO_MAX_ADC = 32768.0
GYRO_MAX_RANGE = 250.0 

# --- 辅助函数 (保持不变) ---
def hex_to_signed_int(hex_str):
    """将16进制字符串转换为有符号的16位整数（十进制）。"""
    val = int(hex_str, 16)
    if val & 0x8000:
        return val - 65536
    return val

def convert_data(data):
    """将提取的十六进制数据转换为最终的传感器值。"""
    
    # 1. 皮肤电 (GSR) - 无符号16位
    gsr_raw = int(data["GSR"], 16) 
    gsr_voltage = (gsr_raw / GSR_MAX_ADC) * GSR_MAX_VOLTAGE

    # 2. 加速度 (AccelX/Y/Z) - 有符号16位
    accel_x_raw = hex_to_signed_int(data["AccelX"])
    accel_y_raw = hex_to_signed_int(data["AccelY"])
    accel_z_raw = hex_to_signed_int(data["AccelZ"])
    accel_x_g = (accel_x_raw / ACC_MAX_ADC) * ACC_MAX_RANGE
    accel_y_g = (accel_y_raw / ACC_MAX_ADC) * ACC_MAX_RANGE
    accel_z_g = (accel_z_raw / ACC_MAX_ADC) * ACC_MAX_RANGE
    
    # 3. 角速度 (GyroX/Y/Z) - 有符号16位
    gyro_x_raw = hex_to_signed_int(data["GyroX"])
    gyro_y_raw = hex_to_signed_int(data["GyroY"])
    gyro_z_raw = hex_to_signed_int(data["GyroZ"])
    gyro_x_s = (gyro_x_raw / GYRO_MAX_ADC) * GYRO_MAX_RANGE
    gyro_y_s = (gyro_y_raw / GYRO_MAX_ADC) * GYRO_MAX_RANGE
    gyro_z_s = (gyro_z_raw / GYRO_MAX_ADC) * GYRO_MAX_RANGE
    
    # 4. 心率 (HR) 和 血氧 (SpO2) - 无符号8位
    hr_val = int(data["HR"], 16) 
    spo2_val = int(data["SpO2"], 16) 

    return {
        "皮肤电": round(gsr_voltage, 4), "加速度x": round(accel_x_g, 4), "加速度y": round(accel_y_g, 4), "加速度z": round(accel_z_g, 4),
        "角速度x": round(gyro_x_s, 4), "角速度y": round(gyro_y_s, 4), "角速度z": round(gyro_z_s, 4),
        "心率": hr_val, "血氧": spo2_val
    }

# --- 主程序 ---

def process_converted_data(input_path, output_path):
    """
    读取已拆分的 TXT 文件，进行严格的 19 字节完整性检查和数据转换。
    """
    print(f"Reading data from: {input_path}")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: The input file was not found at {input_path}")
        return
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return

    all_converted_data = []
    empty_data_fields = {
        "皮肤电": None, "加速度x": None, "加速度y": None, "加速度z": None,
        "角速度x": None, "角速度y": None, "角速度z": None, "心率": None, "血氧": None
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = re.search(r"\[(.*?)\] (.*)", line)
        if not match:
            continue

        timestamp = match.group(1).strip()
        packet_with_spaces = match.group(2).strip()
        packet_hex = packet_with_spaces.replace(' ', '')
        
        row_data = {"时间戳": timestamp}

        # 2. 严格完整性检查 (检查十六进制字符数量是否等于 38)
        if len(packet_hex) != EXPECTED_HEX_CHAR_COUNT:
            # 数据包不完整，填充空数据并继续
            row_data.update(empty_data_fields)
            all_converted_data.append(row_data)
            continue

        # 3. 提取 16 字节的数据体 (32 字符)
        # 完整包结构：FA (2 chars) + Data (32 chars) + Checksum (2 chars) + AF (2 chars)
        # 17 bytes of payload: 16 Data Bytes + 1 Checksum Byte
        # 提取 16 字节数据体，跳过 FA (2) 和 Checksum+AF (4)
        packet_data_body = packet_hex[2:-4] # 长度应为 32 字符 (16 字节数据)
        
        # 4. 数据解析 (严格按照最新的图片字段顺序)
        try:
            data_start_index = 0

            # 4.1 GSR: H/L (2 bytes, 4 chars)
            gsr = packet_data_body[data_start_index: data_start_index + 4] 
            data_start_index += 4
            
            # 4.2 AccelX/Y/Z: H/L (3 * 2 bytes = 6 bytes, 12 chars)
            acc_x = packet_data_body[data_start_index: data_start_index + 4]
            data_start_index += 4
            acc_y = packet_data_body[data_start_index: data_start_index + 4]
            data_start_index += 4
            acc_z = packet_data_body[data_start_index: data_start_index + 4]
            data_start_index += 4
            
            # 4.3 GyroX/Y/Z: H/L (3 * 2 bytes = 6 bytes, 12 chars)
            gyro_x = packet_data_body[data_start_index: data_start_index + 4]
            data_start_index += 4
            gyro_y = packet_data_body[data_start_index: data_start_index + 4]
            data_start_index += 4
            gyro_z = packet_data_body[data_start_index: data_start_index + 4]
            data_start_index += 4
            
            # 4.4 HeartRate: 1 byte (2 chars)
            hr_data = packet_data_body[data_start_index: data_start_index + 2]
            data_start_index += 2
            
            # 4.5 SpO2: 1 byte (2 chars)
            spo2_data = packet_data_body[data_start_index: data_start_index + 2]
            data_start_index += 2

            if data_start_index != 32:
                # Should not happen with fixed 19-byte structure
                raise ValueError("Data slicing failed; index mismatch.")

            data_map = {
                "GSR": gsr, "AccelX": acc_x, "AccelY": acc_y, "AccelZ": acc_z,
                "GyroX": gyro_x, "GyroY": gyro_y, "GyroZ": gyro_z,
                "HR": hr_data, "SpO2": spo2_data
            }
            
            # 5. 数据转换
            converted_values = convert_data(data_map)
            
            # 6. 组合数据
            row_data.update(converted_values)
            all_converted_data.append(row_data)

        except Exception as e:
            print(f"Error: Failed to convert packet {line.strip()}. Error: {e}")
            row_data.update(empty_data_fields)
            all_converted_data.append(row_data)
            continue

    if not all_converted_data:
        print("Warning: No data entries to process.")
        return

    # 7. 转换并保存到 Excel
    df = pd.DataFrame(all_converted_data)

    column_order = [
        "时间戳", "皮肤电", "加速度x", "加速度y", "加速度z", 
        "角速度x", "角速度y", "角速度z", "心率", "血氧"
    ]
    df = df[column_order]
    
    try:
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"Successfully converted {len(df)} data entries.")
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: Failed to save data to Excel file at {output_path}. Error: {e}")


# --- 程序执行部分 ---
if __name__ == "__main__":
    process_converted_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)