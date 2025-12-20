import os
import re
import pandas as pd

# ====== Paths (project-relative, safe for GitHub) ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Recommended repo layout:
#   repo/
#     data/
#       split_sensor_packets_variable_length.txt
#     outputs/
#       converted_sensor_data_final.xlsx
INPUT_FILENAME = "split_sensor_packets_variable_length.txt"
OUTPUT_FILENAME = "converted_sensor_data_final.xlsx"

INPUT_FILE_PATH = os.path.join(BASE_DIR, "..", "data", INPUT_FILENAME)
OUTPUT_FILE_PATH = os.path.join(BASE_DIR, "..", "outputs", OUTPUT_FILENAME)
os.makedirs(os.path.dirname(OUTPUT_FILE_PATH), exist_ok=True)

# ====== Packet format constants ======
# Final confirmed full packet length: 19 bytes (FA + 17 payload bytes + AF)
EXPECTED_BYTE_COUNT = 19
EXPECTED_HEX_CHAR_COUNT = EXPECTED_BYTE_COUNT * 2  # 38 hex characters

# ====== Data conversion parameters (unchanged) ======
GSR_MAX_ADC = 4095.0
GSR_MAX_VOLTAGE = 3.3
ACC_MAX_ADC = 32768.0
ACC_MAX_RANGE = 2.0
GYRO_MAX_ADC = 32768.0
GYRO_MAX_RANGE = 250.0

# ====== Helper functions (unchanged logic) ======
def hex_to_signed_int(hex_str: str) -> int:
    """Convert a hex string to a signed 16-bit integer (decimal)."""
    val = int(hex_str, 16)
    if val & 0x8000:
        return val - 65536
    return val

def convert_data(data: dict) -> dict:
    """Convert extracted hex fields into final sensor values."""
    # 1) GSR (Skin conductance) - unsigned 16-bit
    gsr_raw = int(data["GSR"], 16)
    gsr_voltage = (gsr_raw / GSR_MAX_ADC) * GSR_MAX_VOLTAGE

    # 2) Acceleration (AccelX/Y/Z) - signed 16-bit
    accel_x_raw = hex_to_signed_int(data["AccelX"])
    accel_y_raw = hex_to_signed_int(data["AccelY"])
    accel_z_raw = hex_to_signed_int(data["AccelZ"])
    accel_x_g = (accel_x_raw / ACC_MAX_ADC) * ACC_MAX_RANGE
    accel_y_g = (accel_y_raw / ACC_MAX_ADC) * ACC_MAX_RANGE
    accel_z_g = (accel_z_raw / ACC_MAX_ADC) * ACC_MAX_RANGE

    # 3) Gyroscope (GyroX/Y/Z) - signed 16-bit
    gyro_x_raw = hex_to_signed_int(data["GyroX"])
    gyro_y_raw = hex_to_signed_int(data["GyroY"])
    gyro_z_raw = hex_to_signed_int(data["GyroZ"])
    gyro_x_s = (gyro_x_raw / GYRO_MAX_ADC) * GYRO_MAX_RANGE
    gyro_y_s = (gyro_y_raw / GYRO_MAX_ADC) * GYRO_MAX_RANGE
    gyro_z_s = (gyro_z_raw / GYRO_MAX_ADC) * GYRO_MAX_RANGE

    # 4) Heart rate (HR) and SpO2 - unsigned 8-bit
    hr_val = int(data["HR"], 16)
    spo2_val = int(data["SpO2"], 16)

    # Keep original Chinese column names to match downstream scripts
    return {
        "皮肤电": round(gsr_voltage, 4),
        "加速度x": round(accel_x_g, 4),
        "加速度y": round(accel_y_g, 4),
        "加速度z": round(accel_z_g, 4),
        "角速度x": round(gyro_x_s, 4),
        "角速度y": round(gyro_y_s, 4),
        "角速度z": round(gyro_z_s, 4),
        "心率": hr_val,
        "血氧": spo2_val,
    }

# ====== Main processing ======
def process_converted_data(input_path: str, output_path: str) -> None:
    """
    Read the split TXT file, perform a strict 19-byte integrity check,
    convert data fields, and save to Excel.
    """
    print(f"Reading data from: {input_path}")

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return
    except Exception as e:
        print(f"Error: Failed to read input file: {e}")
        return

    all_converted_data = []
    empty_data_fields = {
        "皮肤电": None, "加速度x": None, "加速度y": None, "加速度z": None,
        "角速度x": None, "角速度y": None, "角速度z": None,
        "心率": None, "血氧": None
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Expected line pattern: "[timestamp] <hex bytes with spaces>"
        match = re.search(r"\[(.*?)\] (.*)", line)
        if not match:
            continue

        timestamp = match.group(1).strip()
        packet_with_spaces = match.group(2).strip()
        packet_hex = packet_with_spaces.replace(" ", "")

        row_data = {"时间戳": timestamp}

        # Strict integrity check: hex length must be exactly 38 chars (19 bytes)
        if len(packet_hex) != EXPECTED_HEX_CHAR_COUNT:
            # Incomplete packet -> fill with empty fields and continue
            row_data.update(empty_data_fields)
            all_converted_data.append(row_data)
            continue

        # Extract the 16-byte data body (32 hex chars)
        # Full packet: FA (2 chars) + Data (32 chars) + Checksum (2 chars) + AF (2 chars)
        packet_data_body = packet_hex[2:-4]  # should be 32 chars

        # Parse fields in the fixed order
        try:
            data_start_index = 0

            # 1) GSR: 2 bytes (4 hex chars)
            gsr = packet_data_body[data_start_index:data_start_index + 4]
            data_start_index += 4

            # 2) AccelX/Y/Z: 3 * 2 bytes = 6 bytes (12 hex chars)
            acc_x = packet_data_body[data_start_index:data_start_index + 4]
            data_start_index += 4
            acc_y = packet_data_body[data_start_index:data_start_index + 4]
            data_start_index += 4
            acc_z = packet_data_body[data_start_index:data_start_index + 4]
            data_start_index += 4

            # 3) GyroX/Y/Z: 3 * 2 bytes = 6 bytes (12 hex chars)
            gyro_x = packet_data_body[data_start_index:data_start_index + 4]
            data_start_index += 4
            gyro_y = packet_data_body[data_start_index:data_start_index + 4]
            data_start_index += 4
            gyro_z = packet_data_body[data_start_index:data_start_index + 4]
            data_start_index += 4

            # 4) HR: 1 byte (2 hex chars)
            hr_data = packet_data_body[data_start_index:data_start_index + 2]
            data_start_index += 2

            # 5) SpO2: 1 byte (2 hex chars)
            spo2_data = packet_data_body[data_start_index:data_start_index + 2]
            data_start_index += 2

            if data_start_index != 32:
                raise ValueError("Data slicing failed: index mismatch (expected 32 hex chars).")

            data_map = {
                "GSR": gsr,
                "AccelX": acc_x, "AccelY": acc_y, "AccelZ": acc_z,
                "GyroX": gyro_x, "GyroY": gyro_y, "GyroZ": gyro_z,
                "HR": hr_data, "SpO2": spo2_data
            }

            converted_values = convert_data(data_map)
            row_data.update(converted_values)
            all_converted_data.append(row_data)

        except Exception as e:
            print(f"Error: Failed to convert packet line: {line}. Error: {e}")
            row_data.update(empty_data_fields)
            all_converted_data.append(row_data)
            continue

    if not all_converted_data:
        print("Warning: No data entries to process.")
        return

    # Convert to DataFrame and save to Excel
    df = pd.DataFrame(all_converted_data)

    column_order = [
        "时间戳", "皮肤电",
        "加速度x", "加速度y", "加速度z",
        "角速度x", "角速度y", "角速度z",
        "心率", "血氧"
    ]
    df = df[column_order]

    try:
        df.to_excel(output_path, index=False, engine="openpyxl")
        print(f"Successfully converted {len(df)} data entries.")
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: Failed to save Excel file: {output_path}. Error: {e}")

# ====== Script entry ======
if __name__ == "__main__":
    process_converted_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)

if __name__ == "__main__":
    process_converted_data(INPUT_FILE_PATH, OUTPUT_FILE_PATH)
