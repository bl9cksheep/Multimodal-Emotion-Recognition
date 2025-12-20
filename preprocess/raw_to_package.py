import re
import os

# ====== Paths (project-relative, safe for GitHub) ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Recommended repo layout:
#   repo/
#     data/
#       SaveWindows2025_11_11_10-16-19.txt
#     outputs/
#       split_sensor_packets_variable_length.txt
INPUT_FILENAME = "SaveWindows2025_11_11_10-16-19.txt"
OUTPUT_FILENAME = "split_sensor_packets_variable_length.txt"

FILE_PATH = os.path.join(BASE_DIR, "..", "data", INPUT_FILENAME)
OUTPUT_PATH = os.path.join(BASE_DIR, "..", "outputs", OUTPUT_FILENAME)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

def split_and_save_packets_variable(file_path: str, output_path: str) -> None:
    """
    Read the raw log file and split the hex stream into independent packets
    based on FA...AF boundaries (variable length). Then save packets to a new file.
    """
    print(f"Reading data from: {file_path}")

    try:
        # Read with 'gbk' encoding (common for some Windows-generated logs)
        with open(file_path, "r", encoding="gbk") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: Input file not found: {file_path}")
        return
    except Exception as e:
        print(f"Error: Failed to read input file: {e}")
        return

    # 1) Preprocess: build a timestamp map and a continuous hex stream
    timestamps_map = []  # list of (hex_start_index_in_stream, timestamp)
    full_hex_data = ""   # concatenated hex string from all lines

    for line in lines:
        # Match pattern: [timestamp]收←◆<hex...>
        match = re.search(r"\[(.*?)\]收←◆(.*)", line)
        if match:
            timestamp = match.group(1).strip()
            # Remove spaces from hex payload and append into the continuous stream
            hex_str = match.group(2).replace(" ", "").strip()

            # Record where this line's hex string starts in the full stream
            timestamps_map.append((len(full_hex_data), timestamp))
            full_hex_data += hex_str

    # 2) Find and extract packets (non-greedy match)
    processed_packets = []
    # Regex: FA (header) + (.*?) (non-greedy) + AF (tail)
    packet_pattern = r"(FA.*?AF)"

    # Find all packets in the continuous hex stream
    for m in re.finditer(packet_pattern, full_hex_data):
        full_packet_hex = m.group(1)  # full "FA...AF" string
        start_index = m.start()

        # Determine associated timestamp:
        # choose the latest timestamp whose start index <= packet start index
        if not timestamps_map:
            associated_timestamp = ""
        else:
            associated_timestamp = timestamps_map[0][1]
            for ts_start_index, ts in timestamps_map:
                if ts_start_index <= start_index:
                    associated_timestamp = ts
                else:
                    break

        # Format output: "[timestamp] <hex bytes separated by spaces>"
        hex_len = len(full_packet_hex)
        spaced_packet = " ".join(full_packet_hex[i:i + 2] for i in range(0, hex_len, 2))
        processed_packets.append(f"[{associated_timestamp}] {spaced_packet}\n")

    if not processed_packets:
        print("Warning: No valid FA...AF packets were found with variable-length matching.")
        return

    # 3) Write to output text file (utf-8)
    try:
        with open(output_path, "w", encoding="utf-8") as outfile:
            outfile.writelines(processed_packets)

        print(f"Successfully split and saved {len(processed_packets)} packets (variable length).")
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"Error: Failed to save output file: {output_path}. Error: {e}")

if __name__ == "__main__":
    split_and_save_packets_variable(FILE_PATH, OUTPUT_PATH)
