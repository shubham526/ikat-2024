import os
import bz2
import json
import glob
import logging
import argparse

# Predefined total lines for each file
totals = {
    "ikat_2023_passages_15.jsonl.bz2": 7311551,
    "ikat_2023_passages_14.jsonl.bz2": 7326852,
    "ikat_2023_passages_11.jsonl.bz2": 7322582,
    "ikat_2023_passages_10.jsonl.bz2": 7312220,
    "ikat_2023_passages_13.jsonl.bz2": 7313577,
    "ikat_2023_passages_12.jsonl.bz2": 7350282,
    "ikat_2023_passages_00.jsonl.bz2": 7311973,
    "ikat_2023_passages_01.jsonl.bz2": 7317527,
    "ikat_2023_passages_02.jsonl.bz2": 7300102,
    "ikat_2023_passages_03.jsonl.bz2": 7291878,
    "ikat_2023_passages_04.jsonl.bz2": 7200535,
    "ikat_2023_passages_05.jsonl.bz2": 7288255,
    "ikat_2023_passages_06.jsonl.bz2": 7320431,
    "ikat_2023_passages_07.jsonl.bz2": 7333325,
    "ikat_2023_passages_09.jsonl.bz2": 7278310,
    "ikat_2023_passages_08.jsonl.bz2": 7259587
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_offsets(bz2_files, output_offset_file):
    offsets = {}

    for file_path in bz2_files:
        file_name = os.path.basename(file_path)
        total_lines = totals.get(file_name, 0)
        offsets[file_name] = []

        logging.info(f"Processing {file_name}...")
        with bz2.open(file_path, 'rt') as f:
            offset = f.tell()
            line_count = 0
            while True:
                line = f.readline()
                if not line:
                    break
                offsets[file_name].append(offset)
                offset = f.tell()
                line_count += 1
                if line_count % 100000 == 0:
                    logging.info(f"Processed {line_count}/{total_lines} lines for {file_name}")

    with open(output_offset_file, 'w') as outfile:
        json.dump(offsets, outfile)

    logging.info(f"Offsets written to {output_offset_file}")

def main():
    parser = argparse.ArgumentParser(description="Calculate and store offsets for bz2 files.")
    parser.add_argument("--data-dir", help="Directory containing bz2 files.", required=True)
    parser.add_argument("--offset-file", help="Output file to store offsets.", required=True)
    args = parser.parse_args()
    bz2_files = glob.glob(os.path.join(args.data_dir, '*.bz2'))

    calculate_offsets(bz2_files, args.offset_file)

if __name__ == '__main__':
    main()
