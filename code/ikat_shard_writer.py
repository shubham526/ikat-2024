import os
import argparse
import json
import bz2
import glob
from tqdm import tqdm
import hashlib
from typing import List

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

seen = set()


def consistent_hash(doc_id):
    """Generate a consistent hash for a given document ID."""
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16)


def process_document(doc_id, text):
    if doc_id not in seen:
        seen.add(doc_id)
        if len(text) and any(letter.isalnum() for letter in text):
            return {'docno': doc_id.strip(), 'text': text.strip()}
    return None


def shard_dataset(file_paths: List[str], id_field: str, text_field: str, num_shards: int, output_dir: str):
    shard_files = [os.path.join(output_dir, f'shard_{i}.jsonl') for i in range(num_shards)]
    shard_writers = [open(shard_file, 'a') for shard_file in shard_files]
    shard_line_counts = {os.path.basename(shard_file): 0 for shard_file in shard_files}

    for file_path in file_paths:
        print(f'Processing file ==> {file_path}')
        file_name = os.path.basename(file_path)
        total_lines = totals.get(file_name, None)

        with bz2.open(file_path, 'rt') as f:
            for line in tqdm(f, total=total_lines, desc=file_name):
                d = json.loads(line)
                doc_id = d[id_field]
                processed_doc = process_document(doc_id, d[text_field])
                if processed_doc:
                    shard_index = consistent_hash(doc_id) % num_shards
                    shard_writers[shard_index].write(json.dumps(processed_doc) + '\n')
                    shard_line_counts[os.path.basename(shard_files[shard_index])] += 1
        print('[Done].')
        print('======================================================')

    for writer in shard_writers:
        writer.close()

    # Save the shard line counts to a JSON file
    line_counts_file = os.path.join(output_dir, 'shard_file_lines.json')
    with open(line_counts_file, 'w') as f:
        json.dump(shard_line_counts, f)


def main():
    parser = argparse.ArgumentParser("Shard iKAT dataset into multiple files.")
    parser.add_argument("--data-dir", help='Directory containing bz2 files.', required=True)
    parser.add_argument("--shard-dir", help='Directory to save shard files.', required=True)
    parser.add_argument('--id-field', help='DocID field in data. Default: doc_id.', default='doc_id', type=str)
    parser.add_argument('--text-field', help='Text field in data. Default: contents.', default='contents', type=str)
    parser.add_argument('--num-shards', help='Number of shards. Default: 4.', default=4, type=int)
    args = parser.parse_args()

    os.makedirs(args.shard_dir, exist_ok=True)
    bz2_files = glob.glob(os.path.join(args.data_dir, '*.bz2'))
    shard_dataset(bz2_files, args.id_field, args.text_field, args.num_shards, args.shard_dir)


if __name__ == '__main__':
    main()
