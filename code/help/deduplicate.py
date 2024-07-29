import os
import bz2
import json
import argparse
from typing import List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from datasketch import MinHash, MinHashLSH
import time
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load SentenceBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

def custom_collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Custom collate function to return the batch as is.
    Args:
        batch (List[Dict[str, Any]]): List of items from Dataset.
    Returns:
        List[Dict[str, Any]]: Collated batch as a list of dictionaries.
    """
    return batch

class InMemoryJSONLDataset(Dataset):
    def __init__(self, data_dir, file_info_path):
        self.data_dir = data_dir
        self.file_info = self._load_file_info(file_info_path)
        self.data = self._load_data()

    def _load_file_info(self, file_info_path):
        with open(file_info_path, 'r') as f:
            file_info = json.load(f)
        return file_info

    def _load_data(self):
        data = []
        for file_name, num_lines in self.file_info.items():
            file_path = os.path.join(self.data_dir, file_name)
            if os.path.exists(file_path) and file_path.endswith('.bz2'):
                with bz2.open(file_path, 'rt', encoding='ascii', errors='ignore') as f:
                    for line in tqdm(f, total=num_lines, desc=f"Reading {file_name}"):
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logging.error(f"JSONDecodeError: {e} in line {line}")
            else:
                logging.warning(f'File {file_path} not found or not a .bz2 file. Skipping.')
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_sbert_embeddings_batch(paragraphs):
    """
    Get SBERT embeddings for a batch of paragraphs.

    Args:
        paragraphs (List[str]): List of input paragraphs.

    Returns:
        numpy.ndarray: The SBERT embeddings of the paragraphs.
    """
    embeddings = sbert_model.encode(paragraphs, convert_to_tensor=True, device=device)
    return embeddings.cpu().numpy()

def paragraph_to_minhash(paragraphs, num_perm=128):
    """
    Convert a batch of paragraphs to MinHash.

    Args:
        paragraphs (List[str]): The input paragraphs.
        num_perm (int): Number of permutations for MinHash.

    Returns:
        List[datasketch.MinHash]: The MinHash representations of the paragraphs.
    """
    embeddings = get_sbert_embeddings_batch(paragraphs)
    minhashes = []
    for emb in embeddings:
        m = MinHash(num_perm=num_perm)
        for val in emb:
            m.update(str(val).encode('utf8'))
        minhashes.append(m)
    return minhashes

def deduplicate(output_file: str, duplicates_file: str, batch_size: int, num_perm: int, threshold: float, data_dir: str, bz2_file_lines: str):
    """
    Deduplicate paragraphs in batches with cross-batch deduplication.

    This function processes paragraphs stored in bz2-compressed JSONL files, computes their MinHash
    signatures using a SentenceBERT model, and uses Locality Sensitive Hashing (LSH) to identify
    and remove duplicate paragraphs across multiple batches.

    Args:
        :param bz2_file_lines:
        :param duplicates_file: Path to the file to save duplicate IDs.
        :param output_file: Path to the output file.
        :param batch_size: Batch size for processing.
        :param num_perm: Number of permutations for MinHash.
        :param threshold: Similarity threshold for MinHash LSH.
        :param data_dir: Path to data directory.

    Workflow:
        1. Load SentenceBERT model for generating embeddings.
        2. Initialize MinHashLSH for deduplication with a specified threshold and number of permutations.
        3. Load offsets from the offset file to identify JSONL entries in bz2 files.
        4. Use DataLoader to iterate over batches of JSONL entries:
            a. For each batch, extract the 'contents' field from each entry.
            b. Compute MinHash signatures for the 'contents'.
            c. Use LSH to check for existing similar entries:
                - If no similar entry exists, insert the MinHash signature and add the entry to the unique list.
        5. Save the deduplicated entries to the specified output file.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    logging.info(f"[START] Creating Dataset...")
    dataset = InMemoryJSONLDataset(data_dir=data_dir, file_info_path=bz2_file_lines)
    logging.info(f"[END] Creating Dataset...")

    logging.info(f"[START] Creating DataLoader. Loading data from file.")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=custom_collate_fn)
    logging.info(f"[END] Creating DataLoader. Loading data from file.")

    print('Finding total batches')
    total_batches = len(dataloader)
    print('[Done]')

    def process_batch(batch, batch_num, outfile, dupfile):
        nonlocal lsh
        paragraphs = [item['contents'] for item in batch]

        logging.info(f"[START] Calculating minhash signatures for batch# {batch_num}...")
        minhashes = paragraph_to_minhash(paragraphs, num_perm)
        logging.info(f"[END] Calculating minhash signatures for batch# {batch_num}...")

        logging.info(f"[START] Finding unique paragraphs in batch# {batch_num}...")
        for idx, (m, item_inner) in enumerate(zip(minhashes, batch)):
            unique_key = item_inner['id']  # Use docno as the unique key
            if len(lsh.query(m)) == 0:
                lsh.insert(unique_key, m)
                json.dump(item_inner, outfile)
                outfile.write('\n')
            else:
                dupfile.write(unique_key + '\n')
        logging.info(f"[END] Finding unique paragraphs in batch# {batch_num}...")

    with open(output_file, 'w') as outfile, open(duplicates_file, 'w') as dupfile:
        for batch_num, batch in enumerate(dataloader, 1):
            start_time = time.time()
            process_batch(batch, batch_num, outfile, dupfile)
            end_time = time.time()
            elapsed_time = end_time - start_time
            hours, rem = divmod(elapsed_time, 3600)
            minutes, seconds = divmod(rem, 60)
            logging.info(
                f'Processed batch {batch_num} of {total_batches} in {int(hours)} hours, {int(minutes)} minutes, and {int(seconds)} seconds.')

    logging.info(f"[END] Deduplicate paragraphs...")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate TREC iKAT collection.")
    parser.add_argument("--data-dir",
                        help='Directory containing bz2 files.',
                        required=True)
    parser.add_argument("--bz2-file-lines",
                        help='JSON file containing number of lines in each bz2 file.',
                        required=True)
    parser.add_argument("--save",
                        help='Deduplicated file.',
                        required=True)
    parser.add_argument("--duplicate-ids",
                        help='File to save duplicate IDs.',
                        required=True)
    parser.add_argument('--batch-size',
                        help='Batch size. Default=1000.',
                        default=1000,
                        type=int)
    parser.add_argument('--num-perm',
                        help='Number of permutations for MinHash algorithm. Default=128',
                        default=128,
                        type=int)
    parser.add_argument('--threshold',
                        help='Similarity threshold for considering two MinHash signatures as similar (or duplicate) items.',
                        default=0.9,
                        type=float)
    args = parser.parse_args()

    deduplicate(
        output_file=args.save,
        duplicates_file=args.duplicate_ids,
        batch_size=args.batch_size,
        num_perm=args.num_perm,
        threshold=args.threshold,
        data_dir=args.data_dir,
        bz2_file_lines=args.bz2_file_lines
    )

if __name__ == '__main__':
    main()


