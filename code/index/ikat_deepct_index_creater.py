import os
import argparse
import pyterrier as pt
from tqdm import tqdm
import json
import bz2
import glob
import time
import logging
from pyterrier_deepct import DeepCT, Toks2Text

if not pt.started():
    pt.init()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_time(seconds):
    """
    Helper function to format time in seconds into hours and minutes.
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    return hours, minutes

def process_document(doc_id, text, url):
    """
    Process a document and return a formatted dictionary if it meets the criteria.
    """
    if len(text) and any(letter.isalnum() for letter in text):
        return {'docno': doc_id.strip(), 'text': text.strip(), 'url': url.strip()}
    return None

def bz2_data_generator(bz2_files, pbar):
    """
    Generator function to load bz2 data line by line from multiple files.
    """
    for bz2_file in bz2_files:
        print(f'Loading bz2 file ==> {bz2_file}')
        with bz2.open(bz2_file, 'rt') as f:
            for line in f:
                doc = json.loads(line)
                if 'id' in doc and 'contents' in doc and 'url' in doc:
                    processed_doc = process_document(doc['id'], doc['contents'], doc['url'])
                    if processed_doc:
                        yield processed_doc
                        pbar.update(1)
                else:
                    logger.warning(f"Document in {bz2_file} missing 'id', 'contents', or 'url' field: {doc}")
        print('[Done]')

def create_index(args, bz2_file_lines):
    logger.info(f'Batch size for indexing = {args.batch_size}')
    device = f"cuda:{args.cuda}"

    logger.info(f"Using device: {device}")

    deep_ct = DeepCT(device=device, batch_size=args.batch_size)
    bz2_files = sorted(glob.glob(os.path.join(args.bz2_dir, '*.bz2')))
    total_lines = sum(bz2_file_lines.get(os.path.basename(file), 0) for file in bz2_files)

    with tqdm(total=total_lines, desc="Indexing documents") as pbar:
        documents_generator = bz2_data_generator(bz2_files, pbar)
        logger.info(f'Starting to index...')

        start_time = time.time()

        iter_indexer = pt.index.IterDictIndexer(
            index_path=args.index_dir,
            verbose=True,
            fields=['text'],
            meta={'docno': 40, 'text': 10000, 'url': 100}
        )

        indexer_pipe = deep_ct >> Toks2Text() >> iter_indexer
        indexer_pipe.index(documents_generator, batch_size=args.batch_size)

        end_time = time.time()
        elapsed_time = end_time - start_time
        total_hours, total_minutes = format_time(elapsed_time)
        logger.info(f'[Done] indexing.')
        logger.info(f'Time taken: {total_hours} hours and {total_minutes} minutes')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

def main():
    parser = argparse.ArgumentParser("Create a PyTerrier DeepCT index for ikat dataset.")
    parser.add_argument("--bz2-dir", help='Directory containing bz2 files.', required=True)
    parser.add_argument("--index-dir", help='Directory where index will be saved.', required=True)
    parser.add_argument('--batch-size', help='Batch size for indexing. Default: 200.', default=200, type=int)
    parser.add_argument('--bz2-file-lines', help='JSON file containing the number of lines per bz2 file.', required=True)
    parser.add_argument('--cuda', help='Which CUDA device to use. Default: 0', default=0, type=int)
    args = parser.parse_args()

    # Load the bz2 file lines information
    with open(args.bz2_file_lines, 'r') as f:
        bz2_file_lines = json.load(f)

    create_index(args, bz2_file_lines)

if __name__ == '__main__':
    main()
