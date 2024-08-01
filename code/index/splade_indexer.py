# This code includes or is derived from code in the SPLADE repository,
# which is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
#
# Original SPLADE repository: https://github.com/naver/splade
# License: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Modifications made:
# - Adapted the CollectionDatasetPreLoad class to create DocumentDatasetPreLoad for document preloading.
# - Adapted the CollectionDataLoader class to create DocumentDataLoader for handling document loading.
# - Adapted the SparseIndexing class for handling indexing of documents with custom processing.
# - Integrated logging and improved error handling.
# - Added support for loading bz2 compressed files.
#
# This code is also licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

import os
import argparse
import json
import logging
import bz2
from tqdm import tqdm
import glob
import time
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from splade.indexing.inverted_index import IndexDictOfArray
from splade.models.transformer_rep import Splade

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

def process_document(doc_id, text):
    """
    Process a document and return a formatted dictionary if it meets the criteria.
    """
    if len(text) and any(letter.isalnum() for letter in text):
        return {'docno': doc_id.strip(), 'text': text.strip()}
    return None

class DocumentDataset(Dataset):
    """
    Dataset to iterate over a document collection.
    Preloads everything in memory at init.
    """

    def __init__(self, bz2_files, total_lines):
        self.data_dict = {}
        with tqdm(total=total_lines, desc="Loading documents") as pbar:
            for bz2_file in bz2_files:
                logger.info(f'Loading bz2 file ==> {bz2_file}')
                with bz2.open(bz2_file, 'rt') as f:
                    for line in f:
                        doc = json.loads(line)
                        if 'id' in doc and 'contents' in doc:
                            processed_doc = process_document(doc['id'], doc['contents'])
                            if processed_doc:
                                self.data_dict[processed_doc['docno']] = processed_doc['text']
                        else:
                            logger.warning(f"Document in {bz2_file} missing 'id' or 'contents' field: {doc}")
                        pbar.update(1)
                logger.info('[Done]')
        self.ids = list(self.data_dict.keys())

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        return id_, self.data_dict[id_]

class DocumentDataLoader(DataLoader):
    def __init__(self, dataset, tokenizer_type, max_length, batch_size, shuffle, num_workers, prefetch_factor=2):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        self.max_length = max_length
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
            prefetch_factor=prefetch_factor
        )

    def collate_fn(self, batch):
        ids, texts = zip(*batch)
        processed_texts = self.tokenizer(
            list(texts),
            add_special_tokens=True,
            padding="longest",
            truncation="longest_first",
            max_length=self.max_length,
            return_tensors="pt",
            return_attention_mask=True
        )
        return {
            **{k: v for k, v in processed_texts.items()},
            "id": list(ids)  # Keep ids as strings
        }

class SparseIndexing:
    """Sparse indexing"""
    def __init__(self, model, index_dir, device, dim_voc=None, force_new=True):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.index_dir = index_dir
        self.sparse_index = IndexDictOfArray(self.index_dir, dim_voc=dim_voc, force_new=force_new)

    def index(self, collection_loader, id_dict=None):
        doc_ids = []
        count = 0
        with torch.no_grad():
            for t, batch in enumerate(tqdm(collection_loader)):
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"id"}}
                # Get the document representation from the model
                batch_documents = self.model(d_kwargs=inputs)["d_rep"]
                row, col = torch.nonzero(batch_documents, as_tuple=True)
                data = batch_documents[row, col]
                row = row + count
                batch_ids = list(batch["id"])
                if id_dict:
                    batch_ids = [id_dict[x] for x in batch_ids]
                count += len(batch_ids)
                doc_ids.extend(batch_ids)
                self.sparse_index.add_batch_document(row.cpu().numpy(), col.cpu().numpy(), data.cpu().numpy(),
                                                     n_docs=len(batch_ids))
        self.sparse_index.save()
        with open(os.path.join(self.index_dir, "doc_ids.pkl"), "wb") as f:
            pickle.dump(doc_ids, f)
        print("Done iterating over the corpus...")
        print("Index contains {} posting lists".format(len(self.sparse_index.index_doc_id)))
        print("Index contains {} documents".format(len(doc_ids)))

def create_index(args, bz2_file_lines):
    logger.info(f'Batch size for indexing = {args.batch_size}')

    model_checkpoint = args.model_checkpoint
    model = Splade(model_type_or_dir=model_checkpoint)
    dim_voc = model.module.output_dim if hasattr(model, "module") else model.output_dim
    model.to(args.device)

    sparse_indexing = SparseIndexing(model=model, index_dir=args.index_dir, device=args.device, dim_voc=dim_voc)

    bz2_files = sorted(glob.glob(os.path.join(args.bz2_dir, '*.bz2')))
    total_lines = sum(bz2_file_lines.get(os.path.basename(file), 0) for file in bz2_files)

    dataset = DocumentDataset(bz2_files=bz2_files, total_lines=total_lines)
    dataloader = DocumentDataLoader(
        dataset=dataset,
        tokenizer_type=model_checkpoint,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=10,
        prefetch_factor=4,
        max_length=512
    )

    logger.info(f'Starting to index...')
    start_time = time.time()
    sparse_indexing.index(dataloader)
    end_time = time.time()
    elapsed_time = end_time - start_time
    total_hours, total_minutes = format_time(elapsed_time)
    logger.info(f'[Done] indexing.')
    logger.info(f'Time taken: {total_hours} hours and {total_minutes} minutes')
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

def main():
    parser = argparse.ArgumentParser("Create a SPLADE index for a dataset using a pre-trained model from Hugging Face.")
    parser.add_argument("--bz2-dir", help='Directory containing bz2 files.', required=True)
    parser.add_argument("--index-dir", help='Directory where index will be saved.', required=True)
    parser.add_argument("--model", help='Name of Hugging Face model checkpoint.', required=True)
    parser.add_argument('--batch-size', help='Batch size for indexing. Default: 200.', default=200, type=int)
    parser.add_argument('--bz2-file-lines', help='JSON file containing the number of lines per bz2 file.', required=True)
    parser.add_argument("--device", default='cpu', help="Device to run the model on (e.g., 'cpu' or 'cuda').")
    args = parser.parse_args()

    # Load the bz2 file lines information
    with open(args.bz2_file_lines, 'r') as f:
        bz2_file_lines = json.load(f)

    create_index(args, bz2_file_lines)

if __name__ == '__main__':
    main()
