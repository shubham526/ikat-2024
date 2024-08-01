# This code includes or is derived from code in the SPLADE repository,
# which is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
#
# Original SPLADE repository: https://github.com/naver/splade
# License: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Modifications made:
# - Simplified the initialization process by directly using provided `index_dir`.
# - Removed complex configuration handling for `index_d`, `compute_stats`, `is_beir`, etc.
# - Simplified the `retrieve` method by removing support for batched queries and statistics computation.
# - Removed dependencies on `Evaluator`, `L0`, and other SPLADE-specific classes/functions.
# - Simplified conversion to Numba typed dictionaries.
#
# This code is also licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

import argparse
import pandas as pd
from tqdm import tqdm
import os
import torch
import pickle
import numpy as np
import numba
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from splade.models.transformer_rep import Splade
from splade.indexing.inverted_index import IndexDictOfArray

class SparseRetrieval:
    """Simplified retrieval from SparseIndexing"""

    @staticmethod
    def select_topk(filtered_indexes, scores, k):
        if len(filtered_indexes) > k:
            sorted_ = np.argpartition(scores, k)[:k]
            filtered_indexes, scores = filtered_indexes[sorted_], -scores[sorted_]
        else:
            scores = -scores
        return filtered_indexes, scores

    @staticmethod
    @numba.njit(nogil=True, parallel=True, cache=True)
    def numba_score_float(inverted_index_ids, inverted_index_floats, indexes_to_retrieve, query_values, threshold, size_collection):
        scores = np.zeros(size_collection, dtype=np.float32)  # initialize array with size = size of collection
        n = len(indexes_to_retrieve)
        for _idx in range(n):
            local_idx = indexes_to_retrieve[_idx]  # which posting list to search
            query_float = query_values[_idx]  # what is the value of the query for this posting list
            retrieved_indexes = inverted_index_ids[local_idx]  # get indexes from posting list
            retrieved_floats = inverted_index_floats[local_idx]  # get values from posting list
            for j in numba.prange(len(retrieved_indexes)):
                scores[retrieved_indexes[j]] += query_float * retrieved_floats[j]
        filtered_indexes = np.argwhere(scores > threshold)[:, 0]  # ideally we should have a threshold to filter
        return filtered_indexes, -scores[filtered_indexes]

    def __init__(self, model, index_dir, device, dim_voc):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

        self.sparse_index = IndexDictOfArray(index_dir, dim_voc=dim_voc)
        self.doc_ids = pickle.load(open(os.path.join(index_dir, "doc_ids.pkl"), "rb"))

        # convert to numba
        self.numba_index_doc_ids = numba.typed.Dict()
        self.numba_index_doc_values = numba.typed.Dict()
        for key, value in self.sparse_index.index_doc_id.items():
            self.numba_index_doc_ids[key] = value
        for key, value in self.sparse_index.index_doc_value.items():
            self.numba_index_doc_values[key] = value

    def retrieve(self, q_loader, top_k, threshold=0):
        res = defaultdict(dict)
        with torch.no_grad():
            for batch in q_loader:
                q_id = batch["id"][0]
                inputs = {k: v.to(self.device) for k, v in batch.items() if k not in {"id"}}
                query = self.model(q_kwargs=inputs)["q_rep"]  # we assume ONE query per batch here

                row, col = torch.nonzero(query, as_tuple=True)
                values = query[row, col]
                filtered_indexes, scores = self.numba_score_float(
                    self.numba_index_doc_ids,
                    self.numba_index_doc_values,
                    col.cpu().numpy(),
                    values.cpu().numpy().astype(np.float32),
                    threshold=threshold,
                    size_collection=self.sparse_index.nb_docs()
                )
                filtered_indexes, scores = self.select_topk(filtered_indexes, scores, k=top_k)
                for id_, sc in zip(filtered_indexes, scores):
                    res[q_id][self.doc_ids[id_]] = float(sc)
        return res

def write_to_file(all_results, run, run_id):
    all_results_df = pd.concat(all_results, ignore_index=True)
    # Increment the rank by 1 to start from 1 instead of 0
    all_results_df = all_results_df.sort_values(by=['qid', 'score'], ascending=[True, False])
    all_results_df['rank'] = all_results_df.groupby('qid').cumcount() + 1
    all_results_df = all_results_df[['qid', 'docno', 'rank', 'score']]
    all_results_df['Q0'] = 'Q0'
    all_results_df['tag'] = run_id
    trec_results = all_results_df[['qid', 'Q0', 'docno', 'rank', 'score', 'tag']]
    trec_results.to_csv(run, sep=' ', header=False, index=False)

class QueryDataset(Dataset):
    def __init__(self, doc_file):
        self.doc_file = doc_file
        self.data_dict = {}
        with open(self.doc_file) as reader:
            for line in tqdm(reader):
                if len(line) > 1:
                    id_, *data = line.split("\t")
                    data = " ".join(" ".join(data).splitlines())
                    self.data_dict[id_.strip()] = data.strip()
        self.ids = list(self.data_dict.keys())

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        return id_, self.data_dict[id_]

class QueryDataLoader(DataLoader):
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
            "id": list(ids)
        }

def retrieve_documents(model, tokenizer_type, index_dir, queries_file, hits, device='cpu'):
    dim_voc = model.module.output_dim if hasattr(model, "module") else model.output_dim
    retriever = SparseRetrieval(model=model, index_dir=index_dir, dim_voc=dim_voc, device=device)
    query_dataset = QueryDataset(queries_file)
    query_loader = QueryDataLoader(
        dataset=query_dataset,
        tokenizer_type=tokenizer_type,
        max_length=512,
        batch_size=1,
        shuffle=False,
        num_workers=1
    )
    all_results = []
    for batch in tqdm(query_loader, total=len(query_loader.dataset), desc="Processing queries"):
        qid = batch['id'][0]
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device)
        results = retriever.retrieve([batch], top_k=hits)
        res_df = pd.DataFrame.from_dict(results[qid], orient='index', columns=['score'])
        res_df = res_df.reset_index().rename(columns={'index': 'docno'})
        res_df['qid'] = qid
        res_df['rank'] = res_df['score'].rank(ascending=False).astype(int) - 1
        all_results.append(res_df)
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Retrieve documents from SPLADE index and write to TREC-style run file.")
    parser.add_argument("--model", help='Name of Hugging Face model checkpoint.', required=True)
    parser.add_argument("--index-dir", required=True, help="Directory where SPLADE index is stored.")
    parser.add_argument("--queries", required=True, help="TSV file containing query_id and query.")
    parser.add_argument("--run", required=True, help="Output file to save the TREC-style results.")
    parser.add_argument("--hits", type=int, default=1000, help="Number of hits to retrieve for each query.")
    parser.add_argument("--device", default='cpu', help="Device to run the model on (e.g., 'cpu' or 'cuda').")
    args = parser.parse_args()
    model = Splade(model_type_or_dir=args.model)
    model.eval()
    model.to(args.device)
    all_results = retrieve_documents(
        model=model,
        tokenizer_type=args.model,
        index_dir=args.index_dir,
        queries_file=args.queries,
        hits=args.hits,
        device=args.device
    )
    write_to_file(all_results, args.run, 'SPLADE-Retrieval')
    print("Done")

if __name__ == '__main__':
    main()
