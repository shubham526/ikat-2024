import pyterrier as pt
import pandas as pd
import argparse
import logging
from tqdm import tqdm

if not pt.started():
    pt.init()

import pyt_splade

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def retrieve_documents(index_dir, queries_file, output_file, run_id="PyTerrier-SPLADE++"):
    factory = pyt_splade.SpladeFactory()

    # Set up the BatchRetrieve component
    br = pt.BatchRetrieve(index_dir, wmodel='Tf')

    # Assuming factory is already defined and query_splade is your query encoder
    query_splade = factory.query()

    # Define the retrieval pipeline
    retr_pipe = query_splade >> br

    # Read the queries from the TSV file
    queries = pd.read_csv(queries_file, sep='\t', header=None, names=['qid', 'query'])

    all_results = []

    # Retrieve results for each query
    for _, row in tqdm(queries.iterrows(), total=queries.shape[0], desc="Processing queries"):
        query = row['query']
        results = retr_pipe.search(query)
        # logger.info(f"Found {len(results)} documents for query {query}")
        all_results.append(results)

    # Concatenate all results into a single DataFrame
    all_results_df = pd.concat(all_results, ignore_index=True)

    # Select and rename the necessary columns to TREC format
    all_results_df = all_results_df[['qid', 'docno', 'rank', 'score']]
    all_results_df['Q0'] = 'Q0'
    all_results_df['tag'] = run_id

    # Reorder columns to match TREC format
    trec_results = all_results_df[['qid', 'Q0', 'docno', 'rank', 'score', 'tag']]

    # Save to TREC file
    trec_results.to_csv(output_file, sep=' ', header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description="Retrieve documents from SPLADE index and write to TREC-style run file.")
    parser.add_argument("--index-dir", help='Directory where SPLADE index is stored.', required=True)
    parser.add_argument("--queries", help='TSV file containing query_id and query.', required=True)
    parser.add_argument("--run", help='Output file to save the TREC-style results.', required=True)
    args = parser.parse_args()

    retrieve_documents(
        index_dir=args.index_dir,
        queries_file=args.queries,
        output_file=args.run
    )

if __name__ == '__main__':
    main()
