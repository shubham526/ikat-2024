import pyterrier as pt
import pandas as pd
import argparse
import logging
from tqdm import tqdm

if not pt.started():
    pt.init()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def retrieve_documents(index_dir, queries_file, output_file, ret_model, run_id, hits):
    logger.info(f'Retrieving documents from {index_dir}')
    logger.info(f'Using retrieval model {ret_model}')
    logger.info(f'Retrieving {hits} documents per query')
    # Initialize the PyTerrier BatchRetrieve with the DeepCT index
    br = pt.BatchRetrieve(index_dir, wmodel=ret_model)

    # Read the queries from the TSV file
    queries = pd.read_csv(queries_file, sep='\t', header=None, names=['qid', 'query'])

    results_list = []

    # Retrieve results for each query with a progress bar
    for _, row in tqdm(queries.iterrows(), total=queries.shape[0], desc="Processing queries"):
        qid = row['qid']
        query = row['query']
        try:
            # Search the index
            results = br.search(query, num_results=hits)

            # Process the search results
            for rank, result in enumerate(results.itertuples(), start=1):
                results_list.append({
                    'qid': qid,
                    'Q0': 'Q0',
                    'docno': result.docno,
                    'rank': rank,  # Start ranking from 1
                    'score': result.score,
                    'tag': run_id
                })
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(f"Skipping this query: {query}")

    logger.info(f"Writing results to {output_file}")

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save results to TREC format file
    results_df.to_csv(output_file, sep=' ', header=False, index=False)

    logger.info("Done")


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve documents from DeepCT index using PyTerrier and write to TREC-style run file.")
    parser.add_argument("--index-dir",
                        help='Directory where DeepCT index is stored.',
                        required=True)
    parser.add_argument("--queries",
                        help='TSV file containing query_id and query.',
                        required=True)
    parser.add_argument("--run",
                        help='Output file to save the TREC-style results.',
                        required=True)
    parser.add_argument("--hits",
                        help='Number of hits to retrieve for each query.',
                        type=int,
                        default=1000)
    parser.add_argument("--ret-model",
                        help='The name of the weighting model. '
                             'Valid values are the Java class name of any Terrier weighting model. '
                             'Terrier provides many, such as "BM25", "PL2". '
                             'Default: BM25',
                        type=str,
                        default='BM25')
    parser.add_argument("--tag",
                        help='Tag for the TREC run file. Default: PyTerrier-SPLADE++',
                        type=str,
                        default='PyTerrier-DeepCT-BM25')
    args = parser.parse_args()

    retrieve_documents(
        index_dir=args.index_dir,
        queries_file=args.queries,
        output_file=args.run,
        hits=args.hits,
        ret_model=args.ret_model,
        run_id=args.tag
    )


if __name__ == '__main__':
    main()
