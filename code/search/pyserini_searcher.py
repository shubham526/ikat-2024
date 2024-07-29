from pyserini.search.lucene import LuceneSearcher
import pandas as pd
import argparse
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def retrieve_documents(
        index_dir,
        queries_file,
        output_file,
        model,
        k1,
        b,
        mu,
        top_k,
        run_id,
):
    # Initialize the Pyserini searcher
    searcher = LuceneSearcher(index_dir)

    # Set retrieval model
    if model == 'bm25':
        searcher.set_bm25(k1, b)
    elif model == 'ql':
        searcher.set_qld(mu)

    # Read the queries from the TSV file
    queries = pd.read_csv(queries_file, sep='\t', header=None, names=['qid', 'query'])

    results_list = []

    # Retrieve results for each query with a progress bar
    for _, row in tqdm(queries.iterrows(), total=queries.shape[0], desc="Processing queries"):
        qid = row['qid']
        query = row['query']
        try:
            # Search the index
            hits = searcher.search(query, k=top_k)

            # Process the search results
            for rank, hit in enumerate(hits):
                results_list.append({
                    'qid': qid,
                    'Q0': 'Q0',
                    'docno': hit.docid,
                    'rank': rank + 1,  # Start ranking from 1
                    'score': hit.score,
                    'tag': run_id
                })
        except Exception as e:
            logger.error(f"Error: {e}")
            logger.error(f"Skipping this query: {query}")

    # Convert results list to DataFrame
    results_df = pd.DataFrame(results_list)

    # Save results to TREC format file
    logger.info(f"Writing results to {output_file}")
    results_df.to_csv(output_file, sep=' ', header=False, index=False)
    logger.info("Done")


def main():
    parser = argparse.ArgumentParser(description="Retrieve documents from Pyserini index and write to TREC-style run file.")
    parser.add_argument("--index-dir", help='Directory where Pyserini index is stored.', required=True)
    parser.add_argument("--queries", help='TSV file containing query_id and query.', required=True)
    parser.add_argument("--run", help='Output file to save the TREC-style results.', required=True)
    parser.add_argument("--model", help='Retrieval model to use (bm25 or ql).', default='bm25')
    parser.add_argument("--k1", help='BM25 k1 parameter.', type=float, default=0.9)
    parser.add_argument("--b", help='BM25 b parameter.', type=float, default=0.4)
    parser.add_argument("--mu", help='QL mu parameter.', type=float, default=1000)
    parser.add_argument("--hits", help='Number of hits to retrieve for each query.', type=int, default=1000)
    parser.add_argument("--tag", help='Tag for the run. Default: Pyserini-Run', type=str, default='Pyserini-Run')
    args = parser.parse_args()

    retrieve_documents(
        index_dir=args.index_dir,
        queries_file=args.queries,
        output_file=args.run,
        model=args.model,
        k1=args.k1,
        b=args.b,
        mu=args.mu,
        top_k=args.hits,
        run_id=args.tag
    )


if __name__ == '__main__':
    main()

