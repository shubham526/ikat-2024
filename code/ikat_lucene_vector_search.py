import argparse
import logging
import torch
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import jnius_config
from jnius import autoclass
import pandas as pd

# Set up the JVM classpath
lucene_classpath = "/home/user/ikat/lucene_jar/lucene-9.11.1/modules/*"
jnius_config.set_classpath(lucene_classpath)

# Configure logging
def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    return logging.getLogger(__name__)

logger = configure_logging()

def load_queries(query_file):
    queries = pd.read_csv(query_file, sep='\t', header=None, names=['query_id', 'query'])
    return queries

def encode_query(model, tokenizer, query_texts, max_length, device):
    query_tokens = tokenizer(query_texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(device)
    with torch.no_grad():
        embeddings = model(**query_tokens)[0][:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()

def numpy_to_java_float_array(np_array):
    try:
        Float = autoclass('java.lang.Float')
        JavaArray = autoclass('java.lang.reflect.Array')
        java_float_array = JavaArray.newInstance(Float.TYPE, len(np_array))
        for i in range(len(np_array)):
            java_float_array[i] = float(np_array[i])
        return java_float_array
    except Exception as e:
        logger.error(f"Error converting numpy array to Java float array: {e}")
        return None

def search_index(args):
    FSDirectory = autoclass('org.apache.lucene.store.FSDirectory')
    DirectoryReader = autoclass('org.apache.lucene.index.DirectoryReader')
    IndexSearcher = autoclass('org.apache.lucene.search.IndexSearcher')
    Path = autoclass('java.nio.file.Paths')
    KnnFloatVectorQuery = autoclass('org.apache.lucene.search.KnnFloatVectorQuery')

    # Load the index
    index_path = FSDirectory.open(Path.get(args.index_dir))
    reader = DirectoryReader.open(index_path)
    searcher = IndexSearcher(reader)

    # Load and encode query
    model_name = 'Snowflake/snowflake-arctic-embed-m'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model.eval()

    device = torch.device(args.device)
    model.to(device)
    logger.info(f"Using device {args.device}.")

    queries = load_queries(args.queries)
    query_texts = queries['query'].tolist()
    query_embeddings = encode_query(model, tokenizer, query_texts, args.max_length, device)

    with open(args.run, 'w') as f:
        tag = "Lucene-KNN"  # You can set this to any tag you want
        for query_id, query_vector in zip(queries['query_id'], query_embeddings):
            java_query_vector = numpy_to_java_float_array(query_vector)
            query = KnnFloatVectorQuery("content_vector", java_query_vector, 1)
            top_docs = searcher.search(query, 100)
            hits = top_docs.scoreDocs

            for rank, hit in enumerate(hits):
                doc = searcher.storedFields().document(hit.doc)
                doc_id = doc.get("docno")  # Assuming docno is stored in the index
                score = hit.score
                f.write(f"{query_id} Q0 {doc_id} {rank + 1} {score} {tag}\n")

def main():
    parser = argparse.ArgumentParser("Search documents in a Lucene index using HuggingFace transformers.")
    parser.add_argument('--index-dir', help='Path to the Lucene index directory.', required=True)
    parser.add_argument('--queries', help='TSV file containing queries.', required=True)
    parser.add_argument('--run', help='Output file for TREC format run results.', required=True)
    parser.add_argument("--device", help='Device to use for computation (e.g., "cpu", "cuda").', default='cpu', type=str)
    parser.add_argument("--max-length", help='Maximum length for model inputs. Default: 512.', default=512, type=int)

    args = parser.parse_args()
    try:
        search_index(args)
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == '__main__':
    main()
