import os
import json
import bz2
import glob
import argparse
import logging
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
# Import jnius_config and set up the JVM classpath before importing jnius
import jnius_config

# Set up the JVM classpath
lucene_classpath = "/home/user/ikat/lucene_jar/lucene-9.11.1/modules/*"
jnius_config.set_classpath(lucene_classpath)

from jnius import autoclass
import jnius

# Configure logging
def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    return logging.getLogger(__name__)

logger = configure_logging()

def process_document(doc_id, text):
    """
    Process a document and return a formatted string if it meets the criteria.
    """
    if len(text) and any(letter.isalnum() for letter in text):
        return {'docno': doc_id.strip(), 'text': text.strip()}
    return None

def document_generator(bz2_files: List[str]) -> dict:
    """
    Generator to load documents from bz2 files one by one.
    """
    for bz2_file in bz2_files:
        with bz2.open(bz2_file, 'rt') as f:
            for line in f:
                doc = json.loads(line)
                if 'contents' in doc:
                    processed_doc = process_document(doc['id'], doc['contents'])
                    if processed_doc:
                        yield processed_doc
                else:
                    logger.warning(f"Document in {bz2_file} missing 'contents' field: {doc}")

def encode_documents(model, tokenizer, document_texts, max_length, device):
    document_tokens = tokenizer(document_texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(device)

    with torch.no_grad():
        embeddings = model(**document_tokens)[0][:, 0]
    embeddings = F.normalize(embeddings, p=2, dim=1)  # Normalize embeddings
    return embeddings.cpu().numpy()

def numpy_to_java_float_array(np_array):
    """
    Convert a numpy array to a Java float array.
    """
    Float = autoclass('java.lang.Float')
    JavaArray = autoclass('java.lang.reflect.Array')
    java_float_array = JavaArray.newInstance(Float.TYPE, len(np_array))
    for i in range(len(np_array)):
        java_float_array[i] = float(np_array[i])
    return java_float_array


def create_lucene_index(index_dir):
    logger.info(f"Creating Lucene index at {index_dir}.")
    FSDirectory = autoclass('org.apache.lucene.store.FSDirectory')
    Path = autoclass('java.nio.file.Paths')
    # StandardAnalyzer = autoclass('org.apache.lucene.analysis.standard.StandardAnalyzer')
    EnglishAnalyzer = autoclass('org.apache.lucene.analysis.en.EnglishAnalyzer')
    IndexWriterConfig = autoclass('org.apache.lucene.index.IndexWriterConfig')
    IndexWriter = autoclass('org.apache.lucene.index.IndexWriter')

    index_path = FSDirectory.open(Path.get(index_dir))
    # analyzer = StandardAnalyzer()
    analyzer = EnglishAnalyzer()  # Initialize the EnglishAnalyzer
    config = IndexWriterConfig(analyzer)
    writer = IndexWriter(index_path, config)
    return writer

def create_index(args, bz2_file_lines):
    bz2_files = sorted(glob.glob(os.path.join(args.data_dir, '*.bz2')))
    total_lines = sum(bz2_file_lines.get(os.path.basename(file), 0) for file in bz2_files)
    total_batches = (total_lines + args.batch_size - 1) // args.batch_size

    # Load model and tokenizer
    model_name = 'Snowflake/snowflake-arctic-embed-m'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model.eval()

    # Move model to CUDA
    device = torch.device(args.device)
    model.to(device)
    logger.info(f"Using device {args.device}.")

    # Create Lucene index
    writer = create_lucene_index(args.index_dir)

    # Initialize variables for batch processing
    document_texts = []
    batch_size = args.batch_size

    # Lucene classes for adding documents
    Document = autoclass('org.apache.lucene.document.Document')
    TextField = autoclass('org.apache.lucene.document.TextField')
    StringField = autoclass('org.apache.lucene.document.StringField')
    Field = autoclass('org.apache.lucene.document.Field')
    FieldStore = autoclass('org.apache.lucene.document.Field$Store')
    KnnFloatVectorField = autoclass('org.apache.lucene.document.KnnFloatVectorField')
    VectorSimilarityFunction = autoclass('org.apache.lucene.index.VectorSimilarityFunction')

    # Process documents in chunks using generator
    doc_generator = document_generator(bz2_files)

    with tqdm(total=total_batches, desc="Encoding and adding documents") as pbar:
        for doc in doc_generator:
            document_texts.append(doc['text'])

            # If batch size is reached, encode and add to index
            if len(document_texts) >= batch_size:
                embeddings = encode_documents(model, tokenizer, document_texts, args.max_length, device)
                for i, embedding in enumerate(embeddings):
                    lucene_doc = Document()
                    lucene_doc.add(StringField("docno", doc['docno'], FieldStore.YES))
                    lucene_doc.add(TextField("text", document_texts[i], FieldStore.YES))
                    java_embedding = numpy_to_java_float_array(embedding)  # Convert numpy array to Java float array
                    lucene_doc.add(
                        KnnFloatVectorField("content_vector", java_embedding, VectorSimilarityFunction.COSINE))
                    writer.addDocument(lucene_doc)
                document_texts = []  # Clear the list for the next batch
                pbar.update(1)

        # Process any remaining documents
        if document_texts:
            embeddings = encode_documents(model, tokenizer, document_texts, args.max_length, device)
            for i, embedding in enumerate(embeddings):
                lucene_doc = Document()
                lucene_doc.add(StringField("docno", doc['docno'], Field.Store.YES))
                lucene_doc.add(TextField("text", document_texts[i], Field.Store.YES))
                java_embedding = numpy_to_java_float_array(embedding)  # Convert numpy array to Java float array
                lucene_doc.add(KnnFloatVectorField("content_vector", java_embedding, VectorSimilarityFunction.COSINE))
                writer.addDocument(lucene_doc)
            pbar.update(1)

    # Commit and close the writer
    writer.commit()
    writer.close()

    logger.info(f"Saved Lucene index to {args.output_dir}")

def main():
    parser = argparse.ArgumentParser("Encode documents from bz2 files using HuggingFace transformers and save to Lucene index.")
    parser.add_argument("--data-dir", help='Path to bz2 files.', required=True)
    parser.add_argument('--bz2-file-lines', help='JSON file containing the number of lines per bz2 file.', required=True)
    parser.add_argument('--index-dir', help='Directory to save the encoded document embeddings and metadata.', required=True)
    parser.add_argument('--batch-size', help='Batch size for encoding. Default: 200.', default=200, type=int)
    parser.add_argument("--max-length", help='Maximum length for model inputs. Default: 512.', default=512, type=int)
    parser.add_argument("--device", help='Device to use for computation (e.g., "cpu", "cuda").', default='cpu', type=str)

    args = parser.parse_args()

    # Load the bz2 file lines information
    with open(args.bz2_file_lines, 'r') as f:
        bz2_file_lines = json.load(f)

    create_index(args, bz2_file_lines)

if __name__ == '__main__':
    main()
