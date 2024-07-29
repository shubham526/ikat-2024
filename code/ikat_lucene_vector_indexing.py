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
import jnius_config

# Set up the JVM classpath
lucene_classpath = "/home/user/ikat/lucene_jar/lucene-9.11.1/modules/*"
jnius_config.set_classpath(lucene_classpath)

from jnius import autoclass

# Configure logging
def configure_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    return logging.getLogger(__name__)

logger = configure_logging()

def process_document(doc_id, text):
    try:
        if len(text) and any(letter.isalnum() for letter in text):
            return {'docno': doc_id.strip(), 'text': text.strip()}
    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}")
    return None

def document_generator(bz2_files: List[str]) -> dict:
    for bz2_file in bz2_files:
        try:
            with bz2.open(bz2_file, 'rt') as f:
                for line in f:
                    doc = json.loads(line)
                    if 'contents' in doc:
                        processed_doc = process_document(doc['id'], doc['contents'])
                        if processed_doc:
                            yield processed_doc
                    else:
                        logger.warning(f"Document in {bz2_file} missing 'contents' field: {doc}")
        except Exception as e:
            logger.error(f"Error reading file {bz2_file}: {e}")

def encode_documents(model, tokenizer, document_texts, max_length, device):
    try:
        document_tokens = tokenizer(document_texts, padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(device)
        with torch.no_grad():
            embeddings = model(**document_tokens)[0][:, 0]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()
    except Exception as e:
        logger.error(f"Error encoding documents: {e}")
        return None

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

def create_lucene_index(index_dir):
    try:
        logger.info(f"Creating Lucene index at {index_dir}.")
        FSDirectory = autoclass('org.apache.lucene.store.FSDirectory')
        Path = autoclass('java.nio.file.Paths')
        EnglishAnalyzer = autoclass('org.apache.lucene.analysis.en.EnglishAnalyzer')
        IndexWriterConfig = autoclass('org.apache.lucene.index.IndexWriterConfig')
        IndexWriter = autoclass('org.apache.lucene.index.IndexWriter')

        index_path = FSDirectory.open(Path.get(index_dir))
        analyzer = EnglishAnalyzer()
        config = IndexWriterConfig(analyzer)
        writer = IndexWriter(index_path, config)
        return writer
    except Exception as e:
        logger.error(f"Error creating Lucene index: {e}")
        return None

def create_index(args, bz2_file_lines):
    try:
        bz2_files = sorted(glob.glob(os.path.join(args.data_dir, '*.bz2')))
        total_lines = sum(bz2_file_lines.get(os.path.basename(file), 0) for file in bz2_files)
        total_batches = (total_lines + args.batch_size - 1) // args.batch_size

        model_name = 'Snowflake/snowflake-arctic-embed-m'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        model.eval()

        device = torch.device(args.device)
        model.to(device)
        logger.info(f"Using device {args.device}.")

        writer = create_lucene_index(args.index_dir)
        if writer is None:
            raise RuntimeError("Failed to create Lucene index writer")

        document_texts = []
        batch_size = args.batch_size

        Document = autoclass('org.apache.lucene.document.Document')
        TextField = autoclass('org.apache.lucene.document.TextField')
        StringField = autoclass('org.apache.lucene.document.StringField')
        Field = autoclass('org.apache.lucene.document.Field')
        FieldStore = autoclass('org.apache.lucene.document.Field$Store')
        KnnFloatVectorField = autoclass('org.apache.lucene.document.KnnFloatVectorField')
        VectorSimilarityFunction = autoclass('org.apache.lucene.index.VectorSimilarityFunction')

        doc_generator = document_generator(bz2_files)

        with tqdm(total=total_batches, desc="Encoding and adding documents") as pbar:
            for doc in doc_generator:
                document_texts.append(doc['text'])

                if len(document_texts) >= batch_size:
                    embeddings = encode_documents(model, tokenizer, document_texts, args.max_length, device)
                    if embeddings is None:
                        continue
                    for i, embedding in enumerate(embeddings):
                        try:
                            lucene_doc = Document()
                            lucene_doc.add(StringField("docno", doc['docno'], FieldStore.YES))
                            lucene_doc.add(TextField("text", document_texts[i], FieldStore.YES))
                            java_embedding = numpy_to_java_float_array(embedding)
                            if java_embedding is None:
                                continue
                            lucene_doc.add(KnnFloatVectorField("content_vector", java_embedding, VectorSimilarityFunction.COSINE))
                            writer.addDocument(lucene_doc)
                        except Exception as e:
                            logger.error(f"Error adding document {doc['docno']}: {e}")
                    document_texts = []
                    pbar.update(1)

            if document_texts:
                embeddings = encode_documents(model, tokenizer, document_texts, args.max_length, device)
                if embeddings is None:
                    return
                for i, embedding in enumerate(embeddings):
                    try:
                        lucene_doc = Document()
                        lucene_doc.add(StringField("docno", doc['docno'], FieldStore.YES))
                        lucene_doc.add(TextField("text", document_texts[i], FieldStore.YES))
                        java_embedding = numpy_to_java_float_array(embedding)
                        if java_embedding is None:
                            continue
                        lucene_doc.add(KnnFloatVectorField("content_vector", java_embedding, VectorSimilarityFunction.COSINE))
                        writer.addDocument(lucene_doc)
                    except Exception as e:
                        logger.error(f"Error adding document {doc['docno']}: {e}")
                pbar.update(1)

        writer.commit()
        writer.close()

        logger.info(f"Saved Lucene index to {args.index_dir}")
    except Exception as e:
        logger.error(f"Error in create_index: {e}")

def main():
    parser = argparse.ArgumentParser("Encode documents from bz2 files using HuggingFace transformers and save to Lucene index.")
    parser.add_argument("--data-dir", help='Path to bz2 files.', required=True)
    parser.add_argument('--bz2-file-lines', help='JSON file containing the number of lines per bz2 file.', required=True)
    parser.add_argument('--index-dir', help='Directory to save the encoded document embeddings and metadata.', required=True)
    parser.add_argument('--batch-size', help='Batch size for encoding. Default: 200.', default=200, type=int)
    parser.add_argument("--max-length", help='Maximum length for model inputs. Default: 512.', default=512, type=int)
    parser.add_argument("--device", help='Device to use for computation (e.g., "cpu", "cuda").', default='cpu', type=str)

    args = parser.parse_args()

    try:
        with open(args.bz2_file_lines, 'r') as f:
            bz2_file_lines = json.load(f)

        create_index(args, bz2_file_lines)
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == '__main__':
    main()
