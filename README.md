# Code for TREC iKAT 2024 track.

## Recreating the Conda Environment

To recreate the conda environment used for this project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ikat-2024.git
   cd ikat-2024
   ```
2. Create the conda environment from the `environment.yml` file:
   ```bash
    conda env create -f environment.yml
   ```
3. Activate the new environment:
   ```bash
    conda activate trec-ikat
   ```

## Running the code

### Deduplication script

The `deduplicate.py` script is designed to remove duplicate paragraphs from a collection of bz2-compressed JSONL files. It leverages SentenceBERT embeddings and MinHashLSH for efficient deduplication.

#### Usage

To run the deduplication script, use the following command:

```bash
python deduplicate.py --data-dir <data_directory> --bz2-file-lines <bz2_file_lines.json> --save <output_file.json> --duplicate-ids <duplicates_file.txt> [--batch-size <batch_size>] [--num-perm <num_perm>] [--threshold <threshold>]
```
#### Arguments
- `--data-dir`: Directory containing bz2 files with JSONL data.
- `--bz2-file-lines`: Path to the JSON file containing the number of lines in each bz2 file.
- `--save`: Path to the output file where deduplicated data will be saved.
- `--duplicate-ids`: Path to the file where duplicate IDs will be saved.
- `--batch-size` (optional): Batch size for processing (default is 1000).
- `--num-perm` (optional): Number of permutations for MinHash (default is 128).
- `--threshold` (optional): Similarity threshold for MinHash LSH (default is 0.9).

### Creating a DeepCT index using PyTerrier

The `ikat_deepct_index_creater.py` script is designed to create a PyTerrier DeepCT index for the iKAT dataset, processing documents stored in bz2-compressed JSONL files.

#### Usage

To run the indexing script, use the following command:

```bash
python indexing.py --bz2-dir <data_directory> --index-dir <index_directory> --bz2-file-lines <bz2_file_lines.json> [--batch-size <batch_size>] [--cuda <cuda_device>]
```
#### Arguments
- `--bz2-dir`: Directory containing bz2 files with JSONL data.
- `--index-dir`: Directory where the index will be saved.
- `--bz2-file-lines`: Path to the JSON file containing the number of lines in each bz2 file.
- `--batch-size` (optional): Batch size for processing (default is 200).
- `--cuda` (optional): CUDA device to use for processing (default is 0).

### Creating a dense vector index using Lucene 9.11.1 

The `ikat_lucene_vector_indexing.py` script is designed to encode documents from bz2-compressed JSONL files using HuggingFace transformers and save the encoded document embeddings and metadata to a Lucene index.

#### Usage

To run the indexing script, use the following command:

```bash
python indexing_lucene.py --data-dir <data_directory> --index-dir <index_directory> --bz2-file-lines <bz2_file_lines.json> [--batch-size <batch_size>] [--max-length <max_length>] [--device <device>]
```
#### Arguments
- `--data-dir`: Path to the directory containing bz2 files with JSONL data.
- `--index-dir`: Directory where the Lucene index will be saved.
- `--bz2-file-lines`: Path to the JSON file containing the number of lines in each bz2 file.
- `--batch-size` (optional): Batch size for encoding (default is 200).
- `--max-length` (optional): Maximum length for model inputs (default is 512).
- `--device` (optional): Device to use for computation (e.g., "cpu", "cuda") (default is "cpu").

### Creating a SPLADE++ index using PyTerrier

The `ikat_splade_index_creater.py` script is designed to create a PyTerrier SPLADE++ index for the iKAT dataset, processing documents stored in bz2-compressed JSONL files.

#### Usage

To run the indexing script, use the following command:

```bash
python indexing_splade.py --bz2-dir <data_directory> --index-dir <index_directory> --bz2-file-lines <bz2_file_lines.json> [--batch-size <batch_size>]
```
#### Arguments
- `--bz2-dir`: Directory containing bz2 files with JSONL data.
- `--index-dir`: Directory where the SPLADE index will be saved.
- `--bz2-file-lines`: Path to the JSON file containing the number of lines in each bz2 file.
- `--batch-size` (optional): Batch size for indexing (default is 200).




