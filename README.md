# Code for TREC iKAT 2024 track

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

### Offset Calculation Script

The `calculate_offsets.py` script is designed to calculate and store offsets for bz2-compressed JSONL files. These offsets can be used to efficiently access specific lines within the bz2 files.

#### Usage

To run the offset calculation script, use the following command:

```bash
python calculate_offsets.py --data-dir <data_directory> --offset-file <output_offset_file.json>
```
#### Arguments
- `--data-dir`: Directory containing bz2 files with JSONL data.
- `--offset-file`: Output file to store the calculated offsets.

### Sharding Script
The `shard_dataset.py` script is designed to divide the iKAT dataset into multiple smaller files (shards) based on a consistent hash of the document IDs. If you find that you are running out of memory when trying to index the iKAT collection, you can shard the dataset and then index the shards individually. 

#### Usage
To run the sharding script, use the following command:
```bash
python shard_dataset.py --data-dir <data_directory> --shard-dir <shard_directory> [--id-field <id_field>] [--text-field <text_field>] [--num-shards <num_shards>]
```
#### Arguments
- `--data-dir`: Directory containing bz2 files with JSONL data.
- `--shard-dir`: Directory to save the shard files.
- `--id-field` (optional): DocID field in data (default is doc_id).
- `--text-field` (optional): Text field in data (default is contents).
- `--num-shards` (optional): Number of shards to create (default is 4).

#### Example usage
Here's how you can create a SPLADE++ index using PyTerrier. 

```python
    for i, shard in enumerate(shards):
        shard_index_name = f"{args.index_name}_shard_{i}"
        shard_index_path = os.path.join(args.index_dir, shard_index_name)
        os.makedirs(shard_index_path, exist_ok=True)

        indexer = pt.index.IterDictIndexer(
            index_path=shard_index_path,
            fields=['text'],
            pretokenised=True,
            meta={'docno': 40, 'text': 10000}
        )

        indexer_pipe = splade.indexing() >> indexer

        print(f'Starting to index shard {shard_index_name}...')
        indexer_pipe.index(
            iter(shard),
            batch_size=args.batch_size
        )
        print(f'[Done] indexing shard {shard_index_name}.')
```



### Creating a DeepCT index using PyTerrier

The `ikat_deepct_index_creater.py` script is designed to create a PyTerrier DeepCT index for the iKAT dataset, processing documents stored in bz2-compressed JSONL files.

#### Usage

To run the indexing script, use the following command:

```bash
python ikat_deepct_index_creater.py --bz2-dir <data_directory> --index-dir <index_directory> --bz2-file-lines <bz2_file_lines.json> [--batch-size <batch_size>] [--cuda <cuda_device>]
```
#### Arguments
- `--bz2-dir`: Directory containing bz2 files with JSONL data.
- `--index-dir`: Directory where the index will be saved.
- `--bz2-file-lines`: Path to the JSON file containing the number of lines in each bz2 file.
- `--batch-size` (optional): Batch size for processing (default is 200).
- `--cuda` (optional): CUDA device to use for processing (default is 0).

### Creating a dense vector index using Lucene 9.11.1 

The `ikat_lucene_vector_indexing.py` script is designed to encode documents from bz2-compressed JSONL files using HuggingFace transformers and save the encoded document embeddings and metadata to a Lucene index. This script will index the documents using Lucene and store the embeddings in a retrievable format. With this setup, you can leverage the full power of Lucene through Python using Pyjnius.

#### Setting Up Java and Pyjnius
Make sure you have Java installed on your system and properly configured. Pyjnius will need to find your Java installation. You may need to set the `JAVA_HOME` environment variable to point to your JDK installation.

To use Lucene with Pyjnius, you need to download the core Lucene JAR files and their dependencies. Specifically, you will need:
1. Lucene Core JAR: This is the main library.
2. Lucene Analysis Common JAR: Provides standard analyzers.
3. Lucene Query Parser JAR: Enables the use of query parsing features.
4. Lucene Backward Codecs JAR: Provides backward compatibility with older Lucene indexes.

You can download these JAR files from the [Apache Lucene website](https://lucene.apache.org/core/downloads.html) or use Maven Central to get the latest versions. Here are the files you need to download:

##### Required JAR Files
- `lucene-core.jar`
- `lucene-analyzers-common.jar`
- `lucene-queryparser.jar`
- `lucene-backward-codecs.jar`

##### Steps to Download
1. Visit the Lucene Download Page:
   Go to the [Apache Lucene download page](https://lucene.apache.org/core/downloads.html).

2. Download the Binaries:
   Download the binary release (`lucene-9.11.1.tgz`).

3. Extract the Archive:
   Extract the downloaded archive to get the required JAR files. You can use a tool like `tar` or any other archive manager.
   ```sh
   tar -xvzf lucene-<version>-bin.tgz
   ```
4. Locate the JAR Files:
   Navigate to the extracted directory, and you will find the JAR files in the `lucene-core-<version>.jar`, `lucene-analyzers-common-<version>.jar`, `lucene- 
   queryparser-<version>.jar`, and `lucene-backward-codecs-<version>.jar`.
   
6. Include JARs in the Classpath:
   Use these paths to set up the JVM classpath in your Python script. For example:
   ```python
   lucene_classpath = "/home/user/ikat/lucene_jar/lucene-9.11.1/modules/*"
   jnius_config.set_classpath(lucene_classpath)
   ```

#### Usage

To run the indexing script, use the following command:

```bash
python ikat_lucene_vector_indexing.py --data-dir <data_directory> --index-dir <index_directory> --bz2-file-lines <bz2_file_lines.json> [--batch-size <batch_size>] [--max-length <max_length>] [--device <device>]
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
python ikat_splade_index_creater.py --bz2-dir <data_directory> --index-dir <index_directory> --bz2-file-lines <bz2_file_lines.json> [--batch-size <batch_size>]
```
#### Arguments
- `--bz2-dir`: Directory containing bz2 files with JSONL data.
- `--index-dir`: Directory where the SPLADE index will be saved.
- `--bz2-file-lines`: Path to the JSON file containing the number of lines in each bz2 file.
- `--batch-size` (optional): Batch size for indexing (default is 200).

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request
   
## License
This project is licensed under the MIT License.



