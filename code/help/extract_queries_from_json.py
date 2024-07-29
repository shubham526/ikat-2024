import json
import pandas as pd
import argparse

def extract_queries(json_file, output_tsv, field):
    # Load the JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract query_id and query
    queries = []
    for entry in data:
        number = entry['number']
        for turn in entry['turns']:
            query_id = f"{number}_{turn['turn_id']}"
            query = turn[field]
            queries.append([query_id, query])

    # Create a DataFrame and write to a TSV file
    queries_df = pd.DataFrame(queries, columns=['query_id', 'query'])
    queries_df.to_csv(output_tsv, sep='\t', header=False, index=False)

def main():
    parser = argparse.ArgumentParser(description="Extract query_id and query from JSON and write to TSV file.")
    parser.add_argument("--json-file", help='Input JSON file containing the data.', required=True)
    parser.add_argument("--output", help='Output TSV file to save the extracted queries.', required=True)
    parser.add_argument("--field", help='Which query field to extract?.', required=True)
    args = parser.parse_args()

    extract_queries(args.json_file, args.output, args.field)

if __name__ == '__main__':
    main()
