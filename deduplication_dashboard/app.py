from flask import Flask, render_template, request
import requests
import json

app = Flask(__name__)

# Read duplicate passage IDs from a file and group by document ID
def read_duplicates(file_path):
    duplicates = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            doc_id, passage_id = line.rsplit(':', 1)
            if doc_id not in duplicates:
                duplicates[doc_id] = []
            duplicates[doc_id].append(passage_id)
    return duplicates

# Fetch the content of a passage using the provided API
def fetch_passage_content(doc_id, passage_id):
    url = f"https://ikat-searcher.grill.science/api/fulltext/{doc_id}/{passage_id}"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"body": "Error parsing content"}
    else:
        return {"body": "Error fetching content"}

@app.route('/')
def index():
    num_docs = int(request.args.get('num_docs', 10))
    page = int(request.args.get('page', 1))
    duplicates = read_duplicates('duplicates.txt')
    
    # Paginate documents
    doc_ids = list(duplicates.keys())[:num_docs]
    total_docs = len(doc_ids)
    if page > total_docs:
        page = total_docs
    if page < 1:
        page = 1
    doc_id = doc_ids[page - 1]

    doc_contents = {}
    passages = []
    for passage_id in duplicates[doc_id]:
        content = fetch_passage_content(doc_id, passage_id)
        passages.append((passage_id, content))
    doc_contents[doc_id] = passages
    
    return render_template('index.html', doc_contents=doc_contents, page=page, num_docs=num_docs, total_docs=total_docs)

if __name__ == '__main__':
    app.run(debug=True)

