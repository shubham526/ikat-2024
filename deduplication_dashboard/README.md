# TREC iKAT ClueWeb22-B Deduplication Dashboard

## Overview

This project provides a web interface for viewing the duplicate passages in the TREC iKAT ClueWeb22-B collection. The interface allows users to view documents and their associated (duplicate) passages.

## Features

- Display documents and their duplicate passages.
- Pagination for easy navigation through documents.
- Option to specify the number of documents to review.
- Fetch passage content dynamically via API.

## Requirements

- Python 3.9+
- Flask
- requests

## Usage

1. Ensure you have a file named `duplicates.txt` in the root directory containing the duplicate passage IDs.
2. Run the Flask application:
   ```bash
   python app.py
   ```
3. Open your web browser and navigate to:
   ```arduino
   http://127.0.0.1:5000/
   ```
4. Specify the number of documents you want to review in the input form and use the pagination controls to navigate through the documents.

## URL Parameters

There are several hundreds of thousands of duplicate passages. You probably don't want to view all of them. You can control the number of documents to display and navigate through the pages by modifying the URL parameters directly in your browser's address bar:

- `num_docs`: Specifies the number of documents to review from the duplicates.txt file.
- `page`: Specifies the page number to view.

### Examples

1. First Page with 10 Documents to Review:
```ruby
http://127.0.0.1:5000/?num_docs=10
```
2. Second Page with 10 Documents to Review:
```ruby
http://127.0.0.1:5000/?page=2&num_docs=10
```
3. First Page with 20 Documents to Review:
```ruby
http://127.0.0.1:5000/?num_docs=20
```
4. Third Page with 15 Documents to Review:
```ruby
http://127.0.0.1:5000/?page=3&num_docs=15
```

## API
The passage content is fetched using the following API:
```php
https://ikat-searcher.grill.science/api/fulltext/<doc_id>/<passage_id>
```

