# Kantify RAG

This module implements a Retrieval Augmented Generation (RAG) system for analyzing and querying philosophical texts, specifically focused on Kant's work.

## Prerequisites

- Python 3.13 or higher
- An OpenAI API key
- uv (optional)

## Installation

1. Clone the repository:

```bash
git clone https://github.com/juanejd/Kantify-app.git
cd Kantify-app/rag
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On Unix/MacOS
source .venv/bin/activate
# Using uv
uv .venv
```

3. Install dependencies:

```bash
pip install -e .
```

4. Set up environment variables:
   - Create a `.env` file in the `rag` folder
   - Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Project Structure

```
rag/
├── data/               # Folder for PDF documents
├── chroma/            # Vector database (git-ignored)
├── main.py            # Main script
├── charge_data.py     # Document loading and processing functions
├── chroma_db.py       # Vector database management functions
└── .env               # Environment variables (git-ignored)
```

## Usage

1. Place your PDF documents in the `data/` folder

2. Run the main script:

```bash
python main.py
uv run main.py
```

The script will:

- Load and process the documents
- Create embeddings using OpenAI's model
- Store the embeddings in a vector database
- Perform a similarity search with the predefined query

## Features

- PDF document processing
- Text chunking for manageable pieces
- Embeddings using OpenAI (text-embedding-3-large)
- Vector storage with Chroma
- Semantic content search

## Notes

- The `chroma/` folder contains the vector database and is git-ignored
- The database is recreated each time you run the script
- Ensure you have enough disk space for the embeddings

## Main Dependencies

- chromadb>=1.0.9
- langchain>=0.3.25
- langchain-openai>=0.3.17
- pypdf>=5.5.0
- python-dotenv>=1.0.0
