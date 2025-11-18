# Coconut RAG (example)

This repository is a minimal Retrieval-Augmented Generation (RAG) skeleton using OpenAI embeddings and FAISS.

Getting started

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and add your OpenAI API key.

3. Ingest data to build the FAISS index:

```powershell
python scripts\ingest.py
```

4. Run the API:

```powershell
python main.py
```

5. Query the API (example using curl):

```powershell
curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"query":"What is Coconut RAG?","top_k":2}'
```

Notes
- This is a minimal starting point. For production use you should add error handling, batching for embeddings, more robust persistence, authentication, and tests.
