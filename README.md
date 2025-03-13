# FASTAPI-rag-boilerplate

This application is a FastAPI service that processes PDF files and answers queries based on the content of those files using a retrieval-augmented generation (RAG) approach.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Running the Application

To run the application, use the following command:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The application will be accessible at `http://localhost:8000`.
