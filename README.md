# TalkWithNepaliDocument

A Retrieval-Augmented Generation (RAG) system for Nepali legal documents. This project uses Vespa as a vector database, FastAPI for the backend, and Streamlit for the frontend.

## Data Ingestion & Embeddings

Before running the application, you need to generate embeddings and upload them to Vespa.

- Please refer to the notebooks
- These notebook contains the steps for processing Nepali documents, creating vector embeddings, and indexing them into the Vespa instance.

## Deployment with Docker

The easiest way to run the entire stack (FastAPI backend and Streamlit frontend) is using Docker Compose.

1. **Prerequisites**: Ensure you have Docker and Docker Compose installed.
2. **Environment Variables**: Copy `.env.example` to `.env` and fill in the required API keys.
   ```bash
   cp .env.example .env
   ```
3. **Run the Application**:
   ```bash
   docker compose up --build
   ```

The Streamlit frontend will be available at `http://localhost:8501`.
The FastAPI backend will be available at `http://localhost:8080`.

## Project Structure

- `backend/`: FastAPI application code.
- `frontend/`: Streamlit interface code.
- `ai/`: Shared AI/RAG logic.
- `Analysis/`: Notebooks for data analysis and Vespa configuration.