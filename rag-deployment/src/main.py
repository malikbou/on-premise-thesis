import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain components for RAG
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models.litellm import ChatLiteLLM


# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
INDEX_DIR = ".rag_cache/faiss_index"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")


# --- Data Models for API ---

class QueryRequest(BaseModel):
    """Request model for the /query endpoint."""
    question: str
    model_name: str = Field(
        default="ollama/phi3:mini",
        description="The model to use for the query (e.g., ollama/phi3:mini, gpt-4o, claude-3-opus)."
    )

class Document(BaseModel):
    """Response model for a single retrieved document."""
    page_content: str
    metadata: dict

class QueryResponse(BaseModel):
    """Response model for the /query endpoint."""
    answer: str
    source_documents: List[Document]


# --- Global Variables ---
rag_resources = {}


# --- Lifespan Management (for Startup and Shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    This function now loads the pre-built FAISS index from disk.
    """
    print("--- RAG API is starting up ---")

    # --- 1. Load the Pre-built FAISS Index ---
    print(f"Loading FAISS index from '{INDEX_DIR}'...")

    if not os.path.exists(INDEX_DIR):
        raise RuntimeError(
            f"FAISS index not found at {INDEX_DIR}. "
            "Please run the index builder service first using: "
            "'docker compose up index-builder'"
        )

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=OLLAMA_BASE_URL
    )

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True # This is required for FAISS
    )
    rag_resources["vectorstore"] = vectorstore
    print("FAISS index loaded successfully.")

    yield # The API is now running

    # --- Shutdown Logic ---
    print("--- RAG API is shutting down ---")
    rag_resources.clear()


# --- FastAPI Application ---

app = FastAPI(title="RAG API", lifespan=lifespan)

@app.get("/health")
def health_check():
    """A simple endpoint to confirm the API is running."""
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_rag_pipeline(request: QueryRequest) -> QueryResponse:
    """
    Receives a question, retrieves relevant context, and generates an answer.
    """
    vectorstore = rag_resources.get("vectorstore")
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store not available.")

    # --- Create the RAG Chain ---
    llm = ChatLiteLLM(model=request.model_name, api_base="http://litellm:4000")
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # --- Invoke the Chain and Return the Response ---
    result = await qa_chain.ainvoke({"query": request.question})

    return QueryResponse(
        answer=result["result"],
        source_documents=[
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in result["source_documents"]
        ]
    )
