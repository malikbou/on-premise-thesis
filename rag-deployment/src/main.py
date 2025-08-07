import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain components
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI # Use the standard OpenAI client

# Load environment variables
load_dotenv()

# --- Configuration ---
INDEX_DIR = ".rag_cache/faiss_index"
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://litellm:4000")

# --- Data Models ---
class QueryRequest(BaseModel):
    question: str
    model_name: str = Field(default="ollama/phi3:mini")

class Document(BaseModel):
    page_content: str
    metadata: dict

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Document]

# --- Global Resources ---
rag_resources = {}

# --- Lifespan Management ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- RAG API is starting up ---")

    print(f"Loading FAISS index from '{INDEX_DIR}'...")
    if not os.path.exists(INDEX_DIR):
        raise RuntimeError(f"FAISS index not found at {INDEX_DIR}. Run the index builder first.")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL_NAME, base_url=OLLAMA_BASE_URL)

    vectorstore = FAISS.load_local(
        INDEX_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
    rag_resources["vectorstore"] = vectorstore
    print("FAISS index loaded successfully.")

    yield

    print("--- RAG API is shutting down ---")
    rag_resources.clear()

# --- FastAPI Application ---
app = FastAPI(title="RAG API", lifespan=lifespan)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/query", response_model=QueryResponse)
async def query_rag_pipeline(request: QueryRequest) -> QueryResponse:
    vectorstore = rag_resources.get("vectorstore")
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Vector store not available.")

    # Use the standard ChatOpenAI client pointed at our LiteLLM proxy
    llm = ChatOpenAI(
        model=request.model_name,
        openai_api_base=LITELLM_API_BASE,
        openai_api_key="anything" # LiteLLM doesn't require a key for local models
    )

    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    result = await qa_chain.ainvoke({"query": request.question})

    return QueryResponse(
        answer=result["result"],
        source_documents=[
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in result["source_documents"]
        ]
    )
