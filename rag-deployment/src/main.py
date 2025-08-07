import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain components for RAG
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.chat_models.litellm import ChatLiteLLM


# Load environment variables from a .env file
load_dotenv()

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

# This dictionary will hold our "heavy" resources, like the vector store.
# We use a dictionary to allow it to be mutated by the lifespan manager.
rag_resources = {}


# --- Lifespan Management (for Startup and Shutdown) ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Handles startup and shutdown events for the FastAPI application.
    This is the recommended way to manage resources.
    """
    print("--- RAG API is starting up ---")

    # --- 1. Load Documents ---
    print("Loading documents from 'data/' directory...")
    loader = DirectoryLoader(
        "data/",
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # --- 2. Split Documents into Chunks ---
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} document chunks.")

    # --- 3. Create Embeddings and Vector Store ---
    # We use a local Ollama model for embeddings, as it's fast and free.
    # This embedding model runs in the `ollama` container.
    print("Creating embeddings and FAISS vector store...")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

    # Ensure the embedding model is pulled locally
    try:
        import requests
        requests.post(f"{ollama_base_url}/api/pull", json={"name": embedding_model_name})
        print(f"Successfully pulled embedding model: {embedding_model_name}")
    except Exception as e:
        print(f"Warning: Could not pull embedding model. It may need to be pulled manually. Error: {e}")

    embeddings = OllamaEmbeddings(
        model=embedding_model_name,
        base_url=ollama_base_url
    )

    # Create the FAISS vector store from the document chunks
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    print("FAISS vector store created successfully.")

    # Store the vector store in our global resources dictionary
    rag_resources["vectorstore"] = vectorstore

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
    # The LLM is accessed via our LiteLLM gateway service.
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
