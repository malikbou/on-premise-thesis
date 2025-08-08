import os
import re
import requests
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

# Load environment variables from a .env file
load_dotenv()

# --- Configuration ---
DATA_DIR = "data/"
INDEX_DIR = os.getenv("INDEX_DIR")  # may be None; we'll derive if missing
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

def main():
    """
    This script builds the FAISS vector store from the documents in the data directory
    and saves it to disk. This is a one-time process that should be run before
    starting the API server for the first time.
    """
    print("--- Starting FAISS Index Build ---")

    # --- 1. Load Documents ---
    print(f"Loading documents from '{DATA_DIR}' directory...")
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True
    )
    docs = loader.load()
    if not docs:
        print("No documents found. Exiting.")
        return
    print(f"Loaded {len(docs)} documents.")

    # --- 2. Split Documents into Chunks ---
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    splits = text_splitter.split_documents(docs)
    print(f"Created {len(splits)} document chunks.")

    # --- 3. Create Embeddings ---
    print(f"Initializing embedding model: {EMBEDDING_MODEL_NAME}...")

    # Ensure the embedding model is pulled locally in the ollama container
    try:
        print(f"Pulling embedding model '{EMBEDDING_MODEL_NAME}' from Ollama...")
        requests.post(f"{OLLAMA_BASE_URL}/api/pull", json={"name": EMBEDDING_MODEL_NAME}, timeout=600)
        print(f"Successfully pulled embedding model.")
    except Exception as e:
        print(f"Warning: Could not pull embedding model. It may need to be pulled manually. Error: {e}")

    # Derive INDEX_DIR if not provided via env
    global INDEX_DIR
    if not INDEX_DIR:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", EMBEDDING_MODEL_NAME.lower())
        INDEX_DIR = f".rag_cache/{slug}/faiss_index"
    print(f"Using index directory: '{INDEX_DIR}'")

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        keep_alive=0,
    )

    # --- 4. Create and Save FAISS Vector Store ---
    print("Creating FAISS vector store from document chunks...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)

    print(f"Saving FAISS index to '{INDEX_DIR}'...")
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)

    print("--- FAISS Index Build Complete ---")

if __name__ == "__main__":
    main()
