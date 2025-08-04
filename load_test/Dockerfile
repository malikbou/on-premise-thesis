# syntax=docker/dockerfile:1

# -------- Base image (arm64) --------
FROM --platform=linux/arm64 python:3.11-slim AS base

# Install system deps (FAISS prereqs) and create non-root user
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# -------- Copy project --------
# Copy only what we need: requirements then code (leverages Docker cache)
COPY requirements.txt ./
COPY load_test/requirements-load.txt ./requirements-load.txt

# Install Python deps (split: core + load-harness extras)
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-load.txt

# Copy the rest of the repository (for src/ imports, data, etc.)
COPY . .

# Default environment vars
ENV PYTHONUNBUFFERED=1 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    EMBEDDING_MODEL=all-minilm \
    CHAT_MODEL=mistral:7b

# -------- Runtime --------
EXPOSE 8000
CMD ["uvicorn", "load_test.rag_server:app", "--host", "0.0.0.0", "--port", "8000"]
