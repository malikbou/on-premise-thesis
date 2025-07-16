#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build (or load) a shared FAISS vector index for the document corpus.

Key points
==========
* Embedding model is **fixed** to `all-MiniLM-L6-v2` (served locally by Ollama).
* Index path and chunking parameters mirror those in benchmark.py so the main
  benchmarking script can reuse the generated index without changes.
* Optional CLI flags allow you to force-rebuild the index, tweak chunk size /
  overlap, and run quick retrieval diagnostics against the existing Ragas
  test-set.

Usage examples
--------------
# Build index (if not already cached)
python -m src.build_embeddings

# Force rebuild with custom chunking and evaluate retrieval quality
python -m src.build_embeddings --force --chunk-size 800 --chunk-overlap 100 \
    --eval --sample 10 --top-k 5
"""

import os
import time
import json
import argparse
import gc
from typing import List

import pandas as pd
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from ragas import evaluate, EvaluationDataset
from ragas.metrics import context_precision, context_recall, ContextRelevance

# -------- Constants (mirrored from benchmark.py) --------
INDEX_DIR = os.path.join(".rag_cache", "faiss_index")
DOCS_PATH = "data/"  # default corpus directory
EMBED_MODEL = "tazarov/all-MiniLM-L6-v2-f32"  # fixed embedding model

DEFAULT_CHUNK_SIZE = 400
DEFAULT_CHUNK_OVERLAP = 50

# --------------------------------------------------------

def build_or_load_index(docs: List, chunk_size: int, chunk_overlap: int):
    """Return (vectorstore, retriever). Creates index if needed."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    index_file = os.path.join(INDEX_DIR, "index.faiss")
    if os.path.exists(index_file):
        try:
            print("Loading existing FAISS index …")
            vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
            return vs, vs.as_retriever()
        except Exception as e:
            print(f"Failed to load existing index (will rebuild): {e}")

    print("Building new FAISS index … this may take a few minutes.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks (avg {chunk_size} chars, overlap {chunk_overlap}).")

    vs = FAISS.from_documents(chunks, embedding=embeddings)
    vs.save_local(INDEX_DIR)
    print(f"Saved index to {INDEX_DIR}")
    return vs, vs.as_retriever()


def quick_retrieval_eval(retriever, testset_file: str, top_k: int, sample: int, judge_model: str):
    """Run lightweight retrieval-only metrics (no LLM answers)."""
    with open(testset_file) as f:
        data = json.load(f)
    df = pd.DataFrame(data).head(sample)
    records = []
    for _, row in df.iterrows():
        q = row["user_input"]
        gt_contexts = row["reference_contexts"]
        docs = retriever.get_relevant_documents(q)
        rec = {
            "user_input": q,
            "retrieved_contexts": [d.page_content for d in docs],
            "reference_contexts": gt_contexts,
            # dummy fields required by EvaluationDataset
            "response": "dummy",
            "reference": row.get("reference", "")
        }
        records.append(rec)

    dataset = EvaluationDataset.from_list(records)
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    # Use separate judge model for metrics that need chat capability
    llm = ChatOllama(model=judge_model, temperature=0)

    relevance_metric = ContextRelevance()
    metrics = [context_precision, context_recall, relevance_metric]
    results = {}
    for metric in metrics:
        print(f"Evaluating {metric.__class__.__name__} …")
        res = evaluate(dataset=dataset, metrics=[metric], llm=llm, embeddings=embeddings, batch_size=1)
        if hasattr(res, "to_pandas"):
            df_res = res.to_pandas()
            col = df_res.columns[-1]
            results[col] = float(df_res[col].mean())
            print(f"  {col}: {results[col]:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Build or load FAISS index with fixed embeddings model.")
    parser.add_argument("--force", action="store_true", help="Rebuild index even if cached version exists.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    parser.add_argument("--docs", type=str, default=DOCS_PATH, help="Path to documents directory.")
    parser.add_argument("--eval", action="store_true", help="Run retrieval diagnostics using testset.")
    parser.add_argument("--judge-model", type=str, default="phi3:mini", help="Chat-capable model to act as judge for Ragas metrics.")
    parser.add_argument("--testset", type=str, default="testset/ragas_testset.json")
    parser.add_argument("--sample", type=int, default=5, help="Number of questions to evaluate (0 = all).")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    # ---- Load documents ----
    print(f"Loading documents from {args.docs} …")
    loader = DirectoryLoader(args.docs, glob="**/*.pdf")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # ---- Build / load index ----
    if args.force and os.path.exists(INDEX_DIR):
        print("--force supplied: removing cached index …")
        for f in os.listdir(INDEX_DIR):
            os.remove(os.path.join(INDEX_DIR, f))
        gc.collect()
    vstore, retriever = build_or_load_index(docs, args.chunk_size, args.chunk_overlap)

    # ---- Optional evaluation ----
    if args.eval:
        sample_n = args.sample or len(docs)
        metrics_res = quick_retrieval_eval(retriever, args.testset, args.top_k, sample_n, args.judge_model)
        print("Retrieval evaluation complete.")

        # Persist metrics to disk for later comparison
        os.makedirs("runs", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join("runs", f"embedding_metrics_{ts}.json")
        payload = {
            "embedding_model": EMBED_MODEL,
            "judge_model": args.judge_model,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "top_k": args.top_k,
            "sample_questions": sample_n,
            "metrics": metrics_res,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
