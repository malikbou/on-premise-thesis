#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory-Optimized Multi-Model Ragas Benchmarking Script

Key Features:
- SHARED EMBEDDINGS: Uses one embedding model for all LLMs for fair comparison.
- MEMORY MANAGEMENT: Monitors RAM usage and provides cleanup options.
- MODEL LIFECYCLE: Efficient pull -> test -> benchmark -> cleanup cycle.
- CONFIGURABLE: Adjustable sample size and cleanup aggressiveness.

Memory Management Modes:
- Conservative (default): Keeps models available for re-testing; Ollama manages memory.
- Aggressive: Completely removes models from the system to free disk space.

Why Shared Embeddings?
- Ensures all LLMs see identical retrieved contexts.
- Eliminates retrieval bias in comparisons.
- Avoids embedding compatibility issues.
- More efficient than recreating vector stores per model.
"""

import os
import json
import pandas as pd
import time
import gc
import subprocess
import traceback
import psutil
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import DirectoryLoader
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    answer_relevancy,  # keep legacy metric for backward-compatibility
    context_precision,
    context_recall,
    AnswerAccuracy,
    ResponseGroundedness,
    ContextRelevance,
)
from tqdm import tqdm

# Configuration
DEFAULT_SAMPLE_SIZE = 3
DEFAULT_TIMEOUT = 180  # allow heavy local models more time to respond
DEFAULT_TEMPERATURE = 0.1
# SHARED_EMBEDDING_MODEL = "phi3:mini"  # Fast, lightweight embedding model for all tests
SHARED_EMBEDDING_MODEL = "all-minilm"  # updated default embedding model as requested

# Retrieval & chunking parameters
CHUNK_SIZE = 400  # characters per chunk
CHUNK_OVERLAP = 50  # overlap between chunks
TOP_K = 3  # number of chunks retrieved per query
# Directory where the persisted FAISS index (and associated metadata) will be stored
# Index dir will be derived from embedding model slug inside the benchmarker
INDEX_DIR = os.path.join(".rag_cache", "faiss_index")
# Directory where raw answers and evaluation artefacts will be stored
RUNS_DIR = "runs"
# Directory for consolidated benchmark summaries
RESULTS_DIR = "results"

class MemoryOptimizedBenchmarker:
    """Memory-optimized benchmarking with shared embeddings and proper model lifecycle"""

    def __init__(self, sample_size: int = DEFAULT_SAMPLE_SIZE,
                 embedding_model: str = SHARED_EMBEDDING_MODEL,
                 aggressive_cleanup: bool = False,
                 judge_model: str = "phi3:mini",
                 include_heavy_metrics: bool = False):
        # Store config
        self.sample_size = sample_size
        self.embedding_model = embedding_model
        self.aggressive_cleanup = aggressive_cleanup  # Whether to completely remove models
        self.judge_model = judge_model  # chat-capable model used for Ragas metric prompts
        self.include_heavy_metrics = include_heavy_metrics  # Toggle for heavyweight metrics

        # Derive per-embedding-model cache directory
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.embedding_model.lower())
        self.index_dir = os.path.join(".rag_cache", slug, "faiss_index")
        # Save the slug so we can create unique filenames per embedding model
        self.embed_slug = slug
        os.makedirs(self.index_dir, exist_ok=True)
        os.makedirs(RUNS_DIR, exist_ok=True)
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "percent_used": memory.percent
        }

    def log_memory_status(self, context: str = ""):
        """Log current memory status"""
        mem = self.get_memory_info()
        print(f"Memory {context}: {mem['used_gb']:.1f}GB/{mem['total_gb']:.1f}GB used ({mem['percent_used']:.1f}%)")

    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                models = []
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]  # First column is model name
                        models.append(model_name)
                return models
            return []
        except Exception as e:
            print(f"Error getting model list: {e}")
            return []

    def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        available_models = self.get_available_models()
        return model_name in available_models

    def unload_model(self, model_name: str) -> bool:
        """Unload model from memory to free GPU space"""
        print(f"Unloading model {model_name} to free memory...")
        try:
            # First check if model exists
            if not self.is_model_available(model_name):
                print(f"Warning: Model {model_name} not found, skipping unload")
                return True

            result = subprocess.run(
                ["ollama", "rm", model_name],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                print(f"Successfully unloaded {model_name}")
                # Give system time to free resources
                time.sleep(2)
                return True
            else:
                print(f"Warning: Failed to unload {model_name}: {result.stderr}")
                return False

        except Exception as e:
            print(f"Error unloading {model_name}: {e}")
            return False

    def pull_model(self, model_name: str) -> bool:
        """Pull model if not available"""
        if self.is_model_available(model_name):
            print(f"Model {model_name} already available")
            return True

        print(f"Pulling model {model_name}...")
        try:
            result = subprocess.run(
                ["ollama", "pull", model_name],
                check=True,
                timeout=300  # 5 minute timeout
            )
            print(f"Successfully pulled {model_name}")
            return True
        except Exception as e:
            print(f"Failed to pull {model_name}: {e}")
            return False

    def test_model_basic(self, model_name: str) -> bool:
        """Quick functionality test for model"""
        print(f"Testing {model_name}...")
        try:
            llm = ChatOllama(model=model_name, temperature=0, timeout=30)
            response = llm.invoke("Say 'OK'")
            print(f"Model test passed: {response.content.strip()[:20]}...")
            return True
        except Exception as e:
            print(f"Model test failed: {e}")
            return False

    def setup_shared_vectorstore(self, _docs: List) -> bool:
        """Load the shared FAISS index built by build_embeddings.py.

        If the index is missing we exit early with instructions, rather than
        re-building it here. This keeps the responsibilities separated: the
        *embedding* script handles indexing; the *benchmark* script loads it.
        """
        print(f"\nLoading shared vector store built with {self.embedding_model} …")

        # Path where build_embeddings.py saves the index
        index_file = os.path.join(self.index_dir, "index.faiss")

        if not os.path.exists(index_file):
            print("ERROR: FAISS index not found. Run `python -m src.build_embeddings` first.")
            return False

        # Make sure the embedding model needed for loading is present
        if not self.pull_model(self.embedding_model):
            return False

        try:
            embeddings = OllamaEmbeddings(model=self.embedding_model)
            self.shared_vectorstore = FAISS.load_local(self.index_dir, embeddings, allow_dangerous_deserialization=True)
            self.shared_retriever = self.shared_vectorstore.as_retriever(search_kwargs={"k": TOP_K})
            print("Shared vector store loaded successfully!")
            return True
        except Exception as e:
            print(f"Failed to load shared vector store: {e}")
            return False

    def generate_answers(self, model_name: str, testset_df: pd.DataFrame,
                         force_rerun: bool = False) -> Optional[Dict[str, Any]]:
        """Generate answers for a single model using the shared retriever and persist them to disk"""

        print(f"\n{'='*60}\nANSWER GENERATION: {model_name}\n{'='*60}")
        self.log_memory_status("before generation")

        # Build per-embedding, per-chat-model filename (JSON, not JSONL)
        chat_slug = model_name.replace(":", "_")
        answers_file = os.path.join(RUNS_DIR, f"{self.embed_slug}_{chat_slug}_answers.json")

        if os.path.exists(answers_file) and not force_rerun:
            print(f"Answers already exist at {answers_file}. Skipping (use force_rerun=True to regenerate)")
            return {"answers_file": answers_file}

        # Model lifecycle: pull → test → generate → unload
        if not self.pull_model(model_name):
            return None
        if not self.test_model_basic(model_name):
            return None

        try:
            llm = ChatOllama(model=model_name, temperature=DEFAULT_TEMPERATURE, timeout=DEFAULT_TIMEOUT)
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.shared_retriever,
                return_source_documents=True
            )

            answers = []
            sample_df = testset_df.head(self.sample_size)
            print(f"Processing {len(sample_df)} questions…")

            for i, row in sample_df.iterrows():
                question = row["user_input"]
                ground_truth = row["reference"]
                print(f"  Q{i+1}: {question[:50]}…")
                try:
                    result = rag_chain.invoke({"query": question})
                    answer_data = {
                        "user_input": question,
                        "retrieved_contexts": [doc.page_content for doc in result["source_documents"]],
                        "response": result["result"],
                        "reference": ground_truth,
                        "reference_contexts": row.get("reference_contexts", [])
                    }
                except Exception as e:
                    print(f"    Error: {e}")
                    answer_data = {
                        "user_input": question,
                        "retrieved_contexts": ["Error retrieving contexts"],
                        "response": f"Error: {str(e)}",
                        "reference": ground_truth,
                        "reference_contexts": row.get("reference_contexts", [])
                    }
                answers.append(answer_data)

            # Persist answers as a single, human-readable JSON file
            with open(answers_file, "w") as f:
                json.dump(answers, f, indent=2)
            print(f"Saved answers to {answers_file}")

            return {"answers_file": answers_file, "questions_processed": len(answers)}

        except Exception as e:
            print(f"Answer generation failed for {model_name}: {e}")
            traceback.print_exc()
            return None

        finally:
            # Clean up model from RAM but keep on disk
            print("Cleaning up…")
            if self.aggressive_cleanup and model_name != self.embedding_model:
                self.unload_model(model_name)
            gc.collect()
            time.sleep(2)
            self.log_memory_status("after generation")

    def benchmark_single_model(self, model_name: str, testset_df: pd.DataFrame,
                             force_rerun: bool = False) -> Optional[Dict[str, Any]]:
        """Benchmark single model using shared vector store"""

        print(f"\n{'='*60}")
        print(f"BENCHMARKING: {model_name}")
        print(f"{'='*60}")

        self.log_memory_status("before benchmark")

        # Check for existing results
        slug = re.sub(r"[^A-Za-z0-9]+", "_", self.embedding_model.lower())
        results_file = f"benchmark_results_{slug}_{model_name.replace(':', '_')}.json"

        if os.path.exists(results_file) and not force_rerun:
            print(f"Loading existing results...")
            try:
                with open(results_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Error loading cached results: {e}")

        # Model lifecycle: pull → test → benchmark → unload
        if not self.pull_model(model_name):
            return None

        if not self.test_model_basic(model_name):
            return None

        start_time = time.time()

        try:
            # Initialize LLM (but use shared embeddings/retriever)
            print("Initializing LLM...")
            llm = ChatOllama(
                model=model_name,
                temperature=DEFAULT_TEMPERATURE,
                timeout=DEFAULT_TIMEOUT
            )

            # Create RAG chain using shared retriever
            print("Creating RAG chain with shared retriever...")
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.shared_retriever,
                return_source_documents=True
            )

            # Process questions
            evaluation_data = []
            sample_df = testset_df.head(self.sample_size)
            print(f"Processing {len(sample_df)} questions...")

            for i, row in sample_df.iterrows():
                question = row['user_input']
                ground_truth = row['reference']

                try:
                    print(f"  Question {i+1}/{len(sample_df)}: {question[:50]}...")
                    q_start = time.time()

                    result = rag_chain.invoke({"query": question})

                    q_end = time.time()
                    print(f"  Completed in {q_end - q_start:.1f}s")

                    evaluation_data.append({
                        "user_input": question,
                        "retrieved_contexts": [doc.page_content for doc in result['source_documents']],
                        "response": result['result'],
                        "reference": ground_truth,
                        "reference_contexts": row.get("reference_contexts", [])
                    })

                except Exception as e:
                    print(f"  Error on question {i+1}: {e}")
                    evaluation_data.append({
                        "user_input": question,
                        "retrieved_contexts": ["Error retrieving contexts"],
                        "response": f"Error: {str(e)}",
                        "reference": ground_truth,
                        "reference_contexts": row.get("reference_contexts", [])
                    })

            # Run Ragas evaluation using shared embedding model
            print("Running Ragas evaluation...")
            evaluation_dataset = EvaluationDataset.from_list(evaluation_data)

            # Use shared embedding model for evaluation consistency
            eval_embeddings = OllamaEmbeddings(model=self.embedding_model)

            # Ensure judge model is available and instantiate it for metrics
            if not self.pull_model(self.judge_model):
                return None
            judge_llm = ChatOllama(model=self.judge_model, temperature=0, timeout=DEFAULT_TIMEOUT)

            results = {}
            metrics = [
                AnswerAccuracy(),
                ContextRelevance(),
                context_precision,
                context_recall,
            ]
            if self.include_heavy_metrics:
                # Heavy / slower metrics that require additional LLM calls
                try:
                    metrics.append(ResponseGroundedness())
                except Exception as metric_err:  # pragma: no cover – best-effort
                    print(f"  Warning: could not add ResponseGroundedness – {metric_err}")

            for metric in metrics:
                try:
                    metric_name = metric.__class__.__name__
                    print(f"  Evaluating {metric_name}...")

                    metric_result = evaluate(
                        dataset=evaluation_dataset,
                        metrics=[metric],
                        llm=judge_llm,
                        embeddings=eval_embeddings,
                        raise_exceptions=True,
                        batch_size=1
                    )

                    # Extract numeric columns and average
                    if hasattr(metric_result, 'to_pandas'):
                        df_result = metric_result.to_pandas()
                        for col in df_result.select_dtypes(include="number").columns:
                            score = df_result[col].mean()
                            results[col] = float(score)
                            print(f"  {col}: {score:.4f}")

                except Exception as e:
                    print(f"  Error evaluating {metric_name}: {e}")
                    results[metric_name] = f"error: {str(e)}"

            # Calculate total time
            total_time = time.time() - start_time
            results["benchmark_time_seconds"] = total_time
            results["questions_processed"] = len(evaluation_data)
            results["embedding_model_used"] = self.embedding_model

            # Save results
            try:
                with open(results_file, "w") as f:
                    json.dump(results, f, indent=2)
                print(f"Results saved to {results_file}")
            except Exception as e:
                print(f"Warning: Error saving results: {e}")

            print(f"Total benchmark time: {total_time:.1f}s")
            self.log_memory_status("after benchmark")

            return results

        except Exception as e:
            print(f"Benchmark failed for {model_name}: {e}")
            traceback.print_exc()
            return None

        finally:
            # Clean up: manage memory based on cleanup policy
            print("Cleaning up...")
            if self.aggressive_cleanup and model_name != self.embedding_model:
                print(f"Warning: Aggressively removing {model_name} from system")
                self.unload_model(model_name)
            elif model_name != self.embedding_model:
                print(f"Keeping {model_name} available (use aggressive_cleanup=True to remove)")
            else:
                print(f"Keeping embedding model {model_name} loaded for other benchmarks")
            gc.collect()
            time.sleep(3)  # Allow system to stabilize
            self.log_memory_status("after cleanup")

    def grade_answers(self, model_name: str, force_rerun: bool = False) -> Optional[Dict[str, Any]]:
        """Run Ragas evaluation on previously generated answers for a given model"""

        print(f"\n{'='*60}\nGRADING: {model_name}\n{'='*60}")
        self.log_memory_status("before grading")

        chat_slug = model_name.replace(":", "_")
        answers_file = os.path.join(RUNS_DIR, f"{self.embed_slug}_{chat_slug}_answers.json")
        scores_file = os.path.join(RUNS_DIR, f"{self.embed_slug}_{chat_slug}_scores.json")

        if os.path.exists(scores_file) and not force_rerun:
            print(f"Scores already exist at {scores_file}. Skipping (use force_rerun=True to regenerate)")
            try:
                with open(scores_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass  # fallthrough to recompute if file corrupted

        if not os.path.exists(answers_file):
            print(f"Error: answers file {answers_file} not found. Run answer generation first.")
            return None

        # Ensure judge model available
        if not self.pull_model(self.judge_model):
            return None

        start_time = time.time()

        try:
            # Load answer records from JSON list (no longer JSONL)
            with open(answers_file, "r") as f:
                try:
                    records = json.load(f)
                except Exception as e:
                    print(f"Error reading answers file: {e}")
                    return None

            if not records:
                print("No answers to grade – skipping")
                return None

            evaluation_dataset = EvaluationDataset.from_list(records)
            embeddings = OllamaEmbeddings(model=self.embedding_model)
            llm = ChatOllama(model=self.judge_model, temperature=0, timeout=DEFAULT_TIMEOUT)

            metrics = [
                AnswerAccuracy(),
                ContextRelevance(),
                context_precision,
                context_recall,
            ]
            if self.include_heavy_metrics:
                try:
                    metrics.append(ResponseGroundedness())
                except Exception as metric_err:  # pragma: no cover
                    print(f"    Warning: could not add ResponseGroundedness – {metric_err}")
            scores: Dict[str, float] = {}

            for metric in metrics:
                metric_name = metric.__class__.__name__
                print(f"  Evaluating {metric_name}…")
                try:
                    metric_result = evaluate(
                        dataset=evaluation_dataset,
                        metrics=[metric],
                        llm=llm,
                        embeddings=embeddings,
                        raise_exceptions=True,
                        batch_size=4
                    )

                    if hasattr(metric_result, "to_pandas"):
                        df_result = metric_result.to_pandas()
                        for col in df_result.select_dtypes(include="number").columns:
                            scores[col] = float(df_result[col].mean())
                            print(f"    {col}: {scores[col]:.4f}")
                except Exception as e:
                    print(f"    Error evaluating {metric_name}: {e}")
                    scores[metric_name] = f"error: {str(e)}"

            scores["questions_graded"] = len(records)
            scores["grading_time_seconds"] = time.time() - start_time

            # Persist scores
            with open(scores_file, "w") as f:
                json.dump(scores, f, indent=2)
            print(f"Saved scores to {scores_file}")

            return scores

        except Exception as e:
            print(f"Grading failed for {model_name}: {e}")
            traceback.print_exc()
            return None

        finally:
            # Unload grading model from RAM (but keep on disk)
            if self.aggressive_cleanup:
                print("Cleaning up grading model… (aggressive)")
                self.unload_model(self.judge_model) # Changed to judge_model
            gc.collect()
            self.log_memory_status("after grading")

    def benchmark_all_models(self, testset_df: pd.DataFrame, docs: List,
                           target_models: Optional[List[str]] = None) -> Dict[str, Any]:
        """Benchmark all available models with memory optimization"""

        print("STARTING MEMORY-OPTIMIZED MULTI-MODEL BENCHMARK")
        print("=" * 70)

        self.log_memory_status("initial")

        # Set up shared vector store once
        if not self.setup_shared_vectorstore(docs):
            print("Failed to setup shared vector store")
            return {}

        # Get models to benchmark (skip embedding-only entries quickly)
        available_models = [m for m in self.get_available_models() if not m.startswith("tazarov/")]

        if target_models:
            models_to_test = [m for m in target_models if m in available_models]
            missing_models = [m for m in target_models if m not in available_models]
            if missing_models:
                print(f"Warning: Missing models: {missing_models}")
        else:
            # Filter out embedding model from LLM testing to avoid conflicts
            models_to_test = [m for m in available_models if m != self.embedding_model]

        print(f"Models to benchmark: {models_to_test}")
        print(f"Shared embedding model: {self.embedding_model}")

        # Benchmark each model
        all_results = {}
        failed_models = []

        for i, model_name in enumerate(models_to_test, 1):
            print(f"\nProgress: {i}/{len(models_to_test)} models")

            gen_info = self.generate_answers(model_name, testset_df, force_rerun=True)

            if gen_info is None:
                failed_models.append(model_name)
                print(f"{model_name} failed during answer generation")
                continue

            # After generating answers, run grading
            score_info = self.grade_answers(model_name)

            if score_info:
                all_results[model_name] = score_info
                print(f"{model_name} graded successfully")
            else:
                failed_models.append(model_name)
                print(f"{model_name} failed during grading")

        # Summary
        print(f"\n{'='*70}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*70}")
        print(f"Successful: {len(all_results)}")
        print(f"Failed: {len(failed_models)}")
        if failed_models:
            print(f"Failed models: {failed_models}")

        # Final cleanup: unload embedding model if aggressive cleanup is enabled
        print("Final cleanup...")
        if self.aggressive_cleanup:
            print("Warning: Aggressively removing embedding model from system")
            self.unload_model(self.embedding_model)
        else:
            print("Keeping embedding model available (use aggressive_cleanup=True to remove)")
        gc.collect()

        self.log_memory_status("final")

        return all_results

def main():
    """Main execution function supporting multiple embedding models."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RAGAS benchmark across one or many embedding vector stores."
    )
    parser.add_argument(
        "--embedding-model",
        action="append",
        help=(
            "Embedding model(s) to evaluate. "
            "Can be provided multiple times or as a comma-separated list. "
            f"Defaults to {SHARED_EMBEDDING_MODEL}."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of questions from the test-set to benchmark against.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="Include heavy metrics like ResponseGroundedness (slower).",
    )
    args = parser.parse_args()

    # ---------------- Prepare CLI inputs ----------------
    sample_size_cli: int = args.sample_size
    embedding_models: List[str] = []
    if args.embedding_model:
        for value in args.embedding_model:
            embedding_models.extend([m.strip() for m in value.split(",") if m.strip()])
    if not embedding_models:
        embedding_models = [SHARED_EMBEDDING_MODEL]

    print("RAGAS MEMORY-OPTIMIZED MULTI-EMBEDDING BENCHMARK")
    print("=" * 70)
    print(f"Embedding models to evaluate: {embedding_models}")
    print(f"Heavy metrics enabled: {args.all_metrics}")
    print(f"Sample size: {sample_size_cli}")

    # ---------------- Load static resources -------------
    print("Loading test-set …")
    with open("testset/ragas_testset.json", "r") as f:
        testset_data = json.load(f)
    testset_df = pd.DataFrame(testset_data)
    print(f"Loaded {len(testset_df)} questions.")

    print("Loading documents …")
    loader = DirectoryLoader("data/", glob="**/*.pdf")
    docs = loader.load()
    print(f"Loaded {len(docs)} documents.")

    # Chat models to benchmark per embedding store – hard-coded for now
    target_models = ["mistral:7b", "qwen3:4b"]

    consolidated_payload: Dict[str, Any] = {"embedding_model_runs": []}

    # ---------------- Main loop -------------------------
    for emb_model in embedding_models:
        print("\n" + "=" * 70)
        print(f"BENCHMARKING EMBEDDING MODEL: {emb_model}")
        print("=" * 70)

        benchmarker = MemoryOptimizedBenchmarker(
            sample_size=sample_size_cli,
            embedding_model=emb_model,
            aggressive_cleanup=False,
            judge_model="phi3:mini",
            include_heavy_metrics=args.all_metrics,
        )

        try:
            results = benchmarker.benchmark_all_models(
                testset_df=testset_df, docs=docs, target_models=target_models
            )
        except Exception as exc:  # pragma: no cover – top-level safety
            print(f"Top-level failure while benchmarking {emb_model}: {exc}")
            traceback.print_exc()
            continue

        # Consolidate per-chat-model scores
        for chat_model_name, metrics in results.items():
            consolidated_payload["embedding_model_runs"].append(
                {
                    "embedding_model": emb_model,
                    "chat_model": chat_model_name,
                    "sample_size": sample_size_cli,
                    "metrics": metrics,
                }
            )

    # ---------------- Persist consolidated results ------
    if consolidated_payload["embedding_model_runs"]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(
            RESULTS_DIR, f"benchmark_results_{timestamp}_multi_emb.json"
        )
        try:
            with open(out_file, "w") as fp:
                json.dump(consolidated_payload, fp, indent=2)
            print(f"\nSaved consolidated results to {out_file}")
        except Exception as e:
            print(f"Error saving consolidated results: {e}")
    else:
        print("No results produced – nothing to write.")

if __name__ == "__main__":
    main()
