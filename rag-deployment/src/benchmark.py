import os
import json
from datetime import datetime
import requests
import pandas as pd
import asyncio
import concurrent.futures
from typing import Dict, Any, List
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI # Use the standard OpenAI client


def _parse_list_env(var_name: str, default_list: list[str]) -> list[str]:
    value = os.getenv(var_name)
    if value is None or value.strip() == "":
        return default_list
    return [item.strip() for item in value.split(",") if item.strip()]


def get_ollama_loaded_models(base_url: str) -> list[str]:
    """Return list of currently loaded models from Ollama (/api/ps)."""
    try:
        url = f"{base_url.rstrip('/')}/api/ps"
        try:
            resp = requests.get(url, timeout=10)
        except Exception:
            resp = requests.post(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        return models
    except Exception as e:
        print(f"WARNING: Could not query Ollama /api/ps: {e}")
        return []


def stop_ollama_model(base_url: str, model_name: str) -> bool:
    """Attempt to unload a model from Ollama using HTTP API (/api/stop). Returns True if request succeeded."""
    try:
        url = f"{base_url.rstrip('/')}/api/stop"
        resp = requests.post(url, json={"name": model_name}, timeout=10)
        if resp.status_code // 100 == 2:
            return True
        print(f"WARNING: Ollama stop returned status {resp.status_code}: {resp.text}")
        return False
    except Exception as e:
        print(f"WARNING: Could not stop Ollama model '{model_name}': {e}")
        return False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def sanitize_filename(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace("*", "_")
    )


def get_metric_retry_config(metric_name: str) -> Dict[str, Any]:
    """
    Get metric-specific retry configuration based on common failure patterns
    """
    retry_configs = {
        "answer_relevancy": {
            "max_retries": 4,  # JSON parsing failures need more retries
            "initial_delay": 1.0,
            "backoff_factor": 2.0,  # Exponential backoff for API rate limits
            "timeout_multiplier": 1.5,
        },
        "faithfulness": {
            "max_retries": 2,  # Usually stable, fewer retries needed
            "initial_delay": 0.5,
            "backoff_factor": 1.5,
            "timeout_multiplier": 1.0,
        },
        "context_precision": {
            "max_retries": 3,  # Moderate complexity
            "initial_delay": 2.0,  # Longer initial delay for complex evaluation
            "backoff_factor": 1.8,
            "timeout_multiplier": 2.0,  # Needs more time
        },
        "context_recall": {
            "max_retries": 3,
            "initial_delay": 1.5,
            "backoff_factor": 1.5,
            "timeout_multiplier": 1.5,
        }
    }

    # Default config for unknown metrics
    default_config = {
        "max_retries": 2,
        "initial_delay": 1.0,
        "backoff_factor": 1.5,
        "timeout_multiplier": 1.0,
    }

    return retry_configs.get(metric_name, default_config)


def evaluate_metric_parallel(metric, evaluation_dataset, judge_llm, ragas_embeddings):
    """
    Evaluate a single metric with smart, metric-specific retry logic
    """
    import time

    metric_name = getattr(metric, "name", getattr(metric, "__name__", str(metric)))
    retry_config = get_metric_retry_config(metric_name)

    max_retries = retry_config["max_retries"]
    initial_delay = retry_config["initial_delay"]
    backoff_factor = retry_config["backoff_factor"]

    for retry_count in range(max_retries + 1):
        try:
            result = evaluate(
                dataset=evaluation_dataset,
                metrics=[metric],
                llm=judge_llm,
                embeddings=ragas_embeddings,
                raise_exceptions=False,
                batch_size=1,
            )

            if hasattr(result, "to_pandas"):
                df = result.to_pandas()
                score_dict = {}
                for col in df.select_dtypes(include="number").columns:
                    score = float(df[col].mean())
                    score_dict[col] = score
                return metric_name, score_dict
            else:
                return metric_name, {metric_name: str(result)}

        except Exception as e:
            if retry_count < max_retries:
                # Calculate delay with exponential backoff
                delay = initial_delay * (backoff_factor ** retry_count)
                print(f"  Retry {retry_count + 1}/{max_retries} for {metric_name} in {delay:.1f}s (reason: {str(e)[:60]}...)")
                time.sleep(delay)
                continue
            else:
                print(f"  ERROR evaluating {metric_name} after {max_retries} retries: {e}")
                return metric_name, {metric_name: f"error: {e}"}

    return metric_name, {metric_name: "error: max retries exceeded"}


def _normalize_base_name(name: str) -> str:
    """Normalize model identifier to base name without registry/prefix or tag.
    Examples: 'ollama/qwen3:4b' -> 'qwen3', 'nomic-embed-text:latest' -> 'nomic-embed-text'
    """
    n = name.strip()
    if "/" in n:
        n = n.split("/", 1)[-1]
    if ":" in n:
        n = n.split(":", 1)[0]
    return n


def stop_models_by_base_name(base_url: str, base_name: str) -> None:
    """Stop all loaded models whose base name matches, irrespective of tag.
    This aligns with how Ollama lists running models like 'name:tag'.
    """
    try:
        loaded = get_ollama_loaded_models(base_url)
        targets = []
        target_base = _normalize_base_name(base_name)
        for loaded_name in loaded:
            if _normalize_base_name(loaded_name) == target_base:
                targets.append(loaded_name)
        if not targets:
            print(f"No loaded models match base '{base_name}' – nothing to stop.")
            return
        for exact in targets:
            print(f"Stopping loaded model: {exact}")
            stop_ollama_model(base_url, exact)
    except Exception as e:
        print(f"WARNING: stop_models_by_base_name failed for '{base_name}': {e}")

# --- Configuration ---
RAG_API_URL = "http://rag-api:8000/query"  # Fallback/default API URL
TESTSET_FILE = "testset/CS_testset_from_markdown_gpt-4o-mini_20250803_175526.json"
DEFAULT_MODELS_TO_TEST = [
    "ollama/gemma3:4b",
]
DEFAULT_EMBEDDING_MODELS = ["nomic-embed-text"]  # Treated as retrieval embeddings list
OLLAMA_HOST_URL = "http://host.docker.internal:11434"
LITELLM_API_BASE = os.getenv("LITELLM_API_BASE", "http://litellm:4000")
NUM_QUESTIONS_TO_TEST = int(os.getenv("NUM_QUESTIONS_TO_TEST", "1"))

# Allow overriding via env (comma-separated)
MODELS_TO_TEST = _parse_list_env("MODELS_TO_TEST", DEFAULT_MODELS_TO_TEST)
EMBEDDING_MODELS = _parse_list_env("EMBEDDING_MODELS", DEFAULT_EMBEDDING_MODELS)
NUM_QUESTIONS_TO_TEST = int(os.getenv("NUM_QUESTIONS_TO_TEST", "1"))
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
RUN_STAMP = os.getenv("RUN_STAMP", datetime.now().strftime("%Y%m%d_%H%M%S"))
RUN_DIR = os.path.join(RESULTS_DIR, RUN_STAMP)

# Optional mapping: embedding -> API base URL (Option A per-embedding API services)
# Example: EMBEDDING_API_MAP="all-minilm=http://rag-api-minilm:8001,nomic-embed-text=http://rag-api-nomic:8002"
EMBEDDING_API_MAP_RAW = os.getenv("EMBEDDING_API_MAP", "")
EMBEDDING_API_MAP: dict[str, str] = {}
for pair in [p.strip() for p in EMBEDDING_API_MAP_RAW.split(",") if p.strip()]:
    if "=" in pair:
        k, v = pair.split("=", 1)
        EMBEDDING_API_MAP[k.strip()] = v.strip()

def load_testset_streaming(file_path: str, num_questions: int):
    """
    Load testset with memory optimization - stream processing for large datasets
    """
    try:
        with open(file_path, 'r') as f:
            testset_data = json.load(f)

        # Memory-efficient slicing
        if num_questions > 0:
            testset_data = testset_data[:num_questions]

        # Process in chunks to reduce memory footprint
        questions = []
        ground_truths = []
        contexts = []

        for item in testset_data:
            questions.append(item['user_input'])
            ground_truths.append(item['reference'])
            contexts.append(item.get('reference_contexts', []))

        print(f"Loaded {len(questions)} question(s) with memory optimization.")
        return questions, ground_truths, contexts

    except Exception as e:
        print(f"ERROR: Could not load testset '{file_path}': {e}")
        return None, None, None


def process_questions_in_batches(questions, api_url, model_name, batch_size=5):
    """
    Process questions in smaller batches to reduce memory usage and improve throughput
    """
    all_answers = []
    all_contexts = []

    print(f"Processing {len(questions)} questions in batches of {batch_size}...")

    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i + batch_size]
        batch_end = min(i + batch_size, len(questions))

        print(f"  Processing batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size} (questions {i+1}-{batch_end})")

        batch_answers = []
        batch_contexts = []

        for q in batch_questions:
            try:
                response = requests.post(
                    api_url,
                    json={"question": q, "model_name": model_name},
                    timeout=600
                )
                response.raise_for_status()
                data = response.json()

                answer = data['answer']
                contexts = [doc['page_content'] for doc in data['source_documents']]

                batch_answers.append(answer)
                batch_contexts.append(contexts)

                # Show progress for long batches
                snippet = (answer or "").strip().replace("\n", " ")[:100]
                print(f"    ✓ Q{len(all_answers) + len(batch_answers)}: {snippet}...")

            except requests.exceptions.RequestException as e:
                print(f"    ✗ Q{len(all_answers) + len(batch_answers) + 1}: API request failed: {e}")
                batch_answers.append("")
                batch_contexts.append([])

        # Add batch results to main lists
        all_answers.extend(batch_answers)
        all_contexts.extend(batch_contexts)

        # Optional: Brief pause between batches to prevent overwhelming the API
        if i + batch_size < len(questions):
            import time
            time.sleep(0.5)

    return all_answers, all_contexts


def main():
    """
    Memory-optimized RAG accuracy benchmark with streaming processing
    """
    print("--- Starting RAG Accuracy Benchmark (Memory Optimized) ---")

    # --- 1. Load the Testset with Memory Optimization ---
    print(f"Loading testset from {TESTSET_FILE}...")
    questions, ground_truths, _ = load_testset_streaming(TESTSET_FILE, NUM_QUESTIONS_TO_TEST)

    if questions is None:
        return

    # --- 2. Run Benchmark for Each Retrieval Embedding ---
    all_results: dict[str, dict[str, dict]] = {}
    for embedding_model in EMBEDDING_MODELS:
        print(f"\n=== Retrieval Embedding: {embedding_model} ===")
        api_base = EMBEDDING_API_MAP.get(embedding_model, RAG_API_URL)
        api_url = api_base.rstrip("/") + "/query"
        print(f"Using RAG API: {api_base}")

        all_results.setdefault(embedding_model, {})

        for model_name in MODELS_TO_TEST:
            print(f"\n--- Benchmarking Model: {model_name} with embedding {embedding_model} ---")
            loaded = get_ollama_loaded_models(OLLAMA_HOST_URL)
            print(f"Ollama loaded models (before generation): {loaded}")

            # Memory-optimized question processing with batching
            batch_size = min(10, max(1, len(questions) // 4))  # Dynamic batch size
            answers, contexts = process_questions_in_batches(
                questions, api_url, model_name, batch_size
            )

            print(f"Answer generation complete. Processed {len(answers)} questions.")
            loaded = get_ollama_loaded_models(OLLAMA_HOST_URL)
            print(f"Ollama loaded models (after generation): {loaded}")

            # Persist raw answers for audit
            ensure_dir(RUN_DIR)
            answers_name = f"answers__{sanitize_filename(embedding_model)}__{sanitize_filename(model_name)}.json"
            answers_path = os.path.join(RUN_DIR, answers_name)
            try:
                records_save = []
                for i in range(len(questions)):
                    records_save.append({
                        "user_input": questions[i],
                        "response": answers[i],
                        "retrieved_contexts": contexts[i],
                        "reference": ground_truths[i],
                    })
                with open(answers_path, "w") as f:
                    json.dump(records_save, f, indent=2)
                print(f"Saved answers to {answers_path}")
            except Exception as e:
                print(f"WARNING: Failed to save answers to {answers_path}: {e}")

            # Memory-efficient dataset creation
            print("Creating evaluation dataset...")
            try:
                # Process in smaller chunks to avoid memory issues with large datasets
                chunk_size = 50
                all_records = []

                for i in range(0, len(questions), chunk_size):
                    chunk_records = []
                    end_idx = min(i + chunk_size, len(questions))

                    for j in range(i, end_idx):
                        chunk_records.append({
                            "user_input": questions[j],
                            "response": answers[j],
                            "retrieved_contexts": contexts[j],
                            "reference": ground_truths[j],
                        })

                    all_records.extend(chunk_records)

                    # Progress indicator for large datasets
                    if len(questions) > 20:
                        print(f"  Processed {end_idx}/{len(questions)} records for evaluation...")

                evaluation_dataset = EvaluationDataset.from_list(all_records)
                print(f"Evaluation dataset created with {len(all_records)} records.")

            except Exception as e:
                print(f"ERROR: Failed to create evaluation dataset: {e}")
                continue

            # Use same embedding for evaluation to avoid bias
            ragas_embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=OLLAMA_HOST_URL,
                keep_alive=0,
            )

                        # Use cloud judge for better JSON parsing reliability
            # Option 1: Cloud judge (GPT-4o-mini - recommended for structured output)
            judge_llm = ChatOpenAI(
                model="gpt-4o-mini",
                openai_api_base=LITELLM_API_BASE,
                openai_api_key=os.getenv("OPENAI_API_KEY", "anything"), # LiteLLM handles the key
                temperature=0,
                timeout=600,
            )
            print(f"Using judge model: gpt-4o-mini @ {LITELLM_API_BASE}")

            # Option 2: Local judge (fallback - may have JSON parsing issues)
            # judge_model_name = model_name.split("/", 1)[-1] if model_name.startswith("ollama/") else model_name
            # judge_llm = ChatOllama(
            #     model=judge_model_name,
            #     base_url=OLLAMA_HOST_URL,
            #     temperature=0,
            #     timeout=600,
            #     keep_alive=0,
            # )
            # print(f"Using judge model: {judge_model_name} @ {OLLAMA_HOST_URL}")
            loaded = get_ollama_loaded_models(OLLAMA_HOST_URL)
            print(f"Ollama loaded models (before evaluation): {loaded}")

            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

            # Parallel evaluation for better performance
            print("Evaluating metrics in parallel...")
            start_time = datetime.now()

            # Use ThreadPoolExecutor for parallel metric evaluation
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(metrics)) as executor:
                # Submit all metric evaluations
                future_to_metric = {
                    executor.submit(
                        evaluate_metric_parallel,
                        metric,
                        evaluation_dataset,
                        judge_llm,
                        ragas_embeddings
                    ): metric for metric in metrics
                }

                scores = {}
                for future in concurrent.futures.as_completed(future_to_metric):
                    metric = future_to_metric[future]
                    try:
                        metric_name, metric_scores = future.result()
                        scores.update(metric_scores)
                        print(f"  ✓ {metric_name}: {list(metric_scores.values())[0]:.4f}"
                              if isinstance(list(metric_scores.values())[0], (int, float))
                              else f"  ✓ {metric_name}: {list(metric_scores.values())[0]}")
                    except Exception as e:
                        metric_name = getattr(metric, "name", str(metric))
                        print(f"  ✗ {metric_name}: error: {e}")
                        scores[metric_name] = f"error: {e}"

            evaluation_time = (datetime.now() - start_time).total_seconds()
            print(f"Parallel evaluation completed in {evaluation_time:.1f}s")

            # Persist per-(embedding, model) scores
            ensure_dir(RUN_DIR)
            out_name = f"scores__{sanitize_filename(embedding_model)}__{sanitize_filename(model_name)}.json"
            out_path = os.path.join(RUN_DIR, out_name)
            try:
                with open(out_path, "w") as f:
                    json.dump(scores, f, indent=2)
                print(f"Saved results to {out_path}")
            except Exception as e:
                print(f"WARNING: Failed to save results to {out_path}: {e}")

            all_results[embedding_model][model_name] = scores
            print(f"Evaluation complete for {model_name} (retrieval embedding={embedding_model}). Scores: {scores}")
            loaded = get_ollama_loaded_models(OLLAMA_HOST_URL)
            print(f"Ollama loaded models (after evaluation): {loaded}")

    # --- 4. Print Final Results ---
    print("\n\n--- Benchmark Summary ---")
    # Persist overall summary
    ensure_dir(RUN_DIR)
    summary_path = os.path.join(RUN_DIR, "summary.json")
    try:
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved summary to {summary_path}")
    except Exception as e:
        print(f"WARNING: Failed to save summary to {summary_path}: {e}")

    for embedding_model, by_model in all_results.items():
        print(f"\nRetrieval Embedding: {embedding_model}")
        print("-" * 20)
        for model_name, results in by_model.items():
            print(f"Model: {model_name}")
            print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
