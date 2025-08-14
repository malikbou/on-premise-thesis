import os
import json
from datetime import datetime
import requests
import pandas as pd
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)
from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings


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

def main():
    """
    This script runs an accuracy benchmark by sending questions to the RAG API
    and evaluating the responses using the Ragas library.
    """
    print("--- Starting RAG Accuracy Benchmark ---")

    # --- 1. Load the Testset ---
    print(f"Loading testset from {TESTSET_FILE}...")
    try:
        with open(TESTSET_FILE, 'r') as f:
            testset_data = json.load(f)

        # Slice the dataset to the desired number of questions for testing
        testset_data = testset_data[:NUM_QUESTIONS_TO_TEST]

        questions = [item['user_input'] for item in testset_data]
        ground_truths = [item['reference'] for item in testset_data]
        print(f"Loaded and sliced to {len(questions)} question(s).")
    except Exception as e:
        print(f"ERROR: Could not load or parse testset '{TESTSET_FILE}'. Details: {e}")
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

            contexts = []
            answers = []
            print(f"Generating answers for {len(questions)} question(s)...")
            for q in questions:
                try:
                    response = requests.post(
                        api_url,
                        json={"question": q, "model_name": model_name},
                        timeout=600
                    )
                    response.raise_for_status()
                    data = response.json()
                    answers.append(data['answer'])
                    contexts.append([doc['page_content'] for doc in data['source_documents']])
                    snippet = (data.get('answer') or "").strip().replace("\n", " ")[:160]
                    print(f"  Answer snippet: {snippet!r}")
                    print(f"  Retrieved contexts: {len(data.get('source_documents', []))}")
                except requests.exceptions.RequestException as e:
                    print(f"  ERROR: API request failed for question '{q[:50]}...'. Error: {e}")
                    answers.append("")
                    contexts.append([])

            print("Answer generation complete.")
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

            # Build records for Ragas
            records = []
            for i in range(len(questions)):
                records.append({
                    "user_input": questions[i],
                    "response": answers[i],
                    "retrieved_contexts": contexts[i],
                    "reference": ground_truths[i],
                })
            evaluation_dataset = EvaluationDataset.from_list(records)

            # Use same embedding for evaluation to avoid bias
            ragas_embeddings = OllamaEmbeddings(
                model=embedding_model,
                base_url=OLLAMA_HOST_URL,
                keep_alive=0,
            )

            # Use the model-under-test as judge (memory efficient)
            judge_model_name = model_name.split("/", 1)[-1] if model_name.startswith("ollama/") else model_name
            judge_llm = ChatOllama(
                model=judge_model_name,
                base_url=OLLAMA_HOST_URL,
                temperature=0,
                timeout=600,
                keep_alive=0,
            )
            print(f"Using judge model: {judge_model_name} @ {OLLAMA_HOST_URL}")
            loaded = get_ollama_loaded_models(OLLAMA_HOST_URL)
            print(f"Ollama loaded models (before evaluation): {loaded}")

            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
            scores = {}
            for metric in metrics:
                metric_name = getattr(metric, "__name__", str(metric))
                try:
                    result = evaluate(
                        dataset=evaluation_dataset,
                        metrics=[metric],
                        llm=judge_llm,
                        embeddings=ragas_embeddings,
                        raise_exceptions=True,
                        batch_size=1,
                    )
                    if hasattr(result, "to_pandas"):
                        df = result.to_pandas()
                        for col in df.select_dtypes(include="number").columns:
                            scores[col] = float(df[col].mean())
                            print(f"  {col}: {scores[col]:.4f}")
                    else:
                        scores[metric_name] = str(result)
                except Exception as e:
                    print(f"  ERROR evaluating {metric_name}: {e}")
                    scores[metric_name] = f"error: {e}"

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
