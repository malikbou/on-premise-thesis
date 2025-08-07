import os
import json
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

# --- Configuration ---
RAG_API_URL = "http://rag-api:8000/query"
TESTSET_FILE = "testset/CS_testset_from_markdown_gpt-4o-mini_20250803_175526.json"
MODELS_TO_TEST = [
    "ollama/gemma3:4b",
    # "ollama/qwen3:4b"
]
# JUDGE_MODEL = "qwen3:4b" # No longer needed, we use the model-under-test as the judge
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_HOST_URL = "http://host.docker.internal:11434"
NUM_QUESTIONS_TO_TEST = 1 # Set to 1 for quick testing

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

    # --- 2. Prepare Ragas Embeddings (LLM is now set per-model) ---
    ragas_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_HOST_URL)

    # --- 3. Run Benchmark for Each Model ---
    all_results = {}
    for model_name in MODELS_TO_TEST:
        print(f"\n--- Benchmarking Model: {model_name} ---")

        contexts = []
        answers = []
        print(f"Generating answers for {len(questions)} question(s)...")
        for q in questions:
            try:
                response = requests.post(
                    RAG_API_URL,
                    json={"question": q, "model_name": model_name},
                    timeout=600
                )
                response.raise_for_status()
                data = response.json()
                answers.append(data['answer'])
                contexts.append([doc['page_content'] for doc in data['source_documents']])
                # Debug visibility of generation
                snippet = (data.get('answer') or "").strip().replace("\n", " ")[:160]
                print(f"  Answer snippet: {snippet!r}")
                print(f"  Retrieved contexts: {len(data.get('source_documents', []))}")
            except requests.exceptions.RequestException as e:
                print(f"  ERROR: API request failed for question '{q[:50]}...'. Error: {e}")
                answers.append("")
                contexts.append([])

        print("Answer generation complete.")

        print("Evaluating generated answers with Ragas...")
        # Build records with column names expected by Ragas metrics
        # Expected keys per errors: 'user_input', 'response', 'retrieved_contexts', 'reference'
        records = []
        for i in range(len(questions)):
            records.append({
                "user_input": questions[i],
                "response": answers[i],
                "retrieved_contexts": contexts[i],
                "reference": ground_truths[i],
            })
        evaluation_dataset = EvaluationDataset.from_list(records)

        # Use the model being tested as its own judge to save memory
        # When calling native Ollama, do NOT include the 'ollama/' prefix
        judge_model_name = model_name.split("/", 1)[-1] if model_name.startswith("ollama/") else model_name
        judge_llm = ChatOllama(model=judge_model_name, base_url=OLLAMA_HOST_URL, temperature=0, timeout=600)
        print(f"Using judge model: {judge_model_name} @ {OLLAMA_HOST_URL}")

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

        all_results[model_name] = scores
        print(f"Evaluation complete for {model_name}. Scores: {scores}")

    # --- 4. Print Final Results ---
    print("\n\n--- Benchmark Summary ---")
    for model_name, results in all_results.items():
        print(f"\nModel: {model_name}")
        print("-" * 20)
        # The result is now a simple dict, so json.dumps will work correctly
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
