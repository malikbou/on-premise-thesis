#!/usr/bin/env python3
"""
Simplified Ollama Model Benchmarking

Tests multiple Ollama models with CS handbook questions to compare performance.
"""

import asyncio
import json
import os
import random
import statistics
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse
import pandas as pd

# Import httpx for async HTTP requests
import httpx
import requests

class OllamaBenchmark:
    def __init__(self, testset_path: str, results_dir: str = "results/ollama_benchmarks"):
        self.testset_path = testset_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load testset
        with open(testset_path, 'r') as f:
            self.testset = json.load(f)

        # Ollama models to test
        self.models = {
            "nemotron-mini": "nemotron-mini:4b",
            "llama3.2": "llama3.2:3b",
            "gemma3": "gemma3:4b",
        }

        self.ollama_base_url = "http://host.docker.internal:11434"

    def stop_ollama_model(self, model_name: str) -> bool:
        """Attempt to unload a model from Ollama using HTTP API (/api/stop). Returns True if request succeeded."""
        try:
            url = f"{self.ollama_base_url.rstrip('/')}/api/stop"
            resp = requests.post(url, json={"name": model_name}, timeout=10)
            if resp.status_code // 100 == 2:
                return True
            print(f"WARNING: Ollama stop returned status {resp.status_code}: {resp.text}")
            return False
        except Exception as e:
            print(f"WARNING: Could not stop Ollama model '{model_name}': {e}")
            return False

    def get_test_prompts(self, num_prompts: int = 10) -> List[str]:
        """Get random questions from testset with context simulation."""
        # Use actual testset questions - these are already CS handbook specific
        questions = [item["user_input"] for item in self.testset]
        selected_questions = random.sample(questions, min(num_prompts, len(questions)))

        # Add context instruction to simulate RAG behavior
        context_instruction = """You are an AI assistant helping students with questions about the UCL Computer Science Student Handbook. Please provide helpful, accurate answers based on typical university policies and procedures. Be concise and practical in your responses.

Question: """

        return [context_instruction + q for q in selected_questions]

    async def chat_completion(
        self,
        client: httpx.AsyncClient,
        model_id: str,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.2
    ) -> Tuple[Optional[float], Optional[int], Optional[str]]:
        """Send one completion request and return (latency, tokens, response)."""
        url = f"{self.ollama_base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }

        t0 = time.perf_counter()
        try:
            r = await client.post(url, headers=headers, json=payload, timeout=60)
            latency = time.perf_counter() - t0
            r.raise_for_status()
            data = r.json()

            # Extract response text
            response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            # Extract token usage
            usage = data.get("usage", {})
            tokens = usage.get(
                "total_tokens",
                usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            )

            return latency, tokens, response_text
        except Exception as e:
            print(f"      Error: {e}")
            return None, None, None

    async def benchmark_model(
        self,
        model_name: str,
        model_id: str,
        prompts: List[str],
        concurrency: int = 2,
        max_tokens: int = 150,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """Benchmark a single Ollama model with multiple prompts."""

        print(f"\nTesting {model_name} ({model_id})")
        print(f"   Concurrency: {concurrency}")
        print(f"   Questions: {len(prompts)}")

        # Collect all results
        all_latencies = []
        all_tokens = []
        all_responses = []

        start_time = time.perf_counter()

        # Test each prompt
        for i, prompt in enumerate(prompts):
            print(f"   Testing question {i+1}/{len(prompts)}... ", end="", flush=True)

            # Use semaphore for concurrency control
            sem = asyncio.Semaphore(concurrency)

            async def worker():
                async with sem:
                    return await self.chat_completion(
                        client, model_id, prompt, max_tokens, temperature
                    )

            try:
                async with httpx.AsyncClient(http2=True, timeout=120) as client:
                    # Run concurrent requests for this prompt
                    tasks = [asyncio.create_task(worker()) for _ in range(concurrency)]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    # Collect successful results
                    for result in results:
                        if isinstance(result, tuple) and result[0] is not None:
                            latency, tokens, response = result
                            all_latencies.append(latency)
                            all_tokens.append(tokens or 0)
                            if response:
                                all_responses.append({
                                    "question": prompt.split("Question: ")[-1],  # Remove context instruction
                                    "response": response,
                                    "latency": latency,
                                    "tokens": tokens
                                })

                print("DONE")

            except Exception as e:
                print(f"ERROR: {e}")

        end_time = time.perf_counter()
        wall_time = end_time - start_time

        # Calculate metrics
        success_count = len(all_latencies)
        total_requests = len(prompts) * concurrency

        metrics = {
            "model": model_name,
            "model_id": model_id,
            "concurrency": concurrency,
            "n_requests": total_requests,
            "n_success": success_count,
            "wall_time_s": wall_time,
            "requests_s": success_count / wall_time if wall_time > 0 else 0,
            "tokens_s": sum(all_tokens) / wall_time if wall_time > 0 else 0,
            "latency_avg_s": statistics.mean(all_latencies) if all_latencies else 0,
            "latency_p50_s": statistics.median(all_latencies) if all_latencies else 0,
            "latency_p95_s": (
                sorted(all_latencies)[int(0.95 * len(all_latencies))]
                if all_latencies else 0
            )
        }

        print(f"   SUCCESS: {success_count}/{total_requests} requests")
        print(f"   PERFORMANCE: {metrics['requests_s']:.2f} req/s, {metrics['tokens_s']:.1f} tok/s")
        print(f"   LATENCY: {metrics['latency_avg_s']:.3f}s avg, {metrics['latency_p95_s']:.3f}s p95")

        # Unload model to free memory
        print(f"   Unloading model {model_id}...")
        self.stop_ollama_model(model_id)

        return metrics, all_responses

    async def run_benchmarks(self, num_questions: int = 5, concurrency: int = 2):
        """Run benchmarks across all Ollama models."""

        print("Ollama Models CS Handbook Benchmark")
        print("=" * 50)

        # Get test prompts
        prompts = self.get_test_prompts(num_questions)
        print(f"Using {len(prompts)} CS handbook questions")

        # Results storage
        all_metrics = []
        all_responses = {}

        # Test each model
        for model_name, model_id in self.models.items():
            try:
                metrics, responses = await self.benchmark_model(
                    model_name, model_id, prompts, concurrency
                )
                all_metrics.append(metrics)
                all_responses[model_name] = responses

            except Exception as e:
                print(f"FAILED to test {model_name}: {e}")

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save metrics as CSV
        df = pd.DataFrame(all_metrics)
        csv_path = self.results_dir / f"ollama_benchmark_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # Save responses
        responses_path = self.results_dir / f"ollama_responses_{timestamp}.json"
        with open(responses_path, 'w') as f:
            json.dump(all_responses, f, indent=2)
        print(f"Responses saved to {responses_path}")

        # Generate summary
        self.print_summary(all_metrics)

        return all_metrics, all_responses

    def print_summary(self, metrics: List[Dict[str, Any]]):
        """Print a summary comparison."""
        print("\n" + "=" * 50)
        print("OLLAMA MODELS BENCHMARK SUMMARY")
        print("=" * 50)

        # Sort by requests per second (performance)
        for m in sorted(metrics, key=lambda x: x['requests_s'], reverse=True):
            print(f"\n{m['model'].upper()} ({m['model_id']})")
            print(f"   Requests/s:  {m['requests_s']:8.2f}")
            print(f"   Tokens/s:    {m['tokens_s']:8.1f}")
            print(f"   Avg Latency:  {m['latency_avg_s']:6.3f}s")
            print(f"   P95 Latency:  {m['latency_p95_s']:6.3f}s")
            print(f"   Success Rate: {m['n_success']}/{m['n_requests']} ({100*m['n_success']/m['n_requests']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Ollama Models CS Handbook Benchmark")
    parser.add_argument("--testset", default="/app/testset/CS_testset_from_markdown_gpt-4o-mini_20250803_175526.json",
                       help="Path to testset JSON file")
    parser.add_argument("--questions", type=int, default=5,
                       help="Number of test questions to use")
    parser.add_argument("--concurrency", type=int, default=2,
                       help="Concurrent requests per question")
    parser.add_argument("--results-dir", default="/app/results/ollama_benchmarks",
                       help="Directory to save results")

    args = parser.parse_args()

    # Initialize benchmarker
    benchmarker = OllamaBenchmark(args.testset, args.results_dir)

    # Run benchmarks
    asyncio.run(benchmarker.run_benchmarks(args.questions, args.concurrency))


if __name__ == "__main__":
    main()
