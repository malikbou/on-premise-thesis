#!/usr/bin/env python3
"""
Load Testing Wrapper for RAG System

Integrates the openai-llm-benchmark tool with our testset and multiple endpoints.
Tests performance across different models and configurations.
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
import matplotlib.pyplot as plt

# Import the benchmark functionality
import sys
sys.path.append('/app/load-testing')
from openai_llm_benchmark import _chat_completion, _report

class RAGLoadTester:
    def __init__(self, testset_path: str, results_dir: str = "results/load_testing"):
        self.testset_path = testset_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Load testset
        with open(testset_path, 'r') as f:
            self.testset = json.load(f)

        # Test configurations - Multiple Ollama models
        self.models = {
            "nemotron-mini": "nemotron-mini:4b",
            "llama3.2": "llama3.2:3b",
            "gemma3": "gemma3:4b",
            "qwen2": "qwen2:3b"
        }

        self.ollama_base_url = "http://host.docker.internal:11434"

    def get_test_prompts(self, num_prompts: int = 10) -> List[str]:
        """Get random questions from testset with context simulation."""
        # Use actual testset questions - these are already CS handbook specific
        questions = [item["user_input"] for item in self.testset]
        selected_questions = random.sample(questions, min(num_prompts, len(questions)))

        # Add context instruction to simulate RAG behavior
        context_instruction = """You are an AI assistant helping students with questions about the UCL Computer Science Student Handbook. Please provide helpful, accurate answers based on typical university policies and procedures. Be concise and practical in your responses.

Question: """

        return [context_instruction + q for q in selected_questions]

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

        print(f"\n🚀 Testing {model_name} ({model_id})")
        print(f"   Concurrency: {concurrency}")
        print(f"   Prompts: {len(prompts)}")

        url = f"{self.ollama_base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type": "application/json"}

        # Collect all results
        all_latencies = []
        all_tokens = []
        all_responses = []

        start_time = time.perf_counter()

        # Test each prompt
        for i, prompt in enumerate(prompts):
            payload = {
                "model": endpoint_config["model"],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }

            print(f"   Testing prompt {i+1}/{len(prompts)}... ", end="", flush=True)

            # Use semaphore for concurrency control
            sem = asyncio.Semaphore(concurrency)

            async def worker():
                async with sem:
                    return await _chat_completion(
                        client, url, headers, payload, capture_responses=True
                    )

            try:
                async with httpx.AsyncClient(http2=True, timeout=60) as client:
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
                                all_responses.append(response)

                print("✓")

            except Exception as e:
                print(f"✗ Error: {e}")

        end_time = time.perf_counter()
        wall_time = end_time - start_time

        # Calculate metrics
        success_count = len(all_latencies)
        total_requests = len(prompts) * concurrency

        metrics = {
            "endpoint": endpoint_name,
            "model": endpoint_config["model"],
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

        print(f"   ✅ {success_count}/{total_requests} requests succeeded")
        print(f"   📊 {metrics['requests_s']:.2f} req/s, {metrics['tokens_s']:.1f} tok/s")
        print(f"   ⏱️  {metrics['latency_avg_s']:.3f}s avg, {metrics['latency_p95_s']:.3f}s p95")

        return metrics, all_responses

    async def run_benchmarks(self, num_prompts: int = 10, concurrency: int = 2):
        """Run benchmarks across all endpoints."""

        print("🎯 RAG System Load Testing")
        print("=" * 50)

        # Get test prompts
        prompts = self.get_test_prompts(num_prompts)
        print(f"📝 Using {len(prompts)} test questions from testset")

        # Results storage
        all_metrics = []
        all_responses = {}

        # Test each endpoint
        for endpoint_name, endpoint_config in self.endpoints.items():
            try:
                metrics, responses = await self.benchmark_endpoint(
                    endpoint_name, endpoint_config, prompts, concurrency
                )
                all_metrics.append(metrics)
                all_responses[endpoint_name] = responses

            except Exception as e:
                print(f"❌ Failed to test {endpoint_name}: {e}")

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save metrics as CSV
        df = pd.DataFrame(all_metrics)
        csv_path = self.results_dir / f"load_test_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n📊 Results saved to {csv_path}")

        # Save responses
        responses_path = self.results_dir / f"load_test_responses_{timestamp}.json"
        with open(responses_path, 'w') as f:
            json.dump(all_responses, f, indent=2)
        print(f"💬 Responses saved to {responses_path}")

        # Generate summary
        self.print_summary(all_metrics)

        return all_metrics, all_responses

    def print_summary(self, metrics: List[Dict[str, Any]]):
        """Print a summary comparison."""
        print("\n" + "=" * 50)
        print("📈 BENCHMARK SUMMARY")
        print("=" * 50)

        for m in sorted(metrics, key=lambda x: x['requests_s'], reverse=True):
            print(f"\n🏆 {m['endpoint'].upper()}")
            print(f"   Requests/s: {m['requests_s']:8.2f}")
            print(f"   Tokens/s:   {m['tokens_s']:8.1f}")
            print(f"   Avg Latency: {m['latency_avg_s']:6.3f}s")
            print(f"   P95 Latency: {m['latency_p95_s']:6.3f}s")
            print(f"   Success:    {m['n_success']}/{m['n_requests']}")


def main():
    parser = argparse.ArgumentParser(description="RAG System Load Testing")
    parser.add_argument("--testset", default="/app/testset/CS_testset_from_markdown_gpt-4o-mini_20250803_175526.json",
                       help="Path to testset JSON file")
    parser.add_argument("--prompts", type=int, default=10,
                       help="Number of test prompts to use")
    parser.add_argument("--concurrency", type=int, default=2,
                       help="Concurrent requests per prompt")
    parser.add_argument("--results-dir", default="/app/results/load_testing",
                       help="Directory to save results")

    args = parser.parse_args()

    # Import httpx here to avoid import issues
    import httpx
    globals()['httpx'] = httpx

    # Initialize tester
    tester = RAGLoadTester(args.testset, args.results_dir)

    # Run benchmarks
    asyncio.run(tester.run_benchmarks(args.prompts, args.concurrency))


if __name__ == "__main__":
    main()
