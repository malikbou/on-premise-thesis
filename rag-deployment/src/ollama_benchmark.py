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
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

# Import httpx for async HTTP requests
import httpx

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

        # Auto-detect environment and set appropriate Ollama URL
        self.ollama_base_url = self._detect_ollama_url()

        # Chart configuration (following professional standards)
        self.figsize_mm = (160, 100)
        self.dpi = 300
        self.marker = "o"
        self.linestyle = "-"
        self.xtick_rot = 30
        self.xtick_fsize = 8

    def _detect_ollama_url(self) -> str:
        """Auto-detect the appropriate Ollama URL based on environment."""
        import socket
        import os

        # Check if we're running inside Docker by looking for container environment
        if os.path.exists('/.dockerenv'):
            # Running inside Docker container - try Docker network first
            possible_urls = [
                "http://ollama:11434",           # Docker Compose service name
                "http://localhost:11434",       # Local fallback
            ]
        else:
            # Running on host system
            possible_urls = [
                "http://host.docker.internal:11434",  # Mac Docker Desktop
                "http://localhost:11434",              # Direct/VM installation
                "http://127.0.0.1:11434",             # Local fallback
            ]

        # Test each URL to find the working one
        for url in possible_urls:
            try:
                # Quick connection test
                import httpx
                with httpx.Client(timeout=2) as client:
                    response = client.get(f"{url}/api/tags")
                    if response.status_code == 200:
                        print(f"Detected Ollama at: {url}")
                        return url
            except Exception:
                continue

        # Fallback to default based on environment
        if os.path.exists('/.dockerenv'):
            default_url = "http://ollama:11434"
        else:
            default_url = "http://host.docker.internal:11434"

        print(f"Could not detect Ollama, using default: {default_url}")
        return default_url

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
        temperature: float = 0.2,
        keep_alive: int = 0
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
            "keep_alive": keep_alive,
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
                        client, model_id, prompt, max_tokens, temperature, keep_alive=0
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
        print(f"   Model automatically unloaded (keep_alive=0)")

        # Additional explicit model unloading for better memory management
        self._explicit_model_unload(model_id)

        return metrics, all_responses

    def _explicit_model_unload(self, model_id: str):
        """Explicitly unload model from memory via Ollama API."""
        try:
            import subprocess
            import time

            # Try to stop the model via ollama stop command
            # This works better than keep_alive=0 for immediate unloading
            stop_url = f"{self.ollama_base_url.replace('/v1/chat/completions', '')}/api/generate"

            # Send stop command
            import httpx
            with httpx.Client(timeout=5) as client:
                try:
                    response = client.post(stop_url, json={
                        "model": model_id,
                        "keep_alive": 0
                    })
                except Exception:
                    pass  # Ignore errors, this is best-effort cleanup

            # Small delay to allow unloading
            time.sleep(1)
            print(f"   Explicit model unload attempted for {model_id}")

        except Exception as e:
            print(f"   Model unload warning: {e}")

    def _mm_to_in(self, mm: float) -> float:
        """Convert millimeters to inches for matplotlib."""
        return mm / 25.4

    def _plain_numbers(self, ax):
        """Set axis formatters to display plain numbers."""
        for axis in (ax.xaxis, ax.yaxis):
            fmt = ScalarFormatter(useOffset=False)
            fmt.set_scientific(False)
            axis.set_major_formatter(fmt)

    def _set_all_xticks(self, ax, vals):
        """Set all x-tick values and format them."""
        vals = sorted(vals)
        ax.set_xticks(vals)
        ax.set_xticklabels(
            [str(v) for v in vals],
            rotation=self.xtick_rot,
            ha="right",
            fontsize=self.xtick_fsize,
        )

    def _set_dense_yticks(self, ax):
        """Set dense y-ticks for better readability."""
        ax.yaxis.set_major_locator(MaxNLocator(nbins=12, integer=True, prune=None))

    def _plot_multi_models(self, ax, df, y, ylabel):
        """Plot multiple models on the same chart."""
        for model, grp in df.groupby("model"):
            g = grp.sort_values("concurrency")
            ax.plot(g["concurrency"], g[y],
                    marker=self.marker, linestyle=self.linestyle, label=model)

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Concurrent requests")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs concurrency")
        ax.grid(True, which="both", ls=":")
        ax.legend(title="Model")

        self._plain_numbers(ax)
        self._set_all_xticks(ax, df["concurrency"].unique())
        self._set_dense_yticks(ax)

    def generate_charts(self, csv_path: Path, fmt="png"):
        """Generate performance comparison charts from benchmark results."""
        print(f"\nGenerating performance charts from {csv_path}")

        df = pd.read_csv(csv_path)

        # Ensure we have the required columns
        required_cols = ["model", "concurrency", "requests_s", "tokens_s", "latency_avg_s", "latency_p95_s"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Error: Missing required columns: {missing_cols}")
            return

        figsize = (self._mm_to_in(self.figsize_mm[0]), self._mm_to_in(self.figsize_mm[1]))

        # Charts to generate
        charts = [
            ("requests_s", "Requests/s", "ollama_requests_vs_concurrency"),
            ("tokens_s", "Tokens/s", "ollama_tokens_vs_concurrency"),
            ("latency_avg_s", "Average latency (s)", "ollama_latency_avg_vs_concurrency"),
            ("latency_p95_s", "P95 latency (s)", "ollama_latency_p95_vs_concurrency"),
        ]

        chart_files = []
        for col, label, filename in charts:
            fig, ax = plt.subplots(figsize=figsize)
            self._plot_multi_models(ax, df, col, label)
            fig.tight_layout()

            chart_path = self.results_dir / f"{filename}.{fmt}"
            fig.savefig(chart_path, dpi=self.dpi if fmt == "png" else None)
            plt.close(fig)
            chart_files.append(chart_path)
            print(f"Generated {chart_path}")

        return chart_files

    async def run_benchmarks(self, num_questions: int = 5, concurrency: int = 2):
        """Run benchmarks across all Ollama models with single concurrency."""
        return await self.run_concurrency_range_benchmarks(
            num_questions, [concurrency], generate_charts=False
        )

    async def run_concurrency_range_benchmarks(
        self,
        num_questions: int = 5,
        concurrency_levels: List[int] = None,
        generate_charts: bool = True
    ):
        """Run benchmarks across all models and concurrency levels."""

        if concurrency_levels is None:
            concurrency_levels = [1, 2, 4, 8, 16]  # Conservative defaults for Mac

        print("Ollama Models CS Handbook Concurrency Benchmark")
        print("=" * 50)
        print(f"Testing concurrency levels: {concurrency_levels}")
        print(f"Models: {list(self.models.keys())}")

        # Get test prompts
        prompts = self.get_test_prompts(num_questions)
        print(f"Using {len(prompts)} CS handbook questions")

        # Results storage
        all_metrics = []
        all_responses = {}

        # Test each concurrency level
        for concurrency in concurrency_levels:
            print(f"\n{'='*20} Concurrency: {concurrency} {'='*20}")

            # Test each model at this concurrency level
            for model_name, model_id in self.models.items():
                try:
                    metrics, responses = await self.benchmark_model(
                        model_name, model_id, prompts, concurrency
                    )

                    # Add backend column for chart compatibility
                    metrics["backend"] = "Ollama"

                    all_metrics.append(metrics)

                    # Store responses with concurrency key
                    response_key = f"{model_name}_c{concurrency}"
                    all_responses[response_key] = responses

                except Exception as e:
                    print(f"FAILED to test {model_name} at concurrency {concurrency}: {e}")

        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")

        # Save metrics as CSV
        df = pd.DataFrame(all_metrics)
        csv_path = self.results_dir / f"ollama_concurrency_benchmark_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")

        # Save responses
        responses_path = self.results_dir / f"ollama_concurrency_responses_{timestamp}.json"
        with open(responses_path, 'w') as f:
            json.dump(all_responses, f, indent=2)
        print(f"Responses saved to {responses_path}")

        # Generate charts if requested
        chart_files = []
        if generate_charts and len(all_metrics) > 0:
            try:
                chart_files = self.generate_charts(csv_path)
                print(f"Generated {len(chart_files)} performance charts")
            except Exception as e:
                print(f"Chart generation failed: {e}")

        # Generate summary
        self.print_summary(all_metrics)

        return all_metrics, all_responses, chart_files

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
                       help="Single concurrency level (for backward compatibility)")
    parser.add_argument("--concurrency-range", type=str, default=None,
                       help="Comma-separated concurrency levels (e.g., '1,2,4,8,16')")
    parser.add_argument("--results-dir", default="/app/results/ollama_benchmarks",
                       help="Directory to save results")
    parser.add_argument("--generate-charts", action="store_true", default=False,
                       help="Generate performance charts automatically")
    parser.add_argument("--chart-format", choices=["png", "pdf", "svg"], default="png",
                       help="Chart output format")

    args = parser.parse_args()

    # Parse concurrency levels
    if args.concurrency_range:
        try:
            concurrency_levels = [int(x.strip()) for x in args.concurrency_range.split(',')]
            print(f"Using concurrency range: {concurrency_levels}")
        except ValueError:
            print("Error: Invalid concurrency range format. Use comma-separated integers.")
            return
    else:
        concurrency_levels = [args.concurrency]
        print(f"Using single concurrency level: {args.concurrency}")

    # Initialize benchmarker
    benchmarker = OllamaBenchmark(args.testset, args.results_dir)

    # Run benchmarks
    if len(concurrency_levels) == 1 and not args.generate_charts:
        # Single concurrency, backward compatibility
        asyncio.run(benchmarker.run_benchmarks(args.questions, concurrency_levels[0]))
    else:
        # Multiple concurrency levels or chart generation requested
        asyncio.run(benchmarker.run_concurrency_range_benchmarks(
            args.questions, concurrency_levels, args.generate_charts
        ))


if __name__ == "__main__":
    main()
