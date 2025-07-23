#!/usr/bin/env python3
"""
show_benchmarks.py – Pretty‐print consolidated benchmark scores from the
`results/` directory.

Usage examples
--------------
python -m src.show_benchmarks                # human-readable table to stdout
python -m src.show_benchmarks --csv summary.csv  # save tidy CSV
python -m src.show_benchmarks --html summary.html # save styled HTML table

The script expects each results file to be a JSON produced by src/benchmark.py
and containing a top-level key `embedding_model_runs`, which is a list of
{embedding_model, chat_model, sample_size, metrics} objects.

It generates a *tidy* DataFrame with one row per (embedding, chat_model) pair
and one column per metric, then prints a pivot-table view (embedding ×
chat_model) for quick comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import glob
from typing import List, Dict, Any

import pandas as pd
from rich import print as rprint
from rich.console import Console
from rich.table import Table

RESULTS_DIR = "results"


def load_results(files: List[str]) -> pd.DataFrame:
    """Return tidy DataFrame with columns: embedding, chat_model, <metric columns>."""
    records: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp) as f:
                payload = json.load(f)
        except Exception as e:
            rprint(f"[red]Warning: could not parse {fp}: {e}")
            continue

        for run in payload.get("embedding_model_runs", []):
            row = {
                "embedding_model": run.get("embedding_model"),
                "chat_model": run.get("chat_model"),
                "sample_size": run.get("sample_size"),
            }
            # Flatten metrics dict into columns
            metrics: Dict[str, Any] = run.get("metrics", {})
            for k, v in metrics.items():
                row[k] = v
            records.append(row)

    if not records:
        raise ValueError("No benchmark records found; check results directory path")

    df = pd.DataFrame(records)
    return df


def to_rich_table(df: pd.DataFrame, metrics: List[str]) -> Table:
    """Return a rich.Table where rows=(embedding, chat_model) and columns=metrics."""
    table = Table(title="RAG Benchmark Summary")
    table.add_column("Embedding")
    table.add_column("Chat Model")
    for m in metrics:
        table.add_column(m, justify="right")

    for _, row in df.iterrows():
        cells = [str(row["embedding_model"]), str(row["chat_model"])]
        for m in metrics:
            val = row.get(m, "-")
            if isinstance(val, float):
                cells.append(f"{val:.3f}")
            else:
                cells.append(str(val))
        table.add_row(*cells)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretty-print consolidated benchmark results.")
    parser.add_argument("files", nargs="*", help="Optional list of benchmark result JSON files to load. If omitted, --results-dir pattern is used.")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="Directory to scan when no files are provided.")
    parser.add_argument("--csv", metavar="PATH", help="Optional path to write the tidy DataFrame as CSV.")
    parser.add_argument("--html", metavar="PATH", help="Optional path to write a styled HTML table.")
    args = parser.parse_args()

    # Decide which files to load
    if args.files:
        # Use user‐provided list (supports glob patterns because shell expands)
        files = args.files
    else:
        pattern = os.path.join(args.results_dir, "benchmark_results_*_multi_emb.json")
        files = sorted(glob.glob(pattern))
        if not files:
            raise SystemExit(f"No result files provided and none matching pattern {pattern}")

    df = load_results(files)

    # Determine metric columns (exclude identifier columns)
    metric_cols = [c for c in df.columns if c not in {"embedding_model", "chat_model", "sample_size"}]

    # Sort for nice display
    df_sorted = df.sort_values(["embedding_model", "chat_model"]).reset_index(drop=True)

    # Print to terminal using rich
    console = Console()
    console.print(to_rich_table(df_sorted, metric_cols))

    # Optional exports
    if args.csv:
        df_sorted.to_csv(args.csv, index=False)
        rprint(f"[green]Wrote CSV to {args.csv}")
    if args.html:
        try:
            styled = df_sorted.style.format({c: "{:.3f}" for c in metric_cols})
            styled.to_html(args.html)
            rprint(f"[green]Wrote HTML table to {args.html}")
        except Exception as e:
            rprint(f"[red]Failed to write HTML: {e}")


if __name__ == "__main__":
    main()
