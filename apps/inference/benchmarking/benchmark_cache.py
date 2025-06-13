#!/usr/bin/env python3
# ABOUTME: Standalone benchmark script to demonstrate layer activation cache performance
# ABOUTME: Run this to generate performance metrics for GitHub PR

"""
Layer Activation Cache Performance Benchmark

This script demonstrates the performance improvements from the layer activation cache
by running a series of timed tests against the inference endpoints.

Usage:
    python benchmark_cache.py [--endpoint <url>] [--model <model_id>]
"""

import argparse
import json
import statistics
import time
from typing import Tuple

import requests
from rich.console import Console
from rich.table import Table

console = Console()


class CacheBenchmark:
    """Benchmark tool for layer activation cache performance."""

    def __init__(
        self, base_url: str = "http://localhost:5002", model: str = "gpt2-small"
    ):
        self.base_url = base_url
        self.model = model
        self.results: dict[str, list[float]] = {}

    def _make_request(self, endpoint: str, payload: dict) -> Tuple[float, bool]:
        """Make a timed request to an endpoint."""
        url = f"{self.base_url}/v1/{endpoint}"
        start = time.time()
        try:
            response = requests.post(url, json=payload)
            elapsed = time.time() - start
            success = response.status_code == 200
            return elapsed, success
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return 0.0, False

    def benchmark_activation_all(self, iterations: int = 5):
        """Benchmark activation/all endpoint."""
        console.print(
            "\n[bold blue]Benchmarking activation/all endpoint...[/bold blue]"
        )

        payload = {
            "prompt": "The development of artificial intelligence has accelerated rapidly in recent years",
            "model": self.model,
            "source_set": "res-jb",
            "selected_sources": [
                "0-res-jb",
                "2-res-jb",
                "4-res-jb",
                "6-res-jb",
                "8-res-jb",
            ],
            "num_results": 20,
            "sort_by_token_indexes": [],
            "ignore_bos": False,
        }

        times = []
        for i in range(iterations):
            elapsed, success = self._make_request("activation/all", payload)
            if success:
                times.append(elapsed)
                status = "[green]✓[/green]" if i == 0 else "[yellow]✓[/yellow]"
                console.print(f"  Run {i+1}: {elapsed*1000:.2f}ms {status}")
            else:
                console.print(f"  Run {i+1}: [red]Failed[/red]")

        self.results["activation/all"] = times
        return times

    def benchmark_activation_single(self, iterations: int = 5):
        """Benchmark activation/single endpoint with multiple layers."""
        console.print(
            "\n[bold blue]Benchmarking activation/single endpoint...[/bold blue]"
        )

        prompt = (
            "Machine learning models have revolutionized natural language processing"
        )
        layers = ["0-res-jb", "3-res-jb", "6-res-jb", "9-res-jb", "11-res-jb"]

        all_times = []
        for layer in layers:
            payload = {
                "prompt": prompt,
                "source": layer,
                "index": 100,
            }

            layer_times = []
            console.print(f"\n  [cyan]Layer {layer}:[/cyan]")

            for i in range(iterations):
                elapsed, success = self._make_request("activation/single", payload)
                if success:
                    layer_times.append(elapsed)
                    all_times.append(elapsed)
                    cache_indicator = "🔵" if i == 0 else "🟢"
                    console.print(
                        f"    Run {i+1}: {elapsed*1000:.2f}ms {cache_indicator}"
                    )

        self.results["activation/single"] = all_times
        return all_times

    def benchmark_mixed_pattern(self):
        """Benchmark a realistic mixed usage pattern."""
        console.print("\n[bold blue]Benchmarking mixed endpoint pattern...[/bold blue]")

        prompt = "Understanding deep neural networks requires knowledge of linear algebra and calculus"

        sequence = [
            (
                "activation/all",
                {
                    "prompt": prompt,
                    "model": self.model,
                    "source_set": "res-jb",
                    "selected_sources": ["0-res-jb", "1-res-jb", "2-res-jb"],
                    "num_results": 10,
                },
            ),
            (
                "activation/single",
                {
                    "prompt": prompt,
                    "source": "1-res-jb",
                    "index": 50,
                },
            ),
            (
                "activation/topk-by-token",
                {
                    "prompt": prompt,
                    "source": "2-res-jb",
                    "top_k": 5,
                },
            ),
            (
                "activation/single",
                {
                    "prompt": prompt,
                    "source": "0-res-jb",
                    "index": 75,
                },
            ),
        ]

        times = []
        for i, (endpoint, payload) in enumerate(sequence):
            elapsed, success = self._make_request(endpoint, payload)
            if success:
                times.append(elapsed)
                cache_status = "COLD" if i == 0 else "WARM"
                console.print(
                    f"  {endpoint:<25} {elapsed*1000:>8.2f}ms [{cache_status}]"
                )

        self.results["mixed_pattern"] = times
        return times

    def get_cache_stats(self) -> dict:
        """Fetch cache statistics from health endpoint."""
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                return data.get("cache_stats", {})
        except Exception:
            pass
        return {}

    def print_summary(self):
        """Print a summary of benchmark results."""
        console.print("\n[bold green]Performance Summary[/bold green]")

        # Create summary table
        table = Table(title="Benchmark Results")
        table.add_column("Endpoint", style="cyan")
        table.add_column("First Run (ms)", style="red")
        table.add_column("Avg Cached (ms)", style="green")
        table.add_column("Improvement", style="yellow")
        table.add_column("Speedup", style="magenta")

        for endpoint, times in self.results.items():
            if len(times) >= 2:
                first_run = times[0] * 1000
                cached_runs = times[1:]
                avg_cached = statistics.mean(cached_runs) * 1000
                improvement = (first_run - avg_cached) / first_run * 100
                speedup = first_run / avg_cached

                table.add_row(
                    endpoint,
                    f"{first_run:.2f}",
                    f"{avg_cached:.2f}",
                    f"{improvement:.1f}%",
                    f"{speedup:.1f}x",
                )

        console.print(table)

        # Print cache statistics
        cache_stats = self.get_cache_stats()
        if cache_stats:
            console.print("\n[bold cyan]Cache Statistics:[/bold cyan]")
            console.print(f"  Hit Rate: {cache_stats.get('hit_rate', 0):.2%}")
            console.print(f"  Total Hits: {cache_stats.get('hits', 0)}")
            console.print(f"  Total Misses: {cache_stats.get('misses', 0)}")
            console.print(
                f"  Cache Size: {cache_stats.get('size', 0)}/{cache_stats.get('max_size', 5)}"
            )
            console.print(f"  Evictions: {cache_stats.get('evictions', 0)}")

    def export_results(self, filename: str = "cache_benchmark_results.json"):
        """Export results to JSON file."""
        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": self.model,
            "results": self.results,
            "cache_stats": self.get_cache_stats(),
            "summary": {},
        }

        # Calculate summary statistics
        for endpoint, times in self.results.items():
            if len(times) >= 2:
                output["summary"][endpoint] = {
                    "first_run_ms": times[0] * 1000,
                    "avg_cached_ms": statistics.mean(times[1:]) * 1000,
                    "improvement_percent": (times[0] - statistics.mean(times[1:]))
                    / times[0]
                    * 100,
                    "sample_size": len(times),
                }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        console.print(f"\n[green]Results exported to {filename}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark layer activation cache performance"
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:5002",
        help="Base URL of the inference server",
    )
    parser.add_argument(
        "--model", default="gpt2-small", help="Model ID to use for testing"
    )
    parser.add_argument(
        "--export", action="store_true", help="Export results to JSON file"
    )

    args = parser.parse_args()

    console.print("[bold]Layer Activation Cache Performance Benchmark[/bold]")
    console.print(f"Server: {args.endpoint}")
    console.print(f"Model: {args.model}")

    # Create benchmark instance
    benchmark = CacheBenchmark(args.endpoint, args.model)

    # Run benchmarks
    try:
        benchmark.benchmark_activation_all()
        benchmark.benchmark_activation_single()
        benchmark.benchmark_mixed_pattern()

        # Print summary
        benchmark.print_summary()

        # Export if requested
        if args.export:
            benchmark.export_results()

        console.print("\n[bold green]✨ Benchmark completed successfully![/bold green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during benchmark: {e}[/red]")


if __name__ == "__main__":
    main()
