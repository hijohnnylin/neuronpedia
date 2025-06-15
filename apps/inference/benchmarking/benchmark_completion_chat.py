#!/usr/bin/env python3
# ABOUTME: Benchmark script to measure performance of completion_chat endpoint before and after optimization
# ABOUTME: Tracks timing for STEERED, DEFAULT, and BOTH response types with various chat configurations

"""
Completion Chat Endpoint Performance Benchmark

This script measures the performance of the /v1/steer/completion-chat endpoint
to establish baseline metrics before optimization and compare after.

Usage:
    python benchmark_completion_chat.py [--before|--after] [--endpoint <url>] [--model <model_id>]
"""

import argparse
import asyncio
import json
import statistics
import time
from typing import Tuple

import aiohttp
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table

console = Console()


class CompletionChatBenchmark:
    """Benchmark tool for completion-chat endpoint performance."""

    def __init__(
        self, base_url: str = "http://localhost:5002", model: str = "gpt2-small"
    ):
        self.base_url = base_url
        self.model = model
        self.results: dict[str, list[float]] = {}
        self.token_counts: dict[str, list[int]] = {}

    async def _make_request(
        self, payload: dict, track_tokens: bool = True
    ) -> Tuple[float, bool, dict | None]:
        """Make an async timed request to completion-chat endpoint."""
        url = f"{self.base_url}/v1/steer/completion-chat"

        # Track initial time
        start = time.time()
        total_tokens = 0

        try:
            headers = {"X-SECRET-KEY": "localhost-secret"}
            async with aiohttp.ClientSession() as session, session.post(
                url, json=payload, headers=headers
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    console.print(f"[red]HTTP {response.status}: {error_text}[/red]")
                    return 0.0, False, None

                # Stream response to count tokens
                result_data = {"steered": "", "default": ""}
                async for line in response.content:
                    if line:
                        try:
                            line_str = line.decode("utf-8").strip()
                            if (
                                line_str.startswith("data: ")
                                and line_str != "data: [DONE]"
                            ):
                                data = json.loads(line_str[6:])
                                if track_tokens and "outputs" in data:
                                    # Parse the completion_chat response format
                                    for output in data["outputs"]:
                                        if (
                                            output.get("type") == "STEERED"
                                            and "raw" in output
                                        ):
                                            result_data["steered"] = output["raw"]
                                        elif (
                                            output.get("type") == "DEFAULT"
                                            and "raw" in output
                                        ):
                                            result_data["default"] = output["raw"]
                        except Exception:
                            pass

                elapsed = time.time() - start

                # Estimate token count (rough approximation)
                if track_tokens:
                    total_tokens = len(result_data.get("steered", "").split()) + len(
                        result_data.get("default", "").split()
                    )

                return elapsed, True, {"tokens": total_tokens, "elapsed": elapsed}

        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return 0.0, False, None

    async def benchmark_single_type(self, iterations: int = 5):
        """Benchmark with only STEERED or only DEFAULT responses."""
        console.print("\n[bold blue]Benchmarking single response type...[/bold blue]")

        # For GPT-2, we'll use simple prompts without roles
        prompt = [
            {
                "role": "user",
                "content": "Explain how neural networks learn in simple terms.",
            }
        ]

        feature = {
            "model": self.model,
            "source": "5-res-jb",  # Use layer 5 SAE source
            "index": 100,
            "strength": 2.0,
            "steering_vector": np.random.randn(768).tolist(),  # GPT-2 small dimension
        }

        for response_type in ["STEERED", "DEFAULT"]:
            console.print(f"\n  [cyan]{response_type} only:[/cyan]")

            payload = {
                "prompt": prompt,
                "model": self.model,
                "features": [feature] if response_type == "STEERED" else [],
                "types": [response_type],
                "steer_method": "SIMPLE_ADDITIVE",
                "normalize_steering": True,
                "strength_multiplier": 1.0,
                "n_completion_tokens": 50,
                "temperature": 0.7,
                "freq_penalty": 0.0,
                "seed": 42,
                "steer_special_tokens": False,
            }

            times = []
            token_counts = []

            for i in range(iterations):
                elapsed, success, data = await self._make_request(payload)
                if success:
                    times.append(elapsed)
                    if data:
                        token_counts.append(data["tokens"])

                    console.print(
                        f"    Run {i+1}: {elapsed*1000:.2f}ms ({data['tokens'] if data else '?'} tokens)"
                    )
                else:
                    console.print(f"    Run {i+1}: [red]Failed[/red]")

            self.results[f"single_{response_type.lower()}"] = times
            self.token_counts[f"single_{response_type.lower()}"] = token_counts

    async def benchmark_both_types(self, iterations: int = 5):
        """Benchmark with both STEERED and DEFAULT responses (the optimization target)."""
        console.print(
            "\n[bold blue]Benchmarking both response types (STEERED + DEFAULT)...[/bold blue]"
        )

        test_cases = [
            {
                "name": "Short conversation",
                "prompt": [{"role": "user", "content": "What is machine learning?"}],
                "n_completion_tokens": 30,
            },
            {
                "name": "Medium conversation",
                "prompt": [{"role": "user", "content": "How do transformers work?"}],
                "n_completion_tokens": 50,
            },
            {
                "name": "Long conversation",
                "prompt": [
                    {
                        "role": "user",
                        "content": "Write a Python function to sort a list.",
                    }
                ],
                "n_completion_tokens": 100,
            },
        ]

        features = [
            {
                "model": self.model,
                "source": "5-res-jb",
                "index": 100,
                "strength": 2.0,
                "steering_vector": np.random.randn(768).tolist(),
            },
            {
                "model": self.model,
                "source": "7-res-jb",
                "index": 200,
                "strength": 1.5,
                "steering_vector": np.random.randn(768).tolist(),
            },
        ]

        for test_case in test_cases:
            console.print(f"\n  [cyan]{test_case['name']}:[/cyan]")

            payload = {
                "prompt": test_case["prompt"],
                "model": self.model,
                "features": features,
                "types": ["STEERED", "DEFAULT"],  # Both types - target for optimization
                "steer_method": "SIMPLE_ADDITIVE",
                "normalize_steering": True,
                "strength_multiplier": 1.0,
                "n_completion_tokens": test_case["n_completion_tokens"],
                "temperature": 0.7,
                "freq_penalty": 0.0,
                "seed": 42,
                "steer_special_tokens": False,
            }

            times = []
            token_counts = []

            for i in range(iterations):
                elapsed, success, data = await self._make_request(payload)
                if success:
                    times.append(elapsed)
                    if data:
                        token_counts.append(data["tokens"])

                    tokens_per_sec = (
                        data["tokens"] / elapsed if data and elapsed > 0 else 0
                    )
                    console.print(
                        f"    Run {i+1}: {elapsed*1000:.2f}ms ({data['tokens'] if data else '?'} tokens, {tokens_per_sec:.1f} tok/s)"
                    )
                else:
                    console.print(f"    Run {i+1}: [red]Failed[/red]")

            key = f"both_{test_case['name'].lower().replace(' ', '_')}"
            self.results[key] = times
            self.token_counts[key] = token_counts

    async def benchmark_stress_test(self):
        """Stress test with many features and long generation."""
        console.print("\n[bold blue]Running stress test...[/bold blue]")

        # Create many steering features
        features = []
        for i in range(10):
            features.append(
                {
                    "model": self.model,
                    "source": f"{i}-res-jb",  # Use different layers (0-9)
                    "index": i * 100,
                    "strength": 1.0 + (i * 0.1),
                    "steering_vector": np.random.randn(768).tolist(),
                }
            )

        prompt = [
            {
                "role": "user",
                "content": "Write a detailed story about artificial intelligence.",
            }
        ]

        payload = {
            "prompt": prompt,
            "model": self.model,
            "features": features,
            "types": ["STEERED", "DEFAULT"],
            "steer_method": "SIMPLE_ADDITIVE",
            "normalize_steering": True,
            "strength_multiplier": 1.0,
            "n_completion_tokens": 200,  # Long generation
            "temperature": 0.7,
            "freq_penalty": 0.0,
            "seed": 42,
            "steer_special_tokens": False,
        }

        console.print(
            f"  Testing with {len(features)} steering features, n_completion_tokens=200"
        )

        elapsed, success, data = await self._make_request(payload)
        if success:
            tokens_per_sec = data["tokens"] / elapsed if data and elapsed > 0 else 0
            console.print(
                f"  Result: {elapsed*1000:.2f}ms ({data['tokens'] if data else '?'} tokens, {tokens_per_sec:.1f} tok/s)"
            )
            self.results["stress_test"] = [elapsed]
            self.token_counts["stress_test"] = [data["tokens"]] if data else []
        else:
            console.print("  Result: [red]Failed[/red]")

    def calculate_memory_usage(self):
        """Estimate memory usage during benchmark."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / 1024 / 1024  # MB
        except ImportError:
            return None

    def print_summary(self):
        """Print a comprehensive summary of benchmark results."""
        console.print("\n[bold green]Performance Summary[/bold green]")

        # Main results table
        table = Table(title="Completion Chat Benchmark Results")
        table.add_column("Test Case", style="cyan")
        table.add_column("Avg Time (ms)", style="yellow")
        table.add_column("Std Dev (ms)", style="blue")
        table.add_column("Min (ms)", style="green")
        table.add_column("Max (ms)", style="red")
        table.add_column("Avg Tokens", style="magenta")
        table.add_column("Tok/s", style="white")

        for test_name, times in self.results.items():
            if times:
                avg_time = statistics.mean(times) * 1000
                std_dev = statistics.stdev(times) * 1000 if len(times) > 1 else 0
                min_time = min(times) * 1000
                max_time = max(times) * 1000

                # Get corresponding token counts
                tokens = self.token_counts.get(test_name, [])
                avg_tokens = statistics.mean(tokens) if tokens else 0
                tokens_per_sec = (
                    avg_tokens / statistics.mean(times)
                    if times and avg_tokens > 0
                    else 0
                )

                table.add_row(
                    test_name,
                    f"{avg_time:.2f}",
                    f"{std_dev:.2f}",
                    f"{min_time:.2f}",
                    f"{max_time:.2f}",
                    f"{avg_tokens:.0f}",
                    f"{tokens_per_sec:.1f}",
                )

        console.print(table)

        # Key metrics for optimization comparison
        console.print("\n[bold cyan]Key Metrics for Optimization:[/bold cyan]")

        # Calculate overhead of generating both types vs single type
        single_steered = self.results.get("single_steered", [])
        single_default = self.results.get("single_default", [])
        both_short = self.results.get("both_short_conversation", [])

        if single_steered and single_default and both_short:
            avg_single_s = statistics.mean(single_steered) * 1000
            avg_single_d = statistics.mean(single_default) * 1000
            avg_both = statistics.mean(both_short) * 1000
            expected_sequential = avg_single_s + avg_single_d
            actual_overhead = avg_both - max(avg_single_s, avg_single_d)

            console.print(f"  Single STEERED avg: {avg_single_s:.2f}ms")
            console.print(f"  Single DEFAULT avg: {avg_single_d:.2f}ms")
            console.print(f"  Both types avg: {avg_both:.2f}ms")
            console.print(f"  Expected if sequential: {expected_sequential:.2f}ms")
            console.print(
                f"  Current overhead: {actual_overhead:.2f}ms ({actual_overhead/avg_both*100:.1f}% of total)"
            )
            console.print(
                f"  [yellow]Optimization potential: ~{expected_sequential - avg_both:.2f}ms reduction[/yellow]"
            )

        # Memory usage
        memory = self.calculate_memory_usage()
        if memory:
            console.print(f"\n  Memory usage: {memory:.1f} MB")

    def export_results(self, filename: str = None, phase: str = "before"):
        """Export results to JSON file."""
        if filename is None:
            filename = f"completion_chat_benchmark_{phase}.json"

        output = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase,
            "model": self.model,
            "endpoint": self.base_url,
            "results": {},
            "token_counts": self.token_counts,
            "summary": {},
        }

        # Include raw timing data
        for test_name, times in self.results.items():
            output["results"][test_name] = {
                "times_ms": [t * 1000 for t in times],
                "count": len(times),
            }

            if times:
                tokens = self.token_counts.get(test_name, [])
                output["summary"][test_name] = {
                    "avg_time_ms": statistics.mean(times) * 1000,
                    "std_dev_ms": statistics.stdev(times) * 1000
                    if len(times) > 1
                    else 0,
                    "min_time_ms": min(times) * 1000,
                    "max_time_ms": max(times) * 1000,
                    "avg_tokens": statistics.mean(tokens) if tokens else 0,
                    "tokens_per_sec": statistics.mean(tokens) / statistics.mean(times)
                    if tokens and times
                    else 0,
                }

        with open(filename, "w") as f:
            json.dump(output, f, indent=2)

        console.print(f"\n[green]Results exported to {filename}[/green]")
        return filename

    def compare_results(self, before_file: str, after_file: str):
        """Compare before and after optimization results."""
        try:
            with open(before_file) as f:
                before = json.load(f)
            with open(after_file) as f:
                after = json.load(f)

            console.print("\n[bold green]Optimization Comparison[/bold green]")

            table = Table(title="Before vs After Optimization")
            table.add_column("Test Case", style="cyan")
            table.add_column("Before (ms)", style="red")
            table.add_column("After (ms)", style="green")
            table.add_column("Improvement", style="yellow")
            table.add_column("Speedup", style="magenta")

            for test_name in before["summary"]:
                if test_name in after["summary"]:
                    before_time = before["summary"][test_name]["avg_time_ms"]
                    after_time = after["summary"][test_name]["avg_time_ms"]
                    improvement = (before_time - after_time) / before_time * 100
                    speedup = before_time / after_time

                    table.add_row(
                        test_name,
                        f"{before_time:.2f}",
                        f"{after_time:.2f}",
                        f"{improvement:.1f}%",
                        f"{speedup:.2f}x",
                    )

            console.print(table)

        except Exception as e:
            console.print(f"[red]Error comparing results: {e}[/red]")


async def main():
    parser = argparse.ArgumentParser(
        description="Benchmark completion-chat endpoint performance"
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
        "--phase", choices=["before", "after"], default="before", help="Benchmark phase"
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare before and after results"
    )
    parser.add_argument("--before-file", help="Path to before optimization results")
    parser.add_argument("--after-file", help="Path to after optimization results")
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations per test"
    )

    args = parser.parse_args()

    if args.compare:
        # Just compare existing results
        before_file = args.before_file or "completion_chat_benchmark_before.json"
        after_file = args.after_file or "completion_chat_benchmark_after.json"
        benchmark = CompletionChatBenchmark()
        benchmark.compare_results(before_file, after_file)
        return

    console.print("[bold]Completion Chat Endpoint Performance Benchmark[/bold]")
    console.print(f"Server: {args.endpoint}")
    console.print(f"Model: {args.model}")
    console.print(f"Phase: {args.phase}")
    console.print(f"Iterations: {args.iterations}")

    # Create benchmark instance
    benchmark = CompletionChatBenchmark(args.endpoint, args.model)

    # Run benchmarks
    try:
        with Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Running benchmarks...", total=4)

            await benchmark.benchmark_single_type(args.iterations)
            progress.advance(task)

            await benchmark.benchmark_both_types(args.iterations)
            progress.advance(task)

            await benchmark.benchmark_stress_test()
            progress.advance(task)

            progress.advance(task)

        # Print summary
        benchmark.print_summary()

        # Export results
        benchmark.export_results(phase=args.phase)

        console.print("\n[bold green]✨ Benchmark completed successfully![/bold green]")

        if args.phase == "before":
            console.print("\n[yellow]Next steps:[/yellow]")
            console.print("1. Implement optimizations")
            console.print("2. Run benchmark again with --phase after")
            console.print(f"3. Compare results with: python {__file__} --compare")

    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error during benchmark: {e}[/red]")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
