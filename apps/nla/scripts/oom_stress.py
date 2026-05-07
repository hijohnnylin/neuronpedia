"""Stress test the running NLA server for OOM and saturation under varying
workload shapes.

Sweeps combinations of:
  * parallel /explain count
  * input prompt length (tokens)
  * positions per /explain
  * max_new_tokens per stream

For each scenario, captures peak GPU VRAM via nvidia-smi and reports
per-request success / 429 / 5xx / timeout / OOM-detected counts plus
mean / p95 / max wall-clock latency.

Runs against an *already-running* server — no model reloads — so each
sweep takes minutes, not hours. Pair with `benchmark/sweep.py` when you
need to compare across server-startup configs.

Examples:
    SECRET=mysecret uv run python scripts/oom_stress.py
    SECRET=mysecret uv run python scripts/oom_stress.py \\
        --parallel 1,2,4,8 \\
        --text-tokens 256,1024,4096 \\
        --positions 8,32,64 \\
        --max-new-tokens 200 \\
        --output oom_report.json

Exit code is 0 if no scenario triggered an OOM/5xx, 1 otherwise (handy
for CI). Use --stop-on-oom to abort the sweep at the first failure
rather than running every remaining scenario after a crash.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

# Reuse the benchmark suite's helpers (VRAM monitor, prompt builder, client).
sys.path.insert(0, str(Path(__file__).resolve().parent / "benchmark"))
import lib  # noqa: E402


# ─── Result types ───────────────────────────────────────────────────────────


@dataclass
class _Outcome:
    """Outcome of one /explain call within a scenario."""

    elapsed_s: float
    status: int | None  # None when no HTTP response was received
    error: str | None  # None on 200
    oom_detected: bool


@dataclass
class ScenarioResult:
    parallel: int
    text_tokens: int
    positions: int
    max_new_tokens: int

    n_total: int
    n_ok: int
    n_429: int
    n_5xx: int
    n_4xx_other: int
    n_timeout: int
    n_oom: int
    n_other_error: int

    elapsed_s_mean: float
    elapsed_s_p95: float
    elapsed_s_max: float

    vram_baseline_used_gb: float
    vram_peak_used_gb: float
    vram_delta_gb: float
    vram_min_free_gb: float
    vram_total_gb: float
    vram_risk: bool

    sample_errors: list[str]


# ─── OOM heuristics ─────────────────────────────────────────────────────────


_OOM_RE = re.compile(
    r"out of memory|cuda.{0,40}oom|cuda.{0,40}memory|hbm full|"
    r"not enough memory|sglang.+pool",
    re.I,
)


def _looks_like_oom(status: int | None, body_text: str | None) -> bool:
    """Heuristic OOM detection from the response.

    Conservative: 5xx + an OOM-shaped error string, OR a 502/503/504 (which
    typically means the server process died mid-response). Connection-level
    errors (RemoteProtocolError, ReadError) are flagged separately at the
    call-site since httpx doesn't surface them as a status code.
    """
    if status in (502, 503, 504):
        return True
    if status is not None and 500 <= status < 600 and body_text:
        if _OOM_RE.search(body_text):
            return True
    return False


# ─── Fire one /explain ──────────────────────────────────────────────────────


async def _one_call(
    client: lib.NLAClient,
    *,
    text: str,
    positions: list[int],
    max_new_tokens: int,
) -> _Outcome:
    t0 = time.perf_counter()
    try:
        res = await client.explain(
            text,
            positions=positions,
            max_new_tokens=max_new_tokens,
        )
        elapsed = time.perf_counter() - t0
        status = int(res["status"])
        body = res["body"]
        if status == 200:
            return _Outcome(elapsed, status, None, False)
        body_text = (
            body.get("detail")
            if isinstance(body, dict) and "detail" in body
            else str(body)
        )
        return _Outcome(
            elapsed_s=elapsed,
            status=status,
            error=f"HTTP {status}: {str(body_text)[:300]}",
            oom_detected=_looks_like_oom(status, str(body_text)),
        )
    except httpx.ReadTimeout as e:
        return _Outcome(time.perf_counter() - t0, None, f"timeout: {e}", False)
    except (httpx.RemoteProtocolError, httpx.ReadError, httpx.ConnectError) as e:
        # Connection died mid-request — most likely the server crashed (OOM
        # is the common cause). Flag as OOM so it shows up in the summary.
        return _Outcome(
            elapsed_s=time.perf_counter() - t0,
            status=None,
            error=f"{type(e).__name__}: {e}",
            oom_detected=True,
        )
    except Exception as e:
        return _Outcome(
            elapsed_s=time.perf_counter() - t0,
            status=None,
            error=f"{type(e).__name__}: {e}",
            oom_detected=False,
        )


# ─── Scenario runner ────────────────────────────────────────────────────────


def _pctl(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = max(0, min(len(xs) - 1, int(math.ceil(p * len(xs))) - 1))
    return xs[k]


async def _run_scenario(
    client: lib.NLAClient,
    *,
    parallel: int,
    text: str,
    actual_text_tokens: int,
    positions: list[int],
    max_new_tokens: int,
    gpu_id: int,
    monitor_interval_s: float,
    cooldown_s: float,
    scenario_idx: int,
    n_scenarios: int,
) -> ScenarioResult:
    print(
        f"[{scenario_idx}/{n_scenarios}] "
        f"parallel={parallel} text≈{actual_text_tokens}t "
        f"positions={len(positions)} mnt={max_new_tokens} → ",
        end="",
        flush=True,
    )
    monitor = lib.VRAMMonitor(device_idx=gpu_id, interval_s=monitor_interval_s)
    monitor.start()
    baseline_used = monitor.samples[0].used_gb if monitor.samples else 0.0
    total = monitor.samples[0].total_gb if monitor.samples else 0.0

    tasks = [
        _one_call(
            client,
            text=text,
            positions=positions,
            max_new_tokens=max_new_tokens,
        )
        for _ in range(parallel)
    ]
    outcomes = await asyncio.gather(*tasks)

    # Allow sglang's KV pool / HF activations to free before the next scenario;
    # peak-VRAM tracking is per-scenario, not cumulative.
    if cooldown_s > 0:
        await asyncio.sleep(cooldown_s)
    monitor.stop()

    elapseds = [o.elapsed_s for o in outcomes]
    n_ok = sum(1 for o in outcomes if o.error is None)
    n_429 = sum(1 for o in outcomes if o.status == 429)
    n_5xx = sum(
        1 for o in outcomes if o.status is not None and 500 <= o.status < 600
    )
    n_4xx_other = sum(
        1 for o in outcomes if o.status is not None and 400 <= o.status < 500 and o.status != 429
    )
    n_timeout = sum(
        1 for o in outcomes if o.error is not None and "timeout:" in o.error
    )
    n_oom = sum(1 for o in outcomes if o.oom_detected)
    # "Other" = something failed but isn't accounted for by the above buckets.
    accounted = n_ok + n_429 + n_5xx + n_4xx_other + n_timeout
    n_other_error = max(0, len(outcomes) - accounted)

    sample_errors: list[str] = []
    for o in outcomes:
        if o.error and len(sample_errors) < 5 and o.error not in sample_errors:
            sample_errors.append(o.error)

    peak = monitor.peak_used_gb
    vram_risk = total > 0 and peak > total * 0.95

    flag = "ok"
    if n_oom > 0:
        flag = "OOM"
    elif vram_risk:
        flag = "VRAM RISK"
    elif n_5xx > 0:
        flag = "5xx"
    elif n_other_error > 0:
        flag = "ERR"
    elif n_timeout > 0:
        flag = "timeout"
    elif n_429 == len(outcomes):
        flag = "all 429"
    elif n_429 > 0:
        flag = "partial 429"
    print(
        f"ok={n_ok}/{parallel} 429={n_429} 5xx={n_5xx} timeout={n_timeout} "
        f"oom={n_oom} peak={peak:.1f}/{total:.0f}GB [{flag}]"
    )

    return ScenarioResult(
        parallel=parallel,
        text_tokens=actual_text_tokens,
        positions=len(positions),
        max_new_tokens=max_new_tokens,
        n_total=len(outcomes),
        n_ok=n_ok,
        n_429=n_429,
        n_5xx=n_5xx,
        n_4xx_other=n_4xx_other,
        n_timeout=n_timeout,
        n_oom=n_oom,
        n_other_error=n_other_error,
        elapsed_s_mean=statistics.fmean(elapseds) if elapseds else 0.0,
        elapsed_s_p95=_pctl(elapseds, 0.95),
        elapsed_s_max=max(elapseds) if elapseds else 0.0,
        vram_baseline_used_gb=baseline_used,
        vram_peak_used_gb=peak,
        vram_delta_gb=peak - baseline_used,
        vram_min_free_gb=monitor.min_free_gb,
        vram_total_gb=total,
        vram_risk=vram_risk,
        sample_errors=sample_errors,
    )


# ─── Scenario matrix ────────────────────────────────────────────────────────


def _matrix(
    parallels: list[int],
    text_tokens: list[int],
    positions: list[int],
    max_new_tokens: list[int],
) -> list[dict]:
    """Cartesian product, skipping nonsensical combos (positions > tokens)."""
    out: list[dict] = []
    for p in parallels:
        for tt in text_tokens:
            for pos in positions:
                if pos > tt:
                    continue
                for mnt in max_new_tokens:
                    out.append(
                        {
                            "parallel": p,
                            "text_tokens": tt,
                            "positions": pos,
                            "max_new_tokens": mnt,
                        }
                    )
    return out


# ─── Summary printing ───────────────────────────────────────────────────────


def _print_summary(results: list[ScenarioResult]) -> None:
    if not results:
        print("(no results)")
        return
    headers = [
        "par",
        "txt",
        "pos",
        "mnt",
        "ok",
        "429",
        "5xx",
        "to",
        "oom",
        "p95",
        "max",
        "peak/tot GB",
        "free GB",
        "flag",
    ]
    rows: list[list[str]] = []
    for r in results:
        flag = "ok"
        if r.n_oom > 0:
            flag = "OOM"
        elif r.vram_risk:
            flag = "VRAM RISK"
        elif r.n_5xx > 0:
            flag = "5xx"
        elif r.n_timeout > 0:
            flag = "timeout"
        elif r.n_4xx_other > 0:
            flag = "4xx"
        elif r.n_429 == r.n_total:
            flag = "all 429"
        elif r.n_429 > 0:
            flag = "partial 429"
        rows.append(
            [
                str(r.parallel),
                str(r.text_tokens),
                str(r.positions),
                str(r.max_new_tokens),
                str(r.n_ok),
                str(r.n_429),
                str(r.n_5xx),
                str(r.n_timeout),
                str(r.n_oom),
                f"{r.elapsed_s_p95:.1f}s",
                f"{r.elapsed_s_max:.1f}s",
                f"{r.vram_peak_used_gb:.1f}/{r.vram_total_gb:.0f}",
                f"{r.vram_min_free_gb:.1f}",
                flag,
            ]
        )
    widths = [
        max(len(c) for c in [h] + [row[i] for row in rows])
        for i, h in enumerate(headers)
    ]
    fmt = " | ".join(f"{{:>{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


# ─── Main ───────────────────────────────────────────────────────────────────


def _parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


async def amain(args: argparse.Namespace) -> int:
    secret = args.secret or os.environ.get("SECRET")
    client = lib.NLAClient(args.url, secret=secret, timeout_s=args.request_timeout_s)

    print(f"[oom_stress] target: {args.url}")
    try:
        health = await client.health()
    except Exception as e:
        print(f"[oom_stress] health check failed: {e}", file=sys.stderr)
        await client.close()
        return 2

    print("[oom_stress] server reports:")
    for k in (
        "verbalizer_model",
        "source_model",
        "verbalizer_quantization",
        "verbalizer_kv_cache_dtype",
        "max_concurrent",
        "max_concurrent_explains",
        "source_max_concurrent",
        "num_cuda_devices",
    ):
        v = health.get(k, "?")
        print(f"  {k}: {v}")
    if "limits" in health:
        print(f"  limits: {health['limits']}")

    # Build scenario list
    parallels = _parse_int_list(args.parallel)
    text_tokens = _parse_int_list(args.text_tokens)
    positions = _parse_int_list(args.positions)
    max_new_tokens = _parse_int_list(args.max_new_tokens)

    matrix = _matrix(parallels, text_tokens, positions, max_new_tokens)
    if not matrix:
        print("[oom_stress] no valid scenarios — check args", file=sys.stderr)
        await client.close()
        return 2

    # Build prompts once per unique text-token target.
    print("[oom_stress] preparing prompts...")
    prompts: dict[int, tuple[str, int]] = {}
    for tt in sorted(set(text_tokens)):
        text, actual = await lib.build_prompt_exact(client, tt)
        prompts[tt] = (text, actual)
        print(
            f"  text_tokens={tt} → {len(text)} chars, actual ~{actual} tokens"
        )

    # Filter scenarios where positions > actual tokens (build_prompt_exact may
    # have produced slightly fewer tokens than requested).
    matrix = [
        s for s in matrix if s["positions"] <= prompts[s["text_tokens"]][1]
    ]
    print(f"[oom_stress] running {len(matrix)} scenario(s)...")

    results: list[ScenarioResult] = []
    aborted = False
    try:
        for i, scen in enumerate(matrix, 1):
            text, actual_tt = prompts[scen["text_tokens"]]
            n_pos = scen["positions"]
            position_idxs = list(range(-n_pos, 0))
            r = await _run_scenario(
                client,
                parallel=scen["parallel"],
                text=text,
                actual_text_tokens=actual_tt,
                positions=position_idxs,
                max_new_tokens=scen["max_new_tokens"],
                gpu_id=args.gpu_id,
                monitor_interval_s=args.monitor_interval_s,
                cooldown_s=args.cooldown_s,
                scenario_idx=i,
                n_scenarios=len(matrix),
            )
            results.append(r)
            if args.stop_on_oom and (r.n_oom > 0 or r.vram_risk or r.n_5xx > 0):
                print(
                    "[oom_stress] stop-on-oom: aborting remaining scenarios"
                )
                aborted = True
                break
    except KeyboardInterrupt:
        print("\n[oom_stress] interrupted")
        aborted = True

    await client.close()

    print("\n=== Summary ===")
    _print_summary(results)

    # Risk recap.
    bad = [r for r in results if r.n_oom or r.n_5xx or r.vram_risk]
    if bad:
        print(f"\n{len(bad)} scenario(s) flagged as risky:")
        for r in bad:
            print(
                f"  parallel={r.parallel} text={r.text_tokens} "
                f"positions={r.positions} mnt={r.max_new_tokens} "
                f"→ ok={r.n_ok}/{r.n_total} 5xx={r.n_5xx} oom={r.n_oom} "
                f"peak={r.vram_peak_used_gb:.1f}/{r.vram_total_gb:.0f}GB"
            )
            for err in r.sample_errors[:2]:
                print(f"      e.g. {err}")
    else:
        print("\nNo OOM / 5xx / VRAM-risk scenarios.")

    if args.output:
        Path(args.output).write_text(
            json.dumps(
                {
                    "url": args.url,
                    "health": health,
                    "args": {
                        "parallel": parallels,
                        "text_tokens": text_tokens,
                        "positions": positions,
                        "max_new_tokens": max_new_tokens,
                    },
                    "aborted": aborted,
                    "results": [asdict(r) for r in results],
                },
                indent=2,
            )
        )
        print(f"\n[oom_stress] wrote report → {args.output}")

    rc = 1 if bad else 0
    return rc


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--url", default="http://localhost:5009")
    p.add_argument(
        "--secret",
        default=None,
        help="Auth secret (or set the SECRET env var).",
    )
    p.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="nvidia-smi index of the GPU sglang/HF use (default: 0).",
    )
    p.add_argument(
        "--parallel",
        default="1,2,4",
        help="Comma-list of concurrent /explain counts (default: 1,2,4).",
    )
    p.add_argument(
        "--text-tokens",
        default="256,1024,4096",
        help="Comma-list of input prompt sizes in tokens (approx).",
    )
    p.add_argument(
        "--positions",
        default="8,32,64",
        help="Comma-list of positions per /explain.",
    )
    p.add_argument(
        "--max-new-tokens",
        default="200",
        help="Comma-list of max_new_tokens values (default: 200).",
    )
    p.add_argument("--request-timeout-s", type=float, default=600.0)
    p.add_argument("--monitor-interval-s", type=float, default=0.1)
    p.add_argument(
        "--cooldown-s",
        type=float,
        default=2.0,
        help="Sleep between scenarios so VRAM settles before peak capture.",
    )
    p.add_argument(
        "--stop-on-oom",
        action="store_true",
        help="Abort on the first OOM/5xx/VRAM-risk scenario.",
    )
    p.add_argument("--output", help="Write JSON report to this path.")
    args = p.parse_args()
    rc = asyncio.run(amain(args))
    sys.exit(rc)


if __name__ == "__main__":
    main()
