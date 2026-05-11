"""Minimal helpers used by `scripts/oom_stress.py`.

Provides three symbols the stress harness imports:
  * ``NLAClient``       — async HTTP client against a running NLA server.
  * ``VRAMMonitor``     — background ``nvidia-smi`` sampler with peak/min stats.
  * ``build_prompt_exact`` — tokenize-driven prompt builder near a target
    token count, using the server's own ``/tokenize`` so counts match the
    actual source-model tokenizer.

Stand-in for an upstream ``benchmark/lib.py`` that ships with the
benchmark suite. If that suite reappears, drop this in favor of it.
"""

from __future__ import annotations

import subprocess
import threading
import time
from dataclasses import dataclass

import httpx


# ─── VRAM monitor ──────────────────────────────────────────────────────────


@dataclass
class _VRAMSample:
    used_gb: float
    total_gb: float
    free_gb: float


class VRAMMonitor:
    """Sample one GPU's VRAM via ``nvidia-smi`` on a background thread.

    ``oom_stress.py`` reads ``samples[0]`` for the baseline and queries
    ``peak_used_gb`` / ``min_free_gb`` after ``stop()``. The constructor
    seeds ``samples`` with one synchronous read so the baseline is
    available even if the caller never starts the sampler thread.
    """

    def __init__(self, device_idx: int = 0, interval_s: float = 0.1) -> None:
        self.device_idx = device_idx
        self.interval_s = interval_s
        self.samples: list[_VRAMSample] = [self._read()]
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _read(self) -> _VRAMSample:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    f"--id={self.device_idx}",
                    "--query-gpu=memory.used,memory.total,memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=5,
            ).strip()
            used_mb, total_mb, free_mb = (float(x) for x in out.split(","))
            return _VRAMSample(used_mb / 1024.0, total_mb / 1024.0, free_mb / 1024.0)
        except Exception:
            return _VRAMSample(0.0, 0.0, 0.0)

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.is_set():
                self.samples.append(self._read())
                time.sleep(self.interval_s)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def peak_used_gb(self) -> float:
        return max((s.used_gb for s in self.samples), default=0.0)

    @property
    def min_free_gb(self) -> float:
        return min(
            (s.free_gb for s in self.samples if s.total_gb > 0),
            default=0.0,
        )


# ─── HTTP client ───────────────────────────────────────────────────────────


class NLAClient:
    """Async HTTP client matching the surface ``oom_stress.py`` uses.

    All endpoint methods either return a parsed JSON body or, for
    ``explain``, a dict of ``{"status": int, "body": dict | str}`` so the
    caller can categorize 200 / 4xx / 5xx without raising.
    """

    def __init__(
        self,
        url: str,
        secret: str | None = None,
        timeout_s: float = 600.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.headers = {"X-SECRET-KEY": secret} if secret else {}
        self._client = httpx.AsyncClient(timeout=timeout_s)

    async def health(self) -> dict:
        r = await self._client.get(f"{self.url}/", headers=self.headers)
        r.raise_for_status()
        return r.json()

    async def tokenize(self, text: str) -> dict:
        r = await self._client.post(
            f"{self.url}/tokenize",
            headers=self.headers,
            json={"text": text},
        )
        r.raise_for_status()
        return r.json()

    async def explain(
        self,
        text: str,
        positions: list[int] | None = None,
        max_new_tokens: int = 200,
    ) -> dict:
        body: dict = {"text": text, "max_new_tokens": max_new_tokens}
        if positions is not None:
            body["positions"] = positions
        r = await self._client.post(
            f"{self.url}/explain",
            headers=self.headers,
            json=body,
        )
        try:
            parsed = r.json()
        except Exception:
            parsed = {"raw": r.text[:2000]}
        return {"status": r.status_code, "body": parsed}

    async def close(self) -> None:
        await self._client.aclose()


# ─── Prompt builder ────────────────────────────────────────────────────────


# Repeat unit: pangrams chosen for token-tokenizer stability across BPE
# variants (Llama / Gemma / Qwen all roughly agree on ~30 tokens here).
_BASE_CHUNK = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
)


async def build_prompt_exact(
    client: NLAClient,
    target_tokens: int,
) -> tuple[str, int]:
    """Build a prompt close to ``target_tokens`` long.

    Uses the server's ``/tokenize`` so counts match the source model's
    actual tokenizer (no client-side BPE guesswork). Returns
    ``(text, actual_token_count)``. May undershoot by a chunk if removing
    one more chunk would underflow the target — ``oom_stress.py`` filters
    scenarios where ``positions > actual`` so this is harmless.
    """
    text = _BASE_CHUNK
    res = await client.tokenize(text)
    actual = int(res["prompt_length"])

    # Grow until we're at or just past the target.
    while actual < target_tokens:
        text += _BASE_CHUNK
        res = await client.tokenize(text)
        actual = int(res["prompt_length"])
        if len(text) > 4_000_000:
            # Safety: avoid pathological growth on absurd targets.
            break

    # Trim whole chunks if we overshot, but keep the count >= target.
    while actual > target_tokens and text.endswith(_BASE_CHUNK):
        candidate = text[: -len(_BASE_CHUNK)]
        res = await client.tokenize(candidate)
        candidate_actual = int(res["prompt_length"])
        if candidate_actual < target_tokens:
            break
        text, actual = candidate, candidate_actual

    return text, actual
