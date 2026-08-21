"""Engine-owned vLLM NLA verbalizer backend (concept injection via ``prompt_embeds``).

The production verbalizer used to run an in-process ``sgl.Engine`` fed ``input_embeds``
(``NLAClient``). This backend does the same concept injection -- splice the (normalized)
activation vector into the prompt's embedding sequence -- but serves it through the
interp-engine's ``VLLMModel`` using vLLM's ``prompt_embeds`` (EmbedsPrompt),
so NLA shares the same engine/vLLM stack as the inference server instead of maintaining a
separate sglang dependency.

Mirrors the subset of the ``NLAClient`` / ``EagerVerbalizer`` interface the NLA server and
``/health`` endpoint depend on (``.cfg`` / ``.tokenizer`` / ``.device`` / ``.embed`` /
``.embed_scale`` / the sglang-only scalars, ``async_generate`` / ``async_generate_stream`` /
``generate`` / ``_extract_text`` / ``shutdown``). Reuses the device-neutral injection +
text-extraction helpers from ``nla_inference``; only generation differs.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from nla_inference import (
    NLAConfig,
    _load_tokenizer,
    build_injected_embeds,
    extract_verbalizer_text,
    load_embedding_only,
    load_nla_config,
    resolve_checkpoint_path,
    resolve_embed_scale,
)

if TYPE_CHECKING:
    # Imported for real inside __init__ instead: `interp_engine.VLLMModel` pulls in
    # vLLM, and this module has to stay importable on hosts that only run the eager backend.
    from interp_engine import VLLMModel

_STOP_SEQUENCES: tuple[str, ...] = ("</explanation>",)


class VLLMVerbalizer:
    """CUDA verbalizer using the engine ``VLLMModel`` + vLLM ``prompt_embeds``.

    Drop-in for ``NLAClient``: same constructor kwargs (the sglang-only ones
    ``tp_size`` / ``kv_cache_dtype`` / ``cuda_graph_max_bs`` / ``enable_torch_compile``
    are accepted and surfaced as attributes for ``/health`` but not all are wired to
    vLLM). ``mem_fraction_static`` maps to vLLM ``gpu_memory_utilization`` -- note that
    this caps vLLM's ENTIRE footprint (weights + activations + CUDA graphs + KV), not
    the KV pool alone. ``max_model_len`` caps the context vLLM sizes that pool against.
    """

    def __init__(
        self,
        verbalizer_model_path: str,
        *,
        nla_config: NLAConfig | None = None,
        injection_scale_override: float | None = None,
        embed_device: str = "cpu",
        device: str | None = None,
        tp_size: int = 1,
        mem_fraction_static: float = 0.85,
        quantization: str | None = None,
        kv_cache_dtype: str | None = None,
        cuda_graph_max_bs: int | None = None,
        enable_torch_compile: bool = False,  # noqa: ARG002 - see note below (bundled with graphs)
        max_model_len: int | None = None,
    ):
        import os

        from interp_engine import VLLMModel

        local_path = resolve_checkpoint_path(verbalizer_model_path)
        self.device = device or "cuda"
        if not self.device.startswith("cuda"):
            raise ValueError(
                f"VLLMVerbalizer requires a CUDA device, got {self.device!r}. Use EagerVerbalizer on CPU/MPS."
            )

        self.tokenizer = _load_tokenizer(local_path)
        if nla_config is not None:
            self.cfg = nla_config
        else:
            self.cfg = load_nla_config(
                local_path,
                self.tokenizer,
                injection_scale_override=injection_scale_override,
            )

        # Embedding table for injection (CPU is fine; tiny).
        self.embed = load_embedding_only(local_path, dtype=torch.bfloat16).to(embed_device)
        self.embed_scale = resolve_embed_scale(local_path)
        assert self.embed.weight.shape[1] == self.cfg.d_model, (
            f"embedding d={self.embed.weight.shape[1]} != config d_model={self.cfg.d_model}."
        )

        # Engine-owned vLLM backend with the EmbedsPrompt path enabled. Unlike the
        # inference server's capture/steer backend (which MUST be enforce_eager so the
        # Python forward-hooks run), the verbalizer uses NO hooks, so we run with CUDA
        # graphs + inductor compile ON by default (enforce_eager=False) -- vLLM bundles
        # both under non-eager. This works with prompt_embeds (forced V1 runner) now that
        # the engine sets the spawn multiproc method + disables prefix caching.
        # Escape hatch: NLA_VERBALIZER_ENFORCE_EAGER=1 disables both (debug/compat).
        enforce_eager = os.environ.get("NLA_VERBALIZER_ENFORCE_EAGER", "").strip().lower() in (
            "1",
            "true",
        )
        extra: dict[str, Any] = {}
        if tp_size and tp_size > 1:
            extra["tensor_parallel_size"] = int(tp_size)
        if quantization:
            extra["quantization"] = quantization
        if kv_cache_dtype:
            # vLLM accepts "fp8"/"fp8_e4m3"/"fp8_e5m2" -- halves KV-pool bytes/token
            # (the memory win the larger verbalizers rely on).
            extra["kv_cache_dtype"] = kv_cache_dtype
        if cuda_graph_max_bs and not enforce_eager:
            # Match/extend vLLM's CUDA-graph capture batch sizes to the expected fan-out
            # (the sglang --cuda-graph-max-bs equivalent). vLLM's default already captures
            # up to 256; this lets you cap capture cost or extend past it.
            n = int(cuda_graph_max_bs)
            sizes = sorted({1, 2, 4} | set(range(8, n + 1, 8)) | {n})
            extra["compilation_config"] = {"cudagraph_capture_sizes": sizes}
        print(
            f"[VLLMVerbalizer] Starting VLLMModel for {verbalizer_model_path} "
            f"(device={self.device}, mem_fraction={mem_fraction_static}, "
            f"quantization={quantization or 'none'}, tp_size={tp_size}, "
            f"enforce_eager={enforce_eager}, cuda_graph_max_bs={cuda_graph_max_bs or 'default'}, "
            f"max_model_len={max_model_len or 'model default'})..."
        )
        # max_model_len is worth setting on any verbalizer whose base model advertises a
        # long context. vLLM refuses to start unless the KV pool can hold one request at
        # the full context (_check_enough_kv_cache_memory), so a 131k-context base like
        # Gemma 3 demands GiB of pool for a length the verbalizer never generates -- its
        # prompt is a fixed template and its output is capped by NLA_MAX_NEW_TOKENS_LIMIT.
        # sglang had no such precondition, which is why this only bites on the vLLM path.
        self.backend = VLLMModel(
            local_path,
            dtype="bfloat16",
            gpu_memory_utilization=mem_fraction_static,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            enable_extraction=False,
            enable_prompt_embeds=True,
            extra_vllm_kwargs=extra or None,
        )

        # Interface parity with NLAClient (read by /health).
        self.quantization = quantization
        self.kv_cache_dtype = kv_cache_dtype
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.enable_torch_compile = enable_torch_compile
        self.max_model_len = max_model_len

        print(
            f"[VLLMVerbalizer] ready: d_model={self.cfg.d_model} "
            f"inj_scale={self.cfg.injection_scale} embed_scale={self.embed_scale:.2f} "
            f"inj_char={self.cfg.injection_char!r}(id={self.cfg.injection_token_id}) "
            f"device={self.device}"
        )

    async def aload(self) -> None:
        """Start the vLLM engine now (warmup so the first request isn't slow).

        ``_ensure_engine`` is the expensive part (~1 min: EngineCore spawn, weight
        load, memory profiling, KV-pool sizing, CUDA-graph capture). The throwaway
        generation additionally warms the per-request path (input processor, sampler
        kernels) by exercising the same ``async_generate`` production callers use.
        """
        await self._require_backend()._ensure_engine()
        try:
            # A zero activation is safe: normalize_activation clamps the norm.
            await self.async_generate(
                np.zeros(self.cfg.d_model, dtype=np.float32),
                extract_explanation=False,
                temperature=0.0,
                max_new_tokens=1,
            )
        except Exception as e:
            print(f"[VLLMVerbalizer] warmup generation failed (engine is up): {e}")

    def shutdown(self):
        """Shut down the vLLM engine (tears down the EngineCore subprocess)."""
        backend = getattr(self, "backend", None)
        engine = getattr(backend, "engine", None) if backend is not None else None
        if engine is not None:
            with contextlib.suppress(Exception):
                engine.shutdown()
        self.backend = None

    def _require_backend(self) -> VLLMModel:
        """The backend, or a clear error if this verbalizer has been shut down.

        ``shutdown()`` drops the reference, so the attribute is Optional for anything that
        runs after it; using it then is a caller bug rather than something to serve.
        """
        if self.backend is None:
            raise RuntimeError("VLLMVerbalizer has been shut down")
        return self.backend

    # ─── Injection ────────────────────────────────────────────────────────────

    def _build_prompt_embeds(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        prompt: str | None,
    ) -> torch.Tensor:
        """Tokenize -> embed -> arch-scale -> inject -> ``[T, d]`` bf16 (vLLM prompt_embeds)."""
        v = torch.as_tensor(np.asarray(activation, dtype=np.float32))
        assert v.numel() == self.cfg.d_model, f"activation length {v.numel()} != d_model {self.cfg.d_model}"
        injected = build_injected_embeds(
            self.cfg, self.tokenizer, self.embed, self.embed_scale, v, prompt
        )  # [1, T, d] fp32 CPU
        return injected[0].to(torch.bfloat16).contiguous()  # [T, d]

    def _sampling_params(self, temperature: float, max_new_tokens: int):
        from vllm import SamplingParams

        return SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_new_tokens),
            stop=list(_STOP_SEQUENCES),
            # Keep the matched </explanation> in the output (parity with the sglang path's
            # no_stop_trim=True) so extraction finds the closing tag.
            include_stop_str_in_output=True,
            skip_special_tokens=False,
        )

    def _make_meta(self, text: str, comp_output: Any, prompt_len: int) -> dict:
        matched_close = "</explanation>" in text
        return {
            "finish_reason": {
                "type": "stop" if matched_close else (comp_output.finish_reason or "length"),
                "matched": "</explanation>" if matched_close else None,
            },
            "completion_tokens": len(comp_output.token_ids),
            "prompt_tokens": prompt_len,
        }

    # ─── Generation ─────────────────────────────────────────────────────────────

    async def async_generate(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        temperature: float = 1.0,
        max_new_tokens: int = 200,
        context: str | None = None,
    ) -> str:
        """Decode one activation vector (async — concurrency-safe: own vLLM request)."""
        embeds = self._build_prompt_embeds(activation, prompt)
        out = await self._require_backend().generate_from_embeds(
            embeds, self._sampling_params(temperature, max_new_tokens)
        )
        comp = out.outputs[0]
        result = {
            "text": comp.text,
            "meta_info": self._make_meta(comp.text, comp, embeds.shape[0]),
        }
        return self._extract_text(result, extract_explanation, context=context)

    async def async_generate_stream(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        temperature: float = 1.0,
        max_new_tokens: int = 200,
    ) -> AsyncGenerator[dict, None]:
        """Yield ``{"text": <cumulative>, "meta_info": {...}}`` as tokens decode."""
        embeds = self._build_prompt_embeds(activation, prompt)
        prompt_len = embeds.shape[0]
        gen = await self._require_backend().generate_from_embeds(
            embeds, self._sampling_params(temperature, max_new_tokens), stream=True
        )
        last = None
        async for out in gen:
            comp = out.outputs[0]
            last = comp
            yield {"text": comp.text, "meta_info": None}
        if last is not None:
            yield {
                "text": last.text,
                "meta_info": self._make_meta(last.text, last, prompt_len),
            }

    def generate(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        temperature: float = 1.0,
        max_new_tokens: int = 200,
        context: str | None = None,
    ) -> str:
        """Decode one activation vector (sync — for CLI / non-async contexts)."""
        return asyncio.run(
            self.async_generate(
                activation,
                prompt=prompt,
                extract_explanation=extract_explanation,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                context=context,
            )
        )

    def _extract_text(self, out: dict, extract_explanation: bool, *, context: str | None = None) -> str:
        return extract_verbalizer_text(
            self.tokenizer,
            out,
            extract_explanation,
            context=context,
            label="VLLMVerbalizer",
        )
