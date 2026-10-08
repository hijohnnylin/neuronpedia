"""The NLA verbalizer over interp-engine's model protocol: one class for every backend.

Concept injection splices a (normalized) activation vector into the prompt's embedding sequence
where the injection token would have been, then generates. The splice is device-neutral math in
``nla_inference``; the generation is ``InterpModel.generate_steps_from_embeds``, which every engine
backend implements. So there is one verbalizer, and which library runs it -- vLLM on CUDA,
transformers on MPS / CPU, MLX on a Mac -- is decided at ``load_model`` and nowhere else.

The interface the server depends on: ``.cfg`` / ``.tokenizer`` / ``.device`` / ``.embed`` /
``.embed_scale``, the scalars ``/health`` reports, ``aload`` / ``async_generate`` /
``async_generate_stream`` / ``generate`` / ``_extract_text`` / ``shutdown``. The legacy sglang
``NLAClient`` exposes the same set.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncGenerator, Iterable
from typing import Any

import numpy as np
import torch
from interp_engine import (
    InterpModel,
    SamplingSettings,
    load_model,
    read_recommended_sampling,
    resolve_sampling,
    sync_model,
)

from nla_inference import (
    NLAConfig,
    _load_tokenizer,
    build_injected_embeds,
    extract_verbalizer_text,
    load_embedding_only,
    load_nla_config,
    resolve_checkpoint_path,
    resolve_dtype_for_device,
    resolve_embed_scale,
)

_STOP_SEQUENCES: tuple[str, ...] = ("</explanation>",)

#: What ``backend="auto"`` resolves to, by device: vLLM where there is a CUDA card, eager
#: transformers everywhere else. MLX is never chosen by inference -- name it.
BACKENDS: tuple[str, ...] = ("auto", "vllm", "eager", "mlx")


def resolve_backend(backend: str, device: str) -> str:
    """The engine backend name for ``backend`` on ``device``.

    ``"vllm"`` becomes the engine's graph-replaying ``vllm-generate``: the verbalizer installs no
    hooks, so CUDA graphs and inductor stay on. ``NLA_VERBALIZER_ENFORCE_EAGER=1`` keeps the hooked
    ``vllm`` engine instead, which is vLLM's ``enforce_eager`` path, for debugging a graph problem.
    """
    if backend not in BACKENDS:
        raise ValueError(f"NLA_VERBALIZER_BACKEND={backend!r} is not one of {', '.join(BACKENDS)} (or 'sglang')")
    if backend == "auto":
        backend = "vllm" if device.startswith("cuda") else "eager"
    if backend == "vllm":
        if not device.startswith("cuda"):
            raise ValueError(f"the vllm verbalizer backend needs a CUDA device, got {device!r}; use eager or mlx")
        enforce_eager = os.environ.get("NLA_VERBALIZER_ENFORCE_EAGER", "").strip().lower() in ("1", "true")
        return "vllm" if enforce_eager else "vllm-generate"
    return backend


class Verbalizer:
    """Decode activation vectors into explanations through any engine backend.

    The sglang-era constructor kwargs are kept so the server builds this and ``NLAClient`` the same
    way. The vLLM ones map onto the engine's vLLM constructor (``mem_fraction_static`` is
    ``gpu_memory_utilization``, which caps vLLM's whole footprint, not the KV pool alone) and are
    ignored by the other backends; ``enable_torch_compile`` is accepted for ``/health`` only, since
    vLLM bundles compile into its non-eager path.
    """

    def __init__(
        self,
        verbalizer_model_path: str,
        *,
        nla_config: NLAConfig | None = None,
        injection_scale_override: float | None = None,
        embed_device: str = "cpu",
        device: str | None = None,
        backend: str = "auto",
        dtype: torch.dtype | None = None,
        tp_size: int = 1,
        mem_fraction_static: float = 0.85,
        quantization: str | None = None,
        kv_cache_dtype: str | None = None,
        cuda_graph_max_bs: int | None = None,
        enable_torch_compile: bool = False,
        max_model_len: int | None = None,
    ):
        local_path = resolve_checkpoint_path(verbalizer_model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.backend = resolve_backend(backend, self.device)
        self.dtype = dtype if dtype is not None else resolve_dtype_for_device(self.device)

        # Before the engine reads the same files: this repairs a malformed tokenizer config on disk.
        self.tokenizer = _load_tokenizer(local_path)
        self.recommended_sampling = read_recommended_sampling(local_path)
        if nla_config is not None:
            self.cfg = nla_config
        else:
            self.cfg = load_nla_config(local_path, self.tokenizer, injection_scale_override=injection_scale_override)

        # Embedding table for injection (CPU is fine; tiny).
        self.embed = load_embedding_only(local_path, dtype=torch.bfloat16).to(embed_device)
        self.embed_scale = resolve_embed_scale(local_path)
        assert self.embed.weight.shape[1] == self.cfg.d_model, (
            f"embedding d={self.embed.weight.shape[1]} != config d_model={self.cfg.d_model}."
        )

        print(
            f"[Verbalizer] Loading {verbalizer_model_path} on backend={self.backend} "
            f"(device={self.device}, dtype={self.dtype}, quantization={quantization or 'none'}, "
            f"kv_cache_dtype={kv_cache_dtype or 'default'}, tp_size={tp_size}, "
            f"mem_fraction={mem_fraction_static}, cuda_graph_max_bs={cuda_graph_max_bs or 'default'}, "
            f"max_model_len={max_model_len or 'model default'})..."
        )
        model = load_model(
            local_path,
            backend=self.backend,
            trust_remote_code=True,
            **self._backend_kwargs(
                tp_size=tp_size,
                mem_fraction_static=mem_fraction_static,
                quantization=quantization,
                kv_cache_dtype=kv_cache_dtype,
                cuda_graph_max_bs=cuda_graph_max_bs,
                max_model_len=max_model_len,
            ),
        )
        assert model.d_model == self.cfg.d_model, f"model d_model={model.d_model} != config d_model={self.cfg.d_model}"
        self.model: InterpModel | None = model

        # Interface parity with NLAClient (read by /health).
        self.quantization = quantization if self.backend.startswith("vllm") else None
        self.kv_cache_dtype = kv_cache_dtype if self.backend.startswith("vllm") else None
        self.cuda_graph_max_bs = cuda_graph_max_bs if self.backend == "vllm-generate" else None
        self.enable_torch_compile = enable_torch_compile
        self.max_model_len = max_model_len

        print(
            f"[Verbalizer] ready: backend={self.backend} d_model={self.cfg.d_model} "
            f"inj_scale={self.cfg.injection_scale} embed_scale={self.embed_scale:.2f} "
            f"inj_char={self.cfg.injection_char!r}(id={self.cfg.injection_token_id}) device={self.device}"
        )

    def _backend_kwargs(
        self,
        *,
        tp_size: int,
        mem_fraction_static: float,
        quantization: str | None,
        kv_cache_dtype: str | None,
        cuda_graph_max_bs: int | None,
        max_model_len: int | None,
    ) -> dict[str, Any]:
        """The ``load_model`` arguments for the resolved backend."""
        if self.backend.startswith("vllm"):
            # `quantization` carries vLLM's own scheme names (awq_marlin, gptq, ...), so it goes to
            # vLLM as is rather than through the engine's short table.
            extra: dict[str, Any] = {}
            if quantization:
                extra["quantization"] = quantization
            if kv_cache_dtype:
                extra["kv_cache_dtype"] = kv_cache_dtype
            if cuda_graph_max_bs and self.backend == "vllm-generate":
                # Match vLLM's CUDA-graph capture sizes to the expected fan-out. Its default already
                # captures up to 256; this caps the capture cost or extends past it.
                n = int(cuda_graph_max_bs)
                sizes = sorted({1, 2, 4} | set(range(8, n + 1, 8)) | {n})
                extra["compilation_config"] = {"cudagraph_capture_sizes": sizes}
            # max_model_len is worth setting on any verbalizer whose base model advertises a long
            # context. vLLM refuses to start unless the KV pool can hold one request at the full
            # context, so a 131k-context base like Gemma 3 demands GiB of pool for a length the
            # verbalizer never generates: its prompt is a fixed template and its output is capped
            # by NLA_MAX_NEW_TOKENS_LIMIT.
            return {
                "dtype": "bfloat16",
                "num_gpus": int(tp_size) if tp_size and tp_size > 1 else 1,
                "gpu_memory_utilization": mem_fraction_static,
                "max_model_len": max_model_len,
                "enable_extraction": False,
                "enable_prompt_embeds": True,
                "extra_vllm_kwargs": extra or None,
            }
        if self.backend == "mlx":
            return {}
        # Eager never reads attention, so the fused kernel is fine and faster than the engine's
        # default of eager attention.
        return {
            "device": self.device,
            "dtype": str(self.dtype).removeprefix("torch."),
            "attn_implementation": "sdpa",
        }

    async def aload(self) -> None:
        """Pay the deferred load now, then run one generation so the first request is not slow.

        On vLLM the first part is the expensive one (EngineCore spawn, weight load, memory
        profiling, KV-pool sizing, CUDA-graph capture); the generation warms the per-request path.
        """
        await self._require_model().warmup()
        try:
            # A zero activation is safe: normalize_activation clamps the norm.
            await self.async_generate(
                np.zeros(self.cfg.d_model, dtype=np.float32),
                extract_explanation=False,
                sampling=self.sampling_settings(temperature=0.0),
                max_new_tokens=1,
            )
        except Exception as e:
            print(f"[Verbalizer] warmup generation failed (model is loaded): {e}")

    def shutdown(self):
        """Release the model's device memory (and vLLM's worker process)."""
        model = getattr(self, "model", None)
        if model is not None:
            sync_model(model).shutdown()
        self.model = None

    def _require_model(self) -> InterpModel:
        if self.model is None:
            raise RuntimeError("Verbalizer has been shut down")
        return self.model

    # ─── Injection ────────────────────────────────────────────────────────────

    def _build_prompt_embeds(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        prompt: str | None,
    ) -> torch.Tensor:
        """Tokenize -> embed -> arch-scale -> inject -> ``[T, d]`` fp32 on CPU.

        The rows carry the family's embedding scale, which is what the engine's
        ``generate_steps_from_embeds`` takes; each backend casts to its own dtype.
        """
        v = torch.as_tensor(np.asarray(activation, dtype=np.float32))
        assert v.numel() == self.cfg.d_model, f"activation length {v.numel()} != d_model {self.cfg.d_model}"
        injected = build_injected_embeds(self.cfg, self.tokenizer, self.embed, self.embed_scale, v, prompt)
        return injected[0].contiguous()

    # ─── Generation ─────────────────────────────────────────────────────────────

    def sampling_settings(
        self,
        *,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        presence_penalty: float | None = None,
    ) -> SamplingSettings:
        """What a generation with these knobs runs with: the caller's value, else the verbalizer
        checkpoint's ``generation_config.json``, else neutral -- the inference server's rule."""
        return resolve_sampling(
            self.recommended_sampling,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            presence_penalty=presence_penalty,
        )

    async def _stream(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None,
        sampling: SamplingSettings | None,
        max_new_tokens: int,
    ) -> AsyncGenerator[tuple[str, dict | None], None]:
        """Yield ``(cumulative_text, meta)`` per token; ``meta`` is set on the last one only.

        The text is decoded from all ids so far rather than joined from per-token strings, so a
        character split across tokens reads whole on every backend. The stop string is matched here,
        on the cumulative text, so the rule is one rule and not one per backend; the matched tag
        stays in the output, where extraction looks for it.
        """
        embeds = self._build_prompt_embeds(activation, prompt)
        prompt_len = int(embeds.shape[0])
        ids: list[int] = []
        text = ""
        finish = "length"
        settings = sampling or self.sampling_settings()
        steps = self._require_model().generate_steps_from_embeds(
            embeds,
            max_tokens=int(max_new_tokens),
            temperature=settings.temperature,
            top_k=settings.top_k,
            top_p=settings.top_p,
            presence_penalty=settings.presence_penalty,
        )
        eos_id = self.tokenizer.eos_token_id
        try:
            async for step in steps:
                ids.append(step.token_id)
                text = self.tokenizer.decode(ids, skip_special_tokens=False)
                if step.token_id == eos_id or any(stop in text for stop in _STOP_SEQUENCES):
                    finish = "stop"
                    break
                yield text, None
        finally:
            # Leaving early on a stop string must end the request, not just this loop. The
            # protocol types the stream as an iterator; every backend's is a generator.
            aclose = getattr(steps, "aclose", None)
            if aclose is not None:
                await aclose()
        yield text, self._make_meta(text, finish, len(ids), prompt_len)

    def _make_meta(self, text: str, finish: str, n_new: int, n_prompt: int) -> dict:
        matched_close = "</explanation>" in text
        return {
            "finish_reason": {
                "type": "stop" if matched_close else finish,
                "matched": "</explanation>" if matched_close else None,
            },
            "completion_tokens": n_new,
            "prompt_tokens": n_prompt,
        }

    async def async_generate(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        sampling: SamplingSettings | None = None,
        max_new_tokens: int = 200,
        context: str | None = None,
    ) -> str:
        """Decode one activation vector (async; each call is its own request on every backend)."""
        result: dict = {"text": "", "meta_info": None}
        async for text, meta in self._stream(
            activation, prompt=prompt, sampling=sampling, max_new_tokens=max_new_tokens
        ):
            result = {"text": text, "meta_info": meta}
        return self._extract_text(result, extract_explanation, context=context)

    async def async_generate_stream(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        sampling: SamplingSettings | None = None,
        max_new_tokens: int = 200,
    ) -> AsyncGenerator[dict, None]:
        """Yield ``{"text": <cumulative>, "meta_info": ...}`` as tokens decode; meta on the last only."""
        async for text, meta in self._stream(
            activation, prompt=prompt, sampling=sampling, max_new_tokens=max_new_tokens
        ):
            yield {"text": text, "meta_info": meta}

    def generate(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        extract_explanation: bool = True,
        sampling: SamplingSettings | None = None,
        max_new_tokens: int = 200,
        context: str | None = None,
    ) -> str:
        """Decode one activation vector (sync -- for CLI / non-async contexts)."""
        return asyncio.run(
            self.async_generate(
                activation,
                prompt=prompt,
                extract_explanation=extract_explanation,
                sampling=sampling,
                max_new_tokens=max_new_tokens,
                context=context,
            )
        )

    # ─── Shared text extraction (server calls this directly) ───────────────────

    def _extract_text(self, out: dict, extract_explanation: bool, *, context: str | None = None) -> str:
        return extract_verbalizer_text(self.tokenizer, out, extract_explanation, context=context, label="Verbalizer")
