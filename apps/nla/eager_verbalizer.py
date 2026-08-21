"""Eager (HF transformers) NLA verbalizer backend for non-CUDA hosts.

The default verbalizer (`NLAClient`) runs an in-process `sgl.Engine`, which is
CUDA-only. This backend provides the same small interface the NLA server
depends on (`.cfg`, `.device`, `async_generate`, `async_generate_stream`,
`_extract_text`, `shutdown`, plus the sglang-only scalars the health endpoint
reads) using plain `transformers` `generate(inputs_embeds=...)`, so NLA can run
on MPS / CPU for local dev and testing.

It reuses the device-neutral injection + text-extraction helpers from
`nla_inference`; only generation is reimplemented. This is intended for
dev/test parity, not production throughput (no batching / KV reuse tricks).
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncGenerator, Iterable
from typing import Any

import numpy as np
import torch
from transformers import TextIteratorStreamer

from nla_inference import (
    NLAConfig,
    _load_tokenizer,
    build_injected_embeds,
    extract_verbalizer_text,
    load_causal_lm,
    load_embedding_only,
    load_nla_config,
    resolve_checkpoint_path,
    resolve_dtype_for_device,
    resolve_embed_scale,
)

_STOP_SEQUENCES: tuple[str, ...] = ("</explanation>",)


class EagerVerbalizer:
    """Drop-in, non-CUDA verbalizer using `transformers` eager generation.

    Matches the subset of the `NLAClient` interface used by `apps/nla/server.py`.
    sglang-only constructor kwargs (`tp_size`, `mem_fraction_static`,
    `quantization`, `kv_cache_dtype`, `cuda_graph_max_bs`, `enable_torch_compile`)
    are accepted and ignored so the two backends are interchangeable at the call
    site; the corresponding attributes are exposed (as None/False) for the health
    endpoint.
    """

    def __init__(
        self,
        verbalizer_model_path: str,
        *,
        nla_config: NLAConfig | None = None,
        injection_scale_override: float | None = None,
        embed_device: str = "cpu",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        tp_size: int = 1,  # noqa: ARG002 - ignored (sglang-only)
        mem_fraction_static: float = 0.85,  # noqa: ARG002 - ignored (sglang-only)
        quantization: str | None = None,  # noqa: ARG002 - ignored (sglang-only)
        kv_cache_dtype: str | None = None,  # noqa: ARG002 - ignored (sglang-only)
        cuda_graph_max_bs: int | None = None,  # noqa: ARG002 - ignored (sglang-only)
        enable_torch_compile: bool = False,  # noqa: ARG002 - ignored (sglang-only)
    ):
        local_path = resolve_checkpoint_path(verbalizer_model_path)
        self.device = device or "cpu"
        self.dtype = dtype if dtype is not None else resolve_dtype_for_device(self.device)

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

        print(f"[EagerVerbalizer] Loading full verbalizer {local_path} (device={self.device}, dtype={self.dtype})...")
        # Annotated rather than inferred because `shutdown()` sets it back to None; without
        # this every generation path below would have to re-prove the model is still loaded.
        self.model: Any = (
            load_causal_lm(
                local_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

        # Interface parity with NLAClient (read by the /health endpoint).
        self.quantization = None
        self.kv_cache_dtype = None
        self.cuda_graph_max_bs = None
        self.enable_torch_compile = False

        print(
            f"[EagerVerbalizer] ready: d_model={self.cfg.d_model} "
            f"inj_scale={self.cfg.injection_scale} embed_scale={self.embed_scale:.2f} "
            f"inj_char={self.cfg.injection_char!r}(id={self.cfg.injection_token_id}) "
            f"device={self.device} dtype={self.dtype}"
        )

    def shutdown(self):
        """Release the model (best effort)."""
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ─── Generation ───────────────────────────────────────────────────────────

    def _prepare(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        prompt: str | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build `(inputs_embeds, attention_mask)` on the model device."""
        v = torch.as_tensor(np.asarray(activation, dtype=np.float32))
        assert v.numel() == self.cfg.d_model, f"activation length {v.numel()} != d_model {self.cfg.d_model}"
        injected = build_injected_embeds(
            self.cfg, self.tokenizer, self.embed, self.embed_scale, v, prompt
        )  # [1, T, d] fp32 CPU
        inputs_embeds = injected.to(self.device, self.model.dtype)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=self.device)
        return inputs_embeds, attention_mask

    def _gen_kwargs(self, temperature: float, max_new_tokens: int) -> dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": pad_id,
            # Keep the </explanation> stop string in the output (parity with the
            # sglang path's no_stop_trim=True) so extraction can find the tag.
            "stop_strings": list(_STOP_SEQUENCES),
            "tokenizer": self.tokenizer,
        }
        if temperature and temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = float(temperature)
        else:
            kwargs["do_sample"] = False
        return kwargs

    def _decode(self, output_ids: torch.Tensor) -> str:
        # generate(inputs_embeds=...) for decoder-only returns ONLY the newly
        # generated token ids. skip_special_tokens=False mirrors the sglang path.
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=False)

    def _make_meta(self, text: str, n_new: int, n_prompt: int) -> dict:
        matched_close = "</explanation>" in text
        return {
            "finish_reason": {
                "type": "stop" if matched_close else "length",
                "matched": "</explanation>" if matched_close else None,
            },
            "completion_tokens": n_new,
            "prompt_tokens": n_prompt,
        }

    def _generate_sync(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None,
        temperature: float,
        max_new_tokens: int,
    ) -> dict:
        inputs_embeds, attention_mask = self._prepare(activation, prompt)
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                **self._gen_kwargs(temperature, max_new_tokens),
            )
        text = self._decode(output_ids)
        meta = self._make_meta(text, output_ids.shape[-1], inputs_embeds.shape[1])
        return {"text": text, "meta_info": meta}

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
        """Decode one activation vector (sync)."""
        out = self._generate_sync(
            activation,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return self._extract_text(out, extract_explanation, context=context)

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
        """Decode one activation vector (async — runs blocking generate off-loop)."""
        out = await asyncio.to_thread(
            self._generate_sync,
            activation,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        return self._extract_text(out, extract_explanation, context=context)

    async def async_generate_stream(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        temperature: float = 1.0,
        max_new_tokens: int = 200,
    ) -> AsyncGenerator[dict, None]:
        """Yield ``{"text": <cumulative>, "meta_info": {...}}`` as tokens decode."""
        inputs_embeds, attention_mask = self._prepare(activation, prompt)
        n_prompt = inputs_embeds.shape[1]
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=False)

        def _run():
            with torch.no_grad():
                self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    streamer=streamer,
                    **self._gen_kwargs(temperature, max_new_tokens),
                )

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        cumulative = ""
        loop = asyncio.get_running_loop()
        it = iter(streamer)
        while True:
            # None is a safe end-of-stream sentinel: the streamer only yields str.
            piece = await loop.run_in_executor(None, lambda: next(it, None))
            if piece is None:
                break
            cumulative += piece
            yield {"text": cumulative, "meta_info": None}

        await asyncio.to_thread(thread.join)
        yield {
            "text": cumulative,
            "meta_info": self._make_meta(cumulative, -1, n_prompt),
        }

    # ─── Shared text extraction (server calls this directly) ───────────────────

    def _extract_text(
        self,
        out: dict,
        extract_explanation: bool,
        *,
        context: str | None = None,
    ) -> str:
        return extract_verbalizer_text(
            self.tokenizer,
            out,
            extract_explanation,
            context=context,
            label="EagerVerbalizer",
        )
