"""NLA verbalizer inference via SGLang Engine (in-process) — no separate server needed.

An NLA (Natural Language Autoencoder) pair is two fine-tuned LMs that together
map activation vectors to natural language and back:

  VERBALIZER     : hidden-state vector  ->  text
  RECONSTRUCTOR  : text  ->  hidden-state vector

This file contains both halves:
  NLAClient         — verbalizer inference via sgl.Engine with input_embeds
  NLAReconstructor  — load reconstructor + reconstruct + score (optional, pure torch)

Checkpoints accept HuggingFace hub paths (e.g. kitft/nla-qwen2.5-7b-actor-step2000)
or local directories. NLA parameters can be loaded from nla_meta.yaml sidecar
or passed explicitly as arguments.
"""

from __future__ import annotations

import gc
import json
import math
import re
import threading
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sglang as sgl
import torch
import yaml
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ─── Constants ──────────────────────────────────────────────────────────────

# Tolerates a missing closing tag (e.g. when generation hits max_new_tokens
# before emitting </explanation>): the closing-tag group is optional, and the
# body is captured non-greedily up to either the closing tag or end-of-text.
EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*(?:</explanation>|\Z)", re.DOTALL)
INJECT_PLACEHOLDER = "<INJECT>"
_EMBED_KEY_SUFFIXES = ("embed_tokens.weight", "wte.weight", "word_embeddings.weight")


def gpu_supports_fp8_native(device_idx: int = 0) -> bool:
    """Whether the GPU at ``device_idx`` supports sglang's FP8 quantization path.

    Sglang's Triton FP8 kernels use the ``fp8e4nv`` dtype (NVIDIA's
    Hopper-style E4M3 FP8). Compute-capability requirements:

      - **Ampere** (sm_80, sm_86: A100, A40, RTX 30xx) — no FP8 hardware at all
      - **Ada Lovelace** (sm_89: RTX 4090, L40, L40S, RTX 6000 Ada) — has FP8
        hardware but only ``fp8e4b15``/``fp8e5`` variants; ``fp8e4nv`` doesn't
        compile (Triton raises ``type fp8e4nv not supported in this architecture``)
      - **Hopper** (sm_90: H100, H200) — fully supported
      - **Blackwell** (sm_100/sm_103: B100/B200/GB200; sm_120: RTX Pro 6000
        Blackwell, RTX 5090) — fully supported

    Returns False on CPU/MPS/no-CUDA setups too. Used to gate
    ``quantization="fp8"`` for the verbalizer (which goes through sglang),
    NOT torchao's weight-only FP8 path used by source/reconstructor — that
    one works on Ampere and up (storage-only, no FP8 compute kernels).
    """
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(device_idx)
    except Exception:
        return False
    return major >= 9


def _parse_cuda_index(device: str) -> int | None:
    """Return the integer GPU index for "cuda:N", or 0 for bare "cuda".

    Returns None for non-CUDA devices ("cpu", "mps", etc.).
    """
    if not isinstance(device, str):
        return None
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        try:
            return int(device.split(":", 1)[1])
        except ValueError:
            return None
    return None


# Default verbalizer prompt template (Qwen NLA verbalizer)
DEFAULT_VERBALIZER_PROMPT_TEMPLATE = """\
You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{injection_char}</concept>

Please provide an explanation."""

# Default reconstructor prompt template
DEFAULT_RECONSTRUCTOR_PROMPT_TEMPLATE = (
    "Summary of the following text: <text>{explanation}</text> <summary>"
)

# This file is OSS-standalone so cannot import arch_adapters; the registry is
# small and the drift hazard of a prefix-match (a hypothetical "phi-gemma-moe"
# would spuriously match .startswith("gemma")) is worse than a duplicated set.
_SCALED_EMBED_MODEL_TYPES = frozenset(
    {
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "t5",
    }
)

# ─── Sidecar config ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class NLAConfig:
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    verbalizer_prompt_template: str
    injection_scale: float


def resolve_checkpoint_path(model_path: str) -> str:
    """Return a local path — downloads from HF Hub if needed.

    Accepts:
      - Local directory path (returned as-is)
      - HF hub ID like 'kitft/nla-qwen2.5-7b-actor-step2000'
    """
    if Path(model_path).is_dir():
        return model_path
    # Treat as HF hub ID — snapshot_download caches locally.
    # Skip Meta's `original/consolidated.*.pth` shards: these are duplicates of
    # the HF-format `*.safetensors` weights we actually load, but ~17.6 GB each
    # (e.g. 8 × 17.6 GB = ~140 GB extra for a Llama-3.x-70B repo).
    print(f"[NLA] Downloading {model_path} from HuggingFace Hub...")
    return snapshot_download(
        model_path,
        ignore_patterns=["original/*", "consolidated.*"],
    )


def make_nla_config(
    *,
    d_model: int,
    injection_scale: float,
    injection_char: str = "㈎",
    injection_token_id: int = 149705,
    injection_left_neighbor_id: int = 29,
    injection_right_neighbor_id: int = 522,
    verbalizer_prompt_template: str = DEFAULT_VERBALIZER_PROMPT_TEMPLATE,
) -> NLAConfig:
    """Build NLAConfig from explicit parameters (no sidecar file needed)."""
    return NLAConfig(
        d_model=d_model,
        injection_char=injection_char,
        injection_token_id=injection_token_id,
        injection_left_neighbor_id=injection_left_neighbor_id,
        injection_right_neighbor_id=injection_right_neighbor_id,
        verbalizer_prompt_template=verbalizer_prompt_template,
        injection_scale=injection_scale,
    )


def load_nla_config(
    checkpoint_dir: str | Path,
    tokenizer: Any,
    injection_scale_override: float | None = None,
) -> NLAConfig:
    """Parse {checkpoint_dir}/nla_meta.yaml and assert against live tokenizer."""
    meta_path = Path(checkpoint_dir) / "nla_meta.yaml"
    assert meta_path.exists(), (
        f"no nla_meta.yaml at {checkpoint_dir!r}. Not an NLA checkpoint — "
        f"use make_nla_config() with explicit params instead."
    )
    meta = yaml.safe_load(meta_path.read_text())

    kind = meta["kind"]
    assert kind in ("nla_model", "nla_dataset"), f"unknown sidecar kind: {kind!r}"
    d_model = meta["d_model"] if kind == "nla_model" else meta["extraction"]["d_model"]

    inj_scale = meta.get("extraction", {}).get("injection_scale")
    if inj_scale is None:
        inj_scale = injection_scale_override
    assert inj_scale is not None, (
        f"nla_meta.yaml at {checkpoint_dir!r} has no extraction.injection_scale. "
        f"Pass injection_scale_override explicitly."
    )

    t = meta["tokens"]
    # Back-compat across sidecar generations:
    #   current canonical: "verbalizer"
    #   legacy:            "actor"
    #   schema_version 2 (kitft): keyed by role ("av"/"ar"); the AV (activation
    #     verbalizer / explainer) template lives under "av".
    prompt_templates = meta["prompt_templates"]
    role = meta.get("role")
    verbalizer_template = (
        prompt_templates.get("verbalizer")
        or prompt_templates.get("actor")
        or (prompt_templates.get(role) if role in ("av",) else None)
    )
    assert verbalizer_template is not None, (
        f"sidecar prompt_templates has no entry for 'verbalizer', 'actor', or "
        f"role={role!r}; got keys {sorted(prompt_templates)!r}."
    )
    cfg = NLAConfig(
        d_model=d_model,
        injection_char=t["injection_char"],
        injection_token_id=t["injection_token_id"],
        injection_left_neighbor_id=t["injection_left_neighbor_id"],
        injection_right_neighbor_id=t["injection_right_neighbor_id"],
        verbalizer_prompt_template=verbalizer_template,
        injection_scale=float(inj_scale),
    )

    _validate_config_against_tokenizer(cfg, tokenizer)
    return cfg


def _tokenize_chat_with_merges(tokenizer: Any, content: str) -> list[int]:
    """Tokenize a single-turn user message via the model's chat template,
    preserving BPE merges for multi-byte tokens.

    Workaround for transformers 5.x: `apply_chat_template(..., tokenize=True)`
    byte-falls-back on multi-byte BPE-merged tokens (e.g. the NLA injection
    char `㎡` U+33A1 → token 105565 for Llama-3.3) instead of consulting
    the merge table — even though `tokenizer.encode()` on the same string
    still merges correctly. We render the template to a string then re-encode.

    `add_special_tokens=False` because the rendered string already contains
    `<|begin_of_text|>` and other markers as literals; the tokenizer
    recognizes them as registered added tokens during encode() and emits
    them as their canonical single-token IDs.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def _validate_config_against_tokenizer(cfg: NLAConfig, tokenizer: Any) -> None:
    """Assert injection char/neighbors match the live tokenizer."""
    live_inj = tokenizer.encode(cfg.injection_char, add_special_tokens=False)
    assert live_inj == [cfg.injection_token_id], (
        f"tokenizer drift: {cfg.injection_char!r} -> {live_inj}, config says "
        f"[{cfg.injection_token_id}]."
    )
    assert live_inj[0] != tokenizer.unk_token_id, f"{cfg.injection_char!r} maps to UNK"

    content = cfg.verbalizer_prompt_template.format(injection_char=cfg.injection_char)
    ids = _tokenize_chat_with_merges(tokenizer, content)
    matches = [i for i, tok in enumerate(ids) if tok == cfg.injection_token_id]
    assert len(matches) == 1, (
        f"injection token appears {len(matches)}x in canonical prompt (expected 1)."
    )
    p = matches[0]
    assert 0 < p < len(ids) - 1
    assert ids[p - 1] == cfg.injection_left_neighbor_id, (
        f"left neighbor drift: {ids[p - 1]} vs config {cfg.injection_left_neighbor_id}"
    )
    assert ids[p + 1] == cfg.injection_right_neighbor_id, (
        f"right neighbor drift: {ids[p + 1]} vs config {cfg.injection_right_neighbor_id}"
    )


# ─── Embedding table ────────────────────────────────────────────────────────


def load_embedding_only(
    checkpoint_dir: str | Path,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Embedding:
    """Load ONLY the input embedding weight tensor from safetensors."""
    root = Path(checkpoint_dir)

    def _find_key(keys: list[str], where: str) -> str:
        m = [k for k in keys if k.endswith(_EMBED_KEY_SUFFIXES)]
        assert len(m) == 1, (
            f"expected exactly one input-embedding key in {where}, got {m!r}"
        )
        return m[0]

    index_path = root / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        key = _find_key(list(weight_map), str(index_path))
        shard = root / weight_map[key]
    else:
        shard = root / "model.safetensors"
        assert shard.exists(), f"no model.safetensors or .index.json at {root!r}"
        with safe_open(str(shard), framework="pt") as f:
            key = _find_key(list(f.keys()), str(shard))

    with safe_open(str(shard), framework="pt") as f:
        weight = f.get_tensor(key).to(dtype)

    vocab, d = weight.shape
    embed = torch.nn.Embedding(vocab, d, _weight=weight)
    embed.requires_grad_(False)
    embed.eval()
    return embed


def resolve_embed_scale(checkpoint_dir: str | Path) -> float:
    """1.0 for Qwen/Llama/Mistral; sqrt(hidden_size) for Gemma-3/T5"""
    config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
    text_cfg = getattr(config, "text_config", config)
    model_type = getattr(text_cfg, "model_type", "") or ""
    if model_type in _SCALED_EMBED_MODEL_TYPES:
        return math.sqrt(text_cfg.hidden_size)
    return 1.0


# ─── Pure injection math ────────────────────────────────────────────────────


def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    """Rescale to target_scale L2-norm. Zeros stay zero. Norm in fp32."""
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    vectors: torch.Tensor,
    inj_id: int,
    left_id: int,
    right_id: int,
) -> torch.Tensor:
    """Overwrite embedding rows at valid injection positions. Clones first."""
    seq_len = input_ids.shape[-1]
    assert input_ids.shape == embeddings.shape[:-1]
    assert vectors.ndim == 2 and vectors.shape[1] == embeddings.shape[-1]
    out = embeddings.clone()
    vectors = vectors.to(out.device, out.dtype)
    vec_idx = 0
    for b, p in (input_ids == inj_id).nonzero().tolist():
        if p == 0 or p == seq_len - 1:
            continue
        if input_ids[b, p - 1] != left_id or input_ids[b, p + 1] != right_id:
            continue
        out[b, p] = vectors[vec_idx]
        vec_idx += 1
    assert vec_idx == vectors.shape[0], (
        f"found {vec_idx} injection sites with correct neighbors, expected "
        f"{vectors.shape[0]}."
    )
    return out


# ─── Client ─────────────────────────────────────────────────────────────────


class NLAClient:
    """NLA verbalizer client using an in-process sgl.Engine (no separate server).

    Uses engine.async_generate() to avoid event-loop conflicts when embedded
    in a FastAPI/uvicorn server (uvicorn runs uvloop; sgl.Engine's sync
    generate() tries to manage async internally via ZMQ, causing conflicts).
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
        enable_torch_compile: bool = False,
    ):
        """
        verbalizer_model_path: HF hub ID (e.g. 'kitft/nla-qwen2.5-7b-actor-step2000')
                          or local directory path.
        nla_config:       Explicit NLAConfig. If None, loads from nla_meta.yaml
                          in the checkpoint dir.
        injection_scale_override: Override sidecar injection_scale (only used
                          when loading from nla_meta.yaml).
        embed_device:     Device for embedding lookup (CPU is fine).
        device:           Verbalizer placement. ``"cuda"`` (=cuda:0) or
                          ``"cuda:N"`` selects the base GPU passed to sglang as
                          ``base_gpu_id``. With ``tp_size>1`` sglang uses GPUs
                          ``[N, N+1, ..., N+tp_size-1]``. ``None`` defers to
                          sglang's default (cuda:0). CPU/MPS unsupported by
                          sglang and not allowed here.
        tp_size:          Tensor parallelism for sgl.Engine.
        mem_fraction_static: GPU memory fraction for KV cache.
        quantization:     sglang quantization mode (e.g. "fp8", "awq",
                          "awq_marlin", "gptq", "gptq_marlin"). None loads in
                          the checkpoint's native dtype (bf16). "fp8" is
                          weight-only on Ampere (memory win, no compute speedup)
                          and full FP8 on Hopper.
        kv_cache_dtype:   sglang KV-cache dtype, e.g. ``"fp8_e5m2"`` or
                          ``"fp8_e4m3"``. None leaves sglang's default (matches
                          model dtype). FP8 KV halves KV-pool bytes-per-token,
                          freeing ~1.5–3 GB on a 27B FP8 verbalizer at batch 32
                          — repurpose into more ``max_concurrent`` or just
                          headroom. Native compute on Hopper/Blackwell; quality
                          cost on a generative head is negligible.
        cuda_graph_max_bs: Max batch size for sglang CUDA-graph capture. None
                          uses sglang's default (often 8, sometimes 32). Bumping
                          to match your peak fan-out (e.g. ``NLA_MAX_CONCURRENT``)
                          keeps decode on the captured fast path instead of
                          falling into eager-mode kernel launches; typically
                          5–15% decode speedup at high concurrency.
        enable_torch_compile: If True, asks sglang to ``torch.compile`` the
                          decode kernels. ~10–20% extra on decode at the cost
                          of a one-time **1–3 minute** warmup at engine boot
                          (longer for bigger models — 27B is ~2 min, 7B is
                          ~45 s). Set ``TORCHINDUCTOR_CACHE_DIR`` to a
                          persistent path to amortize the cost across
                          restarts. Free at runtime; no steady-state VRAM
                          cost.
        """
        # Resolve HF hub path -> local dir
        local_path = resolve_checkpoint_path(verbalizer_model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True
        )

        # Load or use provided NLA config
        if nla_config is not None:
            self.cfg = nla_config
            _validate_config_against_tokenizer(self.cfg, self.tokenizer)
        else:
            self.cfg = load_nla_config(
                local_path,
                self.tokenizer,
                injection_scale_override=injection_scale_override,
            )

        # Load embedding table for injection (lightweight, CPU is fine)
        self.embed = load_embedding_only(local_path, dtype=torch.bfloat16).to(
            embed_device
        )
        self.embed_scale = resolve_embed_scale(local_path)

        assert self.embed.weight.shape[1] == self.cfg.d_model, (
            f"embedding d={self.embed.weight.shape[1]} != config "
            f"d_model={self.cfg.d_model}."
        )

        # Resolve the verbalizer's GPU index for sglang's `base_gpu_id`. None
        # leaves sglang at its default (cuda:0). Non-CUDA strings are rejected
        # — sgl.Engine has no CPU/MPS path.
        base_gpu_id: int | None = None
        if device is not None:
            base_gpu_id = _parse_cuda_index(device)
            if base_gpu_id is None:
                raise ValueError(
                    f"NLAClient device={device!r} is not a CUDA device. "
                    f"sgl.Engine requires CUDA; pass 'cuda', 'cuda:0', "
                    f"'cuda:1', etc."
                )
        self.device = (
            f"cuda:{base_gpu_id}" if base_gpu_id is not None else device or "cuda"
        )

        # Auto-fallback: if FP8 was requested but the verbalizer's GPU compute
        # capability doesn't support sglang's fp8e4nv kernels (Ampere/Ada),
        # warn and run the verbalizer in bf16 instead of crashing at engine
        # init. Check the actual target GPU (matters on heterogeneous boxes).
        if quantization == "fp8" and not gpu_supports_fp8_native(base_gpu_id or 0):
            cap_str = "n/a"
            if torch.cuda.is_available():
                try:
                    cap_str = ".".join(
                        str(x)
                        for x in torch.cuda.get_device_capability(base_gpu_id or 0)
                    )
                except Exception:
                    pass
            print(
                f"[NLAClient] WARNING: quantization='fp8' was requested but "
                f"GPU compute capability {cap_str} on cuda:{base_gpu_id or 0} "
                f"does not support sglang's fp8e4nv kernel (Hopper sm_90+ "
                f"required). Falling back to bf16 verbalizer (no "
                f"quantization). To use FP8, run on H100, H200, RTX Pro 6000 "
                f"Blackwell, B100/B200, or similar."
            )
            quantization = None

        # Launch sgl.Engine in-process — disable radix cache (REQUIRED for
        # input_embeds: radix cache keys on token IDs, which we don't have).
        print(
            f"[NLAClient] Starting sgl.Engine for {verbalizer_model_path} "
            f"(quantization={quantization or 'none'}, "
            f"kv_cache_dtype={kv_cache_dtype or 'default'}, "
            f"cuda_graph_max_bs={cuda_graph_max_bs or 'default'}, "
            f"torch_compile={enable_torch_compile}, "
            f"base_gpu_id={base_gpu_id if base_gpu_id is not None else 'default'}, "
            f"tp_size={tp_size})..."
        )
        engine_kwargs: dict[str, Any] = dict(
            model_path=local_path,
            tp_size=tp_size,
            disable_radix_cache=True,
            mem_fraction_static=mem_fraction_static,
            trust_remote_code=True,
            dtype="bfloat16",
        )
        if quantization is not None:
            engine_kwargs["quantization"] = quantization
        if base_gpu_id is not None:
            engine_kwargs["base_gpu_id"] = base_gpu_id
        if kv_cache_dtype is not None:
            engine_kwargs["kv_cache_dtype"] = kv_cache_dtype
        if cuda_graph_max_bs is not None:
            engine_kwargs["cuda_graph_max_bs"] = cuda_graph_max_bs
        if enable_torch_compile:
            engine_kwargs["enable_torch_compile"] = True
        self.engine = sgl.Engine(**engine_kwargs)

        self.quantization = quantization
        self.kv_cache_dtype = kv_cache_dtype
        self.cuda_graph_max_bs = cuda_graph_max_bs
        self.enable_torch_compile = enable_torch_compile

        print(
            f"[NLAClient] ready: d_model={self.cfg.d_model} "
            f"inj_scale={self.cfg.injection_scale} embed_scale={self.embed_scale:.2f} "
            f"inj_char={self.cfg.injection_char!r}(id={self.cfg.injection_token_id}) "
            f"quantization={quantization or 'none'} "
            f"kv_cache_dtype={kv_cache_dtype or 'default'} "
            f"cuda_graph_max_bs={cuda_graph_max_bs or 'default'} "
            f"torch_compile={enable_torch_compile} device={self.device}"
        )

    def shutdown(self):
        """Shut down the sgl.Engine."""
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    # ─── Core inference step ──────────────────────────────────────────────

    def _build_embeds(
        self, v_raw: torch.Tensor, prompt_content: str | None
    ) -> np.ndarray:
        """Tokenize -> embed -> arch-scale -> inject. Returns embeds [T, d]."""
        if prompt_content is None:
            content = self.cfg.verbalizer_prompt_template.format(
                injection_char=self.cfg.injection_char
            )
        else:
            assert INJECT_PLACEHOLDER in prompt_content, (
                f"custom prompt must contain {INJECT_PLACEHOLDER!r}"
            )
            content = prompt_content.replace(
                INJECT_PLACEHOLDER, self.cfg.injection_char
            )

        input_ids = _tokenize_chat_with_merges(self.tokenizer, content)
        ids_t = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

        with torch.no_grad():
            embeds = (
                self.embed(ids_t.to(self.embed.weight.device)) * self.embed_scale
            ).float()

        assert torch.isfinite(v_raw).all(), "activation has NaN/Inf"
        v_scaled = normalize_activation(
            v_raw.float().view(1, -1), self.cfg.injection_scale
        )

        injected = inject_at_marked_positions(
            ids_t,
            embeds.cpu(),
            v_scaled,
            self.cfg.injection_token_id,
            self.cfg.injection_left_neighbor_id,
            self.cfg.injection_right_neighbor_id,
        )
        # sgl.Engine wants [T, d] unbatched, contiguous float32
        return injected[0].contiguous().numpy()

    # Verbalizer is trained to emit a single <explanation>…</explanation> block.
    # Stopping on the closing tag terminates generation as soon as the block
    # closes, instead of always running to max_new_tokens. Pair this with
    # `no_stop_trim=True` (set in `_make_req`) — sglang trims matched stop
    # strings by default (sglang>=0.5.x), which would silently strip
    # </explanation> from the returned text and trip a spurious "not closed"
    # warning even though the model emitted it correctly.
    _DEFAULT_STOP_SEQUENCES: tuple[str, ...] = ("</explanation>",)

    def _make_req(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        temperature: float = 1.0,
        max_new_tokens: int = 512,
    ) -> "GenerateReqInput":
        """Build a GenerateReqInput with input_embeds.

        Engine.generate() / async_generate() don't expose the input_embeds
        parameter (as of SGLang 0.5.8), so we construct the request object
        directly and feed it to the tokenizer_manager.
        """
        from sglang.srt.managers.io_struct import GenerateReqInput

        v = torch.as_tensor(np.asarray(activation, dtype=np.float32))
        assert v.numel() == self.cfg.d_model, (
            f"activation length {v.numel()} != d_model {self.cfg.d_model}"
        )
        embeds_np = self._build_embeds(v, prompt)

        return GenerateReqInput(
            input_embeds=embeds_np.tolist(),
            sampling_params={
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "skip_special_tokens": False,
                "stop": list(self._DEFAULT_STOP_SEQUENCES),
                # Keep the matched stop string in the returned text so
                # EXPLANATION_RE can find </explanation>. Without this,
                # sglang trims the suffix and _extract_text emits a
                # spurious "not closed" warning.
                "no_stop_trim": True,
            },
        )

    @staticmethod
    def _summarize_meta(out: dict) -> str:
        """Compact one-liner of sglang's `meta_info` for warnings.

        sglang attaches a `meta_info` dict on each generation result with
        diagnostic fields like `finish_reason` ({"type": "stop"|"length"|
        "abort", "matched": <stop-string-or-token-id>}), `completion_tokens`,
        `prompt_tokens`, etc. When generation ends without `</explanation>`
        this is the only way to tell *why* (length truncation vs EOS vs
        stop-string match vs abort) — the raw text alone is ambiguous.
        Returns "" when meta_info is absent (e.g. partial-stream callers).
        """
        meta = out.get("meta_info") if isinstance(out, dict) else None
        if not isinstance(meta, dict):
            return ""
        finish = meta.get("finish_reason")
        ctoks = meta.get("completion_tokens")
        ptoks = meta.get("prompt_tokens")
        return (
            f" finish_reason={finish!r} completion_tokens={ctoks} prompt_tokens={ptoks}"
        )

    def _extract_text(
        self,
        out: dict,
        extract_explanation: bool,
        *,
        context: str | None = None,
    ) -> str:
        text = out["text"]
        if not extract_explanation:
            return text
        ctx = f" [context={context!r}]" if context else ""
        meta_summary = self._summarize_meta(out)
        m = EXPLANATION_RE.search(text)
        if m is None:
            print(
                f"[NLAClient] WARNING: no <explanation> opening tag.{ctx}"
                f"{meta_summary} Raw[:200]={text[:200]!r}"
            )
            return text
        body = m.group(1).strip()
        if "</explanation>" not in m.group(0):
            # If sglang's `</explanation>` stop-string matched, the model
            # DID emit the closing tag — sglang just trimmed it from the
            # output (when no_stop_trim=False). The body is correct; no
            # warning needed. We set no_stop_trim=True in _make_req to
            # avoid this normally, but stay resilient if that flag is ever
            # ignored / removed by an sglang upgrade.
            meta = out.get("meta_info") if isinstance(out, dict) else None
            finish = meta.get("finish_reason") if isinstance(meta, dict) else None
            stop_matched_close_tag = (
                isinstance(finish, dict)
                and finish.get("type") == "stop"
                and finish.get("matched") == "</explanation>"
            )
            if not stop_matched_close_tag:
                # Real partial body: max_new_tokens truncation, EOS without
                # closing tag, abort, etc. Dump head/tail so it's obvious
                # the opening <explanation> IS present (regex group(1)
                # excludes the tag itself by construction).
                head = text[:120]
                tail = text[-120:] if len(text) > 120 else ""
                print(
                    f"[NLAClient] WARNING: <explanation> not closed; "
                    f"returning partial body.{ctx}{meta_summary} "
                    f"Raw_head={head!r} Raw_tail={tail!r} Partial body={body!r}"
                )
        return body

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
        """Decode one activation vector (async — safe inside uvicorn event loop).

        `context`, if given, is included in any truncation/parse warnings
        (e.g. the source token being explained) to make logs actionable.
        """
        obj = self._make_req(
            activation,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        generator = self.engine.tokenizer_manager.generate_request(obj, None)
        out = await generator.__anext__()
        return self._extract_text(out, extract_explanation, context=context)

    async def async_generate_stream(
        self,
        activation: Iterable[float] | np.ndarray | torch.Tensor,
        *,
        prompt: str | None = None,
        temperature: float = 1.0,
        max_new_tokens: int = 200,
    ):
        """Yield sglang output dicts as the verbalizer generates tokens.

        Each yielded value is the full sglang result dict containing at
        least `text` (cumulative decoded text so far) and `meta_info`
        (with `finish_reason`, `completion_tokens`, etc on the final
        chunk). Callers needing just the text should read `out["text"]`;
        passing the final dict to `_extract_text` lets warnings surface
        the actual stop reason instead of guessing.
        """
        obj = self._make_req(
            activation,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        obj.stream = True
        generator = self.engine.tokenizer_manager.generate_request(obj, None)
        async for out in generator:
            yield out

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
        obj = self._make_req(
            activation,
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        generator = self.engine.tokenizer_manager.generate_request(obj, None)
        out = self.engine.loop.run_until_complete(generator.__anext__())
        return self._extract_text(out, extract_explanation, context=context)


# ─── RECONSTRUCTOR ──────────────────────────────────────────────────────────

_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")


_FP8_TYPE_MARKERS = (
    "Float8",
    "AffineQuantized",
    "QuantizedTensor",
    "TensorImpl",
)


def _is_fp8_weight(weight: Any) -> bool:
    """Detect a torchao FP8 weight-only quantized tensor.

    torchao wraps nn.Linear weights in tensor subclasses (e.g.
    `AffineQuantizedTensor`) and KEEPS `weight.dtype` reporting the original
    floating dtype (bf16) for downstream compatibility. So checking
    `weight.dtype` for "float8" reliably under-counts. This helper triangulates
    using three signals: the stringified dtype, the storage element size
    (fp8 = 1 byte, bf16 = 2 bytes), and the type name of the weight + its
    `.data` attribute.
    """
    try:
        if "float8" in str(weight.dtype).lower():
            return True
    except Exception:
        pass
    try:
        if weight.element_size() == 1:
            # 1-byte storage: either FP8 or INT8. We're only ever applying
            # Float8WeightOnlyConfig in this codebase, so 1-byte === fp8 here.
            return True
    except Exception:
        pass
    type_names = [type(weight).__name__]
    data = getattr(weight, "data", None)
    if data is not None:
        type_names.append(type(data).__name__)
    for tn in type_names:
        if any(marker in tn for marker in _FP8_TYPE_MARKERS):
            return True
    return False


def _count_fp8_linear_weights(module: torch.nn.Module) -> tuple[int, int]:
    """Return (fp8_count, total_linear_count) for diagnostic logging.

    See `_is_fp8_weight` for the detection signals.
    """
    fp8_count = 0
    total = 0
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            total += 1
            if _is_fp8_weight(m.weight):
                fp8_count += 1
    return fp8_count, total


def _summarize_extra_modules(model: torch.nn.Module) -> tuple[list[str], float]:
    """Detect *active* non-text submodules (e.g. SigLIP vision tower).

    Returns (list_of_module_names_with_actual_params, total_param_GB).
    Specifically excludes attributes that have been replaced with
    `nn.Identity` — those are inert no-ops and we don't want them
    triggering a "residual" warning after `_strip_vision_components`.
    """
    candidates = (
        "vision_tower",
        "vision_model",
        "multi_modal_projector",
        "image_newline",
        "vision_resampler",
    )
    present: list[str] = []
    inner = getattr(model, "model", None)
    for name in candidates:
        for owner in (model, inner):
            if owner is None:
                continue
            mod = getattr(owner, name, None)
            if mod is None or isinstance(mod, torch.nn.Identity):
                continue
            # Has actual parameters → worth flagging.
            if any(True for _ in mod.parameters()):
                present.append(name)
                break
    if not present:
        return [], 0.0
    bytes_total = 0
    for owner in (model, inner):
        if owner is None:
            continue
        for name in present:
            mod = getattr(owner, name, None)
            if mod is None or isinstance(mod, torch.nn.Identity):
                continue
            for p in mod.parameters():
                bytes_total += _actual_storage_bytes(p)
    return present, bytes_total / (1024**3)


def _text_decoder(inner: torch.nn.Module) -> torch.nn.Module:
    """Drill into the text decoder of a (possibly multimodal) HF backbone.

    HF wraps multimodal models like Gemma 3 / Llama 4 in an outer Model that
    holds .language_model (the actual text decoder with .layers + .norm)
    alongside .vision_tower and .multi_modal_projector. Plain text-only
    models expose .layers / .norm directly. This returns the module that
    has the layer stack and final norm, regardless of which case we're in.
    """
    return getattr(inner, "language_model", inner)


_VISION_SUBMODULE_ATTRS = (
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
    "image_newline",
    "vision_resampler",
)


def _strip_vision_components(backbone: torch.nn.Module) -> list[tuple[str, float]]:
    """Replace any non-text submodules (vision tower etc.) with `nn.Identity()`.

    Call this on a CPU-loaded `AutoModelForCausalLM` BEFORE .to(device) for
    multimodal models (Gemma3-IT, Llava, etc.) that we only use for activation
    extraction. Removing these submodules drops their weights — for Gemma-3-IT
    that's ~0.8 GB of vision tower + projector that would otherwise sit unused
    on the GPU.

    `setattr(..., Identity())` is not enough on its own — `nn.Module._modules`
    keeps a registry that can resurrect references. So we explicitly:
      1. Drop the original module from the parent's `_modules` dict
      2. Re-register the attribute as Identity for forward-compatibility
      3. The parent's `parameters()` traversal will no longer find the old weights

    Returns a list of (attr_name, gb_freed) pairs for diagnostic logging.
    """
    inner = getattr(backbone, "model", backbone)
    freed: list[tuple[str, float]] = []
    for attr in _VISION_SUBMODULE_ATTRS:
        for owner in (backbone, inner):
            mod = getattr(owner, attr, None)
            if mod is None or isinstance(mod, torch.nn.Identity):
                continue
            bytes_total = sum(p.numel() * p.element_size() for p in mod.parameters())
            # Properly remove from _modules registry so parameters() no longer
            # traverses these weights — `setattr` to Identity alone leaves the
            # original Module ref alive in some torch versions.
            if hasattr(owner, "_modules") and attr in owner._modules:
                del owner._modules[attr]
            setattr(owner, attr, torch.nn.Identity())
            freed.append((attr, bytes_total / (1024**3)))
            break  # don't double-count if both owners have the attr
    return freed


def _actual_storage_bytes(p: torch.Tensor) -> int:
    """Return actual GPU storage bytes, drilling into torchao quantized tensors.

    For a torchao `AffineQuantizedTensor`, `p.numel() * p.element_size()` lies:
    the wrapper reports `element_size = 2` (the dequantized bf16 dtype) even
    though the underlying storage is fp8 (1 byte). This helper inspects
    `p.tensor_impl` to sum the actual quantized-data + scale tensors.

    For unquantized tensors, returns the standard byte count.
    """
    # First, try to drill into a torchao tensor subclass.
    impl = getattr(p, "tensor_impl", None)
    if impl is not None:
        # Common attribute names torchao uses for the actual storage.
        candidate_attrs = (
            "float8_data",
            "int_data",
            "qdata",
            "scale",
            "scales",
            "zero_point",
            "zero_points",
        )
        seen_storage: set[int] = set()
        actual = 0
        for attr_name in candidate_attrs:
            attr = getattr(impl, attr_name, None)
            if torch.is_tensor(attr):
                # Avoid double-counting if the same underlying storage is
                # exposed under multiple names (rare but possible).
                sid = attr.untyped_storage().data_ptr()
                if sid not in seen_storage:
                    seen_storage.add(sid)
                    actual += attr.numel() * attr.element_size()
        if actual > 0:
            return actual
    # Default: outer tensor's bytes (correct for unquantized tensors).
    return p.numel() * p.element_size()


def _summarize_top_param_groups(
    model: torch.nn.Module, top_n: int = 8
) -> list[tuple[str, float, int]]:
    """Per-top-module byte breakdown for diagnostic logging.

    Groups parameters by their top-level module path (the first 1-2 segments
    before the deepest layer) and returns a list of (group_name, gb, count)
    sorted by ACTUAL storage size descending. Uses `_actual_storage_bytes`
    so torchao-quantized tensors report their real fp8 storage, not the
    bf16 wrapper size. Used to spot unexpected weights on GPU and to verify
    that quantization is actually saving storage.
    """
    groups: dict[str, tuple[int, int]] = {}
    for name, p in model.named_parameters():
        # First segment, plus second if it's a "layers.N" pattern → "model.layers"
        parts = name.split(".")
        if len(parts) >= 3 and parts[1] == "layers":
            key = f"{parts[0]}.layers"
        elif len(parts) >= 4 and parts[2] == "layers":
            key = f"{parts[0]}.{parts[1]}.layers"
        else:
            key = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
        bytes_count, n = groups.get(key, (0, 0))
        groups[key] = (bytes_count + _actual_storage_bytes(p), n + 1)
    items = sorted(
        ((k, b / (1024**3), n) for k, (b, n) in groups.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return items[:top_n]


class NLAReconstructor:
    """Load an NLA reconstructor and compute reconstruction MSE.

    mse_scale (sqrt(d_model)) makes .mean() produce the d-agnostic 2(1-cos)
    value. MSE range: [0, 4]. ~0.2 = good, ~1.0 = mediocre, 2.0 = orthogonal.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        mse_scale: float | None = None,
        reconstructor_prompt_template: str | None = None,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        fp8: bool = False,
        int4: bool = False,
    ):
        """
        checkpoint_path: HF hub ID or local directory.
        mse_scale:       Override sidecar value (default: loaded from nla_meta.yaml,
                         or sqrt(d_model) if no sidecar).
        reconstructor_prompt_template: Override sidecar template.
        fp8:             If True, apply FP8 weight-only quantization to backbone
                         Linear layers via torchao (Float8WeightOnlyConfig).
                         Roughly halves backbone VRAM. Requires `torchao` and
                         PyTorch >= 2.4.
        int4:            If True, apply INT4 weight-only quantization to
                         backbone Linear layers via torchao (Int4WeightOnlyConfig
                         with group_size=32). Roughly quarters backbone VRAM
                         compared to bf16 (~halves it compared to fp8). Adds
                         5-15% L2 drift to the predicted activation but the
                         reconstructor's output is a soft scalar score (MSE,
                         cosine), so this is acceptable for relative ranking
                         even if absolute scores shift. Mutually exclusive
                         with fp8. Requires `torchao` and PyTorch >= 2.4.
                         The value_head is intentionally left in `dtype`
                         (typically bf16) for both fp8 and int4 paths — it
                         produces the predicted activation vector, so its
                         numerics are quality-critical and the savings (a
                         single d×d projection, ~50-150 MB) aren't worth the
                         risk.
        """
        if fp8 and int4:
            raise ValueError(
                "fp8 and int4 are mutually exclusive — pick at most one for "
                "the reconstructor backbone."
            )
        local_path = resolve_checkpoint_path(checkpoint_path)
        checkpoint_dir = Path(local_path)

        # Try loading from sidecar, fall back to explicit params
        meta_path = checkpoint_dir / "nla_meta.yaml"
        self.extraction_layer_index: int | None = None
        if meta_path.exists():
            meta = yaml.safe_load(meta_path.read_text())
            # Back-compat: role names have changed across schema versions.
            #   schema_version 1 (legacy): "critic"
            #   schema_version 2 (kitft):  "ar" (activation reconstructor)
            #   current canonical:         "reconstructor"
            # Section name + prompt-template key still use legacy "critic" or
            # newer "reconstructor"; kitft v2 sidecars keep "critic" for both.
            role = meta.get("role")
            assert role in ("reconstructor", "critic", "ar"), (
                f"sidecar role={role!r}, expected 'reconstructor', 'critic', or 'ar'."
            )
            self.mse_scale = float(meta["extraction"]["mse_scale"])
            section = meta.get("reconstructor") or meta.get("critic") or {}
            self.extraction_layer_index = section.get("extraction_layer_index")
            # schema_version 2 (kitft) keys prompt_templates by role ("av"/"ar")
            # rather than by "reconstructor"/"critic"; fall back to role lookup.
            prompt_templates = meta["prompt_templates"]
            self.template = (
                prompt_templates.get("reconstructor")
                or prompt_templates.get("critic")
                or prompt_templates.get(role)
            )
            assert self.template is not None, (
                f"sidecar prompt_templates has no entry for 'reconstructor', "
                f"'critic', or role={role!r}; got keys "
                f"{sorted(prompt_templates)!r}."
            )
        else:
            self.template = (
                reconstructor_prompt_template or DEFAULT_RECONSTRUCTOR_PROMPT_TEMPLATE
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True
        )

        # BOS invariant: training tokenized reconstructor prompts with
        # add_special_tokens=True (reward.py, nla_generate.py). For Gemma/Llama
        # this prepends BOS; for Qwen (bos_token=None) it's a no-op. Dropping
        # BOS shifts position-0 meaning → degraded reconstruction everywhere
        # (observed: Gemma fve_nrm 0.31 vs 0.77). reconstruct() below uses
        # add_special_tokens=True — this assert catches if that ever flips.
        probe = self.tokenizer("x", add_special_tokens=True)["input_ids"]
        bos = self.tokenizer.bos_token_id
        assert bos is None or probe[0] == bos, (
            f"tokenizer has bos_token_id={bos} but add_special_tokens=True "
            f"produced first token {probe[0]}. Reconstructor was trained with BOS "
            f"prefix — reconstruct() must match."
        )

        # Single load path: bf16 on CPU. Both FP8 and INT4 reuse this exact
        # load and apply torchao's `quantize_()` directly afterward — same
        # ordering, just a different recipe and a different device for the
        # quantize call. Mirroring the two paths sidesteps the
        # `from_pretrained(quantization_config=..., device_map=...)` codepath
        # for INT4, which had been producing degenerate reconstructions
        # (cosine ≈ 0 across all tokens). With this structure, `lm_head` is
        # replaced with `Identity` *before* `quantize_()` runs in both modes,
        # so torchao never sees lm_head as a Linear and any tied-embedding
        # edge cases (Gemma-style `tie_word_embeddings=True`) cannot affect
        # `embed_tokens` via shared storage.
        backbone = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        # Identity replacements happen BEFORE any quantization so torchao only
        # sees the decoder's interior Linears. Required for correctness with
        # tied embeddings (and harmless / a tiny perf win otherwise).
        backbone.lm_head = torch.nn.Identity()
        inner = _text_decoder(backbone.model)
        for attr in _FINAL_LN_ATTRS:
            if hasattr(inner, attr):
                setattr(inner, attr, torch.nn.Identity())
                break
        else:
            raise AssertionError(
                f"no final-LN attribute on {type(inner).__name__} — tried "
                f"{_FINAL_LN_ATTRS!r}."
            )

        text_cfg = getattr(backbone.config, "text_config", backbone.config)
        d = text_cfg.hidden_size

        # mse_scale: explicit param > sidecar > default sqrt(d)
        if mse_scale is not None:
            self.mse_scale = mse_scale
        elif not hasattr(self, "mse_scale"):
            self.mse_scale = math.sqrt(d)

        self.value_head = torch.nn.Linear(d, d, bias=False, dtype=dtype)
        head_path = checkpoint_dir / "value_head.safetensors"
        assert head_path.exists(), f"no value_head.safetensors at {checkpoint_dir!r}."
        self.value_head.load_state_dict(load_file(str(head_path)))

        self.fp8 = fp8
        self.int4 = int4

        if fp8:
            # FP8 weight-only quantization via torchao on the backbone, applied
            # ON CPU BEFORE the .to(device) transfer to avoid the bf16
            # transient peak on GPU during model load.
            try:
                from torchao.quantization import (  # type: ignore[import-not-found]
                    Float8WeightOnlyConfig,
                    quantize_,
                )
            except ImportError as e:
                raise RuntimeError(
                    "fp8=True requires the `torchao` package. Install with "
                    "`pip install torchao` (needs PyTorch >= 2.4)."
                ) from e
            print(
                "[NLAReconstructor] applying FP8 weight-only quantization "
                "(torchao) to backbone on CPU before GPU transfer "
                "(avoids bf16 transient peak)..."
            )
            quantize_(backbone, Float8WeightOnlyConfig())
            gc.collect()
            self.backbone = backbone.to(device).eval()
        elif int4:
            # INT4 weight-only quantization via torchao. Unlike FP8, the int4
            # conversion op (`aten::_convert_weight_to_int4pack`) has no CPU
            # implementation, so we have to move to GPU first and quantize
            # there. This briefly holds bf16 weights on GPU before they get
            # packed into int4 — a transient peak we accept in exchange for
            # using the same load+identity-swap+quantize_ ordering as FP8
            # (the previous TorchAoConfig+device_map path was producing
            # degenerate reconstructions).
            try:
                from torchao.quantization import (  # type: ignore[import-not-found]
                    Int4WeightOnlyConfig,
                    quantize_,
                )
            except ImportError as e:
                raise RuntimeError(
                    "int4=True requires the `torchao` package. Install with "
                    "`pip install torchao` (needs PyTorch >= 2.4)."
                ) from e
            if not device.startswith("cuda"):
                raise RuntimeError(
                    f"int4=True requires a CUDA device for backbone "
                    f"quantization (torchao's int4 conversion op has no CPU "
                    f"implementation), got device={device!r}."
                )
            print(
                "[NLAReconstructor] moving backbone to "
                f"{device} then applying INT4 weight-only quantization "
                "(torchao, group_size=32) on GPU..."
            )
            backbone = backbone.to(device)
            # group_size=128 broke Llama reconstructor, we use 32
            quantize_(backbone, Int4WeightOnlyConfig(group_size=32, use_hqq=True))
            gc.collect()
            self.backbone = backbone.eval()
        else:
            self.backbone = backbone.to(device).eval()
        self.value_head = self.value_head.to(device).eval()
        self.device = device

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        if fp8 or int4:
            quant_n, lin_n = _count_fp8_linear_weights(self.backbone)
            label = "FP8" if fp8 else "INT4"
            print(
                f"[NLAReconstructor] {label} quantization fingerprint: "
                f"{quant_n}/{lin_n} backbone Linear layers quantized"
            )
            if lin_n > 0 and quant_n == 0:
                print(
                    f"[NLAReconstructor] WARNING: torchao did not quantize "
                    f"any Linear layers — {label.lower()}=True is currently "
                    f"a no-op for this model. Check torchao version "
                    f"compatibility with this architecture."
                )

        quant_state = "fp8" if fp8 else "int4" if int4 else "bf16"
        print(
            f"[NLAReconstructor] {backbone.config.num_hidden_layers} layers  "
            f"d_model={d}  mse_scale={self.mse_scale:.2f}  quant={quant_state}"
        )

    @torch.inference_mode()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        """Explanation text -> predicted activation vector (raw, unnormalized)."""
        prompt = self.template.format(explanation=explanation)
        # add_special_tokens=True: Gemma reconstructor was trained with BOS prefix
        # (reconstructor_prompt_template is a raw string, not chat-template-processed).
        # Qwen has bos_token=None so this is a no-op there. Omitting BOS for
        # Gemma shifts position-0 meaning → degraded reconstruction everywhere.
        ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ].to(self.device)
        out = self.backbone.model(ids, use_cache=False)
        h = out.last_hidden_state[0, -1]
        result = self.value_head(h).float().cpu()
        del ids, out, h
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return result

    @torch.inference_mode()
    def reconstruct_batch(self, explanations: list[str]) -> torch.Tensor:
        """Batched explanation -> predicted activation vectors.

        Returns a (B, d_model) float32 CPU tensor — same numerics as calling
        `reconstruct()` once per row, but with a single padded forward pass
        through the backbone. Right-padding + per-row last-real-token gather
        keeps results identical regardless of batch composition.
        """
        if not explanations:
            return torch.empty(0)
        prompts = [self.template.format(explanation=e) for e in explanations]

        # Batched tokenization needs a pad token. Some checkpoints (Qwen)
        # ship without one; fall back to EOS in that case. This only
        # affects the batch path — single-prompt `reconstruct()` never
        # pads, so its numerics are unchanged.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=True,
            padding=True,
        )
        ids = enc["input_ids"].to(self.device)
        attn = enc["attention_mask"].to(self.device)

        out = self.backbone.model(input_ids=ids, attention_mask=attn, use_cache=False)
        last_hidden = out.last_hidden_state  # (B, T, d)
        # Right-padding: last real token is at attention_mask.sum(-1) - 1.
        # Gather per-row so padded tail tokens never feed the value head.
        last_idx = attn.sum(dim=1) - 1
        rows = torch.arange(last_hidden.size(0), device=self.device)
        h = last_hidden[rows, last_idx]
        result = self.value_head(h).float().cpu()
        del ids, attn, out, last_hidden, h, rows
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        return result

    def score(
        self, explanation: str, original: np.ndarray | torch.Tensor
    ) -> tuple[float, float]:
        """(direction-MSE, cos-sim). MSE = 2(1-cos), range [0, 4]."""
        pred = self.reconstruct(explanation)
        return self._score_from_pred(pred, original)

    def score_batch(
        self,
        explanations: list[str],
        originals: list[np.ndarray | torch.Tensor],
    ) -> list[tuple[float, float]]:
        """Batched (mse, cos) scoring. Equivalent to calling .score() per item."""
        assert len(explanations) == len(originals), (
            f"score_batch: {len(explanations)} explanations vs "
            f"{len(originals)} originals"
        )
        if not explanations:
            return []
        preds = self.reconstruct_batch(explanations)
        return [
            self._score_from_pred(preds[i], originals[i])
            for i in range(len(explanations))
        ]

    def _score_from_pred(
        self,
        pred: torch.Tensor,
        original: np.ndarray | torch.Tensor,
    ) -> tuple[float, float]:
        gold = torch.as_tensor(np.asarray(original, dtype=np.float32))
        pred_n = pred / pred.norm().clamp_min(1e-12) * self.mse_scale
        gold_n = gold / gold.norm().clamp_min(1e-12) * self.mse_scale
        mse = ((pred_n - gold_n) ** 2).mean().item()
        cos = (pred_n @ gold_n / (pred_n.norm() * gold_n.norm())).item()
        return float(mse), float(cos)


# ─── SOURCE MODEL (activation extraction) ───────────────────────────────────


class SourceModel:
    """Load the base model that activations are extracted from.

    The NLA verbalizer was trained on activations from this model's residual stream
    at a specific layer (e.g. layer 20 for Qwen 2.5 7B). To get meaningful
    NLA descriptions, extract activations from THIS model, not the verbalizer.
    """

    def __init__(
        self,
        model_path: str,
        *,
        layer_index: int = 20,
        device: str = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        truncate: bool = True,
        fp8: bool = False,
    ):
        """
        model_path:   HF hub ID (e.g. 'Qwen/Qwen2.5-7B-Instruct') or local dir.
        layer_index:  Which layer's residual stream to extract from (~2/3 depth).
                      Qwen 7B (28 layers): 20. Gemma-3-12B (48 layers): 32.
        truncate:     If True, drop layers past `layer_index` plus the final
                      LayerNorm and `lm_head`. Saves VRAM (e.g. ~25% of layer
                      weights + lm_head for Qwen 7B at layer 20) but disables
                      `model.generate()` — the model produces hidden states,
                      not logits. Required for /completion to be functional.
        fp8:          If True, apply FP8 weight-only quantization to all Linear
                      layers via torchao (Float8WeightOnlyConfig). Roughly
                      halves weight VRAM. Requires `torchao` installed and
                      PyTorch >= 2.4. Quantization runs AFTER truncation, so
                      only the kept layers are quantized.
        """
        local_path = resolve_checkpoint_path(model_path)
        self.layer_index = layer_index
        self.truncated = truncate
        self.fp8 = fp8

        print(
            f"[SourceModel] Loading {model_path} "
            f"(layer {layer_index}, truncate={truncate}, fp8={fp8})..."
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_path, trust_remote_code=True
        )
        # Load on CPU first so truncation drops weights BEFORE the .to(device)
        # transfer — avoids the transient GPU memory spike of loading the full
        # model onto GPU only to immediately free 25%+ of it.
        backbone = AutoModelForCausalLM.from_pretrained(
            local_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )

        text_cfg = getattr(backbone.config, "text_config", backbone.config)
        d = text_cfg.hidden_size
        n_layers = text_cfg.num_hidden_layers
        assert 0 <= layer_index < n_layers, (
            f"layer_index={layer_index} out of range for {n_layers}-layer model"
        )

        # Strip vision tower / multi-modal projector BEFORE .to(device) for
        # multimodal models we only use for text activation extraction. Saves
        # ~1 GB on Gemma-3-IT and ensures these weights never touch the GPU.
        freed_vision = _strip_vision_components(backbone)
        if freed_vision:
            for attr, gb in freed_vision:
                print(
                    f"[SourceModel] stripped non-text submodule {attr}: "
                    f"freed {gb:.2f} GB (replaced with Identity)"
                )
            gc.collect()

        inner = _text_decoder(
            backbone.model if hasattr(backbone, "model") else backbone
        )

        if truncate:
            # Same surgery as NLAReconstructor: lm_head + final norm become
            # Identity, and we additionally chop off everything past the
            # extraction layer since /completion isn't going through this
            # model anymore. The capture hook below sits on layer_index, which
            # is now the LAST layer — its output is the activation we want.
            backbone.lm_head = torch.nn.Identity()
            for attr in _FINAL_LN_ATTRS:
                if hasattr(inner, attr):
                    setattr(inner, attr, torch.nn.Identity())
                    break
            else:
                raise AssertionError(
                    f"no final-LN attribute on {type(inner).__name__} — tried "
                    f"{_FINAL_LN_ATTRS!r}."
                )
            inner.layers = torch.nn.ModuleList(list(inner.layers)[: layer_index + 1])
            # Force GC on CPU before the GPU transfer so the dropped layer
            # tensors are actually released — without this they may linger
            # on CPU and (in pathological multi-modal layouts) get re-attached
            # to the device transfer.
            gc.collect()

        if fp8:
            # FP8 weight-only quantization via torchao, applied ON CPU BEFORE
            # the .to(device) transfer. This is critical on tight-VRAM GPUs:
            # a 27B-class model has ~37 GB of bf16 weights post-truncation,
            # which would otherwise transit through GPU memory before being
            # quantized to ~19 GB. Quantizing first means the .to(device) call
            # only ever moves the smaller fp8 tensors, halving the peak GPU
            # usage during model load. Activations stay bf16 — only nn.Linear
            # weights become torch.float8_e4m3fn.
            try:
                from torchao.quantization import (  # type: ignore[import-not-found]
                    Float8WeightOnlyConfig,
                    quantize_,
                )
            except ImportError as e:
                raise RuntimeError(
                    "fp8=True requires the `torchao` package. Install with "
                    "`pip install torchao` (needs PyTorch >= 2.4)."
                ) from e
            print(
                "[SourceModel] applying FP8 weight-only quantization (torchao) "
                "on CPU before GPU transfer (avoids bf16 transient peak)..."
            )
            quantize_(backbone, Float8WeightOnlyConfig())
            gc.collect()

            fp8_n, lin_n = _count_fp8_linear_weights(backbone)
            print(
                f"[SourceModel] FP8 quantization fingerprint: "
                f"{fp8_n}/{lin_n} Linear layers in fp8"
            )
            if lin_n > 0 and fp8_n == 0:
                print(
                    "[SourceModel] WARNING: torchao did not quantize any Linear "
                    "layers — fp8=True is currently a no-op for this model. "
                    "Check torchao version compatibility with this architecture."
                )

        self.model = backbone.to(device).eval()
        self.device = device

        if device.startswith("cuda"):
            torch.cuda.empty_cache()

        # Detect any remaining non-text components (e.g. SigLIP vision tower
        # for Gemma-3-IT) that survived the strip — should be empty after
        # _strip_vision_components above, but kept as a safety net.
        extra_present, extra_gb = _summarize_extra_modules(self.model)
        if extra_present and extra_gb > 0.05:
            print(
                f"[SourceModel] WARNING: residual non-text submodules "
                f"{extra_present} totalling {extra_gb:.2f} GB still attached "
                f"after stripping. May indicate a model layout this codebase "
                f"hasn't been updated for."
            )

        # Authoritative per-top-module byte breakdown of what is actually on
        # GPU. This is the ground truth for diagnosing "where did my VRAM go".
        # Reports the largest 8 groups by parameter bytes; covers the common
        # culprits (full lm_head, dropped layers still alive, second copy of
        # embeddings, etc.).
        breakdown = _summarize_top_param_groups(self.model, top_n=8)
        if breakdown:
            print("[SourceModel] top-N param byte breakdown (post-load):")
            for name, gb, n in breakdown:
                print(f"    {gb:6.2f} GB  {name:<40s}  ({n} param tensor(s))")

        # Forward hook on the target layer captures its output (post-residual,
        # pre next-layer norm). Equivalent to out.hidden_states[layer_index+1]
        # from output_hidden_states=True, but avoids the GPU memory spike of
        # retaining ALL layer outputs at once.
        #
        # The capture target is thread-local so that concurrent model.generate()
        # calls on other threads (e.g. /completion workers) don't pollute the
        # capture buffer. Hook fires on every forward; thread-local check is a
        # no-op when no extract is in progress on this thread.
        self._capture_local = threading.local()
        target_layer = inner.layers[layer_index]

        def _capture_hook(module, _inputs, output):
            buf = getattr(self._capture_local, "target", None)
            if buf is None:
                return
            # Some decoder layers historically returned (hidden_states, ...).
            # Qwen2/Llama/Gemma in transformers >=4.55 return a bare tensor.
            buf.append(output[0] if isinstance(output, tuple) else output)

        target_layer.register_forward_hook(_capture_hook)

        print(
            f"[SourceModel] ready: d_model={d}  "
            f"layers={len(inner.layers)}/{n_layers}  "
            f"extract_layer={layer_index}  truncated={truncate}  fp8={fp8}"
        )

    @torch.inference_mode()
    def extract(self, text: str) -> list[dict]:
        """Extract activation vectors at every token position.

        Returns list of {token, token_id, position, activation} dicts.
        Activations are from the residual stream at self.layer_index.
        """
        ids = self.tokenizer(text, return_tensors="pt", add_special_tokens=True)[
            "input_ids"
        ].to(self.device)

        # Arm the per-thread capture buffer. The hook on layer[layer_index]
        # appends the layer output during the forward pass.
        capture: list[torch.Tensor] = []
        self._capture_local.target = capture
        try:
            self.model(ids, use_cache=False)
        finally:
            self._capture_local.target = None

        assert len(capture) == 1, (
            f"expected 1 captured activation, got {len(capture)}. "
            f"Concurrent forward on the same thread?"
        )
        hidden = capture[0]  # [1, T, d]
        hidden_cpu = hidden.float().cpu()
        token_ids = ids[0].tolist()
        del capture, hidden, ids
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        results = []
        for pos in range(hidden_cpu.shape[1]):
            vec = hidden_cpu[0, pos]
            results.append(
                {
                    "token": self.tokenizer.decode([token_ids[pos]]),
                    "token_id": token_ids[pos],
                    "position": pos,
                    "activation": vec.tolist(),
                    "l2_norm": float(vec.norm()),
                }
            )
        return results
