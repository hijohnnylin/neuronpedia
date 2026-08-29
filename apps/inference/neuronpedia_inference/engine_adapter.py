"""Adapter bridging the inference app's TransformerLens-style hook names to the
interp-engine (EagerModel) canonical hook points.

SAEs carry a TransformerLens-style hook name (e.g. ``blocks.7.hook_resid_pre``) — a naming
convention we keep. The engine speaks in canonical names (``resid_pre``, ``resid_post``,
``mlp_in``, ...). This module is the single place that maps between the two and produces the
``{hook_name: tensor}`` dicts the existing activation/DFA post-processing consumes.

The per-backend branches below survive the engine's ``InterpModel`` protocol on purpose,
because they are not redundant dispatch:

- **Capture** could go through ``model.capture`` for both, but that protocol method returns
  CPU tensors (vLLM ships them back over ``collective_rpc``, so it has no choice). Eager can
  hand back the activation on the device it was computed on, and the encode that consumes it
  is on that device — so routing eager through the protocol would add a GPU->CPU->GPU round
  trip per hook point on the hottest endpoint.
- **Batched capture** exists eagerly as one padded forward; vLLM has no batched worker
  capture and has to loop per prompt and scatter.
- **DFA / attention** is the capability split the protocol deliberately excludes: eager reads
  the real softmax, vLLM has to recompute it off-kernel from captured q/k/v.

One family of hook names is not a point at all and is recomputed on both backends: TransformerLens'
``blocks.{i}.ln{1,2}.hook_normalized``, which the GemmaScope transcoders are trained at. The engine
owns that arithmetic (``tlens_normalized_hook`` and ``pre_gain_normalized``); what this module adds is
routing it through both capture paths and refusing a model whose norms are not RMS norms. See
:class:`_CapturePoint`.
"""

from __future__ import annotations

from typing import NamedTuple, TypedDict

import einops
import torch
from interp_engine import (
    HOOK_CAPTURE_POINTS,
    Address,
    EagerModel,
    Tokenize,
    UnmappedHook,
    VLLMModel,
    per_head_value,
    point_to_tlens_hook,
    pre_gain_normalized,
    rms_norm_eps_for_model,
    run_with_cache,
    tlens_normalized_hook,
)
from interp_engine import tlens_hook_to_point as engine_tlens_hook_to_point
from interp_engine.points import refusal_reasons, tp_sharded


class BackendUnsupported(ValueError):
    """Raised when the loaded backend cannot serve a requested capture point.

    Endpoints catch this to return a clear 4xx instead of a 500. What vLLM serves is
    ``interp_engine``'s ``HOOK_CAPTURE_POINTS`` (see :data:`_VLLM_CAPTURE_OK`) narrowed by this
    pod's sharding; ``attn_probs`` is served by the off-kernel recompute rather than by capture,
    and the eager-only points are listed with their reasons in the engine's point table.
    """


# Points the engine-owned vLLM worker-hook capture serves. **Derived, not restated**: the engine's
# point table is what both the client-side validation and the worker's hook dispatch answer from,
# so a copy here can only ever be a second opinion -- and was one, having been frozen at the five
# points scripts/vllm_capture_points_check.py happens to validate while the engine grew `attn_out`,
# `attn_out_post`, `mlp_out_post`, `resid_mid` and `value`. Attention-output SAEs read `attn_out`,
# so the stale copy turned them into a 400 that blamed the paged-attention kernel.
#
# `attn_probs` is deliberately absent: it is a recompute rather than a capture, and callers reach
# it through capture_attention / vllm_attention_unsupported_reason below.
_VLLM_CAPTURE_OK = set(HOOK_CAPTURE_POINTS)

# Of those, the ones tensor parallelism splits across ranks -- asked of the same table rather than
# derived from a proxy here. `z` (the output projection's input), `value` (the qkv projection's
# output), the QK-norm points and `mlp_act` are head- or neuron-sharded, so rank 0 -- the only
# payload the capture path reads -- holds 1/tp of the width, and there is no cheap way to
# reassemble them from here (the ranks would have to be concatenated in order).
#
# This used to be `_VLLM_CAPTURE_OK - d_model_wide()`, i.e. "anything not `hidden_size` wide is a
# shard". That proxy holds for the points above and breaks on `router_logits`, which is
# `n_experts` wide off a *replicated* gate: every rank computes the whole thing, so refusing it on a
# multi-GPU pod would be refusing a point that works.
_TP_SHARDED_POINTS = _VLLM_CAPTURE_OK & tp_sharded()


def vllm_served_capture_points(model: object) -> set[str]:
    """Capture points this vLLM instance can serve, narrowed by its GPU sharding."""
    declared = getattr(model, "static_points", ()) or ()
    if declared:
        names = {getattr(a, "name", str(a).split(".", 1)[0]) for a in declared}
        if int(getattr(model, "tensor_parallel_size", 1) or 1) > 1:
            return names - _TP_SHARDED_POINTS
        return names
    if not getattr(model, "hooks_available", True):
        return set()
    if int(getattr(model, "tensor_parallel_size", 1) or 1) > 1:
        return _VLLM_CAPTURE_OK - _TP_SHARDED_POINTS
    return set(_VLLM_CAPTURE_OK)


def vllm_attention_unsupported_reason(model: object) -> str | None:
    """Why this vLLM instance cannot produce attention patterns, or None if it can.

    Separate from the ``unsupported`` list on ``_attn_dims``, which records config terms
    the off-kernel recompute cannot reproduce: this is about how the pod is deployed, not
    about the model. The recompute reshapes rank 0's q/k/v with whole-model head counts,
    which only holds when there is one rank.
    """
    tp_size = int(getattr(model, "tensor_parallel_size", 1) or 1)
    if tp_size > 1:
        return (
            f"the model is sharded across {tp_size} GPUs, which splits attention heads "
            "across ranks; the pattern recompute needs all heads on one rank"
        )
    return None


def assert_hooks_available(model: object, what: str = "This endpoint") -> None:
    """Raise :class:`BackendUnsupported` when the pod cannot run capture/steering hooks at all.

    A whole-pod condition rather than a per-point one: ``GENERATION_ONLY`` loads the engine's
    ``backend="vllm-generate"``, which keeps vLLM's CUDA graphs, and graph replay never calls the
    Python forward the hooks are attached to. The engine refuses these calls too, but as a
    ``RuntimeError`` that would surface as a 500 -- here it is a 400 naming the deployment flag,
    which is what a router or a confused operator needs to see.

    Nothing to do on eager, whose ``hooks_available`` is a constant True. A ``vllm-static`` pod is
    also allowed through: ``hooks_available`` is False there, because the graphs are on, but the
    declared taps and writes still run. Whether *this* request's point is in that set is a
    per-point question, answered further down.
    """
    if getattr(model, "hooks_available", True):
        return
    declared = tuple(getattr(model, "static_points", ()) or ())
    writes = tuple(getattr(model, "static_writes", ()) or ())
    if getattr(model, "graph_replay", False) and (declared or writes):
        return
    raise BackendUnsupported(
        f"{what} needs activation capture, which this pod cannot do: it was started with "
        "GENERATION_ONLY=true, so it runs the engine's vllm-generate backend, which trades the "
        "capture and steering hooks for the CUDA graphs they rule out. It serves completions and "
        "tokenization only; see /capabilities for the full list. A pod that captures at graph speed "
        "is STATIC_POINTS instead, which names the sites up front."
    )


def native_resid_available(model: object) -> bool:
    """True when vLLM native extract is on and can serve ``resid_post``.

    ``capture_resid_post`` exists on every ``VLLMModel`` even with ``enable_extraction=False``
    (the default). That path forces chunked prefill off and misses the final layer. A declared
    ``resid_post`` is the product path; native extract is opt-in.
    """
    return bool(getattr(model, "enable_extraction", False)) and callable(getattr(model, "capture_resid_post", None))


def assert_residual_available(model: object, what: str = "This endpoint", point: str = "resid_post") -> None:
    """Refuse residual reads when this pod has neither hooks, a static ``point``, nor native extract.

    ``/activation/raw`` and ``/lens/logit`` need the residual stream, not the SAE sites. A
    ``vllm-static`` pod can still serve them when it declared the residual point, or through
    ``capture_resid_post`` when ``enable_extraction`` is on.

    ``point`` is which residual address *this* endpoint reads, and it is a parameter because the two
    callers do not read the same one. ``/activation/raw`` asks for ``resid_post`` literally (it builds
    ``blocks.N.hook_resid_post`` names). The logit lens asks for whatever carries the block output on
    the served trunk, which is ``resid_streams`` on a hyper-connection model -- so hard-coding
    ``resid_post`` here 400'd a DeepSeek-V4 pod whose declared tap was exactly what the read-out
    wanted. See ``endpoints/lens/residual_spec.py`` for the point-per-trunk table.

    Native extract answers for ``resid_post`` only. It returns one ``d_model`` vector, and on a
    multi-stream trunk nothing says which collapse of the stack that is -- the failure residual_spec
    exists to prevent -- so a pod relying on it does not get to serve a stream stack.
    """
    if getattr(model, "hooks_available", True):
        return
    declared = tuple(getattr(model, "static_points", ()) or ())
    declared_names = {getattr(a, "name", str(a).split(".", 1)[0]) for a in declared}
    if point in declared_names:
        return
    if point == "resid_post" and native_resid_available(model):
        return
    raise BackendUnsupported(
        f"{what} needs residual-stream reads, and this pod has CUDA graphs on with no {point} to "
        f"read them from. Declared: {sorted(declared_names) or 'nothing'}. Restart it with "
        f"STATIC_POINTS naming {point}, or with STATIC_POINTS unset for the hooked vLLM backend "
        "where every point is reachable."
    )


def declares_static_taps(model: object) -> bool:
    """True when this pod's tap set is fixed, so a per-layer check is worth asking at all.

    Guards the work of resolving which layers a request would touch, not just the answer: on a
    hooked pod that resolution reaches into the SAE manager for each feature's hook name, and a
    check that cannot refuse anything should not be able to raise on the way to saying so.
    """
    return not getattr(model, "hooks_available", True)


def _resid_aliases(address: Address) -> tuple[Address, ...]:
    """The other name for the same residual add: ``resid_pre[L]`` IS ``resid_post[L-1]``.

    Mirrors ``resid_stream_aliases`` in the engine's vllm_backend, which is what the refusals
    these two checks pre-empt match against. Copied rather than imported: it is three lines, and
    the import would reach past the package root into the vLLM backend module.

    Drift is the risk that trades for, so these checks lean PERMISSIVE. A site this fails to
    recognize as declared is one the engine refuses for itself, which is the 500 they exist to
    replace and not a new failure; a site they wrongly call missing would refuse a request the pod
    could have served.
    """
    if address.layer is None:
        return (address,)
    layer = int(address.layer)
    if address.name == "resid_pre" and layer > 0:
        return (address, Address("resid_post", layer - 1))
    if address.name == "resid_post":
        return (address, Address("resid_pre", layer + 1))
    return (address,)


def _undeclared_layers(declared: set[str], layers: list[int], point: str) -> list[int]:
    """Which of ``layers`` no alias of ``point`` covers, sorted and deduplicated."""
    return sorted(
        {
            layer
            for layer in layers
            if not any(str(alias) in declared for alias in _resid_aliases(Address(point, layer)))
        }
    )


def _static_layer_miss(
    model: object,
    layers: list[int],
    point: str,
    *,
    private_attr: str,
    public_attr: str,
) -> tuple[list[int], list[str]] | None:
    """``(missing layers, declared)`` for a static pod, or None when the question does not apply.

    None covers both "hooks run, so every site is reachable" and "nothing declared at all", the
    second because that pod is refused wholesale by :func:`assert_hooks_available` /
    :func:`assert_steering_available` and a layer list would be the wrong reason to give.
    """
    if getattr(model, "hooks_available", True):
        return None
    declared = getattr(model, private_attr, None) or getattr(model, public_attr, ()) or ()
    declared_keys = {str(address) for address in declared}
    if not declared_keys:
        return None
    missing = _undeclared_layers(declared_keys, layers, point)
    return (missing, sorted(declared_keys)) if missing else None


def assert_capture_layers_declared(
    model: object,
    layers: list[int],
    what: str = "This endpoint",
    point: str = "resid_post",
) -> None:
    """Refuse a capture at a layer a static pod did not declare, by layer and not just by name.

    :func:`assert_residual_available` asks whether ``point`` is declared at all, which is the
    right question for an endpoint that reads every layer or none. A readout axis reads exactly
    one, so a pod can declare ``resid_post`` and still not have the layer in hand: the 70B pod
    declared the site its layer-50 SAE reads and was asked for layer 40 by an axis fitted there.

    That distinction only started to matter when axes became database rows. A shipped asset was
    on disk before the graphs were recorded, so the mismatch was a deploy-time fact; an axis that
    arrives with the request makes it a per-request one, and the engine's own refusal comes from
    inside the generate call -- a 500 with a traceback, after the prompt was rendered.
    """
    miss = _static_layer_miss(model, layers, point, private_attr="_static_reads", public_attr="static_points")
    if miss is None:
        return
    missing, declared = miss
    raise BackendUnsupported(
        f"{what} reads {point} at layer(s) {missing}, which this pod did not declare. Its CUDA "
        f"graphs are recorded against a fixed tap set, so no layer can be added while it runs. "
        f"Declared: {declared}. Relaunch it with those layers declared: STATIC_POINTS=auto covers "
        f"every layer, and STATIC_POINTS_EXTRA adds a closed set beside STATIC_POINTS=sae "
        f"(e.g. STATIC_POINTS_EXTRA='[\"{point}.{missing[0]}\"]')."
    )


def assert_steer_layers_declared(
    model: object,
    layers: list[int],
    what: str = "Steered generation",
    point: str = "resid_post",
) -> None:
    """The write-side twin of :func:`assert_capture_layers_declared`.

    :func:`assert_steering_available` asks whether this pod declared ANY write site, which catches
    a generation-only pod and misses the case that actually bit: writes declared, at other layers.
    A projection cap writes wherever the feature or vector it caps was fitted, and an uploaded
    vector can name any layer at all, so the set is not one a pod can enumerate at startup --
    which is why the pod this was written for now declares every layer instead.

    Kept separate from the read check because the two sets genuinely differ: an SAE set declares
    writes under the ``resid_pre[L]`` -> ``resid_post[L-1]`` mapping, so a pod can hold a read at
    a layer and no write there.
    """
    miss = _static_layer_miss(model, layers, point, private_attr="_static_writes", public_attr="static_writes")
    if miss is None:
        return
    missing, declared = miss
    raise BackendUnsupported(
        f"{what} writes {point} at layer(s) {missing}, which this pod did not declare. Its CUDA "
        f"graphs are recorded against a fixed tap set, so no write tap can be added while it "
        f"runs. Declared for writing: {declared}. Relaunch it with STATIC_POINTS=auto, which "
        f"declares every layer to read and to write."
    )


def _vllm_points_use_native_resid(model: object, points: list[_CapturePoint]) -> bool:
    """True when these resid_post reads should go through native extract, not static/hooks."""
    if not points or not all(p.address.name == "resid_post" and p.address.layer is not None for p in points):
        return False
    if getattr(model, "hooks_available", True):
        return False
    declared = set(getattr(model, "static_points", ()) or ())
    if all(p.address in declared for p in points):
        return False
    return native_resid_available(model)


def assert_steering_available(model: object, what: str = "Steered generation") -> None:
    """Refuse STEERED completions when this pod has neither hooks nor static writes.

    Capture static (``static_points``) is not enough: additive steer needs ``static_writes``.
    """
    if getattr(model, "hooks_available", True):
        return
    writes = tuple(getattr(model, "static_writes", ()) or ())
    if getattr(model, "graph_replay", False) and writes:
        return
    raise BackendUnsupported(
        f"{what} needs an additive write into the forward. This pod has CUDA graphs on and declared "
        "no write sites, so there is nowhere to add the vector. Restart it with STATIC_POINTS=sae, "
        "which declares a write beside every SAE read, or with STATIC_POINTS unset for the hooked "
        "vLLM backend. Unsteered completions are unaffected."
    )


def sae_static_addresses(sae_manager: object) -> tuple[list[Address], list[Address]]:
    """Reads/writes to declare for every loaded SAE hook, for a ``STATIC_POINTS=sae`` pod.

    Reads are the engine points those SAEs encode. Writes follow ``/steer/completion``:
    ``resid_pre[L]`` becomes ``resid_post[L-1]`` (layer 0 has no static write). The synthetic
    ``neurons`` set (``mlp.hook_post``) is skipped: it is not an SAE, and wrapping it is not
    what ``STATIC_POINTS=sae`` asked for.
    """
    from interp_engine.vllm_capture import STEERABLE_POINTS
    from interp_engine.vllm_capture.static import (
        ATTN_STATIC_POINT,
        static_unsupported_reason,
        steer_write_for_sae_point,
    )

    neurons = getattr(sae_manager, "NEURONS_SOURCESET", "neurons")
    hooks: list[str] = []
    dfa_hooks: list[str] = []
    is_dfa = getattr(sae_manager, "is_dfa_enabled", lambda _s: False)
    for set_name, sae_ids in getattr(sae_manager, "sae_set_to_saes", {}).items():
        if set_name == neurons:
            continue
        for sae_id in sae_ids:
            hook = sae_manager.get_sae_hook(sae_id)  # type: ignore[union-attr]
            if hook:
                hooks.append(hook)
                if is_dfa(sae_id):
                    dfa_hooks.append(hook)
    if not hooks:
        raise ValueError(
            "STATIC_POINTS=sae needs SAE hooks, but none were loaded. Load SAE_SETS or pass an "
            "explicit static_points list."
        )
    reads: list[Address] = []
    seen: set[str] = set()
    for point in _capture_points(list(dict.fromkeys(hooks))):
        key = str(point.address)
        if key in seen:
            continue
        reason = static_unsupported_reason(point.address.name)
        if reason is not None:
            raise ValueError(f"cannot static SAE hook {point.hook!r} ({point.address}): {reason}")
        seen.add(key)
        reads.append(point.address)
    for point in _capture_points(list(dict.fromkeys(dfa_hooks))):
        if point.address.layer is None:
            continue
        attn = Address(ATTN_STATIC_POINT, int(point.address.layer))
        key = str(attn)
        if key in seen:
            continue
        seen.add(key)
        reads.append(attn)
    writes: list[Address] = []
    write_seen: set[str] = set()
    for address in reads:
        mapped = steer_write_for_sae_point(address)
        if mapped is None or mapped.name not in STEERABLE_POINTS:
            continue
        key = str(mapped)
        if key in write_seen:
            continue
        write_seen.add(key)
        writes.append(mapped)
    return reads, writes


def _why_unserved(model: object, unserved: list[str]) -> str:
    """Why *these* points are refused -- this pod's sharding, or the points' own table entries.

    Two different questions, and printing the wrong one sends the reader after the wrong fix: a
    sharded `z` is a deployment fact that a single-GPU pod would not have, while `attn_probs` is a
    property of the point that no pod serves through capture. This used to print the latter
    unconditionally, so a refusal of `attn_out` -- an ordinary module output, and at the time merely
    missing from a stale allowlist -- blamed the fused paged-attention kernel.
    """
    parts = []
    sharded = sorted(n for n in unserved if n in _TP_SHARDED_POINTS)
    if sharded:
        tp_size = int(getattr(model, "tensor_parallel_size", 1) or 1)
        parts.append(f"sharded across {tp_size} GPUs, which splits {sharded} across ranks")
    rest = [n for n in unserved if n not in _TP_SHARDED_POINTS]
    if rest:
        parts.append(f"per the engine's point table:\n{refusal_reasons(rest)}")
    return "; ".join(parts)


def _assert_vllm_points_supported(model: object, points: list[_CapturePoint]) -> None:
    """Raise :class:`BackendUnsupported` for any point this instance cannot serve.

    Checked against what is actually captured, which for the ``hook_normalized`` hooks is the norm's
    input rather than the hook itself -- while the message names the hooks the caller asked for.
    """
    assert_hooks_available(model, "Activation capture")
    served = vllm_served_capture_points(model)
    bad = [point.hook for point in points if point.address.name not in served]
    if not bad:
        return
    unserved = sorted({point.address.name for point in points if point.address.name not in served})
    raise BackendUnsupported(
        f"vLLM backend cannot capture {bad} on this instance ({_why_unserved(model, unserved)}). "
        f"Serving: {sorted(served)}."
    )


def get_tokenize(model: object) -> Tokenize:
    """Return an engine ``Tokenize`` for the loaded model, regardless of backend.

    Both engine backends build one in their constructor and expose it as ``.tok``. The
    fallback covers anything else that turns up holding only a ``.tokenizer``, since
    message-span computation needs nothing more than that.
    """
    tok = getattr(model, "tok", None)
    if tok is not None:
        return tok
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Loaded model does not expose a tokenizer for chat templating")
    return Tokenize(tokenizer, default_prepend_bos=True, device="cpu")


# The raw sublayer output each block-level TransformerLens hook is the *contribution* twin of. Both
# names collapse onto the raw point when the engine mapper is asked without a model, which is what
# `tlens_hook_to_point` below has to undo -- see its docstring for why it cannot pass one.
#
# Not trusted as a copy: `_assert_contribution_twins` checks each pair against the engine's own
# table at import, because a table that silently stopped agreeing would re-serve the wrong tensor.
_BLOCK_LEVEL_CONTRIBUTION: dict[str, str] = {"mlp_out": "mlp_out_post", "attn_out": "attn_out_post"}


def _assert_contribution_twins() -> None:
    """Fail at import if the engine no longer pairs these points under one TransformerLens name."""
    for raw, post in _BLOCK_LEVEL_CONTRIBUTION.items():
        block_level = point_to_tlens_hook(post, 0)
        if engine_tlens_hook_to_point(block_level) != Address(raw, 0):
            raise AssertionError(
                f"interp-engine no longer maps {block_level!r} to {raw!r} without a model, so "
                f"{post!r} is not {raw!r}'s contribution twin under one hook name. The mapping in "
                f"`engine_adapter._BLOCK_LEVEL_CONTRIBUTION` has to be rechecked against "
                f"`interp_engine.mappers` before SAEs on a sandwich-norm model can be served."
            )


_assert_contribution_twins()


def tlens_hook_to_point(hook_name: str) -> Address:
    """Map an SAE's TransformerLens hook name to an engine :class:`~interp_engine.Address`.

    Delegates to ``interp_engine.mappers``, which owns the vocabulary for both frameworks, then
    resolves the one ambiguity that mapper needs a model to settle. Raises ``UnmappedHook``, a
    ``ValueError`` subclass, so existing handlers are unaffected.

    **TransformerLens' block-level ``hook_mlp_out`` / ``hook_attn_out`` resolve to
    ``mlp_out_post`` / ``attn_out_post`` here, on every architecture.** Those hooks are the
    sublayer's residual *contribution*, which is what the ``*_post`` points are defined as, so on a
    sandwich-norm model (gemma-2/3/4, OLMo-2/3) they fire after the post-sublayer norm. The
    submodule spellings ``mlp.hook_out`` / ``attn.hook_out`` are the raw module outputs and are
    unaffected.

    Gemma Scope's MLP SAEs settle which of the two they read, and it is the contribution: on
    ``gemma-2-2b`` layer 4, ``gemmascope-mlp-16k`` reconstructs ``mlp_out_post`` at FVU 0.26 with an
    L0 of 81 against the SAE's declared 85, and ``mlp_out`` at FVU 9.8 -- worse than predicting the
    mean -- with an L0 of 8. Read off the raw output the whole source is silently dead: the feature
    whose dashboard tops out at 23.5 on "mass-production" fires nowhere in that text at all, which is
    how this was found. Both columns are pinned in
    ``tests/integration/test_gemma_mlp_sae_hook.py`` (marked ``xl``, so run it deliberately).

    **Deliberately does not pass the model**, even though the engine mapper is model-aware, because
    ``mappers.has_sandwich_norms`` reads a flag detected on a real module tree and the vLLM client
    holds no modules -- they live in the worker processes. Asking the ``*_post`` point instead needs
    no such branch and is right on either backend: both resolvers alias it to the raw output where
    the architecture has no post-sublayer norm, so this is a no-op on gpt2 and llama rather than a
    reading they have to opt out of. See "Post-sublayer (sandwich) norms" in the engine's
    ``docs/ARCHITECTURE_QUIRKS.md``, and ``docs/PORTING.md`` for the two MLP-output names.
    """
    address = engine_tlens_hook_to_point(hook_name)
    post = _BLOCK_LEVEL_CONTRIBUTION.get(address.name)
    # The engine landed on a raw sublayer output. If TransformerLens spells that raw point with some
    # *other* name than the one asked for, the caller passed the block-level hook -- the ambiguous
    # one -- and meant the contribution.
    if post is not None and point_to_tlens_hook(address) != hook_name:
        return Address(post, address.layer)
    return address


# TransformerLens fires `hook_normalized` INSIDE a norm: on `x / scale`, before the norm's learned
# gain. No HuggingFace module outputs that tensor, so `tlens_hook_to_point` refuses the name rather
# than answering with `mlp_in`, which is the same tensor times that gain. Real SAEs are trained
# there -- SAELens gives every GemmaScope transcoder (`gemma-2-2b/*-gemmascope-transcoder-16k`)
# `blocks.{i}.ln2.hook_normalized` as its `hook_name` -- so the engine exports the two halves of the
# recompute (`tlens_normalized_hook` for the point to capture, `pre_gain_normalized` for the
# arithmetic) and what is left here is routing them.
#
# Checked against `HookedTransformer` on gemma-2-2b layer 19: the recompute agrees with TL's hook to
# 2e-3 relative -- the residual difference is `resid_mid` itself, which HF's weights and TL's
# converted ones already disagree on by 8e-4 -- and the transcoder fires on exactly the same 170
# features. Encoding `mlp_in` instead gives cosine 0.89, 287 firing features and an L0 of 19 against
# the SAE's declared 12: plausible numbers for the wrong tensor, which is why this is not that.
#
# EXPECT THE SHIPPING DASHBOARDS TO DISAGREE, and do not change this reading to match them. The
# stored dashboards for all 26 `gemma-2-2b/*-gemmascope-transcoder-16k` sources were generated by
# encoding the UNNORMALIZED `resid_mid`, so they read a median 12x higher than the server does:
# `19-gemmascope-transcoder-16k/0` peaks at 409.74 on its own top text where this reads 45. That is
# the wrong tensor rather than an alternative convention -- it reconstructs the transcoder's target
# at an FVU of 6.8e4 against 0.50, at an L0 of 664 against the SAE's declared 12 -- and the
# dashboards' own published density corroborates it: 2.194% for that feature, where the trained
# input gives 0.024%. `scripts/gemma_transcoder_hook_check.py` is the reproduction, and the fix is
# regenerating those sources, not moving this. The difference is also not a rescale a stored
# dashboard could be corrected by: only 39% of features keep the same top-activating token, and
# 6,646 of the 16,384 fire only in the dashboards' reading.


class _CapturePoint(NamedTuple):
    """One requested hook name: where to capture, and what is still owed afterwards."""

    hook: str
    address: Address
    # True for a `ln{1,2}.hook_normalized`, where `address` is the norm's input rather than the
    # tensor the caller asked for, so the normalization still has to be applied.
    normalize: bool


def _capture_points(hook_names: list[str]) -> list[_CapturePoint]:
    """Resolve each SAE hook name to a point this engine can capture."""
    points: list[_CapturePoint] = []
    for hook in hook_names:
        normalized_input = tlens_normalized_hook(hook)
        if normalized_input is None:
            points.append(_CapturePoint(hook, tlens_hook_to_point(hook), normalize=False))
        else:
            points.append(_CapturePoint(hook, normalized_input, normalize=True))
    return points


def _native_resid_layers(points: list[_CapturePoint]) -> list[int]:
    """The layer of each point, for a set :func:`_vllm_points_use_native_resid` has accepted.

    That guard already requires a layer on every point, but it returns a bool, so the narrowing
    does not survive the call and ``Address.layer`` stays ``int | None`` at each use. Restating
    the invariant once here is what keeps the four call sites from casting it away individually,
    and turns a would-be ``None`` into this module's own error rather than a ``TypeError`` deep in
    a dict lookup.
    """
    layers: list[int] = []
    for point in points:
        if point.address.layer is None:
            raise BackendUnsupported(f"native resid capture needs a layer, got {point.address}")
        layers.append(int(point.address.layer))
    return layers


# One model per process, so this holds one entry. Memoized because resolving it walks a config, and
# a batch endpoint asks once per hook per request.
_RMS_NORM_EPS: dict[str, float] = {}


def _rms_norm_eps(model: object) -> float:
    """This model's RMS-norm epsilon, or a refusal if its norms are not RMS norms.

    A model that declares none is a LayerNorm family, where TransformerLens' ``hook_normalized``
    subtracts the mean as well. That is a third tensor, not the one `pre_gain_normalized` computes,
    so this raises rather than normalizing as if the centering were not there. No pod is in that
    position today -- every SAE set on a LayerNorm model reads a residual point -- which is why the
    refusal is a raise rather than a fallback.
    """
    hf_model_id = str(getattr(model, "hf_model_id", "") or type(model).__name__)
    if hf_model_id in _RMS_NORM_EPS:
        return _RMS_NORM_EPS[hf_model_id]
    eps = rms_norm_eps_for_model(model)
    if eps is None:
        raise UnmappedHook(
            f"{type(model).__name__} declares no `rms_norm_eps`, so its norms are not RMS norms. "
            "TransformerLens' `hook_normalized` centers the input on such a model, which is a "
            "different tensor from the one this reproduces."
        )
    _RMS_NORM_EPS[hf_model_id] = eps
    return eps


def _finish(point: _CapturePoint, tensor: torch.Tensor, model: object) -> torch.Tensor:
    return pre_gain_normalized(tensor, _rms_norm_eps(model)) if point.normalize else tensor


def _as_batched(tokens: torch.Tensor) -> torch.Tensor:
    return tokens if tokens.ndim == 2 else tokens.unsqueeze(0)


async def capture_cache_async(model: object, tokens: torch.Tensor, hook_names: list[str]) -> dict[str, torch.Tensor]:
    """Backend-aware capture -> ``{hook_name: tensor[1, seq, d]}`` for any loaded backend.

    EagerModel captures eagerly via ``run_with_cache``; ``VLLMModel`` captures via the
    engine-owned worker forward-hooks (``model.capture``). vLLM points that are not yet at
    eager parity raise :class:`BackendUnsupported`. The returned dict matches exactly what the
    existing ``process_*_activations`` helpers consume, so endpoints stay backend-agnostic.
    """
    points = _capture_points(hook_names)
    if isinstance(model, EagerModel):
        cache = run_with_cache(model, _as_batched(tokens), [point.address for point in points])
        return {point.hook: _finish(point, cache[point.address], model) for point in points}

    if isinstance(model, VLLMModel):
        if _vllm_points_use_native_resid(model, points):
            token_ids = _as_batched(tokens)[0].tolist()
            layers = _native_resid_layers(points)
            resid = await model.capture_resid_post(token_ids, layers)
            return {
                p.hook: _finish(p, resid[layer].unsqueeze(0), model) for p, layer in zip(points, layers, strict=True)
            }
        _assert_vllm_points_supported(model, points)
        token_ids = _as_batched(tokens)[0].tolist()
        raw = await model.capture(token_ids, [point.address for point in points])
        return {point.hook: _finish(point, raw[point.address].unsqueeze(0), model) for point in points}

    raise BackendUnsupported(f"Unsupported model backend for capture: {type(model).__name__}")


async def capture_activation_async(model: object, tokens: torch.Tensor, hook_name: str) -> torch.Tensor:
    """Single-point backend-aware capture -> tensor ``[1, seq, d]`` (see :func:`capture_cache_async`)."""
    cache = await capture_cache_async(model, tokens, [hook_name])
    return cache[hook_name]


async def capture_padded_cache_async(
    model: object,
    padded_tokens: torch.Tensor,
    original_lengths: list[int],
    hook_names: list[str],
) -> dict[str, torch.Tensor]:
    """Backend-aware batched capture -> ``{hook_name: tensor[batch, max_len, d]}`` (right-padded).

    EagerModel captures the padded batch in one eager forward (pads are causally harmless and
    the caller slices to ``original_lengths``). ``VLLMModel`` has no batched worker
    capture, so it captures each prompt at its true length and scatters into the padded tensor.
    """
    points = _capture_points(hook_names)
    if isinstance(model, EagerModel):
        cache = run_with_cache(model, padded_tokens, [point.address for point in points])
        return {point.hook: _finish(point, cache[point.address], model) for point in points}

    if isinstance(model, VLLMModel):
        native = _vllm_points_use_native_resid(model, points)
        if not native:
            _assert_vllm_points_supported(model, points)
        batch, max_len = int(padded_tokens.shape[0]), int(padded_tokens.shape[1])
        # Hoisted: the points are the same for every prompt in the batch.
        layers = _native_resid_layers(points) if native else []
        per_prompt = []
        for i in range(batch):
            ids = padded_tokens[i, : original_lengths[i]].tolist()
            if native:
                resid = await model.capture_resid_post(ids, layers)
                per_prompt.append({p.address: resid[layer] for p, layer in zip(points, layers, strict=True)})
            else:
                per_prompt.append(await model.capture(ids, [point.address for point in points]))
        out: dict[str, torch.Tensor] = {}
        for point in points:
            sample = per_prompt[0][point.address]
            t = torch.zeros(batch, max_len, sample.shape[-1], dtype=sample.dtype)
            for i in range(batch):
                cap = per_prompt[i][point.address]
                t[i, : cap.shape[0]] = cap
            # Normalized after the scatter, not before: the padded rows are zeros, and a norm of a
            # zero row is 0/sqrt(eps) = 0, so they stay the pads the callers slice away.
            out[point.hook] = _finish(point, t, model)
        return out

    raise BackendUnsupported(f"Unsupported model backend for capture: {type(model).__name__}")


def _get_safe_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype == torch.float16 else dtype


class DfaResult(TypedDict):
    """DFA for one feature at one destination position, as the response models consume it."""

    dfa_values: list[float]
    dfa_target_index: int
    dfa_max_value: float


# (value, attn_probs, dims) for one layer, memoized per layer by the batch endpoints.
DfaInputs = tuple[torch.Tensor, torch.Tensor, dict[str, int]]


def _dfa_compute_device(*tensors: torch.Tensor) -> torch.device:
    """Where to run the DFA math: an accelerator if any operand is already on one.

    The operands genuinely disagree on the vLLM backend: value/attn_probs come back from
    the worker as CPU tensors while the SAE weights stay on the serving GPU, so the GPU is
    the side to converge on whenever there is one.
    """
    for tensor in tensors:
        if tensor.device.type != "cpu":
            return tensor.device
    return tensors[0].device


def dfa_from_v_and_probs(
    v: torch.Tensor,
    attn_weights: torch.Tensor,
    W_enc: torch.Tensor,
    index: int,
    max_value_index: int,
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> DfaResult:
    """Shared DFA math: per-head value (GQA-aware) x attention pattern x W_enc.

    ``v`` is ``[batch, src_pos, n_kv_heads, head_dim]`` and ``attn_weights`` is
    ``[batch, n_heads, dest_pos, src_pos]`` -- the same shapes whether sourced from the
    eager engine (``per_head_value`` + ``attn_probs``) or from the vLLM off-kernel
    recompute (``capture_attention``). Mirrors the TransformerLens ``calculate_dfa`` math.

    Only the ``max_value_index`` destination row is ever returned, and the encoder
    direction is fixed, so this contracts ``head_dim`` FIRST and slices that one row out of
    the attention pattern. Peak memory is ``O(src_pos * n_heads)``. Forming the natural
    ``attn_weights * v`` product before the contraction instead costs
    ``O(src_pos^2 * d_model)``, which on a 550-token prompt against a gemma-2-2b attention
    SAE is ~5.6 GiB per feature -- and ``/activation/all`` pays it once per result.
    """
    op_dtype = max(
        _get_safe_dtype(v.dtype),
        _get_safe_dtype(attn_weights.dtype),
        _get_safe_dtype(W_enc.dtype),
        key=lambda x: x.itemsize,
    )
    device = _dfa_compute_device(W_enc, v, attn_weights)

    v = v.to(device=device, dtype=op_dtype)
    attn_weights = attn_weights.to(device=device, dtype=op_dtype)

    # GQA: expand kv-heads up to query-heads (no-op when n_kv == n_heads). Cheap here --
    # `v` is only [batch, src_pos, n_heads, head_dim].
    if n_kv_heads < n_heads:
        expansion_factor = attn_weights.shape[1] // v.shape[2]
        v = v.repeat_interleave(expansion_factor, dim=2)

    # DFA sources are `hook_z` SAEs, whose d_in is the concatenated per-head attention
    # output, so W_enc's rows split evenly into (n_heads, head_dim). Checked rather than
    # assumed: a silent reshape here would return plausible-looking wrong numbers.
    if W_enc.shape[0] != n_heads * head_dim:
        raise ValueError(
            f"DFA expects a hook_z encoder with d_in = n_heads * head_dim "
            f"({n_heads} * {head_dim} = {n_heads * head_dim}), got {W_enc.shape[0]}"
        )
    W_enc_index = W_enc[:, index].to(device=device, dtype=op_dtype)
    w_per_head = W_enc_index.view(n_heads, head_dim)

    # Project each head's value vector onto this feature's encoder direction ...
    z_proj = einops.einsum(
        v,
        w_per_head,
        "batch src_pos n_heads d_head, n_heads d_head -> batch src_pos n_heads",
    )
    # ... then weight each head by its attention from the destination position.
    attn_to_dest = attn_weights[:, :, max_value_index, :]
    per_src_pos_dfa = einops.einsum(
        attn_to_dest,
        z_proj,
        "batch n_heads src_pos, batch src_pos n_heads -> batch src_pos",
    )
    dfa_values = per_src_pos_dfa[0].tolist()
    return {
        "dfa_values": dfa_values,
        "dfa_target_index": max_value_index,
        "dfa_max_value": max(dfa_values),
    }


async def capture_dfa_inputs(model: object, tokens: torch.Tensor, layer_num: int) -> DfaInputs:
    """Backend-aware ``(value, attn_probs, dims)`` for DFA at ``layer_num``.

    Returns ``value [1, src, n_kv, head_dim]``, ``attn_probs [1, n_heads, dest, src]`` and a
    ``dims`` dict (n_heads/n_kv_heads/head_dim). EagerModel captures eagerly (value + attn_probs);
    the vLLM backend uses the off-kernel ``capture_attention`` recompute. Callers memoize this
    per layer and reuse across features (:func:`dfa_from_v_and_probs`).
    """
    if isinstance(model, EagerModel):
        cache = run_with_cache(
            model,
            _as_batched(tokens),
            [("value", layer_num), ("attn_probs", layer_num)],
        )
        v = per_head_value(model, cache, layer_num)  # [batch, src, n_kv, head_dim]
        attn = cache.get("attn_probs", layer_num)  # [batch, n_heads, dest, src]
        dims = {
            "n_heads": model.n_heads,
            "n_kv_heads": model.n_kv_heads,
            "head_dim": model.head_dim,
        }
        return v, attn, dims

    if isinstance(model, VLLMModel):
        assert_hooks_available(model, "DFA")
        reason = vllm_attention_unsupported_reason(model)
        if reason is not None:
            raise BackendUnsupported(f"DFA is not available on this instance: {reason}.")
        ids = _as_batched(tokens)[0].tolist()
        res = await model.capture_attention(ids, [layer_num])
        v = res[layer_num]["value"].unsqueeze(0)  # [1, src, n_kv, head_dim]
        attn = res[layer_num]["probs"].unsqueeze(0)  # [1, n_heads, dest, src]
        ad = model._attn_dims  # type: ignore[attr-defined]
        dims = {
            "n_heads": ad["n_heads"],
            "n_kv_heads": ad["n_kv_heads"],
            "head_dim": ad["head_dim"],
        }
        return v, attn, dims

    raise BackendUnsupported(f"DFA not supported on backend {type(model).__name__}")


async def calculate_dfa(
    model: object,
    sae: object,
    layer_num: int,
    index: int,
    max_value_index: int,
    tokens: torch.Tensor,
) -> DfaResult:
    """Backend-aware DFA (EagerModel or vLLM off-kernel recompute).

    Both indices are in the coordinates of ``tokens`` -- position 0 is whatever position 0 of
    the forward pass was, BOS included. An endpoint that trims its response arrays wants
    :func:`calculate_dfa_for_values` instead.
    """
    v, attn, dims = await capture_dfa_inputs(model, tokens, layer_num)
    return dfa_from_v_and_probs(
        v,
        attn,
        sae.W_enc,  # type: ignore[attr-defined]
        index,
        max_value_index,
        **dims,
    )


async def calculate_dfa_for_values(
    model: object,
    sae: object,
    layer_num: int,
    index: int,
    max_value_index: int,
    tokens: torch.Tensor,
    *,
    n_values: int,
) -> DfaResult:
    """DFA in the coordinates of a response's ``values``, which may be shorter than ``tokens``.

    There are two coordinate systems here, and conflating them is silent rather than loud.
    The attention pattern's dest/src axes are indexed by the FORWARD PASS, so position 0 is
    BOS. But `/activation/single` drops the leading BOS from ``values`` before reporting
    ``max_value_index``, so that index is one SHORT of the row it names.

    Left uncorrected, DFA is attributed to the position before the one that fired, and the
    ``dfa_values`` array comes back one longer than the ``tokens`` it is rendered against --
    which is how a webapp that reads ``dfaValues[i]`` for token ``i`` ends up highlighting the
    neighbouring token as the DFA source. Both halves were visible on
    `gemma-2-2b/10-gemmascope-att-16k/0`, whose shipping dashboard attributes 3.6 to a ','
    where the server returned 2.9 attributed to the ' .' beside it.

    ``n_values`` is the length of the array the caller will return. The offset is derived from
    it rather than from a BOS check, so an endpoint that trims nothing -- ``single-batch`` and
    ``all`` both keep BOS in ``values`` -- gets an exact no-op.
    """
    offset = len(tokens) - n_values
    if offset < 0:
        raise ValueError(
            f"DFA cannot align to a values array longer than its tokens: {n_values} values "
            f"against {len(tokens)} tokens."
        )
    dfa = await calculate_dfa(model, sae, layer_num, index, max_value_index + offset, tokens)
    if offset == 0:
        return dfa

    dfa_values = dfa["dfa_values"][offset:]
    if len(dfa_values) != n_values:
        raise ValueError(
            f"DFA returned {len(dfa['dfa_values'])} source positions for {len(tokens)} tokens; "
            f"trimming {offset} leaves {len(dfa_values)} against {n_values} values."
        )
    # `dfa_max_value` is recomputed rather than carried over: BOS is an attention sink, so it
    # can hold the largest attribution in the untrimmed array and would then name a position
    # no longer in it.
    return DfaResult(
        dfa_values=dfa_values,
        dfa_target_index=dfa["dfa_target_index"] - offset,
        dfa_max_value=max(dfa_values),
    )
