"""Per-endpoint estimates of a request's peak GPU working set, in bytes.

These feed the admission controller (:class:`neuronpedia_inference.shared.VramBudget`), which
holds a request's estimate for the duration of the call. The point is that requests are not
interchangeable: a lens call and an all-layers activation search over 1M-feature SAEs differ
by orders of magnitude, so a flat cap on the NUMBER of concurrent requests either throttles
the cheap ones needlessly or lets the expensive ones OOM. Sizing admission from the request's
own parameters is also what makes a newly added SAE self-throttling -- a wider source raises
its own estimated cost, with no configuration to update.

Three properties matter more than precision:

- Derived from the request, not configured. A new SAE set changes the numbers by itself.
- Conservative. Over-estimating costs throughput; under-estimating costs the process.
- Cheap. This runs before every budgeted request, so no allocation and no device queries.

Accuracy is deliberately uneven. The activation endpoints get real formulas because that is
where the unbounded growth was (source count x d_sae x tokens), and anything that recomputes an
attention pattern gets one because that term is quadratic in the token count -- which covers
`/activation/attention` and, less obviously, every activation endpoint that can run DFA.
Endpoints whose cost is essentially fixed get a flat reservation, which is enough to stop many
of them piling up together.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from neuronpedia_inference.config import Config
from neuronpedia_inference.sae_manager import SAEManager

logger = logging.getLogger(__name__)

_DTYPE_BYTES = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "float8": 1,
    "int8": 1,
}

# Fallbacks for a source whose dims were never recorded (a `neurons` source, or a config
# where load_saes never reached it). Chosen large rather than typical: under-estimating is
# the failure that OOMs.
_FALLBACK_D_SAE = 131_072
_FALLBACK_D_IN = 4096

# Same idea for attention geometry, when the loaded model does not report it.
_FALLBACK_N_HEADS = 64
_FALLBACK_HEAD_DIM = 128

# Slack on every estimate, covering allocator rounding and the small per-op temporaries the
# formulas below do not enumerate individually.
_OVERHEAD_FACTOR = 1.3

# Flat reservations for endpoints whose peak does not scale with request parameters in any
# way worth modelling. Sized to be roughly right, not exact -- their job is to stop a dozen
# of them running concurrently alongside a large activation search.
FLAT_LENS_BYTES = 512 * 1024**2
FLAT_STEER_BYTES = 256 * 1024**2


def _lens_capture_bytes(*, capture_positions: int, n_capture_points: int, d_model: int, n_streams: int) -> int:
    """The captured residuals a lens request holds for its WHOLE sequence.

    The term :func:`lens_cost` used to be missing, and the one that actually OOMs a pod. The
    read-out is chunked, so it was reasonable to treat a lens request as costing a fixed
    amount; the CAPTURE that feeds it is not chunked, and grows with the conversation.

    On the vLLM worker path the harvest lives in ``worker_lens_capture_readout``, which
    concatenates every chunk the hooks produced (``torch.cat(tensors, dim=0)``) and then
    memoizes a stream-reduced copy of each layer beside it. Both are alive while the chunked
    unembed walks them, so a request costs, per captured position:

        n_capture_points x (n_streams + 1) x d_model x model_dtype

    ``n_streams`` is 1 on a conventional trunk, where the reduction is a no-op and returns
    the captured tensor rather than a copy -- so the second term drops out and this is just
    the harvest. It is 4 on a hyper-connection trunk (DeepSeek V4's ``hc_mult``), where a
    capture site is ``n_streams x d_model`` wide and the reduced copy is real. That is what
    makes V4 roughly five times more expensive per token than its width alone suggests:
    43 points x 4 streams x 4096 x 2 B is 1376 KiB per token, plus 344 KiB for the reduction.

    The residual-shipping path (``_iter_residuals_vllm``) accumulates the same sequence's rows
    in the server process instead, already reduced. Charging both at the worker's rate
    over-estimates that path by the stream multiple, which is nothing at ``n_streams`` 1.
    """
    if capture_positions <= 0 or n_capture_points <= 0 or d_model <= 0:
        return 0
    streams = max(1, n_streams)
    # The reduced copy only exists where the reduction has something to do.
    widths = streams + (1 if streams > 1 else 0)
    return n_capture_points * capture_positions * widths * d_model * _model_dtype_bytes()


def lens_cost(
    *,
    staged_positions: int,
    layer_counts: Sequence[int],
    d_model: int,
    capture_positions: int = 0,
    n_capture_points: int = 0,
    n_streams: int = 1,
) -> int:
    """`/v1/lens/prompt`: the captured residuals, plus the staged read-out rows.

    The capture term (:func:`_lens_capture_bytes`) dominates on any real conversation and is
    the one that decides how many of these can run at once. It defaults to zero so a caller
    that cannot determine the capture geometry gets the old behaviour rather than a crash.

    Because it scales with sequence length, a near-limit request on a wide model can estimate
    above the entire budget and be refused as ``RequestTooLarge``. That is the intended
    answer, for the same reason it is in :func:`attention_cost`: such a request was previously
    admitted on the hope that it fit, and on DeepSeek-V4-Flash at 4096 tokens it did not --
    it OOMed inside ``collective_rpc`` partway through the second turn of a conversation.

    The rest is the staged rows, which are device memory.

    A flat reservation was right while the vLLM path staged these rows in host RAM. They
    are on the device now -- that is what moved the Jacobian transport off the CPU -- so the
    peak scales with the request: one ``[batch, n_layers, d_model]`` float32 block per
    requested lens type held at once, plus a transient second copy of whichever type is
    being stacked, because ``torch.stack`` allocates its result while the per-layer blocks
    are still alive.

    At Qwen3.6-27B's 64 layers x 5120 that is 160 MiB per type, so a request asking for both
    peaks near 500 MiB against the 512 MiB flat figure this replaced -- which at
    ``max_concurrent=8`` was admitting 4 GiB of claims for something closer to 5 GiB of use.

    That batch is the EAGER backend's. vLLM now stages in the worker instead, a read-out
    chunk at a time, so its caller passes that smaller number and lands on the floor -- which
    is the right answer there, since what dominates a vLLM lens request is the worker's
    vocab-sized decode chunk, and that is exactly what the flat figure was sized for.

    Floored at the old flat value so this can only ever reserve MORE than before: the rows
    are not the whole working set (the eager backend also holds a vocab-sized decode chunk),
    and that flat number was sized with the rest of it in mind.

    Takes its inputs rather than a request because they are derived, not declared: the
    staging batch is a read-out constant and the layer counts come from resolving "all
    layers for this lens type" against what the lens was actually fit on.
    """
    capture = _lens_capture_bytes(
        capture_positions=capture_positions,
        n_capture_points=n_capture_points,
        d_model=d_model,
        n_streams=n_streams,
    )
    per_type = [staged_positions * n_layers * d_model * _DTYPE_BYTES["float32"] for n_layers in layer_counts]
    if not per_type:
        return max(FLAT_LENS_BYTES, int(_OVERHEAD_FACTOR * capture))
    # Every type's staged block is alive together; one more is in flight while it is stacked.
    modelled = int(_OVERHEAD_FACTOR * (capture + sum(per_type) + max(per_type)))
    return max(FLAT_LENS_BYTES, modelled)


def steer_cost(request) -> int:  # type: ignore[no-untyped-def] # noqa: ARG001
    """`/steer/completion*`: a flat reservation.

    Steering hooks add a delta per forward; the generation itself runs inside vLLM's own
    (already reserved) pool, so the marginal cost out here is the loaded steer vectors and a
    handful of per-layer temporaries, none of which scale with the request in a way worth
    modelling. The reservation exists so several concurrent steers cannot silently consume the
    headroom an activation search was admitted against.
    """
    return FLAT_STEER_BYTES


def _sae_dtype_bytes() -> int:
    return _DTYPE_BYTES.get(Config.get_instance().sae_dtype, 4)


def _model_dtype_bytes() -> int:
    return _DTYPE_BYTES.get(Config.get_instance().model_dtype, 2)


def estimate_tokens(text: str | None) -> int:
    """Upper bound on the token count of `text`, without tokenizing it.

    A token is at least one character, so the character count bounds it, and the activation
    token limit bounds it again (activation endpoints are the ones that cost from this
    estimate). Loose for ordinary English (roughly 4x) but never low, which is the
    direction that matters -- and cheap, which lets this run inline on the request path
    before the prompt has been tokenized.
    """
    config = Config.get_instance()
    limit = getattr(config, "activation_token_limit", None) or config.token_limit
    if not text:
        return 1
    return max(1, min(int(limit), len(text)))


def _widest_source_dims(sources: list[str]) -> tuple[int, int]:
    """(widest d_sae, widest d_in) over `sources`.

    The streaming reduction in /activation/all encodes one source at a time, so the widest
    single source sets the peak rather than the sum over all of them.
    """
    manager = SAEManager.get_instance()
    d_saes = [manager.get_d_sae(s) or _FALLBACK_D_SAE for s in sources]
    d_ins = [manager.get_d_in(s) or _FALLBACK_D_IN for s in sources]
    return (
        max(d_saes) if d_saes else _FALLBACK_D_SAE,
        max(d_ins) if d_ins else _FALLBACK_D_IN,
    )


def _capture_bytes(sources: list[str], n_tokens: int, batch: int = 1) -> int:
    """The capture cache: one [batch, n_tokens, d_in] tensor per DISTINCT hook.

    This is the one term that does still grow with the number of selected sources, because
    all the hooks are captured in a single forward before any of them is encoded. It is small
    next to what a single encode used to cost, but with 26+ sources it is not nothing.
    """
    manager = SAEManager.get_instance()
    widths_by_hook: dict[str, int] = {}
    for source in sources:
        try:
            hook = manager.get_sae_hook(source)
        except Exception:  # noqa: BLE001 - unknown source; the endpoint will reject it
            hook = source
        widths_by_hook[hook] = max(widths_by_hook.get(hook, 0), manager.get_d_in(source) or _FALLBACK_D_IN)
    return batch * n_tokens * sum(widths_by_hook.values()) * _model_dtype_bytes()


def _encode_bytes(d_sae: int, n_tokens: int) -> int:
    """One source's encode. Doubled: the [n_tokens, d_sae] output plus the transient the
    activation function and the top-K row gather need alongside it."""
    return 2 * d_sae * n_tokens * _sae_dtype_bytes()


def activation_all_cost(request) -> int:  # type: ignore[no-untyped-def]
    """`/activation/all`: capture every selected hook, then encode one source at a time.

    Since the reduction became a streaming top-K, the encode term is the WIDEST source rather
    than the sum over all of them -- which is what makes an all-layers search over the full
    source set affordable at all.

    On an attention source set the DFA term dominates all of that: the per-layer memo is never
    evicted, so a search that returns results from many layers holds a quadratic pattern for
    each of them at once.
    """
    sources = _selected_sources(request)
    n_tokens = estimate_tokens(getattr(request, "prompt", None))
    d_sae, _ = _widest_source_dims(sources)
    num_results = max(1, int(getattr(request, "num_results", 1) or 1))

    # Running top-K, the incoming candidate block, and the concatenation of the two.
    result_bytes = 4 * num_results * (n_tokens + 8) * 4

    return int(
        _OVERHEAD_FACTOR
        * (
            _capture_bytes(sources, n_tokens)
            + _encode_bytes(d_sae, n_tokens)
            + result_bytes
            # One memoized capture per layer represented in the results, so `num_results`
            # bounds it as tightly as the source count does.
            + _dfa_bytes(sources, n_tokens, layers_held=num_results)
        )
    )


def activation_all_batch_cost(request) -> int:  # type: ignore[no-untyped-def]
    """`/activation/all-batch`: one padded capture for the batch, then per prompt as above."""
    sources = _selected_sources(request)
    prompts = list(getattr(request, "prompts", None) or [])
    batch = max(1, len(prompts))
    # The padded capture is sized by the LONGEST prompt in the batch.
    n_tokens = max((estimate_tokens(p) for p in prompts), default=1)
    d_sae, _ = _widest_source_dims(sources)
    num_results = max(1, int(getattr(request, "num_results", 1) or 1))

    result_bytes = 4 * num_results * (n_tokens + 8) * 4

    return int(
        _OVERHEAD_FACTOR
        * (
            _capture_bytes(sources, n_tokens, batch=batch)
            + _encode_bytes(d_sae, n_tokens)
            + result_bytes
            # DFA runs per prompt against that prompt's own tokens, and the batch path drops
            # the memo between prompts, so the peak is the longest prompt's worth -- not the
            # batch's.
            + _dfa_bytes(sources, n_tokens, layers_held=num_results)
        )
    )


def activation_single_cost(request) -> int:  # type: ignore[no-untyped-def]
    """`/activation/single`: one hook, one encode, plus one attention pattern if DFA is on.

    That last term is quadratic in the token count, so on an attention source it is the whole
    estimate for anything but a short prompt.
    """
    source = getattr(request, "source", None)
    sources = [source] if source else []
    n_tokens = estimate_tokens(getattr(request, "prompt", None))
    d_sae, _ = _widest_source_dims(sources)
    return int(
        _OVERHEAD_FACTOR
        * (_capture_bytes(sources, n_tokens) + _encode_bytes(d_sae, n_tokens) + _dfa_bytes(sources, n_tokens))
    )


def activation_topk_by_token_cost(request) -> int:  # type: ignore[no-untyped-def]
    """`/activation/topk-by-token`: encode, then a [n_tokens, k] value+index pair."""
    source = getattr(request, "source", None)
    sources = [source] if source else []
    n_tokens = estimate_tokens(getattr(request, "prompt", None))
    d_sae, _ = _widest_source_dims(sources)
    top_k = max(1, min(int(getattr(request, "top_k", None) or 1), d_sae))
    # topk emits fp32 values and int64 indices.
    topk_bytes = n_tokens * top_k * 12
    return int(_OVERHEAD_FACTOR * (_capture_bytes(sources, n_tokens) + _encode_bytes(d_sae, n_tokens) + topk_bytes))


def activation_source_cost(request) -> int:  # type: ignore[no-untyped-def]
    """`/activation/source`: padded capture for the batch, but ONE sequence encoded at a time."""
    source = getattr(request, "source", None)
    sources = [source] if source else []
    prompts = list(getattr(request, "prompts", None) or [])
    batch = max(1, len(prompts))
    n_tokens = max((estimate_tokens(p) for p in prompts), default=1)
    d_sae, _ = _widest_source_dims(sources)
    return int(_OVERHEAD_FACTOR * (_capture_bytes(sources, n_tokens, batch=batch) + _encode_bytes(d_sae, n_tokens)))


def _model_width() -> int:
    """``d_model`` of the loaded model, or the fallback when it cannot be reached.

    Same deferred import and same never-raise contract as :func:`_attention_dims`: an
    estimator that threw would fail admission for a request that would have worked.
    """
    from neuronpedia_inference.shared import Model

    try:
        return int(Model.get_instance().d_model) or _FALLBACK_D_IN
    except Exception:  # noqa: BLE001 - advisory only
        return _FALLBACK_D_IN


def activation_raw_cost(request) -> int:  # type: ignore[no-untyped-def]
    """`/activation/raw`: one padded capture per requested layer, and nothing else.

    No SAE is involved, so unlike every other estimator here there is no encode term -- the
    cost is entirely the capture, which scales with the LAYER COUNT rather than with a source
    set. That matters because the default (no `layers` in the request) is every layer at once.
    """
    prompts = list(getattr(request, "prompts", None) or [])
    batch = max(1, len(prompts))
    n_tokens = max((estimate_tokens(p) for p in prompts), default=1)
    layers = list(getattr(request, "layers", None) or [])
    n_layers = len(layers) or Config.get_instance().num_layers or 1
    d_model = _model_width()

    capture = batch * n_tokens * d_model * n_layers * _model_dtype_bytes()
    # The [batch, d_model] fp32 gather per layer, held until the response is built.
    gathered = batch * d_model * n_layers * 4
    return int(_OVERHEAD_FACTOR * (capture + gathered))


def activation_single_batch_cost(request) -> int:  # type: ignore[no-untyped-def]
    """`/activation/single-batch`: padded capture, and the whole padded batch encoded."""
    source = getattr(request, "source", None)
    sources = [source] if source else []
    prompts = list(getattr(request, "prompts", None) or [])
    batch = max(1, len(prompts))
    n_tokens = max((estimate_tokens(p) for p in prompts), default=1)
    d_sae, _ = _widest_source_dims(sources)
    return int(
        _OVERHEAD_FACTOR
        * (
            _capture_bytes(sources, n_tokens, batch=batch)
            + batch * _encode_bytes(d_sae, n_tokens)
            # One prompt's DFA at a time, each result's capture freed as the next is taken.
            + _dfa_bytes(sources, n_tokens)
        )
    )


def activation_topk_by_token_batch_cost(request) -> int:  # type: ignore[no-untyped-def]
    """`/activation/topk-by-token-batch`: padded capture, then one sequence at a time."""
    source = getattr(request, "source", None)
    sources = [source] if source else []
    prompts = list(getattr(request, "prompts", None) or [])
    batch = max(1, len(prompts))
    n_tokens = max((estimate_tokens(p) for p in prompts), default=1)
    d_sae, _ = _widest_source_dims(sources)
    top_k = max(1, min(int(getattr(request, "top_k", None) or 1), d_sae))
    return int(
        _OVERHEAD_FACTOR
        * (_capture_bytes(sources, n_tokens, batch=batch) + _encode_bytes(d_sae, n_tokens) + n_tokens * top_k * 12)
    )


def _attention_dims() -> tuple[int, int, int]:
    """``(n_heads, n_kv_heads, head_dim)`` of the loaded model.

    Attribute reads from the same places ``/activation/attention`` itself reads them: a dict
    on the vLLM backend, properties on the eager one. No device queries, so this is as cheap
    as the rest of this module.

    Falls back to the constants if the model cannot be reached at all. The DFA callers make
    that reachable: unlike ``/activation/attention``, they can be costed before a model is
    loaded, and an estimator that raised there would fail admission rather than the request
    it was sizing. The fallbacks are the large ones, so this errs toward over-estimating.
    """
    # Imported here, not at module scope: shared.py reaches back into this module for the SAE
    # residency estimate, and defers that import for the same reason.
    from neuronpedia_inference.shared import Model

    try:
        model = Model.get_instance()
    except Exception:
        return _FALLBACK_N_HEADS, _FALLBACK_N_HEADS, _FALLBACK_HEAD_DIM
    dims = getattr(model, "_attn_dims", None)
    if isinstance(dims, dict):
        n_heads = int(dims.get("n_heads") or 0)
        n_kv_heads = int(dims.get("n_kv_heads") or 0)
        head_dim = int(dims.get("head_dim") or 0)
    else:
        n_heads = int(getattr(model, "n_heads", 0) or 0)
        n_kv_heads = int(getattr(model, "n_kv_heads", 0) or 0)
        head_dim = int(getattr(model, "head_dim", 0) or 0)
    n_heads = n_heads or _FALLBACK_N_HEADS
    return n_heads, n_kv_heads or n_heads, head_dim or _FALLBACK_HEAD_DIM


# How many ``[n_heads, seq, seq]`` fp32 buffers are live at the recompute's peak. Tracing
# `recompute_attn_probs`: the scores stay alive while the next op writes its own output, so
# plain causal attention peaks at 2, and the two architectures that add a term -- Gemma's
# logit softcap and gpt-oss's sink column -- peak at 3. The eager backend hands back the
# kernel's own probs and peaks lower. `_OVERHEAD_FACTOR` supplies the margin on top.
_ATTN_PATTERN_BUFFERS = 3


def _attention_pattern_bytes(n_tokens: int, *, held: int = 1) -> int:
    """Peak bytes of the attention-pattern capture, holding ``held`` patterns at once.

    One pattern is ``[n_heads, dest, src]`` in fp32: quadratic in the token count and sized by
    ALL of the layer's heads, whatever the caller ends up reading. Alongside it the recompute
    reads post-rope q/k/v and returns a per-head value tensor, which DFA keeps too.

    ``held`` covers the DFA memo, which caches a whole capture per layer. Only the pattern and
    value of a held capture stay alive; the recompute's other buffers belong to whichever
    capture is in flight, so they are counted once rather than per held pattern.
    """
    n_heads, n_kv_heads, head_dim = _attention_dims()
    pattern = n_heads * n_tokens * n_tokens * 4
    value = n_tokens * n_kv_heads * head_dim * 4
    qkv = n_tokens * (2 * n_heads + n_kv_heads) * head_dim * 4
    in_flight = (_ATTN_PATTERN_BUFFERS - 1) * pattern + qkv
    return max(1, held) * (pattern + value) + in_flight


def _dfa_bytes(sources: list[str], n_tokens: int, *, layers_held: int = 1) -> int:
    """Attention patterns the DFA path holds, or 0 when no source in ``sources`` enables it.

    DFA multiplies attention probs by value, so it recomputes the same pattern
    ``/activation/attention`` does -- the quadratic term these estimates used to omit entirely.
    ``layers_held`` bounds how many of those captures can be memoized at once.
    """
    manager = SAEManager.get_instance()
    dfa_sources = [source for source in sources if manager.is_dfa_enabled(source)]
    if not dfa_sources:
        return 0
    return _attention_pattern_bytes(n_tokens, held=min(len(dfa_sources), max(1, layers_held)))


def attention_cost(request) -> int:  # type: ignore[no-untyped-def]
    """`/activation/attention`: quadratic in the token count, across ALL heads.

    Nothing here scales with an SAE, which is why this endpoint needs its own estimate rather
    than a flat reservation: at the activation token limit on a wide model it is among the
    largest single allocations any endpoint makes, and it used to be admitted against a budget
    that knew nothing about it.

    One consequence worth knowing: on a many-headed model with a high token limit, a
    near-limit prompt can estimate above the whole budget and be refused as `RequestTooLarge`.
    That is the intended answer -- `activation_token_limit` is derived from SAE encode sizes
    and has never had this quadratic term in it, so such a request was previously accepted on
    the hope that it fit.
    """
    return int(_OVERHEAD_FACTOR * _attention_pattern_bytes(estimate_tokens(getattr(request, "prompt", None))))


def similarity_matrix_cost(request) -> int:  # type: ignore[no-untyped-def]
    """`/util/similarity-matrix-pred`: quadratic in the token count.

    The [n_tokens, n_tokens] matrix is the whole cost here, and it is the reason this endpoint
    needed a token limit at all.
    """
    source = getattr(request, "sourceId", None)
    sources = [source] if source else []
    n_tokens = estimate_tokens(getattr(request, "text", None))
    d_sae, _ = _widest_source_dims(sources)
    # The similarity matrix plus the normalized predictions it is built from.
    matrix_bytes = 2 * n_tokens * n_tokens * 4
    return int(_OVERHEAD_FACTOR * (_capture_bytes(sources, n_tokens) + _encode_bytes(d_sae, n_tokens) + matrix_bytes))


def request_sources(request) -> list[str]:  # type: ignore[no-untyped-def]
    """Every source a request might read, across the shapes the endpoints use.

    ``selected_sources``/``source_set`` (the activation searches), a bare ``source`` or
    ``sourceId`` (the single-source endpoints), and the per-feature ``source`` on a steer
    request, which can name several sets in one call.
    """
    sources = list(_selected_sources(request))
    for attr in ("source", "sourceId"):
        value = getattr(request, attr, None)
        if isinstance(value, str) and value:
            sources.append(value)
    for feature in getattr(request, "features", None) or []:
        value = getattr(feature, "source", None)
        if isinstance(value, str) and value:
            sources.append(value)
    return sources


def sae_residency_bytes(request) -> int:  # type: ignore[no-untyped-def]
    """GPU bytes of SAE weights this request needs resident at once.

    The largest single source, not their sum: endpoints read one SAE at a time and the
    paging cache only protects the one currently held (see ``sae_cache``). Returns 0 when
    paging is off, since then the weights are permanently resident and cost the request
    nothing.

    Unlike the working-set estimates above this is a MEASURED size, not a formula -- it is
    recorded when the SAE is loaded -- so a newly added set needs no configuration here
    either.
    """
    manager = SAEManager.get_instance()
    return max(
        (manager.get_sae_nbytes(source) for source in request_sources(request)),
        default=0,
    )


def _selected_sources(request) -> list[str]:  # type: ignore[no-untyped-def]
    """The sources a request will actually read.

    An empty `selected_sources` means the whole set -- the all-layers default -- so it has to
    expand here too, or the widest and most expensive requests would be costed as the
    cheapest.
    """
    selected = list(getattr(request, "selected_sources", None) or [])
    if selected:
        return selected
    source_set = getattr(request, "source_set", None)
    if not source_set:
        return []
    return list(SAEManager.get_instance().sae_set_to_saes.get(source_set, []))
