"""Wire models for ``/v1/lens/*``.

Hand-written rather than generated, so the annotations stay loose (see the note in
``activation.py``). Only the request side lives here; the streamed NDJSON frames stay in
``endpoints/lens/prompt.py`` next to the generator that emits them, because they are not a
response body FastAPI can document.
"""

from enum import StrEnum

from neuronpedia_inference.schemas.common import BaseSchema


class LensType(StrEnum):
    """Which lens's readout direction to use."""

    LOGIT_LENS = "LOGIT_LENS"
    JACOBIAN_LENS = "JACOBIAN_LENS"


class LensChatMessage(BaseSchema):
    """One message of a chat-formatted lens prompt."""

    role: str
    content: str


class LensSteerToken(BaseSchema):
    """A single readout to steer on.

    ``token`` is the EXACT decoded token string (whitespace preserved, e.g.
    ``" cat"``) as it appeared in a read-out slice; the server resolves it back
    to a vocab id via a cached reverse-decode map. ``type`` selects which lens's
    readout direction to use: ``JACOBIAN_LENS`` uses the J-lens direction
    ``J_bar_l^T @ w_t`` at each fitted layer (the residual-space direction whose
    J-lens readout is this token), ``LOGIT_LENS`` uses the plain unembedding
    direction ``w_t``.
    """

    token: str
    type: LensType


class LensPromptRequest(BaseSchema):
    """Everything one lens run needs: what to read out, where, and how to intervene."""

    model: str
    # One or more lens types to compute. When both are requested, the model is
    # run only once (the residuals are shared), so adding LOGIT_LENS alongside
    # JACOBIAN_LENS is essentially free.
    type: list[LensType]
    # Provide exactly one of `prompt` (raw text) or `chat` (chat-formatted).
    prompt: str | None = None
    chat: list[LensChatMessage] | None = None
    top_n: int = 10
    # Layers to read out. Empty (default) = all available layers for the lens
    # type. The model's final layer is ALWAYS included (decoded directly as the
    # model's true output), regardless of this list.
    layers: list[int] = []
    max_seq_len: int | None = None
    prepend_bos: bool = True
    # Whether to enable "thinking" mode when applying a chat template (only
    # relevant for `chat` requests on models whose chat template supports it).
    enable_thinking: bool = False
    # Whether to preserve historical reasoning (`<think>`) blocks when applying a
    # chat template (only relevant for `chat` requests on models whose chat
    # template supports it, e.g. Qwen3.6). Keeping historical think blocks (the
    # default) stabilizes the chat-formatted token prefix across turns: without
    # it, templates strip prior `<think>` blocks from history, shifting token
    # positions every turn.
    preserve_thinking: bool = True
    # Stream results as NDJSON (one message per line). When false, the identical
    # path runs and all messages are buffered into a single JSON object.
    stream: bool = True
    # Sampling temperature for generated tokens. 0 = greedy (argmax).
    temperature: float = 1.0
    # Number of tokens to generate after the prompt. 0 = lens over the prompt
    # only (no generation).
    num_completion_tokens: int = 0
    # Token ids the client already has lens read-outs for (the previous
    # response's prompt + generated tokens, in order). The server computes the
    # longest common prefix with the freshly tokenized prompt and skips the
    # (expensive) per-layer read-out + emission for those positions, so a
    # follow-up turn only recomputes the new tokens. Position reuse is validated
    # by token id, so a divergent prefix simply reuses less (never wrong).
    cached_token_ids: list[int] = []
    # Exact input token ids to read out over, bypassing tokenization. When
    # provided (non-empty), `prompt`/`chat` are ignored, generation is disabled
    # (``num_completion_tokens`` is forced to 0), and the read-out runs over
    # exactly these ids. Used to faithfully reproduce a previously-computed run
    # (e.g. a shared jlens link) without depending on chat-template / tokenizer
    # drift — the lens read-out is a deterministic function of the token ids.
    input_token_ids: list[int] = []
    # Steering: readouts to additively inject (negatively, to suppress) into the
    # residual stream at every position, during prefill AND generation. Empty
    # (default) = no steering. When steering is active, prefix-reuse is disabled
    # (the cached read-outs from an unsteered run are no longer valid).
    steer_tokens: list[LensSteerToken] = []
    # Layers to inject the steering direction at. Empty = the read-out layers.
    steer_layers: list[int] = []
    # Signed steering strength as a fraction of each position's residual norm
    # (negative suppresses the readout). 0 = no steering.
    steer_strength: float = 0.0
    # When true, ABLATE the readout direction: project it out of the residual at
    # every steered layer/position (h <- h - (h.d_hat) d_hat) instead of
    # additively steering. Mutually exclusive with ``steer_strength`` (which is
    # ignored when ablating).
    steer_ablate: bool = False
    # SWAP: when set, replace the source readout (``steer_tokens[0]``) with this
    # target readout at every steered layer/position. The residual's projection
    # onto the source direction is removed and re-added (same magnitude) along
    # the target direction: ``h <- h - (h.s_hat) s_hat + (h.s_hat) t_hat``. This
    # is the causal "lens-vector swap" intervention and takes precedence over
    # ``steer_strength`` / ``steer_ablate`` when present. ``type`` should match
    # the source readout's lens type.
    swap_token: LensSteerToken | None = None
    # Whether to apply the steer/swap intervention to GENERATED tokens too. When
    # false (default), the intervention is applied only to the prompt positions
    # (prefill); generation then proceeds from the steered prompt context but the
    # newly generated positions are not themselves steered/swapped. When true,
    # the intervention is also applied at each generated position as it is
    # produced.
    steer_generated_tokens: bool = False
    # Whether to drop "non-word" tokens (punctuation / whitespace / symbol /
    # special tokens) from each position's per-layer read-out BEFORE selecting
    # the top-n, so the returned tokens are predominantly interesting word
    # tokens. The model's TRUE top-1 (output) token at each layer is always
    # preserved even when it is non-word. Probabilities are computed over the
    # FULL vocab (filtering only changes WHICH tokens are selected, not their
    # reported probabilities). Defaults to True.
    filter_non_word_tokens: bool = True
    # When true, if the server is already processing another request (the global
    # model lock is held), return HTTP 429 immediately instead of queueing and
    # waiting for the lock. This lets a client (e.g. the webapp) fail over to a
    # different inference server for this model. Defaults to False, preserving
    # the original behavior of waiting up to REQUEST_LOCK_TIMEOUT for the lock.
    fail_if_busy: bool = False
