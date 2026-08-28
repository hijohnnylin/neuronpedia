"""Generation for `/steer`, against the interp_engine replacement model.

`/steer` used to be written as if `HookedTransformer` were the only thing a graph pod can hold: it
called `model.generate(...)` and asked that call for `stop_at_eos`, `freq_penalty` and
`return_type="tokens"`, none of which exist anywhere else. `MODEL_ENGINE=interp_engine` is the
default (see `runtime_env`) and is what every pod runs, and there the model is a plain `nn.Module`
wrapping a HuggingFace one, so the endpoint died on the first of those: `AttributeError:
'InterpEngineReplacementModel' object has no attribute 'generate'`.

Removing the attribute error is not enough to steer on that engine, which is why this is a module
rather than one `getattr`. Three further things differ, and each is silent rather than loud:

- `feature_intervention_generate` forwards unrecognized keywords into `transformers`' `generate`,
  which rejects them ("The following `model_kwargs` are not used by the model"). So the request's
  knobs have to be *translated*, not passed along.
- `transformers` starts from the checkpoint's own `generation_config.json`, and several families
  ship sampling defaults there (Qwen3: temperature 0.6, top_p 0.95, top_k 20). The wire contract
  applies none of those, so leaving them would sample from a truncated distribution for a request
  that did not ask for one. `_NEUTRAL_SAMPLING` turns them off.
- It returns the continuation as *text*. `/steer` needs the token ids: it reports one row of top
  logits per token, so it has to know where the boundaries were. Re-tokenizing the text to find
  them is exactly the bug commit bff40b1e ("correctly handle qwen steering") removed, hence
  `_SequenceCollector`.

The request itself is still in TransformerLens' terms, because that is what the steer modal has
always sent and its sliders are calibrated to: `temperature=0` means greedy rather than an error,
and `freq_penalty` subtracts from a logit per earlier occurrence of that token rather than scaling
it. Translating that is most of what this module does.

The other two engines are turned away. `transformerlens` could be supported -- it is where these
keywords come from -- but nothing deploys it, so a second path here would be a second untested
path; a TL pod still generates graphs and fails only on steering. `nnsight` reaches generation
through a tracing context this endpoint never enters, so steering never worked there at all.
"""

from collections.abc import Sequence
from typing import Any

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from transformers.generation.streamers import BaseStreamer

# The sampling knobs `transformers` reads from the checkpoint's `generation_config.json`, set to
# the values that mean "do nothing", so that what `/steer` samples from is decided by the request
# alone. `temperature` is 1.0 because ours is applied inside `_TransformerLensSampling` instead,
# for the ordering reason documented there. Every one of these is ignored unless `do_sample` is on,
# which is why they are applied on that branch alone: `generate` warns about each flag it was given
# and is not going to use, and a line of that per steer request is noise in a pod's logs.
_NEUTRAL_SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": None,
    "typical_p": 1.0,
}


class _TransformerLensSampling(LogitsProcessor):
    """The request's temperature and frequency penalty, in the order `sample_logits` applies them.

    Both in one processor, and the temperature not left to `transformers`, because
    `_get_logits_processor` appends its `TemperatureLogitsWarper` *after* any caller-supplied
    processor (there is a standing TODO in `generation/utils.py` about the ordering).
    TransformerLens divides by the temperature and only then subtracts the penalty, so a separate
    processor would subtract from logits that had not been scaled yet -- a difference the steer
    modal's sliders would show as the penalty quietly changing strength with the temperature.

    `freq_penalty` is a flat subtraction per earlier occurrence of a token, which is not
    `transformers`' `repetition_penalty` -- that one scales a logit instead, so it cannot stand in
    for this. Non-positive penalties are ignored, as `sample_logits` ignores them.
    """

    def __init__(self, temperature: float, freq_penalty: float) -> None:
        self.temperature = temperature
        self.freq_penalty = freq_penalty

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores = scores / self.temperature
        if self.freq_penalty <= 0:
            return scores
        counts = torch.stack([torch.bincount(row, minlength=scores.shape[-1]) for row in input_ids])
        return scores - self.freq_penalty * counts.to(scores.dtype)


class _SequenceCollector(BaseStreamer):
    """Collect the token ids `generate` chose, since the backend only hands back decoded text.

    `streamer` is `generate`'s own way of releasing tokens as they are produced -- the prompt in one
    call, then one call per generated token -- so this reaches through circuit-tracer's `**kwargs`
    without either side having to know about the other. The alternative, re-tokenizing the returned
    string, is what already shifted every logit row by one on Qwen.
    """

    def __init__(self) -> None:
        self._chunks: list[torch.Tensor] = []

    def put(self, value: torch.Tensor) -> None:
        self._chunks.append(value.reshape(-1))

    def end(self) -> None:
        pass

    def sequence(self) -> torch.Tensor:
        """The prompt's tokens followed by the generated ones, as `generate` returns them."""
        assert self._chunks, "generate() finished without handing the streamer any tokens"
        return torch.cat(self._chunks)


def _require_interp_engine(model: Any) -> None:
    """Refuse a model this endpoint cannot generate on, saying which one it got.

    Every circuit-tracer replacement model sets `backend` on the instance, so its absence and its
    value are two different misconfigurations and read as such.
    """
    engine = getattr(model, "backend", None)
    if engine == "interp_engine":
        return
    if engine is None:
        raise NotImplementedError(
            f"/steer needs a circuit-tracer replacement model and got a {type(model).__name__}, "
            "which reports no `backend`. The lm-saes-crm attribution engine loads a model of its "
            "own, and the rest of this handler needs the replacement model's intervention methods "
            "as well as this one."
        )
    raise NotImplementedError(
        f"/steer cannot generate with model engine {engine!r}; it supports 'interp_engine', which "
        "is the default and what every graph pod runs. Graph generation is unaffected, so a pod "
        "started with another engine serves everything but this endpoint."
    )


def _generation_kwargs(max_new_tokens: int, temperature: float, freq_penalty: float) -> dict[str, Any]:
    """The request in `transformers`' terms, sampling the way the wire contract means."""
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        # Honored whether or not sampling is on, unlike the knobs below, and shipped in the
        # `generation_config.json` of families this serves. The request has no repetition penalty
        # to ask for, so this is turned off rather than mirrored.
        "repetition_penalty": 1.0,
    }
    if temperature == 0:
        # `sample_logits` reads temperature 0 as argmax, and the graph steer modal sends 0 by
        # default (`STEER_TEMPERATURE_GRAPH`). `transformers` raises on it instead, so ask for
        # greedy decoding, which is the same thing. The frequency penalty is dropped on this path
        # for the same reason `sample_logits` drops it: there is no distribution left to shape.
        kwargs["do_sample"] = False
        return kwargs
    kwargs["do_sample"] = True
    kwargs.update(_NEUTRAL_SAMPLING)
    kwargs["logits_processor"] = LogitsProcessorList([_TransformerLensSampling(temperature, freq_penalty)])
    return kwargs


def generate_default(
    model: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    freq_penalty: float,
) -> torch.Tensor:
    """The unsteered continuation: the prompt's token ids followed by the generated ones.

    Straight at the HuggingFace model, because this is the baseline and wants none of the
    replacement model's intervention machinery. It is still the same forward pass -- the permanent
    hooks `InterpEngineReplacementModel` installs cache tensors and reroute gradients, leaving
    every value untouched -- so this run and the steered one remain comparable.
    """
    _require_interp_engine(model)
    tokens = model.ensure_tokenized(prompt)
    input_ids = tokens.unsqueeze(0)
    sequences = model.hf_model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
        use_cache=True,
        **_generation_kwargs(max_new_tokens, temperature, freq_penalty),
    )
    return sequences[0]


def generate_steered(
    model: Any,
    prompt: str,
    interventions: Sequence[tuple[Any, Any, Any, Any]],
    max_new_tokens: int,
    temperature: float,
    freq_penalty: float,
    freeze_attention: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The steered continuation, and the logits that chose each generated token.

    `freeze_attention` is handed to the backend rather than acted on here: it holds every layer's
    attention pattern at the values a clean pass over the same prompt produced, for the pass that
    consumes the prompt only. The backend installs those freezes only when there is at least one
    intervention, which is what the handler can end up with after dropping features at unsteerable
    positions.
    """
    _require_interp_engine(model)
    collector = _SequenceCollector()
    _, logits, _ = model.feature_intervention_generate(
        prompt,
        interventions,
        freeze_attention=freeze_attention,
        streamer=collector,
        **_generation_kwargs(max_new_tokens, temperature, freq_penalty),
    )
    # `generate` hands the streamer its tokens on the CPU whatever the model's device; the caller
    # decodes them and runs a forward pass over them, so put them back where the model is.
    return collector.sequence().to(model.device), logits
