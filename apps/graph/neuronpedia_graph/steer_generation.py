"""Generation for `/steer`, against the interp_engine replacement model.

`/steer` used to call `model.generate(...)` straight, which is a `HookedTransformer` method. On the
default `interp_engine` engine the model is a plain `nn.Module` over a HuggingFace one, so the
endpoint raised `AttributeError: 'InterpEngineReplacementModel' object has no attribute 'generate'`.

Everything here exists because the endpoint's *wire contract* is still TransformerLens-shaped even
though nothing executes TransformerLens any more -- the steer modal's sliders are calibrated to it,
and every shared steer result was produced under it -- while the generation underneath is now
`transformers`' `generate`. Three things have to be translated, and each is silent rather than loud:

- `temperature=0` means argmax in `sample_logits` and the steer modal sends it by default, where
  `transformers` rejects it. `freq_penalty` subtracts a flat amount per earlier occurrence of a
  token, which is not `repetition_penalty` -- that scales the logit instead -- so it needs its own
  processor. `stop_at_eos`, `verbose`, `return_type` and `use_past_kv_cache` have no counterpart at
  all, and `feature_intervention_generate` forwards keywords it does not recognize into `generate`,
  which refuses them ("The following `model_kwargs` are not used by the model").
- `generate` starts from the checkpoint's own `generation_config.json`, and several families ship
  sampling defaults there (Qwen3: temperature 0.6, top_p 0.95, top_k 20). TransformerLens applied
  none of them, so the same steer request would otherwise sample from a truncated distribution.
- `feature_intervention_generate` returns the continuation as *text*, and `/steer` needs the token
  ids: it reports one row of top logits per token, so it has to know where the boundaries were.
  Re-tokenizing the text to find them is exactly the bug commit bff40b1e ("correctly handle qwen
  steering") removed, hence `_SequenceCollector`.

The other two engines are refused rather than translated. `transformerlens` could generate here --
it is the engine this endpoint was written for -- but no graph pod has run it since `interp_engine`
became the default, and supporting it meant carrying two spellings of every sampling keyword for a
path nothing exercised. `nnsight` reaches generation through a tracing context this endpoint never
enters, so steering never worked there at all. Both remain fine for generating graphs; only this
endpoint is narrower than the app.
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
_HF_NEUTRAL_SAMPLING: dict[str, Any] = {
    "temperature": 1.0,
    "top_k": 0,
    "top_p": 1.0,
    "min_p": None,
    "typical_p": 1.0,
}


class _TransformerLensSampling(LogitsProcessor):
    """TransformerLens' temperature and frequency penalty, in the order `sample_logits` applies them.

    Named for what it mirrors rather than for what runs it: this is the interp_engine path, and it
    is here because those two knobs are `/steer`'s wire contract, not because TransformerLens is
    involved.

    Both in one processor, and the temperature not left to `transformers`, because
    `_get_logits_processor` appends its `TemperatureLogitsWarper` *after* any caller-supplied
    processor (there is a standing TODO in `generation/utils.py` about the ordering). TransformerLens
    divides by the temperature and only then subtracts the penalty, so a separate processor would
    subtract from logits that had not been scaled yet and answer the same request differently.

    Non-positive penalties are ignored, as `sample_logits` ignores them.
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
    """Refuse a model this cannot generate with. Every circuit-tracer engine sets `backend`."""
    engine = getattr(model, "backend", None)
    if engine == "interp_engine":
        return
    loaded = (
        f"model engine {engine!r}"
        if engine
        else f"a {type(model).__name__}, which is not a circuit-tracer replacement model"
    )
    raise NotImplementedError(
        f"/steer generates through the interp_engine model engine; this pod loaded {loaded}. Every "
        "deployed graph pod omits --model-engine and so gets interp_engine. Graph generation runs "
        "on the other engines; steering does not."
    )


def _hf_kwargs(max_new_tokens: int, temperature: float, freq_penalty: float) -> dict[str, Any]:
    """The request in `transformers`' terms, sampling the way TransformerLens would."""
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        # Honored whether or not sampling is on, unlike the knobs below, and shipped in the
        # `generation_config.json` of families this serves. TransformerLens' `generate` applies no
        # repetition penalty, so neither does this.
        "repetition_penalty": 1.0,
    }
    if temperature == 0:
        # `sample_logits` reads temperature 0 as argmax, and the graph steer modal sends 0 by
        # default (`STEER_TEMPERATURE_GRAPH`). `transformers` raises on it instead, so ask for
        # greedy decoding, which is the same thing. TransformerLens drops the frequency penalty on
        # this path too, so there is nothing to carry over.
        kwargs["do_sample"] = False
        return kwargs
    kwargs["do_sample"] = True
    kwargs.update(_HF_NEUTRAL_SAMPLING)
    kwargs["logits_processor"] = LogitsProcessorList([_TransformerLensSampling(temperature, freq_penalty)])
    return kwargs


def generate_default(
    model: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    freq_penalty: float,
) -> torch.Tensor:
    """The unsteered continuation: the prompt's token ids followed by the generated ones."""
    _require_interp_engine(model)

    # Straight at the HuggingFace model: this is the baseline, so it wants none of the replacement
    # model's intervention machinery. It is still the same forward pass -- the permanent hooks
    # `InterpEngineReplacementModel` installs cache tensors and reroute gradients, leaving every
    # value untouched -- so the two runs remain comparable.
    tokens = model.ensure_tokenized(prompt)
    input_ids = tokens.unsqueeze(0)
    sequences = model.hf_model.generate(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        pad_token_id=model.tokenizer.pad_token_id or model.tokenizer.eos_token_id,
        use_cache=True,
        **_hf_kwargs(max_new_tokens, temperature, freq_penalty),
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

    `freeze_attention` goes to the backend untouched. Worth knowing what it does there, since
    neither this function nor the endpoint can make it more than it is: the freeze applies only to
    the pass that consumes the prompt, so an intervention at the last position has nothing
    downstream of it to hold constant, and it is skipped entirely when `interventions` is empty --
    which is what the handler is left with when every steered position turns out to be unsteerable.
    """
    _require_interp_engine(model)

    collector = _SequenceCollector()
    _, logits, _ = model.feature_intervention_generate(
        prompt,
        interventions,
        freeze_attention=freeze_attention,
        streamer=collector,
        **_hf_kwargs(max_new_tokens, temperature, freq_penalty),
    )
    # `generate` hands the streamer its tokens on the CPU whatever the model's device; the caller
    # decodes them and runs a forward pass over them, so put them back where the model is.
    return collector.sequence().to(model.device), logits
