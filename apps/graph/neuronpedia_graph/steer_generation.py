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
- `transformers` starts from the checkpoint's own `generation_config.json`. That is the right
  default, and the same one the inference server applies: a knob the request leaves unset takes
  the file's value (`interp_engine.resolve_sampling`). Every knob is then passed explicitly, so
  what `generate` samples from is exactly the resolved settings and nothing the file adds on its
  own (`repetition_penalty` in particular is turned off).
- It returns the continuation as *text*. `/steer` needs the token ids: it reports one row of top
  logits per token, so it has to know where the boundaries were. Re-tokenizing the text to find
  them is exactly the bug commit bff40b1e ("correctly handle qwen steering") removed, hence
  `_SequenceCollector`.

`temperature=0` means greedy rather than an error, as the steer modal has always sent it. The one
repetition control is the presence penalty, a flat subtraction from the logit of every token the
generation has produced so far, applied before the temperature and on the greedy path too, as
vLLM and the inference server order it. `interp_engine.hf_generate_kwargs` does that translation,
so this module and the NLA server hand `generate` the same keywords.

The other two engines are turned away. `transformerlens` could be supported -- it is where these
keywords come from -- but nothing deploys it, so a second path here would be a second untested
path; a TL pod still generates graphs and fails only on steering. `nnsight` reaches generation
through a tracing context this endpoint never enters, so steering never worked there at all.
"""

from collections.abc import Sequence
from typing import Any

import torch
from interp_engine import RecommendedSampling, SamplingSettings, hf_generate_kwargs, resolve_sampling
from interp_engine.sampling import parse_recommended_sampling
from transformers.generation.streamers import BaseStreamer


def recommended_sampling(model: Any) -> RecommendedSampling:
    """What the loaded checkpoint's `generation_config.json` states, read off the HF model.

    `transformers` loaded the file onto `hf_model.generation_config` at load time, so this reads
    that rather than the Hub a second time. Empty for a model that carries none.
    """
    hf_model = getattr(model, "hf_model", None)
    config = getattr(hf_model, "generation_config", None)
    if config is None:
        return RecommendedSampling()
    return parse_recommended_sampling(config.to_dict(), source="hf_model.generation_config")


def resolve_request_sampling(
    model: Any,
    *,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    presence_penalty: float | None,
) -> SamplingSettings:
    """The request's knobs decided the way the inference server decides them."""
    return resolve_sampling(
        recommended_sampling(model),
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        presence_penalty=presence_penalty,
    )


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


def _generation_kwargs(max_new_tokens: int, sampling: SamplingSettings, prompt_len: int) -> dict[str, Any]:
    """The resolved settings in `transformers`' terms: the engine's own translation, plus the length."""
    return {"max_new_tokens": max_new_tokens, **hf_generate_kwargs(sampling, prompt_len=prompt_len)}


def generate_default(
    model: Any,
    prompt: str,
    max_new_tokens: int,
    sampling: SamplingSettings,
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
        **_generation_kwargs(max_new_tokens, sampling, prompt_len=int(tokens.shape[-1])),
    )
    return sequences[0]


def generate_steered(
    model: Any,
    prompt: str,
    interventions: Sequence[tuple[Any, Any, Any, Any]],
    max_new_tokens: int,
    sampling: SamplingSettings,
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
    prompt_len = int(model.ensure_tokenized(prompt).shape[-1])
    _, logits, _ = model.feature_intervention_generate(
        prompt,
        interventions,
        freeze_attention=freeze_attention,
        streamer=collector,
        **_generation_kwargs(max_new_tokens, sampling, prompt_len=prompt_len),
    )
    # `generate` hands the streamer its tokens on the CPU whatever the model's device; the caller
    # decodes them and runs a forward pass over them, so put them back where the model is.
    return collector.sequence().to(model.device), logits
