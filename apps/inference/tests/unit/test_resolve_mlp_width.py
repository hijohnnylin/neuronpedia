"""Tests for resolve_mlp_width, the width the memory-cost estimate uses for a `neurons` source."""

from types import SimpleNamespace

from neuronpedia_inference.endpoints.lens.model_specific import mlp_width_is_uniform, resolve_mlp_width


def _model(config: object) -> SimpleNamespace:
    """A model whose `.config` is read directly, as on the eager backend."""
    return SimpleNamespace(config=config)


class _TextConfig(SimpleNamespace):
    """An HF config with `get_text_config`, so `_try_get_hf_config` accepts it as is."""

    def get_text_config(self) -> "_TextConfig":
        """Return the text config, which is this one for a text-only model."""
        return self


class _MultimodalConfig(SimpleNamespace):
    """An HF config whose MLP width lives on a nested text config (Gemma 3/4 style)."""

    def get_text_config(self) -> _TextConfig:
        """Return the nested text config."""
        return self.text_config


def test_reads_intermediate_size():
    """A text-only config gives its intermediate_size."""
    assert resolve_mlp_width(_model(_TextConfig(intermediate_size=25600)), None) == 25600


def test_reads_the_nested_text_config():
    """A multimodal config gives the width of its text model."""
    config = _MultimodalConfig(text_config=_TextConfig(intermediate_size=21504))
    assert resolve_mlp_width(_model(config), None) == 21504


def test_a_per_layer_list_gives_the_widest_layer():
    """A per-layer width list gives the widest entry, so the estimate stays an upper bound."""
    assert resolve_mlp_width(_model(_TextConfig(intermediate_size=[8192, 16384, 8192])), None) == 16384


def test_missing_width_is_none():
    """A config without intermediate_size gives None, and memory cost uses its fallback."""
    assert resolve_mlp_width(_model(_TextConfig()), None) is None


def test_gpt2_gives_n_inner_or_four_times_n_embd():
    """GPT-2 has no intermediate_size: n_inner when set, else 4 * n_embd."""
    assert resolve_mlp_width(_model(_TextConfig(n_embd=768, n_inner=None)), None) == 3072
    assert resolve_mlp_width(_model(_TextConfig(n_embd=768, n_inner=2048)), None) == 2048


def test_a_per_layer_list_is_not_uniform():
    """One width fits every layer only when the config gives one width."""
    assert mlp_width_is_uniform(_model(_TextConfig(intermediate_size=25600)), None)
    assert mlp_width_is_uniform(_model(_TextConfig(intermediate_size=[8192, 8192])), None)
    assert not mlp_width_is_uniform(_model(_TextConfig(intermediate_size=[8192, 16384])), None)
