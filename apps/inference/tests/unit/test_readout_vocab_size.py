"""``_readout_vocab_size`` must prefer the model's padded embedding size."""

from __future__ import annotations

from types import SimpleNamespace

from neuronpedia_inference.endpoints.lens.prompt import _readout_vocab_size


class _Tok:
    def __init__(self, vocab_size: int, length: int):
        self.vocab_size = vocab_size
        self._length = length

    def __len__(self) -> int:
        return self._length


def test_readout_vocab_size_prefers_model_vocab_over_tokenizer():
    # Llama-3.1: tokenizer.vocab_size=128000, embedding table=128256.
    assert _readout_vocab_size(_Tok(128000, 128000), SimpleNamespace(vocab_size=128256)) == 128256


def test_readout_vocab_size_uses_len_tokenizer_when_larger():
    assert _readout_vocab_size(_Tok(128000, 128256), model=None) == 128256


def test_readout_vocab_size_reads_hf_config_on_model():
    model = SimpleNamespace(config=SimpleNamespace(vocab_size=128256))
    assert _readout_vocab_size(_Tok(128000, 128000), model) == 128256
