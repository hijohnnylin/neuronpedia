"""SAE/Transcoder loader wrapper for Neuronpedia Searcher.

This module previously supported only the vanilla SAE objects exposed by
`sae_lens.sae.SAE`. We now extend the functionality to transparently load
three different artifact classes coming from the sae-lens code-base:

* SAE (classic auto-encoder)
* Transcoder
* SkipTranscoder

The heavy lifting is delegated to `load_artifact_from_pretrained`, a new helper
published upstream that inspects the YAML metadata of a given release/sae_id
and automatically returns an instance of the correct class.

For Neuronpedia Inference we treat each artifact uniformly – the caller only
needs the instantiated object and the hook names.  For classic SAEs the single
hook `cfg.hook_name` is sufficient.  Transcoders additionally come with
`cfg.hook_name_out`, the location where the decoder output should be steered.

The `load` method therefore now returns **three** values:

    (artifact, hook_name_in, hook_name_out)

`hook_name_out` is `None` for plain SAEs so users can branch on a simple
truthiness check to detect Transcoder-like artifacts.
"""

from neuronpedia_inference.saes.base import BaseSAE
from sae_lens.toolkit.pretrained_sae_loaders import (  # type: ignore
    load_artifact_from_pretrained,
)
from sae_lens.config import DTYPE_MAP  # type: ignore


class SaeLensSAE(BaseSAE):
    @staticmethod
    def load(release: str, sae_id: str, device: str, dtype: str):
        """Load an artifact (SAE / Transcoder / SkipTranscoder).

        Args:
            release:  The named release on the HF hub (e.g. "sae_lens")
            sae_id:   The specific SAE/Transcoder identifier inside *release*.
            device:   Torch device string, forwarded to the loader.
            dtype:    One of {"float16", "float32", "bfloat16"} – we convert
                      the loaded weights to this dtype after loading.

        Returns:
            artifact:         The initialised model instance (type depends on
                             YAML `type` field).
            hook_name_in:     Where to read encoder activations from.
            hook_name_out:    Where to *write* decoder deltas to when steering.
                             `None` for classic SAEs.
        """

        artifact, _cfg_dict, _sparsity = load_artifact_from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
        )

        # Ensure correct dtype & eval mode
        artifact.to(device, dtype=DTYPE_MAP[dtype])

        # Some classes (SAE, Transcoder, SkipTranscoder) expose this helper –
        # if it does not exist we silently ignore the attribute.
        if hasattr(artifact, "fold_W_dec_norm"):
            try:
                artifact.fold_W_dec_norm()
            except Exception:
                # Folding is a convenience optimization, not critical – we do
                # not want loading to fail if it is not implemented.
                pass

        artifact.eval()

        hook_name_in = artifact.cfg.hook_name
        hook_name_out = getattr(artifact.cfg, "hook_name_out", None) or None

        return artifact, hook_name_in, hook_name_out