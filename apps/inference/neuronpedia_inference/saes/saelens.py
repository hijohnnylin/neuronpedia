import logging
from typing import Any

from sae_lens.saes.sae import SAE

from neuronpedia_inference.saes.base import BaseSAE

logger = logging.getLogger(__name__)


class SaeLensSAE(BaseSAE):
    @staticmethod
    def load(release: str, sae_id: str, device: str, dtype: str) -> tuple[Any, str]:
        # load to cpu first, then GPU - this reduces fragmentation of the GPU memory (saves memory)
        loaded_sae = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device="cpu",
            dtype=dtype,
        )
        loaded_sae.to(device)
        # Attention-output SAEs (`hook_z`: gpt2 att-kk, gemma-2 gemmascope-att) come out of
        # SAELens with input reshaping ON, which folds the last two dims of the input
        # (`... n_heads d_head -> ... (n_heads d_head)`) because TransformerLens' hook_z is
        # [batch, pos, n_heads, d_head]. The engine captures z as the attention output
        # projection's INPUT, which is already concatenated ([batch, pos, n_heads*d_head] ==
        # d_in), so leaving the reshape on folds the position axis into d_in instead and
        # `encode` fails on a `[1, pos*d_in]` tensor. Everything else here (W_enc columns for
        # DFA, W_dec rows for steering) is already in that concatenated space.
        if getattr(loaded_sae.cfg, "reshape_activations", "none") == "hook_z":
            loaded_sae.turn_off_forward_pass_hook_z_reshaping()
        if loaded_sae.cfg.architecture() in ["temporal"]:
            logger.info("Temporal architecture detected, skipping fold_W_dec_norm")
        elif not getattr(loaded_sae.cfg, "rescale_acts_by_decoder_norm", True):
            # Folding W_dec_norm is not safe for TopK SAEs when
            # rescale_acts_by_decoder_norm is False, since it would change which
            # features are selected in the top-k. Leave the weights as trained.
            logger.info("rescale_acts_by_decoder_norm is False, skipping fold_W_dec_norm")
        else:
            loaded_sae.fold_W_dec_norm()
        loaded_sae.eval()

        return loaded_sae, loaded_sae.cfg.metadata.hook_name
