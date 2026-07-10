from typing import Any

from sae_lens.saes.sae import SAE

from neuronpedia_inference.saes.base import BaseSAE


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
        if loaded_sae.cfg.architecture() in ["temporal"]:
            print("Temporal architecture detected, skipping fold_W_dec_norm")
        elif not getattr(loaded_sae.cfg, "rescale_acts_by_decoder_norm", True):
            # Folding W_dec_norm is not safe for TopK SAEs when
            # rescale_acts_by_decoder_norm is False, since it would change which
            # features are selected in the top-k. Leave the weights as trained.
            print(
                "rescale_acts_by_decoder_norm is False, skipping fold_W_dec_norm"
            )
        else:
            loaded_sae.fold_W_dec_norm()
        loaded_sae.eval()

        return loaded_sae, loaded_sae.cfg.metadata.hook_name
