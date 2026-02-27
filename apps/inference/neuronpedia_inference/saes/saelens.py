from typing import Any

from sae_lens.saes.sae import SAE

from neuronpedia_inference.saes.base import BaseSAE


class SaeLensSAE(BaseSAE):
    @staticmethod
    def load(release: str, sae_id: str, device: str, dtype: str) -> tuple[Any, str]:
        loaded_sae = SAE.from_pretrained(
            release=release,
            sae_id=sae_id,
            device=device,
            dtype=dtype,
        )
        if loaded_sae.cfg.architecture() in ["temporal"]:
            print("Temporal architecture detected, skipping fold_W_dec_norm")
        else:
            loaded_sae.fold_W_dec_norm()
        loaded_sae.eval()

        return loaded_sae, loaded_sae.cfg.metadata.hook_name
