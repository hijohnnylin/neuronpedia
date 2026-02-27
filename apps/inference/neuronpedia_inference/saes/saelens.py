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
        else:
            loaded_sae.fold_W_dec_norm()
        loaded_sae.eval()

        return loaded_sae, loaded_sae.cfg.metadata.hook_name
