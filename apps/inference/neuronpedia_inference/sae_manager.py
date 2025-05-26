import logging
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from neuronpedia_inference.config import (
    Config,
    get_sae_lens_ids_from_neuronpedia_id,
    get_saelens_neuronpedia_directory_df,
)
from neuronpedia_inference.saes.saelens import SaeLensSAE  # type: ignore

logger = logging.getLogger(__name__)

# TODO: this should be in SAELens
# if we find this in the neuronpedia ID, we enable DFA
DFA_ENABLED_NP_ID_SEGMENT = "-att-"
DFA_ENABLED_NP_ID_SEGMENT_ALT = "-att_"


class SAE_TYPE:
    NEURONS = "neurons"
    SAELENS = "saelens-1"


class SAEManager:
    NEURONS_SOURCESET = "neurons"

    _instance = None  # Class variable to store the singleton instance

    @classmethod
    def get_instance(cls):
        """Get the global SAEManager instance, creating it if it doesn't exist"""
        if cls._instance is None:
            cls._instance = SAEManager()
        return cls._instance

    def __init__(
        self,
        num_layers: int = 0,
        device: str = "cuda",
    ):
        self.config = Config.get_instance()
        self.num_layers = num_layers
        self.device = device
        self.max_loaded_saes = self.config.MAX_LOADED_SAES

        self.sae_data = {}  # New consolidated dictionary
        self.sae_set_to_saes = {}
        self.valid_sae_sets = []
        self.loaded_saes = OrderedDict()  # Keep track of loaded SAEs
        # self.load_saes()

    def load_saes(self):
        server_cfg = Config.get_instance().SAE_CONFIG

        self.setup_neuron_layers()

        all_sae_ids = []
        for sae_set in server_cfg:
            logger.info(f"Processing SAE set: {sae_set['set']}")
            self.valid_sae_sets.append(sae_set["set"])
            all_sae_ids.extend(sae_set["saes"])
            self.sae_set_to_saes[sae_set["set"]] = sae_set["saes"]

        starting_saes = self.get_starting_saes(all_sae_ids)

        # Load and immediately unload all SAEs not in starting_saes
        for sae_id in all_sae_ids:
            if sae_id not in starting_saes:
                self.load_sae(
                    (
                        self.config.CUSTOM_HF_MODEL_ID
                        if self.config.CUSTOM_HF_MODEL_ID
                        else self.config.MODEL_ID
                    ),
                    sae_id,
                )
                self.unload_sae(sae_id)

        # Load starting SAEs
        for sae_id in starting_saes:
            self.load_sae(
                (
                    self.config.CUSTOM_HF_MODEL_ID
                    if self.config.CUSTOM_HF_MODEL_ID
                    else self.config.MODEL_ID
                ),
                sae_id,
            )

        logger.info(f"Loaded {len(self.loaded_saes)} SAEs")

        self.print_sae_status()

    def get_starting_saes(self, all_sae_ids: list[str]) -> list[str]:
        return all_sae_ids[: (self.max_loaded_saes)]

    def load_sae(self, model_id: str, sae_id: str) -> None:
        start_time = time.time()
        logger.info(f"Loading SAE: {sae_id}")

        sae_lens_release, sae_lens_id = get_sae_lens_ids_from_neuronpedia_id(
            model_id=model_id,
            neuronpedia_id=sae_id,
            df_exploded=get_saelens_neuronpedia_directory_df(),
        )

        loaded_sae, hook_in, hook_out = SaeLensSAE.load(
            release=sae_lens_release,
            sae_id=sae_lens_id,
            device=self.device,
            dtype=self.config.SAE_DTYPE,
        )

        self.sae_data[sae_id] = {
            "sae": loaded_sae,
            "hook": hook_in,
            "hook_out": hook_out,
            "neuronpedia_id": loaded_sae.cfg.neuronpedia_id,
            "type": SAE_TYPE.SAELENS,
            # TODO: this should be in SAELens
            "dfa_enabled": (
                loaded_sae.cfg.neuronpedia_id is not None
                and (
                    DFA_ENABLED_NP_ID_SEGMENT in loaded_sae.cfg.neuronpedia_id
                    or DFA_ENABLED_NP_ID_SEGMENT_ALT in loaded_sae.cfg.neuronpedia_id
                )
            ),
            "transcoder": hook_out is not None
        }

        self.loaded_saes[sae_id] = None  # We're using OrderedDict as an OrderedSet
        if len(self.loaded_saes) > self.max_loaded_saes:
            lru_sae = next(iter(self.loaded_saes))
            self.unload_sae(lru_sae)

        end_time = time.time()

        logger.info(
            f"Successfully loaded SAE: {sae_id} in {end_time - start_time:.2f} seconds"
        )

    def unload_sae(self, sae_id: str) -> None:
        start_time = time.time()
        logger.info(f"Starting to unload SAE: {sae_id}")

        if sae_id in self.sae_data:
            self.sae_data[sae_id]["sae"] = None

        if sae_id in self.loaded_saes:
            del self.loaded_saes[sae_id]

        end_time = time.time()
        logger.info(
            f"Successfully unloaded SAE: {sae_id} in {end_time - start_time:.2f} seconds"
        )

    def get_sae(self, source: str) -> Any:
        if source not in self.loaded_saes:
            self.load_sae(
                (
                    self.config.CUSTOM_HF_MODEL_ID
                    if self.config.CUSTOM_HF_MODEL_ID
                    else self.config.MODEL_ID
                ),
                source,
            )
        else:
            self.loaded_saes.move_to_end(source)
        return self.sae_data.get(source, {}).get("sae")

    def setup_neuron_layers(self):
        neurons_sourceset = []
        for layer in range(self.num_layers):
            layer_str = str(layer)
            neurons_sourceset.append(layer_str)
            self.sae_data[layer_str] = {
                "sae": None,
                "neuronpedia_id": None,
                "dfa_enabled": False,
                "transcoder": False,
                "type": SAE_TYPE.NEURONS,
                "hook": f"blocks.{layer_str}.mlp.hook_post",
            }

        self.sae_set_to_saes[self.NEURONS_SOURCESET] = neurons_sourceset
        return neurons_sourceset

    def download_sae_set(self, sae_set: dict[str, str | bool]) -> Path:
        sae_set_id = sae_set["set"]
        sae_set_dir = f"{self.config.MODEL_ID}__{sae_set_id}"
        sae_set_path = Path(self.config.SAES_PATH).joinpath(sae_set_dir)  # type: ignore

        if (
            not sae_set["local"] and sae_set["type"] != "saelens-1"
        ):  # SAE Lens one will download prior to loading.
            snapshot_download(
                repo_id=f"{self.config.HUGGINGFACE_ACCOUNT}/{sae_set_dir}",  # type: ignore
                local_dir=str(sae_set_path),
            )

        return sae_set_path

    def get_supported_neuronpedia_ids(self):
        """
        Get a list of all Neuronpedia IDs supported by the manager.

        Returns:
            list: A list of all supported Neuronpedia IDs.
        """
        neuronpedia_ids = []
        for data in self.sae_data.values():
            neuronpedia_id = data.get("neuronpedia_id")
            if neuronpedia_id is not None:
                neuronpedia_ids.append(neuronpedia_id)
        return neuronpedia_ids  # type: ignore

    def get_sae_id_by_neuronpedia_id(self, neuronpedia_id: str):
        """
        Retrieve the SAE ID corresponding to a given Neuronpedia ID.

        Args:
            neuronpedia_id (str): The Neuronpedia ID to search for.

        Returns:
            str or None: The corresponding SAE ID if found, None otherwise.
        """
        for sae_id, data in self.sae_data.items():
            if data.get("neuronpedia_id") == neuronpedia_id:
                return sae_id
        return None

    def print_sae_status(self):
        """
        Print a nicely formatted status of loadable and loaded SAEs.
        """
        print("\nSAE Status:")
        print("===========")

        print("\nLoadable SAEs:")
        for sae_set, sae_ids in self.sae_set_to_saes.items():
            if sae_set == self.NEURONS_SOURCESET:
                continue
            print(f"  {sae_set}:")
            for sae_id in sae_ids:
                status = "Loaded" if sae_id in self.loaded_saes else "Not Loaded"
                print(f"    - {sae_id}: {status}")

        print("\nCurrently Loaded SAEs:")
        for i, sae_id in enumerate(self.loaded_saes, 1):
            print(f"  {i}. {sae_id}")

        print(f"\nTotal Loaded: {len(self.loaded_saes)} / {self.max_loaded_saes}")

    # Utility methods
    def get_sae_type(self, sae_id: str) -> str:
        return self.sae_data.get(sae_id, {}).get("type")

    def get_sae_hook(self, sae_id: str) -> str:
        return self.sae_data.get(sae_id, {}).get("hook")
    
    def get_decoder_hook(self, sae_id):
        data = self.sae_data.get(sae_id, {})
        return data.get("hook_out") or data.get("hook")

    def is_dfa_enabled(self, sae_id: str) -> bool:
        return self.sae_data.get(sae_id, {}).get("dfa_enabled", False)

    def get_valid_sae_sets(self):
        return self.valid_sae_sets
