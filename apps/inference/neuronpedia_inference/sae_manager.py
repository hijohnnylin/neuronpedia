import logging
import time
from collections import OrderedDict
from typing import Any

from neuronpedia_inference.config import (
    Config,
    get_sae_lens_ids_from_neuronpedia_id,
    get_saelens_neuronpedia_directory_df,
    resolve_saelens_model_id,
)
from neuronpedia_inference.sae_cache import sae_cache
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
        sae_gpu_budget_bytes: int = 0,
        sae_pinned_host_bytes: int = 0,
    ):
        self.config = Config.get_instance()
        self.num_layers = num_layers
        self.device = device
        self.max_loaded_saes = self.config.max_loaded_saes

        # Paging (sae_cache.py): master copies in host RAM, a byte-budgeted LRU on the GPU.
        # Off by default, in which case everything below behaves exactly as it always has:
        # `max_loaded_saes` SAEs held on the GPU, evicted by count.
        sae_cache.configure(
            budget_bytes=sae_gpu_budget_bytes,
            device=device,
            pinned_host_bytes=sae_pinned_host_bytes,
        )
        self.paging_enabled = sae_cache.enabled

        self.sae_data = {}  # New consolidated dictionary
        self.sae_set_to_saes = {}
        self.valid_sae_sets = []
        self.loaded_saes = OrderedDict()  # Keep track of loaded SAEs
        # self.load_saes()

    def load_saes(self):
        server_cfg = Config.get_instance().sae_config

        self.setup_neuron_layers()

        if not server_cfg:
            # A pod started for capture / lens / steer only. Every SAE-backed endpoint then
            # rejects its source set as invalid, which is the correct answer, and the rest of
            # the server is unaffected.
            logger.info("No SAE sets configured; skipping SAE loading")
            return

        all_sae_ids = []
        for sae_set in server_cfg:
            logger.info(f"Processing SAE set: {sae_set['set']}")
            self.valid_sae_sets.append(sae_set["set"])
            all_sae_ids.extend(sae_set["saes"])
            self.sae_set_to_saes[sae_set["set"]] = sae_set["saes"]

        model_id = self.config.custom_hf_model_id or self.config.model_id

        if self.paging_enabled:
            # Every SAE is loaded once, to the host. Which of them are on the GPU is then a
            # runtime decision, so there is no startup set to choose and nothing to unload.
            for sae_id in all_sae_ids:
                self.load_sae(model_id, sae_id)
            self._warn_if_host_memory_tight()
            sae_cache.warm(all_sae_ids)
        else:
            starting_saes = self.get_starting_saes(all_sae_ids)

            # Load and immediately unload all SAEs not in starting_saes
            for sae_id in all_sae_ids:
                if sae_id not in starting_saes:
                    self.load_sae(model_id, sae_id)
                    self.unload_sae(sae_id)

            # Load starting SAEs
            for sae_id in starting_saes:
                self.load_sae(model_id, sae_id)

            logger.info(f"Loaded {len(self.loaded_saes)} SAEs")

        self.print_sae_status()

    def _warn_if_host_memory_tight(self) -> None:
        """Paging trades VRAM for host RAM; say so loudly if the trade looks bad.

        The masters are as big in host RAM as they used to be in VRAM, so a box with plenty
        of GPU memory but little RAM is the one configuration where paging is a downgrade.
        """
        host_bytes = sae_cache.host_bytes
        pinned = sae_cache.pinner.used_bytes
        try:
            import psutil

            available = int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001 - psutil optional
            return
        logger.info(
            "[SAE-CACHE] %.2f GiB of SAEs held in host RAM (%.2f GiB page-locked); %.2f GiB still available",
            host_bytes / 1024**3,
            pinned / 1024**3,
            available / 1024**3,
        )
        if available < host_bytes * 0.25:
            logger.warning(
                "[SAE-CACHE] only %.2f GiB of host RAM left after loading %.2f GiB of SAEs. "
                "Lower the number of SAE sets or move to a host with more RAM -- paging "
                "moves this memory off the GPU, it does not shrink it.",
                available / 1024**3,
                host_bytes / 1024**3,
            )

    def get_starting_saes(self, all_sae_ids: list[str]) -> list[str]:
        return all_sae_ids[: (self.max_loaded_saes)]

    def load_sae(self, model_id: str, sae_id: str) -> None:
        start_time = time.time()
        logger.info(f"Loading SAE: {sae_id}")

        directory_df = get_saelens_neuronpedia_directory_df()
        saelens_model_id = resolve_saelens_model_id(model_id, directory_df)
        sae_lens_release, sae_lens_id = get_sae_lens_ids_from_neuronpedia_id(
            model_id=saelens_model_id,
            neuronpedia_id=sae_id,
            df_exploded=directory_df,
        )

        # Under paging the SAE never gets a permanent home on the GPU: it is loaded and
        # transformed on the host, and sae_cache moves it across on demand.
        loaded_sae, hook_name = SaeLensSAE.load(
            release=sae_lens_release,
            sae_id=sae_lens_id,
            device="cpu" if self.paging_enabled else self.device,
            dtype=self.config.sae_dtype,
        )

        nbytes = sae_cache.register(sae_id, loaded_sae) if self.paging_enabled else 0

        self.sae_data[sae_id] = {
            # Owned by sae_cache when paging: reading this directly would hand out weights
            # that may be on the host. Go through get_sae(), which stages them in.
            "sae": None if self.paging_enabled else loaded_sae,
            "hook": hook_name,
            # GPU bytes this source occupies while resident, for admission control.
            "nbytes": nbytes,
            "neuronpedia_id": loaded_sae.cfg.metadata.neuronpedia_id,
            "type": SAE_TYPE.SAELENS,
            # Recorded so request cost estimation (memory_cost.py) can size an encode
            # without holding the SAE: `unload_sae` clears "sae" but leaves these, and
            # `load_saes` touches every configured SAE once at startup, so after startup
            # every source has dims here. d_in doubles as the capture width for this hook.
            "d_sae": int(loaded_sae.cfg.d_sae),
            "d_in": int(loaded_sae.cfg.d_in),
            # TODO: this should be in SAELens
            "dfa_enabled": (
                loaded_sae.cfg.metadata.neuronpedia_id is not None
                and (
                    DFA_ENABLED_NP_ID_SEGMENT in loaded_sae.cfg.metadata.neuronpedia_id
                    or DFA_ENABLED_NP_ID_SEGMENT_ALT in loaded_sae.cfg.metadata.neuronpedia_id
                )
            ),
            "transcoder": False,  # You might want to set this based on some condition
        }

        self.loaded_saes[sae_id] = None  # We're using OrderedDict as an OrderedSet
        # Count-based eviction only makes sense when "loaded" means "on the GPU". With
        # paging, loaded means host-resident and GPU residency is bounded in bytes instead.
        if not self.paging_enabled and len(self.loaded_saes) > self.max_loaded_saes:
            lru_sae = next(iter(self.loaded_saes))
            self.unload_sae(lru_sae)

        end_time = time.time()

        logger.info(f"Successfully loaded SAE: {sae_id} in {end_time - start_time:.2f} seconds")

    def unload_sae(self, sae_id: str) -> None:
        start_time = time.time()
        logger.info(f"Starting to unload SAE: {sae_id}")

        if sae_id in self.sae_data:
            self.sae_data[sae_id]["sae"] = None

        if self.paging_enabled:
            sae_cache.unregister(sae_id)

        if sae_id in self.loaded_saes:
            del self.loaded_saes[sae_id]

        end_time = time.time()
        logger.info(f"Successfully unloaded SAE: {sae_id} in {end_time - start_time:.2f} seconds")

    def ensure_source(self, source: str) -> None:
        """Make sure ``sae_data[source]`` exists, WITHOUT bringing weights to the GPU.

        Callers that only need a source's hook name or dims use this. It matters under
        paging: the activation endpoints prime metadata for every selected source before
        capturing, and doing that through ``get_sae`` would stage all 26 of them in and
        evict each other for nothing.
        """
        if source in self.sae_data and not self._needs_reload(source):
            return
        self.load_sae(self.config.custom_hf_model_id or self.config.model_id, source)

    def _needs_reload(self, source: str) -> bool:
        """True when metadata exists but the weights behind it are gone.

        Only possible under paging, and only after an explicit `unload_sae`: the metadata
        entry survives, so the presence check alone would report a source the cache no
        longer holds as ready to use.
        """
        if not self.paging_enabled:
            return False
        if self.sae_data[source].get("type") != SAE_TYPE.SAELENS:
            return False
        return not sae_cache.is_registered(source)

    def get_sae(self, source: str) -> Any:
        """The SAE for ``source``, GPU-resident and ready to encode.

        Under paging this may stage weights in from the host (tens of milliseconds) and
        evict another source to make room, and it ends the caller's claim on whatever
        source it asked for previously -- see the module docstring in ``sae_cache``.
        """
        if self.paging_enabled:
            self.ensure_source(source)
            return sae_cache.acquire(source)

        if source not in self.loaded_saes:
            self.load_sae(
                (self.config.custom_hf_model_id if self.config.custom_hf_model_id else self.config.model_id),
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
                "nbytes": 0,
                "neuronpedia_id": None,
                "dfa_enabled": False,
                "transcoder": False,
                "type": SAE_TYPE.NEURONS,
                "hook": f"blocks.{layer_str}.mlp.hook_post",
            }

        self.sae_set_to_saes[self.NEURONS_SOURCESET] = neurons_sourceset
        return neurons_sourceset

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

        if self.paging_enabled:
            stats = sae_cache.stats()
            print(
                f"\nTotal Loaded: {len(self.loaded_saes)} "
                f"({stats['host_bytes'] / 1024**3:.2f} GiB in host RAM, "
                f"{stats['pinned_bytes'] / 1024**3:.2f} GiB page-locked)"
            )
            print(
                f"GPU residency: {stats['resident_count']} SAEs, "
                f"{stats['resident_bytes'] / 1024**3:.2f} of "
                f"{stats['budget_bytes'] / 1024**3:.2f} GiB"
            )
        else:
            print(f"\nTotal Loaded: {len(self.loaded_saes)} / {self.max_loaded_saes}")

    # Utility methods
    def get_sae_type(self, sae_id: str) -> str:
        return self.sae_data.get(sae_id, {}).get("type")

    def get_sae_hook(self, sae_id: str) -> str:
        return self.sae_data.get(sae_id, {}).get("hook")

    def is_dfa_enabled(self, sae_id: str) -> bool:
        return self.sae_data.get(sae_id, {}).get("dfa_enabled", False)

    def get_d_sae(self, sae_id: str) -> int | None:
        """Feature count for a source, or None if unknown (e.g. a `neurons` source).

        Available whether or not the SAE is currently loaded -- see the note in `load_sae`.
        """
        return self.sae_data.get(sae_id, {}).get("d_sae")

    def get_d_in(self, sae_id: str) -> int | None:
        """Input width for a source (also the capture width for its hook), or None."""
        return self.sae_data.get(sae_id, {}).get("d_in")

    def get_sae_nbytes(self, sae_id: str) -> int:
        """GPU bytes this source occupies while resident. 0 when paging is off."""
        return int(self.sae_data.get(sae_id, {}).get("nbytes") or 0)

    def widest_activation_dims(self) -> tuple[int, int, int]:
        """``(d_sae, d_in, n_hooks)`` for the most expensive all-layers search we serve.

        Walks every configured source set (the empty-``selected_sources`` default expands to
        a whole set). ``n_hooks`` is the largest number of DISTINCT capture hooks any one
        set needs -- several sources on the same layer share a hook, so counting SAEs would
        overstate the capture term. Returns ``(0, 0, 0)`` when no SAE dims were recorded.
        """
        best_d_sae = 0
        best_d_in = 0
        best_n_hooks = 0
        for set_name, sae_ids in self.sae_set_to_saes.items():
            if set_name == self.NEURONS_SOURCESET:
                continue
            d_saes = [self.get_d_sae(s) for s in sae_ids]
            d_ins = [self.get_d_in(s) for s in sae_ids]
            known_sae = [d for d in d_saes if d]
            known_in = [d for d in d_ins if d]
            if not known_sae or not known_in:
                continue
            hooks = {self.get_sae_hook(s) for s in sae_ids if self.get_sae_hook(s) is not None}
            best_d_sae = max(best_d_sae, max(known_sae))
            best_d_in = max(best_d_in, max(known_in))
            best_n_hooks = max(best_n_hooks, len(hooks))
        return best_d_sae, best_d_in, best_n_hooks

    def get_valid_sae_sets(self):
        return self.valid_sae_sets
