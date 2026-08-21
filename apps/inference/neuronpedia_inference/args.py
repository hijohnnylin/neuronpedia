import argparse
import json
import os

from neuronpedia_inference.config import get_saelens_neuronpedia_directory_df


def parse_env_and_args():
    args = argparse.Namespace()

    args.host = os.getenv("HOST", "0.0.0.0")
    args.port = int(os.getenv("PORT", "5002"))
    # A Hugging Face repo id, since this is what reaches load_model. The SAELens short
    # id ("gpt2-small") is not resolvable on the Hub, so it cannot be the default here.
    args.model_id = os.getenv("MODEL_ID", "openai-community/gpt2")
    args.override_model_id = os.getenv("OVERRIDE_MODEL_ID", None)
    args.custom_hf_model_id = os.getenv("CUSTOM_HF_MODEL_ID", None)
    # An empty list is valid and means "load no SAEs": the model-only endpoints
    # (activation/raw, lens, steer, tokenize) work without them.
    args.sae_sets = json.loads(os.getenv("SAE_SETS", '["res-jb"]'))
    # "auto" loads each checkpoint in its native dtype (e.g. gemma-2/3, qwen3 = bf16; gpt2 = fp32)
    # rather than forcing fp32. Override with MODEL_DTYPE=float32 for tight fp32 numerics if needed.
    args.model_dtype = os.getenv("MODEL_DTYPE", "auto")
    args.sae_dtype = os.getenv("SAE_DTYPE", "float32")
    args.token_limit = int(os.getenv("TOKEN_LIMIT", "200"))
    # Separate cap for the lens endpoints only (logit/jacobian lens). Defaults to
    # 1024 and is independent of TOKEN_LIMIT so JLens conversations can be longer
    # (or shorter) than the limit used by the other endpoints.
    args.lens_token_limit = int(os.getenv("LENS_TOKEN_LIMIT", "1024"))
    # Left as None when DEVICE is unset so interp_engine.select_backend() can
    # auto-pick (cuda -> vLLM/eager; else mps for fp16/fp32-native; else cpu).
    # Set DEVICE explicitly to override the auto-selection.
    args.device = os.getenv("DEVICE")
    args.include_sae = json.loads(os.getenv("INCLUDE_SAE", "[]"))
    args.exclude_sae = json.loads(os.getenv("EXCLUDE_SAE", "[]"))
    args.model_from_pretrained_kwargs = os.getenv("MODEL_FROM_PRETRAINED_KWARGS", "{}")
    args.list_models = os.getenv("LIST_MODELS", "").lower() == "true"
    args.max_loaded_saes = int(os.getenv("MAX_LOADED_SAES", "300"))
    # SAE paging (see sae_cache.py). Unset = off: every SAE stays on the GPU for the life of
    # the process. A number of GiB, or "auto" to derive it from the card and the vLLM
    # reservation, keeps the master copies in host RAM and caches that many bytes on the GPU.
    args.sae_gpu_budget_gib = os.getenv("SAE_GPU_BUDGET_GIB") or None
    # Cap on page-locked host memory for those master copies. Unset = measured at startup.
    _pinned_host = os.getenv("SAE_PINNED_HOST_GIB")
    args.sae_pinned_host_gib = float(_pinned_host) if _pinned_host else None
    args.sentry_dsn = os.getenv("SENTRY_DSN")
    # Backend force override (None when unset => auto-select). "vllm" forces the
    # engine-owned vLLM backend; "eager" forces the EagerModel core. Set via
    # --force-vllm / --force-eager (start.py) -> FORCE_BACKEND. The final choice
    # (args.backend) is written in server.py after select_backend() runs.
    args.force_backend = os.getenv("FORCE_BACKEND") or None
    # Number of GPUs to shard the model across on one node: vLLM tensor_parallel_size,
    # or EagerModel device_map="auto". 1 = single GPU.
    args.num_gpus = int(os.getenv("NUM_GPUS", "1"))
    # Trade every interpretability endpoint for decode speed on a vLLM pod: selects the engine's
    # backend="vllm-generate", which keeps vLLM's CUDA graphs instead of the forward hooks they rule
    # out. Worth up to +249% decode on a 1B model and ~nothing at 4B+, so this is for a pod serving
    # small-model completions only. Refused at startup with SAE sets, and every capture/steer/lens
    # endpoint reports unavailable -- see server.py and endpoints/capabilities.py.
    args.generation_only = os.getenv("GENERATION_ONLY", "").lower() == "true"
    # Declared tap set for the vLLM backend: selects the engine's backend="vllm-static", which keeps
    # the CUDA graphs and cuts holes in them at these sites only. Unset keeps the hooked vLLM, where
    # every site is reachable and none are fast. "auto" declares resid_post at every layer
    # (resid_streams on a hyper-connection trunk); "sae" declares the SAE hook sites once those SAEs
    # load; a JSON list of [name, layer] pairs is an explicit set. Declare attn at a layer to keep
    # attention/DFA. Anything not declared 400s, so this is a routing decision, not a tuning knob.
    args.static_points = os.getenv("STATIC_POINTS")
    # Renamed in interp-engine 1.3 along with the backend it drives. Refused rather than aliased,
    # because a deploy that sets the old name is asking for a tap set and would otherwise get a
    # hooked pod: same endpoints, several times slower, and no line in the log saying why.
    if os.getenv("FREEZE_POINTS"):
        raise ValueError(
            "FREEZE_POINTS was renamed to STATIC_POINTS (interp-engine 1.3, where the vllm-freeze "
            "backend became vllm-static). Rename the variable in this pod's deploy config; the "
            "values it takes are unchanged."
        )

    # Lens endpoints (logit lens / jacobian lens)
    # Skip loading the fitted Jacobian lens at startup. The server still starts
    # and LOGIT_LENS requests work; JACOBIAN_LENS requests return an error.
    args.jlens_skip = os.getenv("JLENS_SKIP", "").lower() == "true"
    # Optional absolute path to a local directory containing a fitted lens
    # (e.g. .../<np_model_id>/jlens/Salesforce-wikitext). When set, this is used
    # instead of downloading from Hugging Face.
    args.jlens_source = os.getenv("JLENS_SOURCE")
    # Dataset folder name the lens was fit on (used in the HF path / local path).
    args.jlens_dataset = os.getenv("JLENS_DATASET", "Salesforce-wikitext")
    # Hugging Face model repo holding fitted lenses, keyed by neuronpedia model id
    # under "<np_model_id>/jlens/<dataset>/<slug>_jacobian_lens.pt".
    args.jlens_hf_repo = os.getenv("JLENS_HF_REPO", "neuronpedia/jacobian-lens")
    # Optional exact path (within the HF repo) to the lens .pt file. When set, this
    # is used verbatim instead of deriving it from the model id / dataset.
    args.jlens_hf_path = os.getenv("JLENS_HF_PATH")
    # GPU memory the lens may keep its per-layer J_bar in: GiB, "auto" (measure what is
    # left of the card once the model and SAEs are up), or off. A read-out transports
    # through every fitted layer on every batch, so a lens that does not fit here is
    # re-copied across PCIe rather than kept resident.
    args.jlens_gpu_budget_gib = os.getenv("JLENS_GPU_BUDGET_GIB", "auto")
    # Explicit neuronpedia model id (used to build the HF path). Only needed when
    # np_model_to_hf.json is not present at the repo root.
    args.neuronpedia_model_id = os.getenv("NEURONPEDIA_MODEL_ID")

    return args


def list_available_options():
    df = get_saelens_neuronpedia_directory_df()
    df = df[df["neuronpedia_id"].notna()]  # Remove rows with None neuronpedia_id
    models = df["model"].unique()  # type: ignore
    df = df.sort_values(by=["model", "neuronpedia_set"])  # type: ignore

    print("Available models and SAE sets:")
    for model in models:
        print(f"  {model}:")
        model_df = df[df["model"] == model]
        sae_sets = model_df["neuronpedia_set"].unique()  # type: ignore
        for sae_set in sae_sets:
            set_size = len(model_df[model_df["neuronpedia_set"] == sae_set])
            print(f"    - {sae_set} ({set_size} SAEs)")

        print("-" * 80)
