# this script launches the uvicorn server and allows us to pass in arguments instead of using environment variables
# it is often easier to pass in arguments than to set environment variables
# but environment variables will always override the passed in arguments
# example usages
# python start.py --model_id gpt2-small --sae_sets res-jb --max_loaded_saes 200  --reload --reload-dir neuronpedia_inference --include_sae 5-res-jb --include_sae 4-res-jb
# export INCLUDE_SAE='["9-res-jb"]' && python start.py --reload --reload-dir neuronpedia_inference
# deepseek example
# python start.py --device mps --model_dtype bfloat16 --sae_dtype bfloat16 --model_id meta-llama/Llama-3.1-8B --custom_hf_model_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --sae_sets llamascope-r1-res-32k --max_loaded_saes 200  --reload --reload-dir neuronpedia_inference --include_sae 15-llamascope-slimpj-res-32k
# gemma 2 2b it example
# python start.py --device mps --model_id gemma-2-2b --model_dtype bfloat16 --sae_dtype bfloat16 --override_model_id gemma-2-2b-it --sae_sets gemmascope-res-16k --max_loaded_saes 200  --reload --reload-dir neuronpedia_inference --include_sae 5-gemmascope-res-16k

import argparse
import json
import os
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize server configuration for Neuronpedia Inference Server."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5002,
        help="Port number for the server to listen on",
    )
    parser.add_argument(
        "--model_id",
        default="gpt2-small",
        help="The ID of the base model to use (e.g., 'gpt2-small', 'gemma-2-2b')",
    )
    parser.add_argument(
        "--override_model_id",
        default=None,
        help="Optional: Override the model ID for instantiation. This is used to run the Gemma-2-2B SAEs on the Gemma-2-2B-Instruct model.",
    )
    parser.add_argument(
        "--custom_hf_model_id",
        default=None,
        help="Optional: Use a custom HF model ID that is not directly supported by TransformerLens. This is used to run the deepseek-ai/DeepSeek-R1-Distill-Llama-8B model.",
    )
    parser.add_argument(
        "--sae_sets",
        default=["res-jb"],
        nargs="+",
        help="List of SAE sets to load. Can specify multiple.",
    )
    parser.add_argument(
        "--model_dtype",
        default="float32",
        help="Data type for model computations",
    )
    parser.add_argument(
        "--sae_dtype",
        default="float32",
        help="Data type for SAE computations",
    )
    parser.add_argument(
        "--token_limit",
        type=int,
        default=200,
        help="Maximum number of tokens to process",
    )
    parser.add_argument(
        "--lens_token_limit",
        type=int,
        default=1024,
        help="Maximum number of tokens for the lens endpoints only (logit/jacobian lens). Independent of --token_limit.",
    )
    parser.add_argument(
        "--device",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--include_sae",
        action="append",
        default=[],
        help="Regex pattern to include SAEs",
    )
    parser.add_argument(
        "--exclude_sae",
        action="append",
        default=[],
        help="Regex pattern to exclude SAEs",
    )
    parser.add_argument(
        "--model_from_pretrained_kwargs",
        default="{}",
        help="JSON string of additional keyword arguments",
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help="List available models and SAE sets",
    )
    parser.add_argument(
        "--max_loaded_saes",
        type=int,
        default=500,
        help="Maximum number of SAEs to keep loaded",
    )
    # Uvicorn specific arguments
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--reload-dir",
        default="neuronpedia_inference",
        help="Directory to watch for changes when reload is enabled",
    )
    parser.add_argument(
        "--nnsight",
        action="store_true",
        help="Use nnsight. Not all models are currently supported.",
    )
    parser.add_argument(
        "--nnsight_max_memory",
        type=str,
        default=None,
        help="Max GPU memory in GB for nnsight model loading. Single value applies to all GPUs (e.g. '48'), comma-separated for per-GPU (e.g. '16,46'). If not specified, uses the nnsight default (~90%% of available memory).",
    )
    parser.add_argument(
        "--chatspace",
        action="store_true",
        help="Use chatspace engine.",
    )
    # Lens endpoints (logit lens / jacobian lens)
    parser.add_argument(
        "--jlens_skip",
        action="store_true",
        help="Skip loading the fitted Jacobian lens at startup. LOGIT_LENS still works; JACOBIAN_LENS requests return an error.",
    )
    parser.add_argument(
        "--jlens_source",
        default=None,
        help="Optional absolute path to a local directory containing a fitted lens (e.g. .../<np_model_id>/jlens/Salesforce-wikitext). When set, used instead of downloading from Hugging Face.",
    )
    parser.add_argument(
        "--jlens_dataset",
        default="Salesforce-wikitext",
        help="Dataset folder name the lens was fit on (used in the HF path / local path).",
    )
    parser.add_argument(
        "--jlens_hf_repo",
        default="neuronpedia/jacobian-lens",
        help="Hugging Face model repo holding fitted lenses, keyed by neuronpedia model id under '<np_model_id>/jlens/<dataset>/<slug>_jacobian_lens.pt'.",
    )
    parser.add_argument(
        "--jlens_hf_path",
        default=None,
        help="Optional exact path (within the HF repo) to the lens .pt file. When set, used verbatim instead of deriving it from the model id / dataset.",
    )
    parser.add_argument(
        "--neuronpedia_model_id",
        default=None,
        help="Explicit neuronpedia model id (used to build the HF path). Only needed when np_model_to_hf.json is not present at the repo root.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Only set environment variables if they don't already exist
    if "MODEL_ID" not in os.environ:
        os.environ["MODEL_ID"] = args.model_id
    if args.override_model_id and "OVERRIDE_MODEL_ID" not in os.environ:
        os.environ["OVERRIDE_MODEL_ID"] = args.override_model_id
    if "SAE_SETS" not in os.environ:
        os.environ["SAE_SETS"] = json.dumps(args.sae_sets)
    if "MODEL_DTYPE" not in os.environ:
        os.environ["MODEL_DTYPE"] = args.model_dtype
    if "SAE_DTYPE" not in os.environ:
        os.environ["SAE_DTYPE"] = args.sae_dtype
    if "TOKEN_LIMIT" not in os.environ:
        os.environ["TOKEN_LIMIT"] = str(args.token_limit)
    if "LENS_TOKEN_LIMIT" not in os.environ:
        os.environ["LENS_TOKEN_LIMIT"] = str(args.lens_token_limit)
    if "DEVICE" not in os.environ and args.device is not None:
        os.environ["DEVICE"] = args.device
    if "INCLUDE_SAE" not in os.environ:
        os.environ["INCLUDE_SAE"] = json.dumps(args.include_sae)
    if "EXCLUDE_SAE" not in os.environ:
        os.environ["EXCLUDE_SAE"] = json.dumps(args.exclude_sae)
    if "MODEL_FROM_PRETRAINED_KWARGS" not in os.environ:
        os.environ["MODEL_FROM_PRETRAINED_KWARGS"] = args.model_from_pretrained_kwargs
    if "MAX_LOADED_SAES" not in os.environ:
        os.environ["MAX_LOADED_SAES"] = str(args.max_loaded_saes)
    if "CUSTOM_HF_MODEL_ID" not in os.environ and args.custom_hf_model_id is not None:
        os.environ["CUSTOM_HF_MODEL_ID"] = str(args.custom_hf_model_id)
    if "NNSIGHT" not in os.environ:
        os.environ["NNSIGHT"] = "true" if args.nnsight else "false"
    if "NNSIGHT_MAX_MEMORY" not in os.environ and args.nnsight_max_memory is not None:
        os.environ["NNSIGHT_MAX_MEMORY"] = args.nnsight_max_memory
    if "CHATSPACE" not in os.environ:
        os.environ["CHATSPACE"] = "true" if args.chatspace else "false"
    if "JLENS_SKIP" not in os.environ:
        os.environ["JLENS_SKIP"] = "true" if args.jlens_skip else "false"
    if "JLENS_SOURCE" not in os.environ and args.jlens_source is not None:
        os.environ["JLENS_SOURCE"] = args.jlens_source
    if "JLENS_DATASET" not in os.environ:
        os.environ["JLENS_DATASET"] = args.jlens_dataset
    if "JLENS_HF_REPO" not in os.environ:
        os.environ["JLENS_HF_REPO"] = args.jlens_hf_repo
    if "JLENS_HF_PATH" not in os.environ and args.jlens_hf_path is not None:
        os.environ["JLENS_HF_PATH"] = args.jlens_hf_path
    if "NEURONPEDIA_MODEL_ID" not in os.environ and args.neuronpedia_model_id is not None:
        os.environ["NEURONPEDIA_MODEL_ID"] = args.neuronpedia_model_id

    if args.list_models:
        from neuronpedia_inference.args import list_available_options

        list_available_options()
        return

    uvicorn_args = [
        "uvicorn",
        "neuronpedia_inference.server:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.reload:
        uvicorn_args.extend(["--reload"])
        if args.reload_dir:
            uvicorn_args.extend(["--reload-dir", args.reload_dir])

    subprocess.run(uvicorn_args)


if __name__ == "__main__":
    main()
