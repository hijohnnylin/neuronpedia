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
        help="Use nnsight with nnterp StandardizedTransformer. Not all models are currently supported.",
    )
    parser.add_argument(
        "--nnsight-vllm",
        action="store_true",
        dest="nnsight_vllm",
        help="Use nnsight with nnterp StandardizedVLLM (vLLM backend). Provides faster inference.",
    )
    parser.add_argument(
        "--chatspace",
        action="store_true",
        help="Use chatspace engine.",
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
    if "NNSIGHT_VLLM" not in os.environ:
        os.environ["NNSIGHT_VLLM"] = "true" if args.nnsight_vllm else "false"
    if "CHATSPACE" not in os.environ:
        os.environ["CHATSPACE"] = "true" if args.chatspace else "false"
        
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
