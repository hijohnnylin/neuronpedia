# This script launches the uvicorn server and allows us to pass in arguments instead of using environment variables
# It is often easier to pass in arguments than to set environment variables
# But environment variables will always override the passed in arguments

# Two independent choices, and they are easy to confuse:
#   --attribution-engine  which algorithm builds the graph, so what the graph contains
#   --model-engine        which implementation executes the model, which changes nothing in the graph
#
# Example usages:
# circuit-tracer attribution (default):
#   python start.py --model_id google/gemma-2-2b --transcoder_set gemma
# CRM attribution (lm-saes with Lorsa + Transcoders):
#   python start.py --attribution-engine lm-saes-crm --model_id Qwen/Qwen3-1.7B --sae_repo OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B --sae_expansion 8x --sae_topk k64
# Either of the above, on a different model engine:
#   python start.py --model_id google/gemma-2-2b --transcoder_set gemma --model-engine transformerlens

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize server configuration for Graph Server.")

    # Backend selection
    parser.add_argument(
        "--attribution-engine",
        default="circuit-tracer",
        choices=["circuit-tracer", "lm-saes-crm"],
        help=(
            "Which attribution algorithm builds the graph, and so what the graph contains: "
            "circuit-tracer (default) decomposes MLPs with transcoders; lm-saes-crm additionally "
            "decomposes attention with Lorsa. Separate from --model-engine, which only picks what "
            "executes the model."
        ),
    )

    # Server configuration
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind the server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5004,
        help="Port number for the server to listen on",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes",
    )

    # Model configuration
    parser.add_argument(
        "--model_id",
        default="google/gemma-2-2b",
        help="The HuggingFace ID of the model to load. Default is google/gemma-2-2b.",
    )
    parser.add_argument(
        "--model_dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="The dtype to load the model in. Default is bfloat16.",
    )
    parser.add_argument(
        "--device",
        help="Device to run the model(s) on.",
    )
    parser.add_argument(
        "--model-engine",
        default="interp_engine",
        choices=["interp_engine", "transformerlens", "nnsight"],
        help=(
            "Which implementation executes the model, for either attribution engine. This does "
            "not change what the graph contains. interp_engine (default) hooks the HuggingFace "
            "model in place, reaching any AutoModelForCausalLM; transformerlens converts the "
            "weights, so it only reaches architectures it has a conversion for; nnsight is "
            "circuit-tracer only. Values are underscored to match the libraries' own literals. "
            "The /steer endpoint generates on interp_engine only; graph generation runs on all three."
        ),
    )
    parser.add_argument(
        "--lazy-encoder",
        action="store_true",
        help=(
            "Load transcoder encoder weights only when needed, trading speed for VRAM. Unrelated "
            "to --model-engine: it is passed to the transcoder loader, not the model. Ignored "
            "with a warning for GemmaScope2-format sets (mwhanna/gemma-scope-2-*), whose key "
            "names the lazy path cannot read."
        ),
    )

    # Transcoders configuration (circuit-tracer attribution engine)
    parser.add_argument(
        "--transcoder_set",
        help="Either HF repo ID eg mwhanna/qwen3-4b-transcoders or shortcuts 'gemma' and 'llama'",
    )

    # CRM configuration (lm-saes-crm backend)
    parser.add_argument(
        "--np_model_id",
        default="qwen3-1.7b",
        help="Neuronpedia model ID (used as 'scan' in graph metadata). Only for lm-saes-crm backend.",
    )
    parser.add_argument(
        "--sae_repo",
        default="OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B",
        help="HuggingFace repo for SAE/Lorsa weights. Only for lm-saes-crm backend.",
    )
    parser.add_argument(
        "--sae_expansion",
        default="8x",
        choices=["8x", "32x"],
        help="SAE expansion factor. Only for lm-saes-crm backend.",
    )
    parser.add_argument(
        "--sae_topk",
        default="k64",
        choices=["k64", "k128", "k256"],
        help="SAE top-k sparsity. Only for lm-saes-crm backend.",
    )
    parser.add_argument(
        "--np_transcoder_source_set",
        help="Neuronpedia source set name for transcoder features. Only for lm-saes-crm backend.",
    )
    parser.add_argument(
        "--np_lorsa_source_set",
        help="Neuronpedia source set name for lorsa features. Only for lm-saes-crm backend.",
    )

    # Common settings
    parser.add_argument(
        "--token_limit",
        type=int,
        default=64,
        help="Maximum number of tokens to process",
    )
    parser.add_argument(
        "--max_feature_nodes",
        type=int,
        default=10000,
        help="Default maximum feature nodes for graph generation",
    )
    parser.add_argument(
        "--update_interval",
        type=int,
        default=1000,
        help="Update interval for progress reporting",
    )

    # Uvicorn specific arguments
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--reload-dir",
        default="apps/graph",
        help="Directory to watch for changes when reload is enabled",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Only set environment variables if they don't already exist
    if "ATTRIBUTION_ENGINE" not in os.environ:
        os.environ["ATTRIBUTION_ENGINE"] = args.attribution_engine

    if "MODEL_ENGINE" not in os.environ:
        os.environ["MODEL_ENGINE"] = args.model_engine

    if "LAZY_ENCODER" not in os.environ and args.lazy_encoder:
        os.environ["LAZY_ENCODER"] = "1"

    if "MODEL_ID" not in os.environ:
        os.environ["MODEL_ID"] = args.model_id

    if "TOKEN_LIMIT" not in os.environ:
        os.environ["TOKEN_LIMIT"] = str(args.token_limit)

    if "MAX_FEATURE_NODES" not in os.environ:
        os.environ["MAX_FEATURE_NODES"] = str(args.max_feature_nodes)

    if "UPDATE_INTERVAL" not in os.environ:
        os.environ["UPDATE_INTERVAL"] = str(args.update_interval)

    if "DEVICE" not in os.environ and args.device is not None:
        os.environ["DEVICE"] = args.device

    if "MODEL_DTYPE" not in os.environ:
        os.environ["MODEL_DTYPE"] = args.model_dtype

    if "TRANSCODER_SET" not in os.environ and args.transcoder_set is not None:
        os.environ["TRANSCODER_SET"] = args.transcoder_set

    # CRM-specific env vars
    if "NP_MODEL_ID" not in os.environ:
        os.environ["NP_MODEL_ID"] = args.np_model_id
    if "SAE_REPO" not in os.environ:
        os.environ["SAE_REPO"] = args.sae_repo
    if "SAE_EXPANSION" not in os.environ:
        os.environ["SAE_EXPANSION"] = args.sae_expansion
    if "SAE_TOPK" not in os.environ:
        os.environ["SAE_TOPK"] = args.sae_topk
    if "NP_TRANSCODER_SOURCE_SET" not in os.environ and args.np_transcoder_source_set is not None:
        os.environ["NP_TRANSCODER_SOURCE_SET"] = args.np_transcoder_source_set
    if "NP_LORSA_SOURCE_SET" not in os.environ and args.np_lorsa_source_set is not None:
        os.environ["NP_LORSA_SOURCE_SET"] = args.np_lorsa_source_set

    # Expose bind address to the server module for the LOADING COMPLETE banner.
    if "SERVER_HOST" not in os.environ:
        os.environ["SERVER_HOST"] = args.host
    if "SERVER_PORT" not in os.environ:
        os.environ["SERVER_PORT"] = str(args.port)

    # Build uvicorn command
    uvicorn_args = [
        "python",
        "-m",
        "uvicorn",
        "neuronpedia_graph.server:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--workers",
        str(args.workers),
    ]

    if args.reload:
        uvicorn_args.append("--reload")
        if args.reload_dir:
            uvicorn_args.extend(["--reload-dir", args.reload_dir])

    try:
        subprocess.run(uvicorn_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting server: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)


if __name__ == "__main__":
    main()
