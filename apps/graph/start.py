# This script launches the uvicorn server and allows us to pass in arguments instead of using environment variables
# It is often easier to pass in arguments than to set environment variables
# But environment variables will always override the passed in arguments

# Example usages:
# python start.py --model_id google/gemma-2-2b --port 5003
# python start.py --model_id meta-llama/Llama-3.2-1B --reload --reload-dir apps/graph
# python start.py --model_id google/gemma-2-2b --host 127.0.0.1 --port 5004 --workers 2

import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Initialize server configuration for Circuit Tracer Server."
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
        help="The ID of the transformerlens model to use. Default is google/gemma-2-2b.",
    )
    parser.add_argument(
        "--model_dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="The dtype of the transformerlens model to use. Default is bfloat16.",
    )
    parser.add_argument(
        "--device",
        help="Device to run the model(s) on.",
    )

    # Transcoders configuration
    parser.add_argument(
        "--transcoder_set",
        help="Either HF repo ID eg mwhanna/qwen3-4b-transcoders or shortcuts 'gemma' and 'llama'",
    )

    # Circuit tracer specific settings
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
