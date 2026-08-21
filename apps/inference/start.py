# this script launches the uvicorn server and allows us to pass in arguments instead of using environment variables
# it is often easier to pass in arguments than to set environment variables
# but environment variables will always override the passed in arguments
# run it with uv (which resolves/activates the project's virtualenv): `uv run python start.py ...`
# example usages
# uv run python start.py --model_id openai-community/gpt2 --sae_sets res-jb --max_loaded_saes 200  --reload --reload-dir neuronpedia_inference --include_sae 5-res-jb --include_sae 4-res-jb
# graph-mode SAE static (vLLM CUDA graphs + additive steer / SAE capture at those sites)
# PYTHONPATH=/root/interp-engine uv run python start.py --static_points sae --jlens_skip
# export INCLUDE_SAE='["9-res-jb"]' && uv run python start.py --reload --reload-dir neuronpedia_inference
# deepseek example
# uv run python start.py --device mps --model_dtype bfloat16 --sae_dtype bfloat16 --model_id meta-llama/Llama-3.1-8B --custom_hf_model_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --sae_sets llamascope-r1-res-32k --max_loaded_saes 200  --reload --reload-dir neuronpedia_inference --include_sae 15-llamascope-slimpj-res-32k
# gemma 2 2b it example
# uv run python start.py --device mps --model_id google/gemma-2-2b-it --model_dtype bfloat16 --sae_dtype bfloat16 --sae_sets gemmascope-res-16k --max_loaded_saes 200  --reload --reload-dir neuronpedia_inference --include_sae 5-gemmascope-res-16k
# no-SAE example (raw activation capture / lens / steer only)
# uv run python start.py --model_id meta-llama/Llama-3.1-8B-Instruct --model_dtype bfloat16 --no_saes --token_limit 2048

import argparse
import json
import os
import subprocess
import sys


def gpu_memory_fraction(value: str) -> float:
    """Parse a GPU memory fraction, catching the percent-instead-of-fraction mistake.

    vLLM wants 0-1. Passing 73 (meaning 73%) otherwise surfaces much later as a
    confusing out-of-memory failure during engine startup.
    """
    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number between 0 and 1, got {value!r}") from None
    if not 0 < parsed <= 1:
        # Only suggest the fraction for values that plausibly ARE percentages (73, 90);
        # for 1.5 "did you mean 0.015?" would be nonsense.
        hint = f"; did you mean {parsed / 100:g}?" if 10 <= parsed <= 100 else ""
        raise argparse.ArgumentTypeError(f"expected a fraction between 0 and 1, got {parsed:g}{hint}")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(description="Initialize server configuration for Neuronpedia Inference Server.")
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
        default="openai-community/gpt2",
        help="Hugging Face repo id of the base model (e.g., 'openai-community/gpt2', 'google/gemma-2-2b')",
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
        "--no_saes",
        action="store_true",
        help="Load no SAEs at all (ignores --sae_sets). The model-only endpoints "
        "(activation/raw, lens, steer, tokenize) still work; SAE-backed ones reject their source set.",
    )
    parser.add_argument(
        "--model_dtype",
        default="auto",
        help="Model dtype: 'auto' loads each checkpoint's native dtype (gemma/qwen bf16, gpt2 fp32); "
        "pass e.g. float32/bfloat16 to force one.",
    )
    parser.add_argument(
        "--sae_dtype",
        default="bfloat16",
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
    parser.add_argument(
        "--sae_gpu_budget_gib",
        default=None,
        help=(
            "Enable SAE paging with this much GPU residency (GiB), or 'auto' to derive it. "
            "Unset keeps every SAE on the GPU for the life of the process."
        ),
    )
    parser.add_argument(
        "--sae_pinned_host_gib",
        type=float,
        default=None,
        help="Cap on page-locked host memory for SAE master copies (default: measured)",
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
    # Backend force override (mutually exclusive). When neither is passed the backend
    # is auto-selected (vLLM on CUDA for vLLM-supported archs, else EagerModel).
    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--force-vllm",
        dest="force_vllm",
        action="store_true",
        default=False,
        help="Force the engine-owned vLLM backend (fast serving). Overrides auto-select.",
    )
    backend_group.add_argument(
        "--force-eager",
        dest="force_eager",
        action="store_true",
        default=False,
        help="Force the interp-engine EagerModel backend (raw transformers, eager PyTorch). Overrides auto-select.",
    )
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=gpu_memory_fraction,
        default=None,
        help="Fraction of GPU memory vLLM may use for model weights + KV cache (vLLM backend "
        "only; default 0.9). Lower it to leave room for the SAEs, which are allocated outside "
        "vLLM's accounting and will otherwise OOM during engine warmup. "
        "local_scripts/sae_memory.py computes per-model values.",
    )
    parser.add_argument(
        "--num-gpus",
        dest="num_gpus",
        type=int,
        default=None,
        help="Shard the model across N GPUs on one node: vLLM tensor_parallel_size, or "
        "EagerModel device_map='auto'. Default 1 (single GPU). e.g. Llama-70B on 4x A40: 4.",
    )
    parser.add_argument(
        "--static_points",
        default=None,
        help="Declared tap set for the engine's vllm-static backend: CUDA graphs kept, holes cut "
        "at these sites only. Omit for the hooked vLLM backend, where every site is reachable and "
        "none are fast. 'auto' declares resid_post at every layer (resid_streams on a "
        "hyper-connection trunk); 'sae' declares the SAE hook sites once those SAEs load; a JSON "
        "list of [name, layer] pairs is an explicit set. Declare ('attn', layer) to keep "
        "attention/DFA. Anything not declared 400s. Set GENERATION_ONLY=true for graphs with no "
        "taps at all. "
        "Env STATIC_POINTS overrides this flag.",
    )
    parser.add_argument(
        "--vllm_cudagraph_capture_sizes",
        default=None,
        help="Comma-separated CUDA-graph batch sizes for vllm-static / vllm-generate pods "
        "(passed to vLLM compilation_config.cudagraph_capture_sizes). Default "
        "1,2,4,8,16,32,64,128,256. vLLM's denser 1..256 ladder is tens of GiB on "
        "DeepSeek-V4. Decode uses 1; a prompt longer than the largest size still runs, "
        "eager for that prefill. Env VLLM_CUDAGRAPH_CAPTURE_SIZES overrides this flag.",
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
        "--jlens_gpu_budget_gib",
        default=None,
        help=(
            "GPU memory (GiB) the lens may keep its per-layer J_bar in, or 'auto' (the default) "
            "to measure what is left of the card once the model and SAEs are up, or 'off'. "
            "Layers that do not fit are re-copied from host memory on every read-out batch."
        ),
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
        os.environ["SAE_SETS"] = json.dumps([] if args.no_saes else args.sae_sets)
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
    if "SAE_GPU_BUDGET_GIB" not in os.environ and args.sae_gpu_budget_gib is not None:
        os.environ["SAE_GPU_BUDGET_GIB"] = str(args.sae_gpu_budget_gib)
    if "SAE_PINNED_HOST_GIB" not in os.environ and args.sae_pinned_host_gib is not None:
        os.environ["SAE_PINNED_HOST_GIB"] = str(args.sae_pinned_host_gib)
    if "CUSTOM_HF_MODEL_ID" not in os.environ and args.custom_hf_model_id is not None:
        os.environ["CUSTOM_HF_MODEL_ID"] = str(args.custom_hf_model_id)
    # Backend force override -> FORCE_BACKEND (only when a flag was passed; leaving it
    # unset lets the server auto-select). --force-vllm / --force-eager are mutually
    # exclusive (argparse-enforced).
    if "FORCE_BACKEND" not in os.environ:
        if args.force_vllm:
            os.environ["FORCE_BACKEND"] = "vllm"
        elif args.force_eager:
            os.environ["FORCE_BACKEND"] = "eager"
    if "NUM_GPUS" not in os.environ and args.num_gpus is not None:
        os.environ["NUM_GPUS"] = str(args.num_gpus)
    if "STATIC_POINTS" not in os.environ and args.static_points is not None:
        os.environ["STATIC_POINTS"] = args.static_points
    if "VLLM_GPU_MEMORY_UTILIZATION" not in os.environ and args.vllm_gpu_memory_utilization is not None:
        os.environ["VLLM_GPU_MEMORY_UTILIZATION"] = str(args.vllm_gpu_memory_utilization)
    if "VLLM_CUDAGRAPH_CAPTURE_SIZES" not in os.environ and args.vllm_cudagraph_capture_sizes is not None:
        os.environ["VLLM_CUDAGRAPH_CAPTURE_SIZES"] = str(args.vllm_cudagraph_capture_sizes)
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
    if "JLENS_GPU_BUDGET_GIB" not in os.environ and args.jlens_gpu_budget_gib is not None:
        os.environ["JLENS_GPU_BUDGET_GIB"] = str(args.jlens_gpu_budget_gib)
    if "NEURONPEDIA_MODEL_ID" not in os.environ and args.neuronpedia_model_id is not None:
        os.environ["NEURONPEDIA_MODEL_ID"] = args.neuronpedia_model_id

    # Passed through so the server can print the address it is serving on in its
    # "loading complete" banner.
    if "SERVER_HOST" not in os.environ:
        os.environ["SERVER_HOST"] = args.host
    if "SERVER_PORT" not in os.environ:
        os.environ["SERVER_PORT"] = str(args.port)

    if args.list_models:
        from neuronpedia_inference.args import list_available_options

        list_available_options()
        return

    # Invoke uvicorn via the current interpreter (`python -m uvicorn`) so it
    # resolves from the active (uv-managed) virtualenv without relying on the
    # `uvicorn` console script being on PATH.
    uvicorn_args = [
        sys.executable,
        "-m",
        "uvicorn",
        "neuronpedia_inference.server:app",
        "--host",
        os.environ["SERVER_HOST"],
        "--port",
        os.environ["SERVER_PORT"],
    ]

    if args.reload:
        uvicorn_args.extend(["--reload"])
        if args.reload_dir:
            uvicorn_args.extend(["--reload-dir", args.reload_dir])

    subprocess.run(uvicorn_args)


if __name__ == "__main__":
    main()
