import argparse
import json
import os

import torch

from neuronpedia_inference.config import get_saelens_neuronpedia_directory_df


def parse_env_and_args():
    args = argparse.Namespace()

    args.host = os.getenv("HOST", "0.0.0.0")
    args.port = int(os.getenv("PORT", "5002"))
    args.model_id = os.getenv("MODEL_ID", "gpt2-small")
    args.override_model_id = os.getenv("OVERRIDE_MODEL_ID", None)
    args.custom_hf_model_id = os.getenv("CUSTOM_HF_MODEL_ID", None)
    args.sae_sets = json.loads(os.getenv("SAE_SETS", '["res-jb"]'))
    args.model_dtype = os.getenv("MODEL_DTYPE", "float32")
    args.sae_dtype = os.getenv("SAE_DTYPE", "float32")
    args.token_limit = int(os.getenv("TOKEN_LIMIT", "200"))
    args.device = os.getenv("DEVICE")
    if args.device is None:
        # set device to mps or cuda if available, otherwise cpu
        if torch.backends.mps.is_available():
            args.device = "mps"
        elif torch.cuda.is_available():
            args.device = "cuda"
        else:
            args.device = "cpu"
    args.include_sae = json.loads(os.getenv("INCLUDE_SAE", "[]"))
    args.exclude_sae = json.loads(os.getenv("EXCLUDE_SAE", "[]"))
    args.model_from_pretrained_kwargs = os.getenv("MODEL_FROM_PRETRAINED_KWARGS", "{}")
    args.list_models = os.getenv("LIST_MODELS", "").lower() == "true"
    args.max_loaded_saes = int(os.getenv("MAX_LOADED_SAES", "300"))
    args.sentry_dsn = os.getenv("SENTRY_DSN")
    args.nnsight = os.getenv("NNSIGHT", "").lower() == "true"
    args.chatspace = os.getenv("CHATSPACE", "").lower() == "true"

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
