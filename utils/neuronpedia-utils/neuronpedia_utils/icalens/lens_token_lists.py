"""Builds and loads lens token lists (NeuronTokenList rows) for ICA Lens components and MLP neurons.

Each ICA component, and each MLP neuron's down-projection column, is a residual-stream direction d
at the block output of its layer. A lens reads it as the tokens it raises and lowers:

    J-Lens, J++ Lens:  W_U (g * J_l d), with J_l from neuronpedia/jacobian-lens
    R-Lens:            the lists that the ICA Lens authors computed with their R-lens, from the
                       component profiles in their HF repos (sida/icalens-*). ICA only.

g is the final norm gain, and GPT-2's LayerNorm also centers the direction. POSITIVE is the
tokens that d raises, highest first. NEGATIVE is the tokens that d lowers, lowest first, with
signed values, as the Top Logits columns show them. R-Lens rows keep the ranking only: the
authors' values are full logits after the final norm, so a raised token can have a negative value.

    build <model> [--set S]: writes the token-list files under EXPORTS:
                             icalens: <model>/<layer>-icalens-res/token-lists/{jlens,jpp,rlens}.jsonl.gz
                             neurons: <model>/neurons/<layer>/token-lists/{jlens,jpp}.jsonl.gz
    types:                   upserts the ExplanationType rows into DB_URL.
    load <model> [--set S]:  upserts the rows of the built files into DB_URL. Each neuron must
                             already have its Neuron row.

Model weights come from the HF cache when a shard is there. If not, only the needed tensors are
read from HF with range requests, and kept under TENSOR_CACHE.

Run with:
    uv run --no-project --with torch --with safetensors --with huggingface_hub --with tokenizers \
        --with requests --with "psycopg[binary]" python lens_token_lists.py <step> [model] [--set S]
"""

import argparse
import functools
import gzip
import hashlib
import json
import os
import struct
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

EXPORTS = Path("/data/exports")
TENSOR_CACHE = Path("/data/hf-tensors")
SOURCE_SET = "icalens-res"
NEURON_SET = "neurons"
CREATOR_ID = "clkht01d40000jv08hvalcvly"  # bot-neuronpedia
TOP = 20
COMPONENT_CHUNK = 1024
LOAD_BATCH = 5000
RANGE_CHUNK = 16 << 20
RANGE_WORKERS = 16
JLENS_REPO = "neuronpedia/jacobian-lens"
# Mean top-10 overlap of the plain readout with the dashboards' Top Logits, below which the
# direction, its orientation or the norm is wrong.
MIN_TOP_LOGITS_MATCH = 0.5
# Mean top-10 overlap of the authors' logit lens with ours, below which their components do not line up.
MIN_ALIGNMENT = 0.6

JLENS_TYPE = "jacobian-lens"
RLENS_TYPE = "icalens"
EXPLANATION_TYPES = [
    {
        "name": RLENS_TYPE,
        "displayName": "ICA Lens",
        "description": (
            "Tokens that each ICA Lens component writes, read through an R-lens (Blank, Bhatia and "
            "Nanda) that the ICA Lens authors fitted on 25 Pile-10k prompts. Gemma 2 2B IT uses the "
            "R-lens of Gemma 2 2B."
        ),
        "url": "https://github.com/liusida/ica-lens-paper",
        "creatorName": "Sida Liu and Feijiang Han",
        "citation": (
            "Liu, S. and Han, F. (2026). ICA Lens: Interpreting Language Models Without Training "
            "Another Dictionary. arXiv:2606.11722"
        ),
        "license": "MIT",
        "licenseUrl": "https://github.com/liusida/ica-lens-paper/blob/main/LICENSE",
    },
    {
        "name": JLENS_TYPE,
        "displayName": "Jacobian Lens",
        "description": (
            "Tokens that each feature writes. A fitted lens map carries the feature direction to the "
            "last layer, then the final norm and unembedding read it. J-Lens is Anthropic's Jacobian "
            "lens. J++ Lens is by Ayonrinde and Lindsey."
        ),
        "url": "https://huggingface.co/neuronpedia/jacobian-lens",
        "creatorName": "Neuronpedia",
        "citation": None,
        "license": None,
        "licenseUrl": None,
    },
]

MODELS = {
    "gpt2-small": {
        "hf": "openai-community/gpt2",
        "unembed": "wte.weight",
        "norm": "ln_f.weight",
        "norm_kind": "layernorm",
        "lenses": {
            "J-Lens": "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"
        },
        "rlens": "sida/icalens-gpt2-small-pile10k",
    },
    "gemma-2-2b": {
        "hf": "google/gemma-2-2b",
        "unembed": "model.embed_tokens.weight",
        "norm": "model.norm.weight",
        "norm_kind": "rms1p",
        "lenses": {
            "J-Lens": "gemma-2-2b/jlens/Salesforce-wikitext/gemma-2-2b_jacobian_lens.pt"
        },
        "rlens": "sida/icalens-gemma-2-2b-pile10k",
    },
    "gemma-2-2b-it": {
        "hf": "google/gemma-2-2b-it",
        "unembed": "model.embed_tokens.weight",
        "norm": "model.norm.weight",
        "norm_kind": "rms1p",
        "lenses": {
            "J-Lens": "gemma-2-2b-it/jlens/Salesforce-wikitext/gemma-2-2b-it_jacobian_lens.pt"
        },
        "rlens": "sida/icalens-gemma-2-2b-it-ultrachat-1m",
    },
    "qwen3.5-2b-pt": {
        "hf": "Qwen/Qwen3.5-2B-Base",
        "unembed": "model.language_model.embed_tokens.weight",
        "norm": "model.language_model.norm.weight",
        "norm_kind": "rms1p",
        "lenses": {
            "J-Lens": "qwen3.5-2b-pt/jlens/Salesforce-wikitext/Qwen3.5-2B-Base_jacobian_lens.pt"
        },
        "rlens": "sida/icalens-qwen3.5-2b-base-pile10k",
    },
    "qwen3.5-9b-pt": {
        "hf": "Qwen/Qwen3.5-9B-Base",
        "unembed": "lm_head.weight",
        "norm": "model.language_model.norm.weight",
        "norm_kind": "rms1p",
        "lenses": {
            "J-Lens": "qwen3.5-9b-pt/jlens/Salesforce-wikitext/Qwen3.5-9B-Base_jacobian_lens.pt"
        },
        "rlens": "sida/icalens-qwen3.5-9b-base-pile10k",
    },
    "qwen3.6-27b": {
        "hf": "Qwen/Qwen3.6-27B",
        "unembed": "lm_head.weight",
        "norm": "model.language_model.norm.weight",
        "norm_kind": "rms1p",
        "lenses": {
            "J-Lens": "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt",
            "J++ Lens": "qwen3.6-27b/jpp/Salesforce-wikitext/Qwen3.6-27B_jpp_lens.pt",
        },
        "rlens": None,
    },
}

# MLP neurons. "down" is the weight that writes each neuron into the residual stream, with one
# neuron per column, or per row where "rows_are_neurons".
NEURON_MODELS = {
    "gpt2-small": {
        "hf": "openai-community/gpt2",
        "unembed": "wte.weight",
        "norm": "ln_f.weight",
        "norm_kind": "layernorm",
        "down": "h.{}.mlp.c_proj.weight",
        "rows_are_neurons": True,
        "lenses": {
            "J-Lens": "gpt2-small/jlens/Salesforce-wikitext/gpt2_jacobian_lens.pt"
        },
    },
    "llama3.1-8b-it": {
        "hf": "meta-llama/Llama-3.1-8B-Instruct",
        "unembed": "lm_head.weight",
        "norm": "model.norm.weight",
        "norm_kind": "rms",
        "down": "model.layers.{}.mlp.down_proj.weight",
        "lenses": {
            "J-Lens": "llama3.1-8b-it/jlens/Salesforce-wikitext/Llama-3.1-8B-Instruct_jacobian_lens.pt",
            "J++ Lens": "llama3.1-8b-it/jpp/Salesforce-wikitext/Llama-3.1-8B-Instruct_jpp_lens.pt",
        },
    },
    "llama3.3-70b-it": {
        "hf": "meta-llama/Llama-3.3-70B-Instruct",
        "unembed": "lm_head.weight",
        "norm": "model.norm.weight",
        "norm_kind": "rms",
        "down": "model.layers.{}.mlp.down_proj.weight",
        "lenses": {
            "J-Lens": "llama3.3-70b-it/jlens/Salesforce-wikitext/Llama-3.3-70B-Instruct_jacobian_lens.pt"
        },
    },
    "qwen3-8b": {
        "hf": "Qwen/Qwen3-8B",
        "unembed": "lm_head.weight",
        "norm": "model.norm.weight",
        "norm_kind": "rms",
        "down": "model.layers.{}.mlp.down_proj.weight",
        "lenses": {
            "J-Lens": "qwen3-8b/jlens/Salesforce-wikitext/Qwen3-8B_jacobian_lens.pt"
        },
    },
    "qwen3-14b": {
        "hf": "Qwen/Qwen3-14B",
        "unembed": "lm_head.weight",
        "norm": "model.norm.weight",
        "norm_kind": "rms",
        "down": "model.layers.{}.mlp.down_proj.weight",
        "lenses": {
            "J-Lens": "qwen3-14b/jlens/Salesforce-wikitext/Qwen3-14B_jacobian_lens.pt"
        },
    },
    "qwen3-32b": {
        "hf": "Qwen/Qwen3-32B",
        "unembed": "lm_head.weight",
        "norm": "model.norm.weight",
        "norm_kind": "rms",
        "down": "model.layers.{}.mlp.down_proj.weight",
        "lenses": {
            "J-Lens": "qwen3-32b/jlens/Salesforce-wikitext/Qwen3-32B_jacobian_lens.pt"
        },
    },
    "qwen3.5-4b": {
        "hf": "Qwen/Qwen3.5-4B",
        "unembed": "model.language_model.embed_tokens.weight",
        "norm": "model.language_model.norm.weight",
        "norm_kind": "rms1p",
        "down": "model.language_model.layers.{}.mlp.down_proj.weight",
        "lenses": {
            "J-Lens": "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens.pt",
            "J++ Lens": "qwen3.5-4b/jpp/Salesforce-wikitext/Qwen3.5-4B_jpp_lens.pt",
        },
    },
}
SETS = {SOURCE_SET: MODELS, NEURON_SET: NEURON_MODELS}
LENS_FILE_STEM = {"J-Lens": "jlens", "J++ Lens": "jpp", "R-Lens": "rlens"}


def db_url() -> str:
    """The database URL, never printed."""
    return os.environ["DB_URL"]


def source_layers(model: str, source_set: str) -> list[int]:
    """The layers of the model that have a source in the set, from the database."""
    import psycopg

    with psycopg.connect(db_url()) as conn:
        rows = conn.execute(
            'SELECT id FROM "Source" WHERE "modelId" = %s AND "setName" = %s',
            (model, source_set),
        ).fetchall()
    return sorted(int(r[0].split("-")[0]) for r in rows)


def layer_id(layer: int, source_set: str) -> str:
    """The Neuron layer (Source id) of a layer in the set."""
    return f"{layer}-{SOURCE_SET}" if source_set == SOURCE_SET else str(layer)


def layer_dir(model: str, layer: int, source_set: str) -> Path:
    """The export folder of one layer of the set."""
    if source_set == SOURCE_SET:
        return EXPORTS / model / f"{layer}-{SOURCE_SET}"
    return EXPORTS / model / NEURON_SET / str(layer)


def score(v: float) -> float:
    """A rounded score. Scores below 1e-3 keep 3 significant digits."""
    v = float(v)
    return round(v, 4) if abs(v) >= 1e-3 else float(f"{v:.3g}")


def row(
    model: str,
    layer: str,
    index: int,
    type_name: str,
    name: str,
    pole: str,
    tokens,
    values,
) -> dict:
    """A NeuronTokenList row with an id that is the same on each run."""
    key = json.dumps([model, layer, index, type_name, name, pole], ensure_ascii=False)
    return {
        "id": "tl" + hashlib.sha1(key.encode()).hexdigest()[:23],
        "modelId": model,
        "layer": layer,
        "index": str(index),
        "typeName": type_name,
        "creatorId": CREATOR_ID,
        "name": name,
        "pole": pole,
        "aspect": "STEERING",
        "tokens": list(tokens),
        "values": [score(v) for v in values],
        "metadata": None,
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    """Writes rows as gzipped JSON lines, by way of a temp file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt", compresslevel=5) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


@functools.cache
def weight_map(repo: str) -> dict[str, str]:
    """The shard of each tensor of a HF model. A model with one file has no index."""
    from huggingface_hub import hf_hub_download, list_repo_files

    name = next(
        (f for f in list_repo_files(repo) if f.endswith("safetensors.index.json")), None
    )
    if name is None:
        return {}
    with open(hf_hub_download(repo, name)) as f:
        return json.load(f)["weight_map"]


def http_range(session, url: str, start: int, end: int) -> bytes:
    """Bytes start..end (inclusive) of a URL, retrying a slow or short read up to 10 times."""
    import requests

    for attempt in range(10):
        try:
            r = session.get(
                url, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 60)
            )
            r.raise_for_status()
            if len(r.content) == end - start + 1:
                return r.content
            error = f"short read {len(r.content)}"
        except requests.RequestException as exc:
            error = str(exc)
        print(f"    range {start}-{end} retry {attempt}: {error}", flush=True)
    raise RuntimeError(f"range {start}-{end} of {url} failed")


def fetch_tensor(repo: str, shard: str, key: str, out: Path) -> None:
    """Reads one tensor from a HF safetensors shard with range requests, into its own file."""
    import requests
    import torch
    from safetensors.torch import save_file

    token = os.environ.get("HF_TOKEN")
    auth = {"Authorization": f"Bearer {token}"} if token else {}
    head = requests.head(
        f"https://huggingface.co/{repo}/resolve/main/{shard}",
        headers=auth,
        allow_redirects=True,
        timeout=60,
    )
    head.raise_for_status()
    session = requests.Session()
    n = struct.unpack("<Q", http_range(session, head.url, 0, 7))[0]
    info = json.loads(http_range(session, head.url, 8, 7 + n))[key]
    begin, end = (8 + n + o for o in info["data_offsets"])
    buf = bytearray(end - begin)

    def part(offset: int) -> None:
        stop = min(offset + RANGE_CHUNK, len(buf))
        buf[offset:stop] = http_range(
            session, head.url, begin + offset, begin + stop - 1
        )

    with ThreadPoolExecutor(RANGE_WORKERS) as pool:
        list(pool.map(part, range(0, len(buf), RANGE_CHUNK)))
    dtype = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}[
        info["dtype"]
    ]
    tensor = torch.frombuffer(buf, dtype=dtype).reshape(info["shape"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    save_file({key: tensor}, str(tmp))
    tmp.replace(out)


def get_tensor(repo: str, key: str):
    """One tensor of a HF model, from the HF cache if its shard is there, else by range reads."""
    from huggingface_hub import try_to_load_from_cache
    from safetensors import safe_open
    from safetensors.torch import load_file

    shard = weight_map(repo).get(key, "model.safetensors")
    cached = try_to_load_from_cache(repo, shard)
    if isinstance(cached, str):
        with safe_open(cached, framework="pt") as f:
            return f.get_tensor(key)
    local = TENSOR_CACHE / repo / f"{key}.safetensors"
    if not local.exists():
        fetch_tensor(repo, shard, key, local)
    return load_file(str(local))[key]


def load_weights(spec: dict, device: str):
    """W_U [vocab, d] and the final norm gain [d]."""
    W_U = get_tensor(spec["hf"], spec["unembed"]).float().to(device)
    norm = get_tensor(spec["hf"], spec["norm"]).float().to(device)
    return W_U, 1.0 + norm if spec["norm_kind"] == "rms1p" else norm


def load_lens(path: str) -> dict[int, "object"]:
    """The per-layer maps of a lens file, in either the Jacobian lens or the J++ layout."""
    import torch
    from huggingface_hub import hf_hub_download

    # EXPORTS keeps some lens files at the same path as the HF repo.
    local = EXPORTS / path
    file = local if local.exists() else hf_hub_download(JLENS_REPO, path)
    checkpoint = torch.load(file, map_location="cpu", weights_only=True)
    maps = (
        checkpoint["J"] if "J" in checkpoint else checkpoint["parameters"]["jacobians"]
    )
    return {int(k): v for k, v in maps.items()}


def ica_decoder(model: str, layer: int, device: str):
    """The unit-norm ICA directions of one layer, [n_components, d], in Neuron index order."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    path = hf_hub_download(
        f"neuronpedia/icalens-{model}",
        f"blocks.{layer}.hook_resid_post/sae_weights.safetensors",
    )
    return load_file(path)["W_dec"].float().to(device)


def neuron_directions(model: str, layer: int, device: str):
    """What each MLP neuron of one layer writes, [n_neurons, d], in Neuron index order."""
    spec = NEURON_MODELS[model]
    W = get_tensor(spec["hf"], spec["down"].format(layer))
    return (W if spec.get("rows_are_neurons") else W.T).float().to(device)


def read_out(directions, W_U, gain, center: bool, n_vocab: int):
    """Top and bottom TOP token ids and logits of each direction, all on the GPU."""
    import torch

    tops, bottoms = [], []
    for start in range(0, directions.shape[0], COMPONENT_CHUNK):
        v = directions[start : start + COMPONENT_CHUNK]
        if center:
            v = v - v.mean(-1, keepdim=True)
        logits = (v * gain) @ W_U[:n_vocab].T
        tops.append(logits.topk(TOP, dim=-1))
        bottoms.append((-logits).topk(TOP, dim=-1))
    top_ids = torch.cat([t.indices for t in tops]).cpu()
    top_vals = torch.cat([t.values for t in tops]).cpu()
    bottom_ids = torch.cat([b.indices for b in bottoms]).cpu()
    bottom_vals = -torch.cat([b.values for b in bottoms]).cpu()
    return top_ids, top_vals, bottom_ids, bottom_vals


def dashboard_top_logits(model: str, layer: int) -> dict[int, list[str]]:
    """The first 10 Top Logits tokens of each neuron in the layer's feature exports, if there are any."""
    out = {}
    for path in (layer_dir(model, layer, NEURON_SET) / "features").glob(
        "batch-*.jsonl.gz"
    ):
        with gzip.open(path, "rt") as f:
            for line in f:
                r = json.loads(line)
                out[int(r["index"])] = r["pos_str"][:10]
    return out


def rlens_rows(model: str, layer: int, plain_top_ids) -> tuple[list[dict], float]:
    """The authors' R-lens rows of one layer, and how well their components line up with ours."""
    from huggingface_hub import hf_hub_download

    repo = MODELS[model]["rlens"]
    path = hf_hub_download(
        repo, f"component_profiles/resid_post/layer_{layer:02d}.json.gz"
    )
    with gzip.open(path, "rt") as f:
        profile = json.load(f)
    rows, overlaps = [], []
    for comp in profile["components"]:
        index = int(comp["component"])
        theirs = {
            t["token_id"] for t in comp["logit_lens"]["dominant"]["top_tokens"][:10]
        }
        overlaps.append(len(theirs & set(plain_top_ids[index][:10].tolist())) / 10)
        r_lens = comp.get("r_lens")
        if not r_lens:
            continue
        dominant = r_lens["dominant"]
        for pole, key in (("POSITIVE", "top_tokens"), ("NEGATIVE", "bottom_tokens")):
            tokens = dominant[key][:TOP]
            rows.append(
                row(
                    model,
                    layer_id(layer, SOURCE_SET),
                    index,
                    RLENS_TYPE,
                    "R-Lens",
                    pole,
                    [t["token"] if "\x00" in t["text"] else t["text"] for t in tokens],
                    [],
                )
            )
    return rows, sum(overlaps) / len(overlaps)


def build(model: str, source_set: str) -> None:
    """Writes the token-list files of every lens at every source of the model in the set."""
    import torch
    from tokenizers import Tokenizer

    spec = SETS[source_set][model]
    neurons = source_set == NEURON_SET
    device = "cuda"
    layers = source_layers(model, source_set)
    print(f"{model} {source_set}: {len(layers)} sources, layers {layers}", flush=True)
    tokenizer = Tokenizer.from_pretrained(spec["hf"], token=os.environ.get("HF_TOKEN"))
    n_vocab = tokenizer.get_vocab_size(with_added_tokens=True)
    decoded: dict[int, str] = {}

    def text(i: int) -> str:
        if i not in decoded:
            s = tokenizer.decode([i], skip_special_tokens=False)
            # Postgres text cannot hold NUL, so keep the vocab string, as the dashboards do.
            decoded[i] = tokenizer.id_to_token(i) if "\x00" in s else s
        return decoded[i]

    def directions(layer: int):
        if neurons:
            return neuron_directions(model, layer, "cpu")
        return ica_decoder(model, layer, "cpu")

    W_U, gain = load_weights(spec, device)
    center = spec["norm_kind"] == "layernorm"
    n_vocab = min(n_vocab, W_U.shape[0])
    lenses = {name: load_lens(path) for name, path in spec["lenses"].items()}
    last_layer = max(max(maps) for maps in lenses.values()) + 1

    with ThreadPoolExecutor(1) as prefetch, torch.no_grad():
        upcoming = prefetch.submit(directions, layers[0])
        for i, layer in enumerate(layers):
            D = upcoming.result().to(device)
            if i + 1 < len(layers):
                upcoming = prefetch.submit(directions, layers[i + 1])
            lid = layer_id(layer, source_set)
            folder = layer_dir(model, layer, source_set) / "token-lists"
            plain = read_out(D, W_U, gain, center, n_vocab)
            if neurons and (theirs := dashboard_top_logits(model, layer)):
                overlaps = [
                    len(
                        {text(int(t)) for t in plain[0][index][:10]}
                        & set(theirs[index])
                    )
                    / 10
                    for index in theirs
                ]
                match = sum(overlaps) / len(overlaps)
                print(
                    f"  L{layer} plain readout vs dashboard Top Logits: {match:.2f}",
                    flush=True,
                )
                if match < MIN_TOP_LOGITS_MATCH:
                    raise ValueError(
                        f"L{layer}: plain readout matches Top Logits at only {match:.2f}"
                    )
            for name, maps in lenses.items():
                if layer in maps:
                    J = maps[layer].float().to(device)
                    transported = D @ J.T
                elif layer == last_layer:
                    transported = D
                else:
                    raise ValueError(f"{name} has no map for layer {layer}")
                top_ids, top_vals, bottom_ids, bottom_vals = read_out(
                    transported, W_U, gain, center, n_vocab
                )
                rows = []
                for index in range(D.shape[0]):
                    for pole, ids, vals in (
                        ("POSITIVE", top_ids[index], top_vals[index]),
                        ("NEGATIVE", bottom_ids[index], bottom_vals[index]),
                    ):
                        rows.append(
                            row(
                                model,
                                lid,
                                index,
                                JLENS_TYPE,
                                name,
                                pole,
                                [text(int(i)) for i in ids],
                                vals.tolist(),
                            )
                        )
                write_rows(folder / f"{LENS_FILE_STEM[name]}.jsonl.gz", rows)
                same = (
                    (top_ids[:, :10, None] == plain[0][:, None, :10])
                    .any(-1)
                    .float()
                    .mean()
                )
                print(
                    f"  L{layer} {name}: {len(rows):,} rows, top-10 overlap with Top Logits {same:.2f}",
                    flush=True,
                )
            if not neurons and spec["rlens"]:
                rows, alignment = rlens_rows(model, layer, plain[0])
                if alignment < MIN_ALIGNMENT:
                    raise ValueError(
                        f"L{layer}: authors' components line up at only {alignment:.2f}"
                    )
                write_rows(folder / f"{LENS_FILE_STEM['R-Lens']}.jsonl.gz", rows)
                print(
                    f"  L{layer} R-Lens: {len(rows):,} rows, alignment {alignment:.2f}",
                    flush=True,
                )


def upsert_types() -> None:
    """Inserts or updates the ExplanationType rows."""
    import psycopg

    with psycopg.connect(db_url()) as conn:
        for t in EXPLANATION_TYPES:
            conn.execute(
                'INSERT INTO "ExplanationType" (name, "displayName", description, url, "creatorName", '
                '"creatorId", citation, license, "licenseUrl", featured, "isAttention", "updatedAt") '
                "VALUES (%(name)s, %(displayName)s, %(description)s, %(url)s, %(creatorName)s, %(creatorId)s, "
                "%(citation)s, %(license)s, %(licenseUrl)s, false, false, now()) "
                'ON CONFLICT (name) DO UPDATE SET "displayName" = EXCLUDED."displayName", '
                'description = EXCLUDED.description, url = EXCLUDED.url, "creatorName" = EXCLUDED."creatorName", '
                'citation = EXCLUDED.citation, license = EXCLUDED.license, "licenseUrl" = EXCLUDED."licenseUrl", '
                '"updatedAt" = now()',
                {**t, "creatorId": CREATOR_ID},
            )
        print("upserted", [t["name"] for t in EXPLANATION_TYPES])


def explanation_type_lines() -> list[str]:
    """The ExplanationType rows as lines for config/explanation_type.jsonl."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    return [
        json.dumps(
            {
                **t,
                "creatorId": CREATOR_ID,
                "createdAt": now,
                "updatedAt": now,
                "settings": None,
                "featured": False,
                "explainerModelSettingsId": None,
                "isAttention": False,
            },
            ensure_ascii=False,
        )
        for t in EXPLANATION_TYPES
    ]


def built_files(model: str, source_set: str) -> list[Path]:
    """The token-list files that build wrote for the model in the set."""
    if source_set == SOURCE_SET:
        return sorted((EXPORTS / model).glob(f"*-{SOURCE_SET}/token-lists/*.jsonl.gz"))
    stems = [LENS_FILE_STEM[name] for name in NEURON_MODELS[model]["lenses"]]
    return sorted(
        p
        for stem in stems
        for p in (EXPORTS / model / NEURON_SET).glob(f"*/token-lists/{stem}.jsonl.gz")
    )


def load(model: str, source_set: str, workers: int) -> None:
    """Upserts the rows of the model's built files, checking each batch's row count."""
    import psycopg

    columns = [
        "id",
        "modelId",
        "layer",
        "index",
        "typeName",
        "creatorId",
        "name",
        "pole",
        "aspect",
        "tokens",
        "values",
        "metadata",
    ]
    cols = ", ".join(f'"{c}"' for c in columns)
    sql = (
        f'INSERT INTO "NeuronTokenList" ({cols}) SELECT {cols} '
        'FROM jsonb_populate_recordset(NULL::"NeuronTokenList", %s::jsonb) '
        'ON CONFLICT ("modelId", "layer", "index", "typeName", "name", "pole") DO UPDATE SET '
        'tokens = EXCLUDED.tokens, "values" = EXCLUDED."values", metadata = EXCLUDED.metadata, '
        '"updatedAt" = now()'
    )

    def load_file(path: Path) -> int:
        with gzip.open(path, "rt") as f:
            rows = [json.loads(line) for line in f]
        if any("\x00" in t for r in rows for t in r["tokens"]):
            raise ValueError(f"{path}: a token holds NUL; rebuild it")
        with psycopg.connect(db_url()) as conn:
            for start in range(0, len(rows), LOAD_BATCH):
                batch = rows[start : start + LOAD_BATCH]
                with conn.transaction():
                    cur = conn.execute(sql, (json.dumps(batch, ensure_ascii=False),))
                    if cur.rowcount != len(batch):
                        raise ValueError(
                            f"{path}: wrote {cur.rowcount} of {len(batch)} rows"
                        )
        print(f"  {path.relative_to(EXPORTS)}: {len(rows):,} rows", flush=True)
        return len(rows)

    files = built_files(model, source_set)
    with ThreadPoolExecutor(workers) as pool:
        total = sum(pool.map(load_file, files))
    print(f"{model}: {total:,} rows in {len(files)} files", flush=True)


def main() -> None:
    """Parses the step and runs it."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("step", choices=["build", "types", "type-lines", "load"])
    parser.add_argument(
        "model", nargs="?", choices=sorted(set(MODELS) | set(NEURON_MODELS))
    )
    parser.add_argument(
        "--set", dest="source_set", choices=sorted(SETS), default=SOURCE_SET
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="parallel connections for load"
    )
    args = parser.parse_args()
    if args.step == "types":
        upsert_types()
    elif args.step == "type-lines":
        print("\n".join(explanation_type_lines()))
    elif args.model is None:
        parser.error(f"{args.step} needs a model")
    elif args.model not in SETS[args.source_set]:
        parser.error(f"{args.model} has no {args.source_set} spec")
    elif args.step == "build":
        build(args.model, args.source_set)
    else:
        load(args.model, args.source_set, args.workers)


if __name__ == "__main__":
    main()
