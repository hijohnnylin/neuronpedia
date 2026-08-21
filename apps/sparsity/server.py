"""
FastAPI server for analyzing MLP neuron connections in sparse circuit models.
"""

import os
import sys
import types
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

import sentry_sdk
import torch
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from transformers import AutoModelForCausalLM
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from schemas import (
    ChannelConnectionsResponse,
    ChannelNeuron,
    HealthResponse,
    NeuronConnectionsResponse,
    TraceNode,
)

if TYPE_CHECKING:
    from sentry_sdk._types import Event, Hint

# Load .env file if present
load_dotenv()

# Callers authenticate with a shared secret in `x-secret-key`, and it reaches Sentry by two
# routes. The obvious one is `request.headers`: the SDK redacts Authorization, Cookie and
# X-Api-Key, but its list does not include this header. The other is stack-trace locals -- an
# error inside a route serializes every frame's variables, and the raw ASGI `scope` (headers
# included) is a local in roughly twenty of them, so scrubbing by header name alone still ships
# the secret about twenty times over. The value sweep is what covers that; the key passes above
# are still worth keeping, since they redact the header on transactions, which carry no locals.
SENTRY_SCRUBBED_HEADERS = frozenset({"x-secret-key"})
_SENTRY_HEADER_ATTR_PREFIX = "http.request.header."
_SENTRY_FILTERED = "[Filtered]"


def _redact_sentry_value(node: "Any", secret: str) -> None:
    """Replace every occurrence of `secret` in a nested event payload, in place."""
    if isinstance(node, dict):
        pairs = list(node.items())
    elif isinstance(node, list):
        pairs = list(enumerate(node))
    else:
        return
    for key, value in pairs:
        if isinstance(value, str) and secret in value:
            node[key] = value.replace(secret, _SENTRY_FILTERED)
        else:
            _redact_sentry_value(value, secret)


def scrub_sentry_event(event: "Event", _hint: "Hint") -> "Event":
    """Redact the auth header wherever it reaches an event: headers, span attributes, locals."""
    request: Any = event.get("request")
    headers = request.get("headers") if isinstance(request, dict) else None
    if isinstance(headers, dict):
        for key in headers:
            if key.lower() in SENTRY_SCRUBBED_HEADERS:
                headers[key] = _SENTRY_FILTERED

    # Transactions carry the same headers again as span attributes, so scrubbing only `request`
    # would still ship the secret on the sampled quarter of traffic.
    contexts: Any = event.get("contexts") or {}
    trace: Any = contexts.get("trace") or {}
    spans: Any = event.get("spans") or []
    for data in [trace.get("data"), *(span.get("data") for span in spans)]:
        if not isinstance(data, dict):
            continue
        for key in data:
            if key.lower().removeprefix(_SENTRY_HEADER_ATTR_PREFIX) in SENTRY_SCRUBBED_HEADERS:
                data[key] = _SENTRY_FILTERED

    # Guarded on non-empty: `"".replace` would rewrite every string in the event.
    secret = os.environ.get("SECRET")
    if secret:
        _redact_sentry_value(event, secret)
    return event


# Error reporting. Off entirely without a DSN, so a local run needs no Sentry account. This has
# to precede the `FastAPI(...)` below: the integration wraps route handlers as they are
# registered, so an app built before init reports nothing.
if os.getenv("SENTRY_DSN"):
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_DSN"),
        environment=os.getenv("SENTRY_ENVIRONMENT", "development"),
        release=os.getenv("SENTRY_RELEASE"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.25")),
        profile_session_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.25")),
        profile_lifecycle="trace",
        before_send=scrub_sentry_event,
        before_send_transaction=scrub_sentry_event,
    )

MODEL_HF_ID = "openai/circuit-sparsity"
SECRET = os.environ.get("SECRET")

# Fields `to_circuit_config()` passes that the published `GPTConfig` does not declare, mapped to
# the only value each one can arrive as. See `install_circuit_sparsity_gpt_shim` below.
UNPUBLISHED_CIRCUIT_CONFIG_FIELDS: dict[str, Any] = {
    "unembed_rank": None,
    "bigram_table_rank": None,
    "afrac_ste": False,
    "afrac_ste_only_non_neurons": False,
    "afrac_approx": False,
    "rtopk": False,
    "mup": False,
    "mup_width_multiplier": None,
    "enable_fp8_linear": False,
    "scale_invariance": False,
}

# Global model reference. `None` until the lifespan handler loads it, and `Any` because the
# checkpoint is a trust_remote_code model whose attribute tree (`circuit_model.transformer`) is
# defined by the downloaded code and so is invisible to a type checker.
model: Any = None


def install_circuit_sparsity_gpt_shim() -> None:
    """Stand up the `circuit_sparsity.gpt` module the checkpoint's remote code imports.

    That module only exists inside OpenAI's own tree. The public package spells it
    `circuit_sparsity.inference.gpt`, and the copy of `gpt.py` the checkpoint ships is reachable
    only as a relative import from within its own remote code -- so `to_circuit_config()` raises
    `ModuleNotFoundError: No module named 'circuit_sparsity.gpt'` on any machine, whether or not
    the published package is installed. Standing the module up here is what makes the checkpoint
    loadable at all; we point it at the checkpoint's own `gpt.py`, the same file
    `modeling_circuitgpt.py` builds its `GPT` from, so the config and the module that consumes it
    are never two different versions.

    The internal `GPTConfig` also has ten fields the published one dropped, and
    `to_circuit_config()` passes all of them. Each arrives as the value in
    `UNPUBLISHED_CIRCUIT_CONFIG_FIELDS` -- eight are hardcoded in the remote code and the two
    ranks are null in the checkpoint's `config.json` -- and each of those is what the published
    dataclass defaults to anyway, so dropping them yields the same model rather than a quietly
    different one. A checkpoint that ever sends a real value fails startup instead.
    """
    if "circuit_sparsity.gpt" in sys.modules:
        return

    published_config: Any = get_class_from_dynamic_module("gpt.GPTConfig", MODEL_HF_ID)

    # A function rather than a subclass: the remote code only ever calls `GPTConfig(**kwargs)`,
    # and this way what reaches `GPT` is an instance of the checkpoint's own dataclass.
    def GPTConfig(**kwargs: Any) -> Any:
        for field, only_supported_value in UNPUBLISHED_CIRCUIT_CONFIG_FIELDS.items():
            passed = kwargs.pop(field, only_supported_value)
            if passed != only_supported_value:
                raise RuntimeError(
                    f"{MODEL_HF_ID} asks for {field}={passed!r}, which the published GPTConfig "
                    f"cannot express (it only supports {only_supported_value!r})"
                )
        return published_config(**kwargs)

    gpt_module = types.ModuleType("circuit_sparsity.gpt")
    gpt_module.__dict__["GPTConfig"] = GPTConfig
    package = types.ModuleType("circuit_sparsity")
    package.__dict__["__path__"] = []
    package.__dict__["gpt"] = gpt_module
    # `setdefault` on the parent so an installed `circuit_sparsity` keeps its own module object;
    # only the submodule that does not exist upstream is ours.
    sys.modules.setdefault("circuit_sparsity", package)
    sys.modules["circuit_sparsity.gpt"] = gpt_module


def load_model() -> Any:
    """Load the model and move to appropriate device."""
    print(f"Loading model: {MODEL_HF_ID}")
    install_circuit_sparsity_gpt_shim()
    m: Any = AutoModelForCausalLM.from_pretrained(MODEL_HF_ID, trust_remote_code=True, torch_dtype="auto")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    m = m.to(device).eval()
    print(f"Model loaded on {device}")
    return m


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model
    model = load_model()
    yield
    # Cleanup if needed
    model = None


async def verify_secret(x_secret_key: str | None = Header(default=None)):
    """Verify the secret key header if SECRET is configured."""
    if SECRET is not None:
        if x_secret_key is None:
            raise HTTPException(status_code=401, detail="X-SECRET-KEY header required")
        if x_secret_key != SECRET:
            raise HTTPException(status_code=403, detail="Invalid secret key")


app = FastAPI(
    title="Sparse Circuit Analyzer",
    description="Analyze MLP neuron connections in sparse circuit models",
    lifespan=lifespan,
    dependencies=[Depends(verify_secret)],
)


def get_layers() -> Any:
    """Get transformer layers from model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    return model.circuit_model.transformer["h"]


def neuron_get_connected_reschannels(layer_idx: int, neuron_idx: int):
    """Get residual stream channels connected to a specific MLP neuron."""
    layers = get_layers()
    c_fc_weights = layers[layer_idx].mlp.c_fc.weight[neuron_idx, :].tolist()
    c_proj_weights = layers[layer_idx].mlp.c_proj.weight[:, neuron_idx].tolist()

    return {
        "in": [{"channel_id": i, "weight": w} for i, w in enumerate(c_fc_weights) if w != 0],
        "out": [{"channel_id": i, "weight": w} for i, w in enumerate(c_proj_weights) if w != 0],
    }


def find_neurons_reading_channel(layer_idx: int, channel_id: int, top_k: int = 5) -> list[ChannelNeuron]:
    """Find neurons in a layer that read from a specific channel (via c_fc)."""
    layers = get_layers()
    c_fc_weights = layers[layer_idx].mlp.c_fc.weight[:, channel_id].tolist()

    neurons = [ChannelNeuron(neuron_id=i, weight=w) for i, w in enumerate(c_fc_weights) if w != 0]
    neurons.sort(key=lambda n: abs(n.weight), reverse=True)
    return neurons[:top_k]


def find_neurons_writing_channel(layer_idx: int, channel_id: int, top_k: int = 5) -> list[ChannelNeuron]:
    """Find neurons in a layer that write to a specific channel (via c_proj)."""
    layers = get_layers()
    c_proj_weights = layers[layer_idx].mlp.c_proj.weight[channel_id, :].tolist()

    neurons = [ChannelNeuron(neuron_id=i, weight=w) for i, w in enumerate(c_proj_weights) if w != 0]
    neurons.sort(key=lambda n: abs(n.weight), reverse=True)
    return neurons[:top_k]


def trace_circuit_forward(start_layer: int, start_neuron: int, depth: int = 3, top_k: int = 3) -> list[TraceNode]:
    """Trace a circuit forward from a starting neuron."""
    layers = get_layers()
    num_layers = len(layers)

    def trace_from_neuron(layer: int, neuron: int, remaining_depth: int) -> list[TraceNode] | None:
        if remaining_depth == 0 or layer >= num_layers - 1:
            return None

        connections = neuron_get_connected_reschannels(layer, neuron)
        out_channels = sorted(connections["out"], key=lambda x: abs(x["weight"]), reverse=True)[:top_k]

        result: list[TraceNode] = []
        for ch in out_channels:
            channel_id = ch["channel_id"]
            write_weight = ch["weight"]

            for next_layer in range(layer + 1, num_layers):
                readers = find_neurons_reading_channel(next_layer, channel_id, top_k=top_k)
                for reader in readers:
                    result.append(
                        TraceNode(
                            layer=next_layer,
                            neuron=reader.neuron_id,
                            read_weight=reader.weight,
                            via_channel=channel_id,
                            write_weight=write_weight,
                            children=trace_from_neuron(next_layer, reader.neuron_id, remaining_depth - 1),
                        )
                    )

        result.sort(key=lambda n: abs(n.write_weight) * abs(n.read_weight), reverse=True)
        return result[: top_k * 2] if result else None

    return trace_from_neuron(start_layer, start_neuron, depth) or []


def trace_circuit_backward(start_layer: int, start_neuron: int, depth: int = 3, top_k: int = 3) -> list[TraceNode]:
    """Trace a circuit backward from a starting neuron."""

    def trace_from_neuron(layer: int, neuron: int, remaining_depth: int) -> list[TraceNode] | None:
        if remaining_depth == 0 or layer <= 0:
            return None

        connections = neuron_get_connected_reschannels(layer, neuron)
        in_channels = sorted(connections["in"], key=lambda x: abs(x["weight"]), reverse=True)[:top_k]

        result: list[TraceNode] = []
        for ch in in_channels:
            channel_id = ch["channel_id"]
            read_weight = ch["weight"]

            for prev_layer in range(layer - 1, -1, -1):
                writers = find_neurons_writing_channel(prev_layer, channel_id, top_k=top_k)
                for writer in writers:
                    result.append(
                        TraceNode(
                            layer=prev_layer,
                            neuron=writer.neuron_id,
                            write_weight=writer.weight,
                            via_channel=channel_id,
                            read_weight=read_weight,
                            parents=trace_from_neuron(prev_layer, writer.neuron_id, remaining_depth - 1),
                        )
                    )

        result.sort(key=lambda n: abs(n.write_weight) * abs(n.read_weight), reverse=True)
        return result[: top_k * 2] if result else None

    return trace_from_neuron(start_layer, start_neuron, depth) or []


@app.get("/", responses={200: {"model": HealthResponse}})
async def root():
    """Health check and model info."""
    layers = get_layers()
    return HealthResponse(
        status="ok",
        model=MODEL_HF_ID,
        num_layers=len(layers),
        mlp_size=layers[0].mlp.c_fc.weight.shape[0],
        d_model=layers[0].mlp.c_proj.weight.shape[0],
    )


@app.get("/neuron/{layer}/{neuron}", responses={200: {"model": NeuronConnectionsResponse}})
async def get_neuron_connections(
    layer: int,
    neuron: int,
    trace_depth: int = Query(default=2, description="Depth of circuit trace"),
    trace_k: int = Query(default=3, description="Top K channels/neurons per step in trace"),
):
    """
    Get circuit traces for a specific MLP neuron.

    Returns forward trace (downstream neurons) and backward trace (upstream neurons).
    """
    layers = get_layers()
    num_layers = len(layers)

    if layer < 0 or layer >= num_layers:
        raise HTTPException(status_code=400, detail=f"Layer must be between 0 and {num_layers - 1}")

    mlp_size = layers[layer].mlp.c_fc.weight.shape[0]
    if neuron < 0 or neuron >= mlp_size:
        raise HTTPException(status_code=400, detail=f"Neuron index must be between 0 and {mlp_size - 1}")

    return NeuronConnectionsResponse(
        layer=layer,
        neuron=neuron,
        trace_forward=trace_circuit_forward(layer, neuron, depth=trace_depth, top_k=trace_k),
        trace_backward=trace_circuit_backward(layer, neuron, depth=trace_depth, top_k=trace_k),
    )


@app.get("/channel/{channel_id}", responses={200: {"model": ChannelConnectionsResponse}})
async def get_channel_connections(
    channel_id: int,
    top_k: int = Query(default=10, description="Number of top neurons to return per layer"),
):
    """
    Get all neurons that read from or write to a specific residual channel.
    """
    layers = get_layers()
    num_layers = len(layers)
    d_model = layers[0].mlp.c_proj.weight.shape[0]

    if channel_id < 0 or channel_id >= d_model:
        raise HTTPException(status_code=400, detail=f"Channel must be between 0 and {d_model - 1}")

    readers_by_layer: dict[int, list[ChannelNeuron]] = {}
    writers_by_layer: dict[int, list[ChannelNeuron]] = {}

    for layer_idx in range(num_layers):
        readers = find_neurons_reading_channel(layer_idx, channel_id, top_k=top_k)
        writers = find_neurons_writing_channel(layer_idx, channel_id, top_k=top_k)

        if readers:
            readers_by_layer[layer_idx] = readers
        if writers:
            writers_by_layer[layer_idx] = writers

    return ChannelConnectionsResponse(
        channel_id=channel_id,
        readers_by_layer=readers_by_layer,
        writers_by_layer=writers_by_layer,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5005)
