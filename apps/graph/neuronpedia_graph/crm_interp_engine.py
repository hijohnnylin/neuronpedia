"""Run llamascopium's CRM attribution against a plain HuggingFace model, via interp-engine.

llamascopium's circuit tracing is written against TransformerLens: its ``LanguageModel`` is a
``HookedTransformer`` subclass, and the algorithm names the tensors it needs as TransformerLens hook
names (``blocks.3.ln2.hook_normalized``, ``blocks.3.attn.hook_pattern``, ...). That ties the CRM to
the architectures TransformerLens has a weight conversion for.

The attribution *algorithm* is not tied to any of that. It reaches the model through three methods --
``apply_saes``, ``detach_at``, ``run_with_ref_cache`` -- plus a few attributes (``cfg``, ``blocks``,
``tokenizer``, ``device``). Everything else it does is arithmetic on tensors it got back from the
cache. So this module supplies those three methods on top of an ``interp_engine.EagerModel``, which
hooks an ``AutoModelForCausalLM`` where it stands, and llamascopium's ``attribute`` runs on top
unmodified. QK tracing is the exception and asks by a different route -- see :meth:`hooks`.

What the algorithm asks for, and how each is answered here:

* **The replacement modules' input**, ``blocks.N.ln{1,2}.hook_normalized``. TransformerLens splits a
  norm into ``x / scale`` and ``* w`` with a hook between them, so this tensor is a module boundary
  there and is not one on HuggingFace, whose norm does both in one call. It is recomputed instead,
  from :func:`interp_engine.capture.rms_norm_parts` -- the same tensor and, because the scale is
  detached, the same gradient. See :meth:`InterpEngineLanguageModel._read_normalized`.
* **The replacement modules' output**, ``blocks.N.hook_{attn,mlp}_out``, which are interp-engine's
  ``attn_out``/``mlp_out`` points.
* **The freezes.** Attribution needs the attention pattern and every norm's scale held constant, so
  that a feature's measured effect flows through the values it moves rather than through the softmax
  it shifts. All three sites are local variables inside a HuggingFace forward rather than module
  boundaries: :func:`_frozen_eager_attention` handles the pattern, and :meth:`_norm_freeze` handles
  both the block norms (``ln1``, ``ln2``, any sandwich norms, ``ln_final``) and the Qwen3-style
  QK-norm scales, which llamascopium freezes and circuit-tracer does not.
* **Leaves.** ``hook_embed`` and every replacement module's error term and feature activations are
  detached into graph leaves; that is what makes the backward pass measure the replacement model
  rather than the original. The module-side ones are HookPoints inside ``SparseDictionary``, which is
  a TransformerLens hooked module whoever runs it, so those are driven through ``sae.hook_dict``
  as-is.

Only the above is new. ``SparseDictionary`` (the transcoders and Lorsa), ``NodeIndexedMatrix``,
``prune_attribution`` and ``attribute`` itself are imported from llamascopium and run as written.

Derived from llamascopium (OpenMOSS, MIT licensed): the splice and detach
structure in :meth:`InterpEngineLanguageModel.apply_saes` and
:meth:`InterpEngineLanguageModel.detach_at` follows ``llamascopium/circuits/hooks.py``, and the set
of freeze sites follows ``ln_detach_hooks``/``attn_detach_hooks`` in
``llamascopium/circuits/attribution.py``. See NOTICE.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import torch
from interp_engine import hooks as ie_hooks
from interp_engine import mappers
from interp_engine.capture import is_rms_norm, rms_norm_parts
from interp_engine.model import EagerModel
from llamascopium.models.sparse_dictionary import SparseDictionary
from torch import nn
from transformer_lens.hook_points import HookPoint

FROZEN_ATTN_IMPL = "crm_frozen_eager"
# Set on a norm module while its gain is being measured, to keep the freeze hook re-entrant.
_PROBING = "_crm_probing"


def _input_tensor(args: tuple, kwargs: dict) -> torch.Tensor:
    """The hidden-state argument of a module call, wherever it was passed.

    Norm modules take it positionally on every architecture in transformers today, but attention
    modules take it by keyword, and nothing guarantees which convention a given family picks.
    """
    if args:
        return args[0]
    for key in ("hidden_states", "x", "input"):
        if key in kwargs:
            return kwargs[key]
    raise ValueError(f"Could not find the input tensor among args={len(args)} kwargs={list(kwargs)}")


def _repeat_kv(hidden: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden
    batch, n_kv_heads, seq_len, head_dim = hidden.shape
    return (
        hidden[:, :, None, :, :]
        .expand(batch, n_kv_heads, n_rep, seq_len, head_dim)
        .reshape(batch, n_kv_heads * n_rep, seq_len, head_dim)
    )


def _frozen_eager_attention(module, query, key, value, attention_mask, **kwargs):
    """Eager attention whose probabilities are constants with respect to the graph.

    The probabilities are a local variable inside a HuggingFace attention forward, so no hook
    reaches them -- but they *are* the second return value of the attention interface transformers
    dispatches through. This calls the family's own eager implementation, keeping its scaling,
    masking, softcapping and attention sinks exactly as written, then redoes only the final
    ``probs @ value`` contraction with the probabilities detached. That contraction is the whole
    freeze, and it is the one line of attention math that does not vary between families.
    """
    family = sys.modules[type(module).__module__]
    eager = getattr(family, "eager_attention_forward", None)
    if eager is None:
        raise NotImplementedError(
            f"{type(module).__name__} lives in {family.__name__}, which defines no "
            "`eager_attention_forward`, so its attention pattern cannot be frozen."
        )

    _, attn_weights = eager(module, query, key, value, attention_mask, **kwargs)
    if attn_weights is None:
        raise NotImplementedError(
            f"{family.__name__}'s eager attention returned no probabilities, so they cannot be frozen."
        )

    n_rep = getattr(module, "num_key_value_groups", None) or query.shape[1] // value.shape[1]
    frozen = torch.matmul(attn_weights.detach(), _repeat_kv(value, n_rep))
    return frozen.transpose(1, 2).contiguous(), attn_weights


def _register_frozen_attention() -> None:
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

    if FROZEN_ATTN_IMPL in ALL_ATTENTION_FUNCTIONS:
        return
    AttentionInterface.register(FROZEN_ATTN_IMPL, _frozen_eager_attention)
    # transformers looks the mask builder up under the same key as the attention implementation.
    # Without this the model runs with no causal mask at all -- and still produces numbers.
    ALL_MASK_ATTENTION_FUNCTIONS.register(FROZEN_ATTN_IMPL, ALL_MASK_ATTENTION_FUNCTIONS["eager"])


def _is_norm(module: nn.Module) -> bool:
    """Whether ``module`` is a norm at all, as opposed to whatever a hook path resolved to.

    Separate from :func:`interp_engine.capture.is_rms_norm`, which answers *which kind* of norm a
    norm is and assumes it got one. The paths here come from SAE configs, so a mistyped one can
    resolve to an MLP, and this is what notices.
    """
    return "norm" in type(module).__name__.lower()


class _Cfg:
    """The ``HookedTransformerConfig`` fields the attribution reads."""

    def __init__(self, n_layers: int, d_model: int, use_qk_norm: bool):
        self.n_layers = n_layers
        self.d_model = d_model
        self.use_qk_norm = use_qk_norm


class _Attn:
    """Stands in for ``block.attn``, carrying the QK norms under their TransformerLens names."""

    def __init__(self, module: nn.Module, q_norm: nn.Module | None, k_norm: nn.Module | None):
        self.module = module
        if q_norm is not None:
            self.q_norm = q_norm
        if k_norm is not None:
            self.k_norm = k_norm


class _Block:
    """Stands in for a block, mapping TransformerLens norm names onto HuggingFace modules.

    This exists so that llamascopium's ``ln_detach_hooks`` and ``attn_detach_hooks`` -- which
    enumerate freeze sites by probing ``hasattr(block, "ln1")`` and friends -- produce exactly the
    right list of names for this model, with no per-architecture branching here. The same mapping
    then resolves those names back to modules in :meth:`InterpEngineLanguageModel._resolve_norm`.
    """

    def __init__(self, norms: dict[str, nn.Module], attn: _Attn):
        for name, module in norms.items():
            setattr(self, name, module)
        self.attn = attn


class InterpEngineLanguageModel:
    """A llamascopium ``LanguageModel`` whose forward pass is a HuggingFace model.

    Covers the attribution path only. The training and analysis surfaces (``trace``,
    ``to_activations``, ``preprocess_raw_data``) are deliberately absent, because nothing on that
    path calls them.
    """

    def __init__(
        self,
        model_name: str,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device = "cuda",
    ):
        _register_frozen_attention()
        self.engine = EagerModel(
            model_name,
            device=str(device),
            dtype=dtype,
            attn_implementation=FROZEN_ATTN_IMPL,
        )
        self.arch = self.engine.arch
        self.hf_model = self.engine.hf_model
        self.tokenizer = self.engine.tokenizer
        self.device = torch.device(device)
        self.dtype = dtype
        # llamascopium branches on this to decide whether to route tensors through DTensor. Tensor
        # parallelism is a separate question from which backend runs the forward pass, and this
        # backend does not do it.
        self.device_mesh = None

        self.cfg = _Cfg(
            n_layers=self.arch.n_layers,
            d_model=self.arch.d_model,
            use_qk_norm=self.arch.quirks.qk_norm is not None,
        )
        self._pre_norms = self._find_pre_sublayer_norms()
        self.blocks = [self._block(layer) for layer in range(self.arch.n_layers)]
        self.ln_final = self.arch.final_norm

        self._cache: dict[str, torch.Tensor] = {}
        # What each replacement module encoded this forward pass, keyed by id(module): the input hook
        # fires at the norm and the output hook needs the result a sublayer later.
        self._encoded: dict[int, Any] = {}
        self._sae_by_out: dict[str, SparseDictionary] = {}
        self._error_hooks: dict[str, HookPoint] = {}
        # Keyed by id(norm module): what each norm should do beyond freezing its scale.
        self._norm_listeners: dict[int, list[Callable[[torch.Tensor], None]]] = {}
        self._scale_names: dict[int, str] = {}
        self._norm_hooked: set[int] = set()

    # --- the TransformerLens-shaped view of the model ---------------------------

    def _block(self, layer: int) -> _Block:
        pre = self._pre_norms[layer]
        norms: dict[str, nn.Module] = {}
        if (ln1 := pre.get("attn")) is not None:
            norms["ln1"] = ln1
        if (ln2 := pre.get("mlp")) is not None:
            norms["ln2"] = ln2
        if (post_attn := self.arch.post_attn_norm(layer)) is not None:
            norms["ln1_post"] = post_attn
        if (post_mlp := self.arch.post_mlp_norm(layer)) is not None:
            norms["ln2_post"] = post_mlp

        q_norm = k_norm = None
        if self.arch.quirks.qk_norm is not None:
            q_norm = self.arch.qk_norm_module(layer, "q")
            k_norm = self.arch.qk_norm_module(layer, "k")
        return _Block(norms, _Attn(self.arch.attn_module(layer), q_norm, k_norm))

    def _find_pre_sublayer_norms(self) -> list[dict[str, nn.Module]]:
        """Which norm module feeds each sublayer, per layer.

        ``ArchSpec`` names the norms *after* each sublayer (the sandwich norms) and the one before
        the MLP, but not the one before attention, and no naming convention identifies that one
        reliably across families. So it is found structurally: run one tiny forward, record which
        tensor each norm produced and which tensor each sublayer consumed, and match by identity.

        Identity, not equality -- two norms can produce equal tensors on a short prompt. And strong
        references to the tensors are held throughout, because the ``id()`` of a freed tensor gets
        recycled, which silently matches the wrong pair.
        """
        produced: list[tuple[torch.Tensor, int, nn.Module]] = []
        consumed: list[dict[str, torch.Tensor]] = [{} for _ in range(self.arch.n_layers)]
        handles: list[Any] = []

        def watch_norm(layer: int, norm: nn.Module):
            def hook(_module, _args, output):
                produced.append((output, layer, norm))

            return hook

        def watch_sublayer(layer: int, kind: str):
            def hook(_module, args, kwargs):
                consumed[layer][kind] = _input_tensor(args, kwargs)

            return hook

        try:
            for layer in range(self.arch.n_layers):
                for _, module in self.arch.decoder_layers[layer].named_modules():
                    if _is_norm(module):
                        handles.append(module.register_forward_hook(watch_norm(layer, module)))
                for kind, module in (
                    ("attn", self.arch.attn_module(layer)),
                    ("mlp", self.arch.mlp_module(layer)),
                ):
                    handles.append(module.register_forward_pre_hook(watch_sublayer(layer, kind), with_kwargs=True))

            ids = torch.tensor([[self.tokenizer.bos_token_id or 0, 1, 2]], device=self.device)
            with torch.no_grad():
                self.hf_model(ids, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        found: list[dict[str, nn.Module]] = []
        for layer in range(self.arch.n_layers):
            layer_norms: dict[str, nn.Module] = {}
            for kind, tensor in consumed[layer].items():
                # A sublayer whose input came from no norm in its own block -- a parallel block
                # sharing one norm, say -- simply gets no entry, and asking for it raises.
                for candidate, produced_layer, norm in produced:
                    if candidate is tensor and produced_layer == layer:
                        layer_norms[kind] = norm
                        break
            found.append(layer_norms)
        return found

    def _resolve_norm(self, path: str) -> nn.Module:
        """Turn a TransformerLens norm path (``blocks.3.attn.q_norm``, ``ln_final``) into a module."""
        if path == "ln_final":
            return self.ln_final
        head, _, rest = path.partition(".")
        if head != "blocks":
            raise NotImplementedError(f"Unrecognized norm path {path!r}.")
        layer_str, _, attrs = rest.partition(".")
        target: Any = self.blocks[int(layer_str)]
        for part in attrs.split("."):
            target = getattr(target, part)
        return target

    # --- the three methods the attribution drives the model through -------------

    @contextmanager
    def apply_saes(self, saes: list[SparseDictionary]) -> Iterator[InterpEngineLanguageModel]:
        """Splice each replacement module in, in place of the sublayer it reconstructs.

        Follows llamascopium's ``apply_saes``: encode at the module's input point, and at its output
        point return ``reconstructed + error``, with the error passing through a ``HookPoint`` so
        that ``detach_at`` can make it a leaf.
        """
        handles: list[Any] = []
        self._sae_by_out = {}
        self._error_hooks = {}
        try:
            for sae in saes:
                out_name = sae.cfg.hook_point_out
                self._sae_by_out[out_name] = sae
                error_hook = HookPoint()
                self._error_hooks[out_name] = error_hook
                handles.append(self._read_normalized(sae.cfg.hook_point_in, sae))
                handles.append(self._splice(out_name, sae, error_hook))
            yield self
        finally:
            for handle in handles:
                handle.remove()
            self._norm_listeners.clear()

    def _read_normalized(self, hook_point_in: str, sae: SparseDictionary) -> Any:
        """Hand a replacement module the normalized residual TransformerLens would have shown it.

        ``blocks.N.ln2.hook_normalized`` is ``x / scale``, before the norm's weight is applied and
        with ``scale`` detached (llamascopium detaches ``ln2.hook_scale`` separately). No HuggingFace
        module boundary is that tensor, so it is recomputed from the norm's input and handed over
        without disturbing the forward pass. It remains a differentiable function of the residual,
        so the gradient path from this module's features back into the model is the one
        TransformerLens would have built.
        """
        suffix = ".hook_normalized"
        if not hook_point_in.endswith(suffix):
            raise NotImplementedError(
                f"This backend reads replacement-module inputs at {suffix} points; {hook_point_in!r} is not one."
            )
        norm = self._resolve_norm(hook_point_in.removesuffix(suffix))

        def listener(normalized: torch.Tensor) -> None:
            self._encoded[id(sae)] = sae.encode(normalized, hook_attn_scores=True)

        return self._norm_freeze(norm, listener=listener)

    def _splice(self, out_name: str, sae: SparseDictionary, error_hook: HookPoint) -> Any:
        """Replace a sublayer's output with the replacement module's reconstruction plus its error."""
        address = mappers.tlens_hook_to_point(out_name, self.engine)
        module, side = self.engine.resolve_point(address.name, address.layer, stream=address.stream)
        if side != "output":
            raise NotImplementedError(
                f"{out_name!r} resolves to the {side} of a module, and only outputs can be spliced."
            )

        def hook(_module, _args, output):
            encoded = self._encoded.pop(id(sae), None)
            assert encoded is not None, f"{out_name}: the input hook did not run before the output hook."
            tensor = ie_hooks.extract_hidden(output)
            reconstructed = sae.decode(encoded)
            return ie_hooks.replace_hidden(output, reconstructed + error_hook(tensor - reconstructed))

        return module.register_forward_hook(hook)

    @contextmanager
    def detach_at(self, hook_points: list[str]) -> Iterator[InterpEngineLanguageModel]:
        """Make each named tensor a graph leaf, recording it on both sides of the cut.

        Follows llamascopium's ``detach_at``, which mounts a ``pre``/``post`` HookPoint pair around
        the cut so the cache holds both the in-graph tensor and the leaf. Here the pair is two cache
        writes, since this backend owns its cache rather than borrowing TransformerLens'
        name-indexed one.

        The names divide by who owns the tensor. Anything under ``{hook_point_out}.sae`` or
        ``{hook_point_out}.error`` belongs to a replacement module, and those genuinely are
        HookPoints, so they are driven directly. The rest name model-internal tensors that no
        HuggingFace hook reaches, each handled by the mechanism that can reproduce it.
        """
        handles: list[Any] = []
        try:
            for name in hook_points:
                handles.extend(self._install_detach(name))
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def _install_detach(self, name: str) -> list[Any]:
        def cut(tensor: torch.Tensor) -> torch.Tensor:
            self._cache[name + ".pre"] = tensor
            leaf = tensor.detach().requires_grad_()
            self._cache[name + ".post"] = leaf
            return leaf

        def on_output(_module: Any, _args: Any, output: torch.Tensor) -> torch.Tensor:
            return cut(output)

        if name == "hook_embed":
            module, _ = self.engine.resolve_point("embeddings")
            return [module.register_forward_hook(on_output)]

        if name.endswith(".error"):
            error_hook = self._error_hooks[name.removesuffix(".error")]
            return [error_hook.register_forward_hook(on_output)]

        if (found := self._find_module_hook_point(name)) is not None:
            return [found.register_forward_hook(on_output)]

        if name.endswith(".hook_scale"):
            return [
                self._norm_freeze(
                    self._resolve_norm(name.removesuffix(".hook_scale")),
                    scale_name=name,
                )
            ]

        if name.endswith(".attn.hook_pattern"):
            # Frozen for the whole run by the attention implementation the model was loaded with,
            # which is the only place the probabilities exist. Nothing to install per name, but the
            # freeze is asserted rather than assumed.
            running = self.engine.attn_implementation
            if running != FROZEN_ATTN_IMPL:
                raise RuntimeError(f"{name} must be frozen, but the model is running {running!r} attention.")
            return []

        raise NotImplementedError(f"This backend does not know how to detach at {name!r}.")

    def _find_module_hook_point(self, name: str) -> HookPoint | None:
        """The HookPoint a ``{hook_point_out}.sae.*`` name refers to, inside its replacement module."""
        for out_name, sae in self._sae_by_out.items():
            prefix = out_name + ".sae."
            if not name.startswith(prefix):
                continue
            relative = name.removeprefix(prefix)
            hook_point = sae.hook_dict.get(relative)
            if hook_point is None:
                raise KeyError(
                    f"{type(sae).__name__} at {out_name} has no hook point {relative!r}; "
                    f"it has {sorted(sae.hook_dict)}."
                )
            return hook_point
        return None

    # --- the TransformerLens plumbing the bias-leaf helpers reach for directly ---
    #
    # QK tracing turns each bias into a batched leaf tensor so the tracing can attribute to it. Its
    # two helpers -- `replace_model_biases_with_leaves` and `replace_sae_biases_with_leaves` -- are
    # module-level functions rather than methods on the model, so unlike `apply_saes` and
    # `detach_at` they cannot be answered by supplying a different implementation. They address the
    # model through `mod_dict`, `setup()` and `hooks()`, so those are what this provides.
    #
    # The biases that matter here belong to the replacement modules (`b_Q`, `b_K` on a Lorsa, `b_D`
    # on either), and those are `SparseDictionary` attributes on TransformerLens hooked modules --
    # the same objects whichever backend runs the model. Model-side biases are the part that would
    # need real work, and :meth:`_refuse_unreachable_bias_leaves` refuses rather than dropping them.

    @property
    def mod_dict(self) -> dict[str, Any]:
        """Enough of TransformerLens' module index for a replacement module to be mounted on."""
        return {f"{out_name}.sae": sae for out_name, sae in self._sae_by_out.items()}

    def setup(self) -> None:
        """Re-index each replacement module, so a freshly mounted HookPoint gets a name.

        TransformerLens rebuilds one ``hook_dict`` for the whole tree, with the replacement modules
        mounted inside it. Here the modules are addressed as themselves, so it is their own
        ``hook_dict`` that has to be refreshed.
        """
        for sae in self._sae_by_out.values():
            sae.setup()

    @contextmanager
    def hooks(
        self,
        fwd_hooks: list[tuple[Any, Callable]] | None = None,
        bwd_hooks: list[tuple[Any, Callable]] | None = None,
        reset_hooks_end: bool = True,
        clear_contexts: bool = False,
    ) -> Iterator[InterpEngineLanguageModel]:
        """Register TransformerLens-style ``(name, fn)`` forward hooks by name."""
        if bwd_hooks:
            raise NotImplementedError("This backend registers forward hooks only.")
        handles: list[Any] = []
        try:
            for name, fn in fwd_hooks or []:
                handles.append(self._register_named_hook(name, fn))
            yield self
        finally:
            for handle in handles:
                handle.remove()

    def _register_named_hook(self, name: str, fn: Callable) -> Any:
        """Register one ``fn(tensor, hook=...)`` at a name, replacing the tensor if it returns one."""
        hook_point = self._find_module_hook_point(name)
        if hook_point is not None:

            def on_hook_point(_module, _args, output):
                return fn(output, hook=hook_point)

            return hook_point.register_forward_hook(on_hook_point)

        address = mappers.tlens_hook_to_point(name, self.engine)
        module, side = self.engine.resolve_point(address.name, address.layer, stream=address.stream)
        if side != "output":
            raise NotImplementedError(f"Cannot hook {name!r}: it resolves to the {side} of a module.")

        def on_module(_module, _args, output):
            replaced = fn(ie_hooks.extract_hidden(output), hook=None)
            return output if replaced is None else ie_hooks.replace_hidden(output, replaced)

        return module.register_forward_hook(on_module)

    def _refuse_unreachable_bias_leaves(self) -> None:
        """Refuse QK tracing on a model whose own output biases would be silently dropped.

        ``replace_model_biases_with_leaves`` looks for ``b_O``/``b_out`` on each block's ``attn`` and
        ``mlp``, which are TransformerLens' folded names for the attention and MLP output-projection
        biases. This backend's blocks do not carry them, so on a model that *has* such biases the
        helper would find nothing and quietly produce a graph missing those source nodes.

        Nothing here needs them today, so this refuses instead of implementing a mapping nothing
        exercises. The affected architectures are the GPT-2-era ones -- GPT-2, GPT-NeoX/Pythia,
        Starcoder2, and GPT-J on the MLP side -- since Llama dropped output-projection biases and
        its descendants followed: Llama, Qwen2/3, Gemma 2/3, Mistral, Phi-3 and StableLM 2 all have
        none. None of the affected families has published Lorsa + transcoder checkpoints, so none
        of them can run a CRM at all.
        """
        biased = [
            f"layer {layer} {kind}"
            for layer in range(self.arch.n_layers)
            for kind, module in (
                ("attn.o_proj", self.arch.attn_out_proj(layer)),
                ("mlp.down_proj", self.arch.mlp_projection(layer, "down")),
            )
            if getattr(module, "bias", None) is not None
        ]
        if biased:
            raise NotImplementedError(
                f"{self.arch.architecture} has output-projection biases ({', '.join(biased[:4])}"
                f"{', ...' if len(biased) > 4 else ''}), which QK tracing attributes to as source "
                "nodes. This backend does not expose them under the names the tracing looks for, so "
                "it would silently omit those nodes. Use the transformerlens backend for QK tracing "
                "on this architecture."
            )

    def _norm_freeze(
        self,
        norm: nn.Module,
        listener: Callable[[torch.Tensor], None] | None = None,
        scale_name: str | None = None,
    ) -> Any:
        """Install, or extend, the one hook per norm that holds its scale constant.

        A norm can be asked for two things at once: its scale frozen, and its normalized value
        handed to a replacement module. Both come out of the same decomposition, so they share a
        single hook -- which also keeps the two consistent, rather than letting one recompute a
        scale the other has already divided out.

        The output is rebuilt as ``x / scale.detach() * gain`` rather than the module's own return
        value. ``gain`` is measured rather than read off ``weight``, because the Llama lineage
        applies ``weight`` where Gemma's applies ``1 + weight``; see ``rms_norm_parts``.
        """
        if not (_is_norm(norm) and is_rms_norm(norm)):
            raise NotImplementedError(
                f"{type(norm).__name__} is not an RMS-style norm, so its scale cannot be frozen by "
                "recomputing it. Only RMS-normed architectures are supported by this backend."
            )

        key = id(norm)
        if listener is not None:
            self._norm_listeners.setdefault(key, []).append(listener)
        if scale_name is not None:
            self._scale_names[key] = scale_name
        if key in self._norm_hooked:
            return _NullHandle()

        def hook(module, args, kwargs, output):
            # `rms_norm_parts` measures the gain by running the module on a unit vector, which
            # re-enters this hook. The probe wants the module's own answer, so step aside for it.
            if getattr(module, _PROBING, False):
                return None
            x = _input_tensor(args, kwargs)
            setattr(module, _PROBING, True)
            try:
                scale, gain = rms_norm_parts(module, x)
            finally:
                setattr(module, _PROBING, False)
            scale = scale.detach()
            normalized = (x.float() / scale).to(x.dtype)
            if (recorded := self._scale_names.get(id(module))) is not None:
                self._cache[recorded] = scale
            for fn in self._norm_listeners.get(id(module), ()):
                fn(normalized)
            return (normalized * gain).to(output.dtype)

        self._norm_hooked.add(key)
        return _TrackedHandle(norm.register_forward_hook(hook, with_kwargs=True), self._norm_hooked, key)

    def run_with_ref_cache(
        self, tokens: torch.Tensor, names_filter: list[str] | None = None
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the forward pass, returning references to the requested tensors.

        References, not clones: the attribution differentiates what this returns, so it has to hand
        back the tensors that are in the graph. ``names_filter`` is honoured for names the enclosing
        ``detach_at`` does not already populate; the rest of the requested names are either already
        present or are ones this backend does not produce, which the attribution tolerates.
        """
        self._cache = {}
        self._encoded: dict[int, Any] = {}
        handles: list[Any] = []
        try:
            for name in set(names_filter or ()):
                handles.extend(self._install_reader(name))
            output = self.hf_model(tokens, use_cache=False)
            logits = output.logits if hasattr(output, "logits") else output
        finally:
            for handle in handles:
                handle.remove()
        return logits, self._cache

    def _install_reader(self, name: str) -> list[Any]:
        """Cache a tensor that is not being detached -- a plain read, no graph surgery."""
        if name.endswith((".pre", ".post")) or self._scale_names_contains(name):
            return []
        hook_point = self._find_module_hook_point(name)
        if hook_point is None:
            return []

        def record(_m, _a, output):
            self._cache[name] = output

        return [hook_point.register_forward_hook(record)]

    def _scale_names_contains(self, name: str) -> bool:
        """Norm scales are recorded by the freeze hook, which already holds the value."""
        return name in self._scale_names.values()

    # --- entry points ----------------------------------------------------------

    def attribute(
        self,
        inputs: torch.Tensor | str,
        replacement_modules: list[SparseDictionary],
        max_n_logits: int = 10,
        desired_logit_prob: float = 0.95,
        batch_size: int = 512,
        max_features: int | None = None,
        enable_qk_tracing: bool = False,
        qk_top_fraction: float = 0.6,
        qk_topk: int = 10,
    ):
        from llamascopium.circuits.attribution import attribute

        if enable_qk_tracing:
            self._refuse_unreachable_bias_leaves()

        return attribute(
            self,  # type: ignore[arg-type]
            inputs,
            replacement_modules,
            max_n_logits,
            desired_logit_prob,
            batch_size,
            max_features,
            enable_qk_tracing,
            qk_top_fraction,
            qk_topk,
        )

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        output = self.hf_model(tokens, use_cache=False)
        return output.logits if hasattr(output, "logits") else output


class _NullHandle:
    def remove(self) -> None:
        pass


class _TrackedHandle:
    """Removes a norm hook and forgets the norm was hooked, so a later run reinstalls it."""

    def __init__(self, handle: Any, registry: set[int], key: int):
        self._handle, self._registry, self._key = handle, registry, key

    def remove(self) -> None:
        self._handle.remove()
        self._registry.discard(self._key)
