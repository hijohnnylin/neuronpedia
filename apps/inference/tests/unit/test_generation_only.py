"""``GENERATION_ONLY``: a pod that trades every capture endpoint for decode speed.

The flag loads the engine's ``backend="vllm-generate"``, which keeps the CUDA graphs the hooked vLLM
backend gives up so its forward hooks run. That is worth up to +249% decode on a 1B model and
roughly nothing at 4B+ (the engine's ``benchmarks/results-latest.md``), and it costs every feature
that reads or writes an activation, because graph replay never calls the Python forward the hooks are
attached to.

What has to be true for that trade to be safe, and what these tests pin:

- **Two configurations are errors, not modes.** SAE sets are unusable without capture, and there is
  no generate-only variant of the eager backend -- so both are refused at startup rather than
  discovered per request.
- **The reduced endpoint set is advertised.** A router reading ``/capabilities`` can send capture
  traffic to a pod that serves it; one that only sees 400s can only retry.
- **Completions still work.** Serving them faster is the entire point, so the steer endpoints stay
  available for their unsteered types and refuse only the steered one.

The engine-side half of this -- the refusal itself, on every hook-dependent method -- is in
the engine's ``tests/test_vllm_hook_availability.py``.
"""

from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest

from neuronpedia_inference.config import Config
from neuronpedia_inference.engine_adapter import (
    BackendUnsupported,
    _assert_vllm_points_supported,
    assert_hooks_available,
    assert_residual_available,
)
from neuronpedia_inference.server import _resolve_generation_only


def _args(**overrides: object) -> Namespace:
    args = Namespace(generation_only=True, backend="vllm", sae_sets=[], static_points=None)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestTheStartupValidation:
    def test_off_by_default(self):
        assert _resolve_generation_only(_args(generation_only=False)) is False

    def test_a_vllm_pod_with_no_saes_is_the_supported_shape(self):
        assert _resolve_generation_only(_args()) is True

    def test_sae_sets_are_refused(self):
        with pytest.raises(ValueError, match="SAE_SETS") as excinfo:
            _resolve_generation_only(_args(sae_sets=["res-jb"]))
        # The remedy matters more than the diagnosis: an operator who wanted the SAEs and an operator
        # who wanted the speed need to be told different things.
        assert "SAE_SETS='[]'" in str(excinfo.value)
        assert "unset GENERATION_ONLY" in str(excinfo.value)

    def test_the_eager_backend_is_refused(self):
        # Not a no-op we could ignore: on eager the flag would turn the endpoints off and buy nothing
        # back, since eager hooks the module tree in-process and has no CUDA graphs to keep.
        with pytest.raises(ValueError, match="only meaningful on the vLLM backend"):
            _resolve_generation_only(_args(backend="eager"))

    def test_an_absent_flag_reads_as_off(self):
        # Config objects and Namespaces from older callers may not carry the attribute at all.
        assert _resolve_generation_only(Namespace(backend="vllm", sae_sets=[])) is False

    def test_a_declared_set_is_refused_alongside_generation_only(self):
        with pytest.raises(ValueError, match="STATIC_POINTS"):
            _resolve_generation_only(_args(static_points="sae"))


class TestTheRequestLevelRefusal:
    def test_a_vllm_static_pod_is_allowed(self):
        from interp_engine.address import Address

        assert_hooks_available(
            SimpleNamespace(
                hooks_available=False,
                graph_replay=True,
                static_points=(Address("resid_post", 0),),
                static_writes=(),
            )
        )

    def test_steered_generation_needs_declared_writes_not_just_reads(self):
        from interp_engine.address import Address

        from neuronpedia_inference.engine_adapter import assert_steering_available

        with pytest.raises(BackendUnsupported, match="declared no write sites"):
            assert_steering_available(
                SimpleNamespace(
                    hooks_available=False,
                    graph_replay=True,
                    static_points=(Address("resid_post", 0),),
                    static_writes=(),
                )
            )
        assert_steering_available(
            SimpleNamespace(
                hooks_available=False,
                graph_replay=True,
                static_points=(),
                static_writes=(Address("resid_post", 0),),
            )
        )

    def test_a_vector_layer_the_pod_did_not_declare_is_refused_by_layer_not_by_point_name(self):
        """The 70B case: `resid_post` IS declared, at the layer its SAE reads and not the one the
        vector was fitted at. Checked by address, or the pod passes here and dies inside generate."""
        from interp_engine.address import Address

        from neuronpedia_inference.engine_adapter import assert_capture_layers_declared

        pod = SimpleNamespace(
            hooks_available=False,
            graph_replay=True,
            static_points=(Address("resid_post", 50),),
            static_writes=(),
        )
        with pytest.raises(BackendUnsupported) as excinfo:
            assert_capture_layers_declared(pod, [40], "Readout vector ['lu_assistant-axis']")
        message = str(excinfo.value)
        assert "lu_assistant-axis" in message
        assert "[40]" in message
        # Names the way out, since this is a relaunch and not a code change.
        assert "STATIC_POINTS_EXTRA" in message

        assert_capture_layers_declared(pod, [50], "Readout vector ['x']")

    def test_a_hooked_pod_declares_nothing_and_can_capture_anywhere(self):
        from neuronpedia_inference.engine_adapter import assert_capture_layers_declared

        assert_capture_layers_declared(SimpleNamespace(), [40])
        assert_capture_layers_declared(SimpleNamespace(hooks_available=True), [40])

    def test_a_steer_at_an_undeclared_layer_is_refused_before_the_stream_opens(self):
        """The write-side case: a projection cap writes where its vector was fitted, which is not
        the layer the vector reads. The 70B pod declared 40 to read and was asked to write 32."""
        from interp_engine.address import Address

        from neuronpedia_inference.engine_adapter import assert_steer_layers_declared

        pod = SimpleNamespace(
            hooks_available=False,
            graph_replay=True,
            static_points=(Address("resid_post", 40), Address("resid_post", 50)),
            static_writes=(Address("resid_post", 40), Address("resid_post", 50)),
        )
        with pytest.raises(BackendUnsupported) as excinfo:
            assert_steer_layers_declared(pod, [32])
        message = str(excinfo.value)
        assert "[32]" in message
        assert "STATIC_POINTS=auto" in message

        assert_steer_layers_declared(pod, [40, 50])

    def test_the_read_and_write_sets_are_asked_separately(self):
        """An SAE set declares writes under the resid_pre[L] -> resid_post[L-1] mapping, so a pod
        can hold a read at a layer and no write there. One check for both would miss it."""
        from interp_engine.address import Address

        from neuronpedia_inference.engine_adapter import (
            assert_capture_layers_declared,
            assert_steer_layers_declared,
        )

        pod = SimpleNamespace(
            hooks_available=False,
            graph_replay=True,
            static_points=(Address("resid_post", 40),),
            static_writes=(),
        )
        assert_capture_layers_declared(pod, [40])
        # No writes declared at all is assert_steering_available's refusal, not a layer list.
        assert_steer_layers_declared(pod, [40])

    def test_a_hooked_pod_is_not_asked_which_layers_a_request_would_write(self):
        """The gate exists so resolving those layers -- which reads each feature's hook out of the
        SAE manager -- never runs on a pod where the answer could not refuse anything."""
        from neuronpedia_inference.engine_adapter import declares_static_taps

        assert declares_static_taps(SimpleNamespace(hooks_available=False)) is True
        assert declares_static_taps(SimpleNamespace(hooks_available=True)) is False
        assert declares_static_taps(SimpleNamespace()) is False

    def test_a_layer_declared_under_the_other_residual_name_still_counts(self):
        """`resid_pre[L]` IS `resid_post[L-1]`, and the engine matches on that. Checking the
        literal name would refuse a request the pod can serve."""
        from interp_engine.address import Address

        from neuronpedia_inference.engine_adapter import assert_steer_layers_declared

        pod = SimpleNamespace(
            hooks_available=False,
            graph_replay=True,
            static_points=(),
            static_writes=(Address("resid_pre", 33),),
        )
        assert_steer_layers_declared(pod, [32])
        with pytest.raises(BackendUnsupported):
            assert_steer_layers_declared(pod, [33])

    def test_an_unknown_backend_is_assumed_capable(self):
        # Anything without the attribute predates it and hooks in-process (eager, and the test
        # doubles for it). Defaulting to "refuse" here would break every such caller.
        assert_hooks_available(SimpleNamespace())

    def test_a_generation_only_pod_refuses_with_the_flag_named(self):
        with pytest.raises(BackendUnsupported) as excinfo:
            assert_hooks_available(SimpleNamespace(hooks_available=False), "The logit lens")
        message = str(excinfo.value)
        assert "The logit lens" in message
        assert "GENERATION_ONLY=true" in message

    def test_the_capture_path_refuses_before_it_checks_points(self):
        # Point-by-point support is the wrong answer here -- the pod cannot capture *anything* -- so
        # the whole-pod condition has to be checked first or the error would name a red herring.
        model = SimpleNamespace(hooks_available=False, tensor_parallel_size=1)
        with pytest.raises(BackendUnsupported, match="GENERATION_ONLY"):
            _assert_vllm_points_supported(model, [])


class TestResidualReads:
    def test_native_extract_is_enough_for_lens(self):
        async def capture_resid_post(*_a, **_k):
            return {}

        assert_residual_available(
            SimpleNamespace(
                hooks_available=False,
                static_points=(),
                enable_extraction=True,
                capture_resid_post=capture_resid_post,
            ),
            "The logit lens",
        )

    def test_a_callable_without_enable_extraction_is_not_native_extract(self):
        async def capture_resid_post(*_a, **_k):
            return {}

        with pytest.raises(BackendUnsupported, match="residual-stream"):
            assert_residual_available(
                SimpleNamespace(
                    hooks_available=False,
                    static_points=(),
                    enable_extraction=False,
                    capture_resid_post=capture_resid_post,
                ),
                "The logit lens",
            )

    def test_a_declared_resid_post_is_enough_without_native_extract(self):
        from interp_engine.address import Address

        assert_residual_available(
            SimpleNamespace(hooks_available=False, static_points=(Address("resid_post", 0),)),
            "The logit lens",
        )

    def test_generation_only_without_native_extract_is_refused(self):
        with pytest.raises(BackendUnsupported, match="residual-stream"):
            assert_residual_available(SimpleNamespace(hooks_available=False, static_points=()), "The logit lens")

    def test_a_hyper_connection_trunk_is_served_by_its_declared_stream_stack(self):
        """`STATIC_POINTS: auto` declares resid_streams on DeepSeek-V4; the lens reads exactly that."""
        from interp_engine import vllm_residual_basis
        from interp_engine.address import Address

        from neuronpedia_inference.endpoints.lens.residual_spec import block_output_point

        model = SimpleNamespace(
            hooks_available=False,
            static_points=(Address("resid_streams", 0),),
            residual_basis=vllm_residual_basis(n_residual_streams=4, architecture="DeepseekV4ForCausalLM"),
        )
        assert_residual_available(model, "The logit lens", point=block_output_point(model))
        # A model that says nothing about its trunk is one stream, which is what the stubs in the
        # template tests rely on -- they are not about residual structure and do not declare it.
        assert block_output_point(SimpleNamespace()) == "resid_post"
        # ...and resid_post is not a substitute for it in either direction.
        with pytest.raises(BackendUnsupported, match="no resid_post to read them from"):
            assert_residual_available(model, "/activation/raw")

    def test_native_extract_does_not_stand_in_for_a_stream_stack(self):
        """It returns one d_model vector, and which collapse of the stack that is, nobody says."""

        async def capture_resid_post(*_a, **_k):
            return {}

        with pytest.raises(BackendUnsupported, match="no resid_streams to read them from"):
            assert_residual_available(
                SimpleNamespace(
                    hooks_available=False,
                    static_points=(),
                    enable_extraction=True,
                    capture_resid_post=capture_resid_post,
                ),
                "The logit lens",
                point="resid_streams",
            )


class TestWhatCapabilitiesReports:
    """The advertised contract, assembled from the same code path the endpoint runs."""

    @staticmethod
    def _report(*, hooks: bool, **model_extra: object) -> dict:
        from neuronpedia_inference.endpoints import capabilities as module

        config = Config(
            model_id="gpt2-small",
            sae_sets=[],
            model_dtype="float16",
            device="cpu",
            token_limit=200,
            generation_only=not hooks,
        )
        model = SimpleNamespace(
            **{
                "hooks_available": hooks,
                "graph_replay": not hooks,
                "static_points": (),
                "static_writes": (),
                "grad_support": SimpleNamespace(describe=lambda: {}),
                "tensor_parallel_size": 1,
                **model_extra,
            }
        )
        Config._instance = config
        module.Model._instance = model  # type: ignore[assignment]
        try:
            import asyncio

            return asyncio.run(module.capabilities())
        finally:
            Config._instance = None
            module.Model._instance = None  # type: ignore[assignment]

    def test_no_capture_points_are_advertised(self):
        assert self._report(hooks=False)["capture_points"] == []

    def test_the_reason_is_reported_alongside_the_endpoints(self):
        report = self._report(hooks=False)
        assert report["hooks_available"] is False
        assert report["generation_only"] is True

    def test_every_activation_endpoint_is_off(self):
        endpoints = self._report(hooks=False)["endpoints"]
        off = [name for name, available in endpoints.items() if name.startswith("activation_") and available]
        assert off == []
        assert endpoints["dfa"] is False
        assert endpoints["lens_logit"] is False
        assert endpoints["lens_jacobian"] is False

    def test_what_the_pod_exists_for_stays_on(self):
        endpoints = self._report(hooks=False)["endpoints"]
        assert endpoints["tokenize"] is True
        # True because they serve the DEFAULT completion type on any pod; the STEERED type is refused
        # per request, in the endpoints themselves.
        assert endpoints["steer_completion"] is True
        assert endpoints["steer_completion_chat"] is True

    def test_the_pre_rename_key_names_are_still_served(self):
        """They shipped before 1.3 renamed them, and a router keying off the old names would read a
        missing list as "captures nothing" and quietly stop routing here. Delete with them."""
        from interp_engine.address import Address

        report = self._report(hooks=False, static_points=(Address("resid_post", 3),))
        assert report["frozen_points"] == report["static_points"] == ["resid_post.3"]
        assert report["writes_available"] == report["static_writes"] == []

    def test_an_ordinary_pod_is_unaffected(self):
        report = self._report(hooks=True)
        assert report["hooks_available"] is True
        assert report["generation_only"] is False
        assert report["endpoints"]["activation_all"] is True
        assert report["endpoints"]["lens_logit"] is True

    def test_native_extract_advertises_raw_and_lens(self):
        async def capture_resid_post(*_a, **_k):
            return {}

        endpoints = self._report(hooks=False, capture_resid_post=capture_resid_post, enable_extraction=True)[
            "endpoints"
        ]
        assert endpoints["activation_raw"] is True
        assert endpoints["lens_logit"] is True
        assert endpoints["activation_all"] is False
        assert endpoints["activation_attention"] is False

    def test_a_callable_without_enable_extraction_does_not_advertise_raw(self):
        async def capture_resid_post(*_a, **_k):
            return {}

        endpoints = self._report(hooks=False, capture_resid_post=capture_resid_post, enable_extraction=False)[
            "endpoints"
        ]
        assert endpoints["activation_raw"] is False
        assert endpoints["lens_logit"] is False


class TestSaeStaticAddresses:
    def test_resid_pre_hooks_become_reads_and_writes(self):
        from interp_engine.address import Address

        from neuronpedia_inference.engine_adapter import sae_static_addresses

        manager = SimpleNamespace(
            sae_set_to_saes={"res-jb": ["7-res-jb", "8-res-jb"]},
            get_sae_hook=lambda sae_id: f"blocks.{sae_id.split('-')[0]}.hook_resid_pre",
        )
        reads, writes = sae_static_addresses(manager)
        assert reads == [Address("resid_pre", 7), Address("resid_pre", 8)]
        assert writes == [Address("resid_post", 6), Address("resid_post", 7)]

    def test_resid_pre_layer_0_is_a_read_without_a_write(self):
        from interp_engine.address import Address

        from neuronpedia_inference.engine_adapter import sae_static_addresses

        manager = SimpleNamespace(
            sae_set_to_saes={"res-jb": ["0-res-jb"]},
            get_sae_hook=lambda _sae_id: "blocks.0.hook_resid_pre",
        )
        reads, writes = sae_static_addresses(manager)
        assert reads == [Address("resid_pre", 0)]
        assert writes == []

    def test_neurons_set_is_not_declared(self):
        from interp_engine.address import Address

        from neuronpedia_inference.engine_adapter import sae_static_addresses

        manager = SimpleNamespace(
            NEURONS_SOURCESET="neurons",
            sae_set_to_saes={
                "res-jb": ["7-res-jb"],
                "neurons": ["0", "1"],
            },
            get_sae_hook=lambda sae_id: (
                "blocks.7.hook_resid_pre" if sae_id == "7-res-jb" else f"blocks.{sae_id}.mlp.hook_post"
            ),
        )
        reads, writes = sae_static_addresses(manager)
        assert reads == [Address("resid_pre", 7)]
        assert writes == [Address("resid_post", 6)]

    def test_duplicate_hooks_are_deduped(self):
        from interp_engine.address import Address

        from neuronpedia_inference.engine_adapter import sae_static_addresses

        manager = SimpleNamespace(
            sae_set_to_saes={"a": ["7-res-jb"], "b": ["7-res-jb"]},
            get_sae_hook=lambda _sae_id: "blocks.7.hook_resid_pre",
        )
        reads, writes = sae_static_addresses(manager)
        assert len(reads) == 1
        assert writes == [Address("resid_post", 6)]

    def test_dfa_sources_also_declare_attn(self):
        from interp_engine.address import Address

        from neuronpedia_inference.engine_adapter import sae_static_addresses

        manager = SimpleNamespace(
            sae_set_to_saes={"att-kk": ["5-att-kk"]},
            get_sae_hook=lambda _sae_id: "blocks.5.attn.hook_z",
            is_dfa_enabled=lambda _sae_id: True,
        )
        reads, writes = sae_static_addresses(manager)
        assert Address("z", 5) in reads
        assert Address("attn", 5) in reads
        assert Address("z", 5) in writes

    def test_no_hooks_is_refused(self):
        from neuronpedia_inference.engine_adapter import sae_static_addresses

        with pytest.raises(ValueError, match="STATIC_POINTS=sae"):
            sae_static_addresses(SimpleNamespace(sae_set_to_saes={}, get_sae_hook=lambda _s: None))
