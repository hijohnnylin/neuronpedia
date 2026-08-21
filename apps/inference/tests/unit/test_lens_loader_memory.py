"""How ``LoadedJacobianLens`` spends memory, which is what decides where it can run.

A lens is ``n_layers`` matrices of ``d_model**2``. At d_model 8192 x 80 layers that is
9.9 GiB at 16-bit and 19.8 at 32, so the two things pinned here — the dtype it is held at
and how much of it sits on the compute device — are the difference between a lens that
loads on a big pod and one that does not.

The upstream fitter writes fp16 unconditionally, so neither of these can be inferred from
the file.

The device budget is in BYTES rather than layers, and its eviction order is deliberate.
A read-out sweeps every fitted layer in the same order on every batch, so a cache that
cannot hold the whole sweep is the worst case for LRU: the entry it drops is the one
wanted next. The count this replaced was 8 layers, which on a 63-layer lens missed every
single time and re-copied 3 GiB across PCIe per batch.

Where the transport RUNS is the other half, and it is a property of the lens rather than
of the residual it is handed. The vLLM backend returns residuals as CPU tensors, so a
transport that followed the residual ran a 396 GFLOP host matmul per read-out batch at
Qwen3.6-27B's shape -- 4.3s, against 4.8ms for the same sweep on an A100.
"""

from __future__ import annotations

import pytest
import torch

from neuronpedia_inference import startup_memory
from neuronpedia_inference.endpoints.lens.lens_loader import LoadedJacobianLens
from neuronpedia_inference.startup_memory import resolve_jlens_gpu_budget_bytes

D_MODEL = 8
# One bf16 J_bar at D_MODEL, the unit the budgets below are expressed in.
LAYER_BYTES = D_MODEL * D_MODEL * 2


def _lens(
    n_layers: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    device_budget_bytes: int = 1 << 30,
) -> LoadedJacobianLens:
    return LoadedJacobianLens(
        jacobians={layer: torch.randn(D_MODEL, D_MODEL, dtype=torch.float16) for layer in range(n_layers)},
        source_layers=list(range(n_layers)),
        n_prompts=1,
        d_model=D_MODEL,
        dtype=dtype,
        device_budget_bytes=device_budget_bytes,
    )


def _sweep(lens: LoadedJacobianLens, device: torch.device) -> None:
    """One read-out batch: every fitted layer, in the order the read-out walks them."""
    for layer in lens.source_layers:
        lens.jacobian_on(layer, device)


def _sweep_admitting(lens: LoadedJacobianLens) -> None:
    """The same sweep, as ``jacobian_on`` runs it where ``.to()`` really copies.

    Off CUDA there is nothing to copy, so the admission path cannot be reached through
    ``jacobian_on``; this drives it directly, admitting only on a miss exactly as it does.
    """
    for layer in lens.source_layers:
        if layer not in lens._device_cache:
            lens._admit(layer, lens.jacobians[layer].clone())


class TestDtype:
    def test_holds_the_requested_dtype_not_the_files(self):
        # Widening the fp16 the fitter wrote recovers no precision — the values were
        # already rounded — and doubles the largest allocation the lens makes.
        lens = _lens()
        assert all(J.dtype is torch.bfloat16 for J in lens.jacobians.values())

    def test_defaults_to_bfloat16(self):
        # Not fp16: the transport accumulates over d_model against late-layer
        # residuals, and fp16 tops out at 65504.
        lens = LoadedJacobianLens(
            jacobians={0: torch.zeros(D_MODEL, D_MODEL, dtype=torch.float16)},
            source_layers=[0],
            n_prompts=1,
            d_model=D_MODEL,
        )
        assert lens.dtype is torch.bfloat16

    def test_fp32_is_honored_when_asked_for(self):
        lens = _lens(dtype=torch.float32)
        assert all(J.dtype is torch.float32 for J in lens.jacobians.values())

    def test_resident_bytes_tracks_the_dtype(self):
        narrow = _lens(n_layers=4, dtype=torch.bfloat16)
        wide = _lens(n_layers=4, dtype=torch.float32)
        assert narrow.resident_bytes == 4 * D_MODEL * D_MODEL * 2
        assert wide.resident_bytes == 2 * narrow.resident_bytes


class TestTransport:
    def test_returns_float32_whatever_the_lens_dtype(self):
        # Callers stack a transported layer beside a directly-decoded one, so the
        # output dtype cannot follow the lens.
        residual = torch.randn(3, D_MODEL)
        assert _lens().transport(residual, 0).dtype is torch.float32
        assert _lens(dtype=torch.float32).transport(residual, 0).dtype is torch.float32

    def test_shape_is_the_readout_basis(self):
        out = _lens().transport(torch.randn(5, D_MODEL), 0)
        assert out.shape == (5, D_MODEL)

    def test_matches_a_float32_reference(self):
        # bf16 has ~3 decimal digits, and the fit's own convergence tolerance is
        # ~1%, so the transport only has to agree to well inside that.
        lens = _lens()
        residual = torch.randn(4, D_MODEL)
        reference = residual @ lens.jacobians[0].float().T
        torch.testing.assert_close(lens.transport(residual, 0), reference, rtol=3e-2, atol=3e-2)


class TestDeviceBudget:
    def test_fits_on_device_compares_bytes_not_layers(self):
        # The distinction the old layer count could not draw: the same number of layers
        # is affordable at one width and not at another.
        assert _lens(n_layers=10, device_budget_bytes=10 * LAYER_BYTES).fits_on_device()
        assert not _lens(n_layers=10, device_budget_bytes=9 * LAYER_BYTES).fits_on_device()

    def test_budget_is_bounded(self):
        lens = _lens(n_layers=10, device_budget_bytes=4 * LAYER_BYTES)
        _sweep_admitting(lens)
        assert lens.device_resident_bytes <= lens.device_budget_bytes
        assert len(lens._device_cache) == 4

    def test_a_lens_that_fits_stays_whole(self):
        # The regression this budget exists for: a read-out walks every fitted layer, so
        # anything short of all of them means re-copying part of the lens per batch.
        lens = _lens(n_layers=10, device_budget_bytes=10 * LAYER_BYTES)
        for _ in range(3):
            _sweep_admitting(lens)
        assert len(lens._device_cache) == 10

    def test_a_cyclic_sweep_keeps_a_stable_prefix(self):
        # LRU here would hold nothing across sweeps: each miss evicts the layer wanted
        # next, so the second sweep misses all ten. Evicting the newest keeps the layers
        # admitted first, so the cache is worth its size instead of worth nothing.
        lens = _lens(n_layers=10, device_budget_bytes=4 * LAYER_BYTES)
        for _ in range(3):
            _sweep_admitting(lens)
        assert set(lens._device_cache) >= {0, 1, 2}

    def test_re_admitting_a_layer_does_not_double_count(self):
        lens = _lens(n_layers=4, device_budget_bytes=4 * LAYER_BYTES)
        for _ in range(3):
            lens._admit(0, lens.jacobians[0].clone())
        assert lens.device_resident_bytes == LAYER_BYTES

    def test_a_layer_larger_than_the_budget_is_not_cached(self):
        # And does not evict the rest to make room for something that still would not fit.
        lens = _lens(n_layers=2, device_budget_bytes=1)
        lens._admit(0, lens.jacobians[0].clone())
        assert lens._device_cache == {}
        assert lens.device_resident_bytes == 0

    def test_eviction_does_not_lose_the_jacobian(self):
        # The device copy is a cache, not the storage; a re-read must still be correct.
        lens = _lens(n_layers=6, device_budget_bytes=2 * LAYER_BYTES)
        residual = torch.randn(2, D_MODEL)
        first = lens.transport(residual, 0)
        _sweep_admitting(lens)
        torch.testing.assert_close(lens.transport(residual, 0), first)

    def test_host_storage_is_not_rationed(self):
        # On the vLLM backend the residuals are CPU tensors, so `.to()` hands back the
        # lens's own storage. Counting that as a device copy would ration memory that was
        # never allocated -- and would evict on a device where nothing needs evicting.
        lens = _lens(n_layers=4, device_budget_bytes=0)
        _sweep(lens, torch.device("cpu"))
        assert lens.device_resident_bytes == 0
        for layer in lens.source_layers:
            assert lens.jacobian_on(layer, torch.device("cpu")) is lens.jacobians[layer]


class TestPlaceOnDevice:
    def test_a_cpu_device_leaves_the_transport_where_it_is(self):
        # Nothing to place: the lens is already the thing a CPU transport would read.
        lens = _lens(n_layers=4)
        lens.place_on_device(torch.device("cpu"))
        assert lens.transport_device is None
        assert lens.device_resident_bytes == 0

    def test_a_zero_budget_leaves_the_transport_where_it_is(self):
        lens = _lens(n_layers=4)
        lens.place_on_device(torch.device("cuda:0"), device_budget_bytes=0)
        assert lens.transport_device is None

    def test_placing_overrides_the_budget_it_was_constructed_with(self):
        # The load reads the file before free VRAM can be measured, so the real budget
        # only arrives here.
        lens = _lens(n_layers=4, device_budget_bytes=0)
        lens.place_on_device(torch.device("cpu"), device_budget_bytes=7 * LAYER_BYTES)
        assert lens.device_budget_bytes == 7 * LAYER_BYTES


class TestResolveDeviceBudget:
    """``JLENS_GPU_BUDGET_GIB`` -> bytes."""

    @pytest.mark.parametrize(
        ("setting", "expected"),
        [
            ("2", 2 * 1024**3),
            ("1.5", int(1.5 * 1024**3)),
            ("off", 0),
            ("0", 0),
            ("none", 0),
        ],
    )
    def test_explicit_settings(self, setting, expected):
        assert resolve_jlens_gpu_budget_bytes(setting, device="cuda:0") == expected

    def test_is_off_on_a_non_cuda_device(self):
        # Nothing to ration: `.to()` hands back the lens's own storage.
        assert resolve_jlens_gpu_budget_bytes("auto", device="cpu") == 0
        assert resolve_jlens_gpu_budget_bytes("8", device="cpu") == 0

    def test_unset_means_auto_not_off(self, monkeypatch):
        # Opposite default to the SAE budget, whose paging is opt-in. Keeping every layer
        # resident is not an optional extra here -- the read-out reads them all regardless.
        monkeypatch.setattr(startup_memory, "detect_free_memory_bytes", lambda _device: 20 * 1024**3)
        assert resolve_jlens_gpu_budget_bytes(None, device="cuda:0") > 0

    def test_nonsense_falls_back_to_auto(self, monkeypatch):
        monkeypatch.setattr(startup_memory, "detect_free_memory_bytes", lambda _device: 20 * 1024**3)
        assert resolve_jlens_gpu_budget_bytes("banana", device="cuda:0") > 0

    def test_auto_holds_back_the_request_reserve_and_unwarmed_saes(self, monkeypatch):
        monkeypatch.setattr(startup_memory, "detect_free_memory_bytes", lambda _device: 20 * 1024**3)
        budget = resolve_jlens_gpu_budget_bytes("auto", device="cuda:0", reserved_bytes=6 * 1024**3)
        assert budget == 20 * 1024**3 - startup_memory.AUTO_JLENS_TRANSIENT_RESERVE_BYTES - 6 * 1024**3

    def test_auto_never_goes_negative(self, monkeypatch):
        monkeypatch.setattr(startup_memory, "detect_free_memory_bytes", lambda _device: 1024**3)
        assert resolve_jlens_gpu_budget_bytes("auto", device="cuda:0") == 0


@pytest.mark.cuda
class TestDeviceBudgetOnCuda:
    """The budget only does anything where `.to(device)` is a real copy."""

    def test_whole_lens_stays_resident_across_sweeps(self):
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        device = torch.device("cuda:0")
        lens = _lens(n_layers=10, device_budget_bytes=10 * LAYER_BYTES)
        _sweep(lens, device)
        resident = {layer: lens.jacobian_on(layer, device).data_ptr() for layer in lens.source_layers}
        _sweep(lens, device)
        # Same storage on the second sweep: no layer was evicted and re-copied.
        assert {layer: lens.jacobian_on(layer, device).data_ptr() for layer in lens.source_layers} == resident

    def test_transport_is_correct_when_the_lens_cannot_all_fit(self):
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        device = torch.device("cuda:0")
        lens = _lens(n_layers=6, device_budget_bytes=2 * LAYER_BYTES)
        residual = torch.randn(2, D_MODEL, device=device)
        expected = {layer: lens.transport(residual, layer) for layer in lens.source_layers}
        for _ in range(2):
            for layer in lens.source_layers:
                torch.testing.assert_close(lens.transport(residual, layer), expected[layer])
        assert lens.device_resident_bytes <= lens.device_budget_bytes

    def test_placing_uploads_the_whole_lens_up_front(self):
        # Eager, so the first request does not stall on the upload and -- more importantly --
        # so the transient request budget measured at the end of startup sees this memory
        # as taken rather than handing it out twice.
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lens = _lens(n_layers=6, device_budget_bytes=0)
        lens.place_on_device(torch.device("cuda:0"), device_budget_bytes=6 * LAYER_BYTES)
        assert lens.transport_device is not None and lens.transport_device.type == "cuda"
        assert lens.device_resident_bytes == lens.resident_bytes
        assert all(lens.jacobian_on(layer, torch.device("cuda:0")).is_cuda for layer in lens.source_layers)

    def test_placing_stops_at_the_budget_instead_of_evicting(self):
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lens = _lens(n_layers=6, device_budget_bytes=0)
        lens.place_on_device(torch.device("cuda:0"), device_budget_bytes=3 * LAYER_BYTES)
        assert sorted(lens._device_cache) == [0, 1, 2]
        assert not lens.fits_on_device()

    def test_an_index_less_cuda_device_still_hits_the_cache(self):
        # `--device cuda` is how these pods are configured, and a tensor moved there
        # reports `cuda:0`, so the two forms compare unequal. Left un-normalized, every
        # sweep re-copied all 63 layers while reporting the lens as fully resident.
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lens = _lens(n_layers=6, device_budget_bytes=0)
        lens.place_on_device(torch.device("cuda"), device_budget_bytes=6 * LAYER_BYTES)

        resident = {layer: lens.jacobian_on(layer, torch.device("cuda")).data_ptr() for layer in lens.source_layers}
        _sweep(lens, torch.device("cuda"))
        after = {layer: lens.jacobian_on(layer, torch.device("cuda")).data_ptr() for layer in lens.source_layers}

        assert after == resident
        assert lens.device_resident_bytes == lens.resident_bytes

    def test_asking_for_another_device_does_not_evict_the_resident_copy(self):
        # The steer/swap path asks for every layer on the CPU, because vLLM's unembedding
        # rows land there. Serving that by evicting meant one swap request undid the whole
        # startup placement, and every read-out after it re-uploaded the lens.
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lens = _lens(n_layers=6, device_budget_bytes=0)
        lens.place_on_device(torch.device("cuda"), device_budget_bytes=6 * LAYER_BYTES)

        for layer in lens.source_layers:
            host = lens.jacobian_on(layer, torch.device("cpu"))
            assert host.device.type == "cpu"

        assert sorted(lens._device_cache) == lens.source_layers
        assert lens.device_resident_bytes == lens.resident_bytes

    def test_a_cpu_residual_is_transported_on_the_device(self):
        # The regression this whole placement exists for. vLLM hands back CPU residuals;
        # before, that decided where the matmul ran. Now the lens decides, and a CUDA
        # `J_bar` means the matmul cannot have happened on the host.
        if not torch.cuda.is_available():
            pytest.skip("needs CUDA")
        lens = _lens(n_layers=4, device_budget_bytes=0)
        residual = torch.randn(3, D_MODEL)
        reference = residual @ lens.jacobians[0].float().T

        lens.place_on_device(torch.device("cuda:0"), device_budget_bytes=4 * LAYER_BYTES)
        out = lens.transport(residual, 0)

        assert lens._device_cache[0].is_cuda
        # Returned where the caller's residual lives, so callers that have not staged a
        # whole batch keep working unchanged.
        assert out.device == residual.device
        torch.testing.assert_close(out, reference, rtol=3e-2, atol=3e-2)
