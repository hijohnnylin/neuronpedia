#!/usr/bin/env python3
"""Test direct GPU→shared memory writes via PyTorch tensors.

Validates that we can eliminate the intermediate CPU buffer copy.
"""

import numpy as np
import torch
from multiprocessing.shared_memory import SharedMemory
import time

def test_shm_backed_tensor_writes():
    """Test writing to PyTorch tensor backed by shared memory."""
    print("=" * 60)
    print("Test 1: CPU tensor writes to shm-backed tensor")
    print("=" * 60)

    # Setup
    shape = (100, 512)
    dtype = torch.float32
    np_dtype = np.float32
    nbytes = np.prod(shape) * np.dtype(np_dtype).itemsize

    # Create shared memory
    shm = SharedMemory(create=True, size=nbytes)
    print(f"Created shared memory: {shm.name} ({nbytes / 1024:.1f} KB)")

    try:
        # Create numpy array backed by shm
        shm_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
        print(f"Created numpy array: shape={shm_array.shape}, dtype={shm_array.dtype}")

        # Create PyTorch tensor from numpy (shares memory)
        shm_tensor = torch.from_numpy(shm_array)
        print(f"Created PyTorch tensor: shape={shm_tensor.shape}, dtype={shm_tensor.dtype}")
        print(f"Tensor writable: {shm_tensor.data_ptr() != 0}")
        print(f"Tensor is_contiguous: {shm_tensor.is_contiguous()}")

        # Test 1: Direct assignment
        print("\n--- Test 1a: Direct assignment ---")
        source_data = torch.randn(shape, dtype=dtype)
        shm_tensor.copy_(source_data)
        print(f"✓ copy_() succeeded")
        print(f"Data written to shm: mean={shm_tensor.mean():.4f}, std={shm_tensor.std():.4f}")

        # Verify via numpy view
        verification = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
        assert np.allclose(verification, source_data.numpy()), "Data mismatch!"
        print(f"✓ Data verified via independent numpy view")

    finally:
        shm.close()
        shm.unlink()
        print(f"\n✓ Cleanup complete\n")


def test_gpu_to_shm_direct():
    """Test GPU→shared memory direct copy (the key optimization)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU test")
        return

    print("=" * 60)
    print("Test 2: GPU→SHM direct copy (key optimization)")
    print("=" * 60)

    shape = (1000, 2048)  # ~8MB
    dtype = torch.float32
    np_dtype = np.float32
    nbytes = np.prod(shape) * np.dtype(np_dtype).itemsize

    # Create shared memory
    shm = SharedMemory(create=True, size=nbytes)
    print(f"Created shared memory: {shm.name} ({nbytes / (1024**2):.1f} MB)")

    try:
        # Create shm-backed tensor
        shm_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
        shm_tensor = torch.from_numpy(shm_array)

        # Create GPU tensor
        gpu_tensor = torch.randn(shape, dtype=dtype, device='cuda')
        print(f"Created GPU tensor: {gpu_tensor.device}")

        # Method 1: Synchronous copy
        print("\n--- Method 1: Synchronous GPU→SHM ---")
        t0 = time.perf_counter()
        shm_tensor.copy_(gpu_tensor)  # GPU→CPU (shm-backed)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        print(f"✓ Synchronous copy succeeded: {(t1-t0)*1000:.2f}ms")
        print(f"Throughput: {nbytes/(1024**2)/(t1-t0):.1f} MB/s")

        # Verify data
        gpu_cpu = gpu_tensor.cpu()
        assert torch.allclose(shm_tensor, gpu_cpu, rtol=1e-5), "Data mismatch!"
        print(f"✓ Data verified against GPU→CPU copy")

        # Method 2: Non-blocking copy
        print("\n--- Method 2: Non-blocking GPU→SHM ---")
        gpu_tensor2 = torch.randn(shape, dtype=dtype, device='cuda')

        t0 = time.perf_counter()
        shm_tensor.copy_(gpu_tensor2, non_blocking=True)
        torch.cuda.synchronize()  # Wait for completion
        t1 = time.perf_counter()
        print(f"✓ Non-blocking copy succeeded: {(t1-t0)*1000:.2f}ms")
        print(f"Throughput: {nbytes/(1024**2)/(t1-t0):.1f} MB/s")

        # Verify
        gpu_cpu2 = gpu_tensor2.cpu()
        assert torch.allclose(shm_tensor, gpu_cpu2, rtol=1e-5), "Data mismatch!"
        print(f"✓ Data verified")

    finally:
        shm.close()
        shm.unlink()
        print(f"\n✓ Cleanup complete\n")


def test_bfloat16_shm():
    """Test bfloat16 dtype (common in LLMs)."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping bfloat16 test")
        return

    print("=" * 60)
    print("Test 3: bfloat16 GPU→SHM (LLM dtype)")
    print("=" * 60)

    import ml_dtypes

    shape = (512, 1024)
    nbytes = np.prod(shape) * 2  # bfloat16 = 2 bytes

    shm = SharedMemory(create=True, size=nbytes)
    print(f"Created shared memory: {shm.name} ({nbytes / 1024:.1f} KB)")

    try:
        # Create GPU bfloat16 tensor
        gpu_tensor = torch.randn(shape, dtype=torch.bfloat16, device='cuda')
        print(f"GPU tensor: dtype={gpu_tensor.dtype}, device={gpu_tensor.device}")

        # Approach 1: View as uint16 for numpy compatibility
        print("\n--- Approach 1: uint16 view ---")
        shm_array = np.ndarray(shape, dtype=ml_dtypes.bfloat16, buffer=shm.buf)

        # Convert to uint16 view for torch
        shm_array_u16 = shm_array.view(np.uint16)
        shm_tensor = torch.from_numpy(shm_array_u16).view(torch.bfloat16)

        print(f"SHM tensor: dtype={shm_tensor.dtype}, is_contiguous={shm_tensor.is_contiguous()}")

        # Copy GPU→SHM
        shm_tensor.copy_(gpu_tensor)
        torch.cuda.synchronize()
        print(f"✓ bfloat16 copy succeeded")

        # Verify
        gpu_cpu = gpu_tensor.cpu()
        assert torch.allclose(shm_tensor, gpu_cpu, rtol=1e-2), "Data mismatch!"
        print(f"✓ Data verified (mean absolute error: {(shm_tensor.float() - gpu_cpu.float()).abs().mean():.6f})")

    finally:
        shm.close()
        shm.unlink()
        print(f"\n✓ Cleanup complete\n")


def benchmark_copy_methods():
    """Benchmark old vs new approach."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmark")
        return

    print("=" * 60)
    print("Benchmark: Old (GPU→CPU→SHM) vs New (GPU→SHM)")
    print("=" * 60)

    shape = (2048, 2048)  # ~16MB float32
    dtype = torch.float32
    np_dtype = np.float32
    nbytes = np.prod(shape) * 4

    print(f"Tensor size: {nbytes / (1024**2):.1f} MB")

    # Warmup
    gpu_tensor = torch.randn(shape, dtype=dtype, device='cuda')
    _ = gpu_tensor.cpu()
    torch.cuda.synchronize()

    # Old approach: GPU→CPU→SHM (2 copies)
    print("\n--- Old approach (2 copies) ---")
    iterations = 10
    old_times = []

    for _ in range(iterations):
        shm = SharedMemory(create=True, size=nbytes)
        try:
            gpu_tensor = torch.randn(shape, dtype=dtype, device='cuda')

            t0 = time.perf_counter()
            cpu_tensor = gpu_tensor.cpu()  # COPY 1
            np_array = cpu_tensor.numpy()
            shm_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
            shm_array[:] = np_array  # COPY 2
            t1 = time.perf_counter()

            old_times.append(t1 - t0)
        finally:
            shm.close()
            shm.unlink()

    old_mean = np.mean(old_times) * 1000
    old_std = np.std(old_times) * 1000
    print(f"Time: {old_mean:.2f} ± {old_std:.2f} ms")
    print(f"Throughput: {nbytes/(1024**2)/np.mean(old_times):.1f} MB/s")

    # New approach: GPU→SHM (1 copy)
    print("\n--- New approach (1 copy) ---")
    new_times = []

    for _ in range(iterations):
        shm = SharedMemory(create=True, size=nbytes)
        try:
            gpu_tensor = torch.randn(shape, dtype=dtype, device='cuda')

            shm_array = np.ndarray(shape, dtype=np_dtype, buffer=shm.buf)
            shm_tensor = torch.from_numpy(shm_array)

            t0 = time.perf_counter()
            shm_tensor.copy_(gpu_tensor)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            new_times.append(t1 - t0)
        finally:
            shm.close()
            shm.unlink()

    new_mean = np.mean(new_times) * 1000
    new_std = np.std(new_times) * 1000
    print(f"Time: {new_mean:.2f} ± {new_std:.2f} ms")
    print(f"Throughput: {nbytes/(1024**2)/np.mean(new_times):.1f} MB/s")

    # Comparison
    print("\n--- Speedup ---")
    speedup = old_mean / new_mean
    print(f"New approach is {speedup:.2f}x faster")
    print(f"Time saved: {old_mean - new_mean:.2f} ms per {nbytes/(1024**2):.1f}MB tensor")


if __name__ == "__main__":
    test_shm_backed_tensor_writes()
    test_gpu_to_shm_direct()
    test_bfloat16_shm()
    benchmark_copy_methods()

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("GPU→SHM direct write is viable for optimization")
    print("=" * 60)
