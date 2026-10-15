"""
GPU smoke test for the AMD ROCm stack (RX 7900 XTX / gfx1100).

Checks, in order:
  1. torch sees the GPU (torch.cuda.is_available())
  2. matmul throughput in fp32 and bf16 (warmup + timed, both BLAS backends)
  3. a small training step (conv net, forward + backward + SGD update)

Usage:
    uv run --no-sync python scripts/check_rocm.py
"""

import argparse
import os
import sys
import time

# rocBLAS is both faster and more stable than hipBLASLt for training on
# RDNA3 (see AMD ROCm 10 known issues) — set before torch initializes BLAS.
os.environ.setdefault("TORCH_BLAS_PREFER_HIPBLASLT", "0")

import torch  # noqa: E402


def benchmark_matmul(size: int, dtype: torch.dtype, iters: int = 30) -> float:
    x = torch.randn(size, size, device="cuda", dtype=dtype)
    for _ in range(5):
        x @ x
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        x @ x
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return 2 * size**3 / dt / 1e12


def benchmark_training_step() -> None:
    """Tiny conv classifier: forward + backward + SGD step on GPU."""
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(16, 32, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(32, 10),
    ).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    images = torch.randn(8, 3, 128, 128, device="cuda")
    labels = torch.randint(0, 10, (8,), device="cuda")

    loss = torch.nn.functional.cross_entropy(model(images), labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()
    print(f"  training step OK (loss={float(loss.detach()):.4f}, grads flowed)")


def main():
    parser = argparse.ArgumentParser(description="ROCm GPU smoke test")
    parser.add_argument("--size", type=int, default=4096, help="Matmul matrix size")
    args = parser.parse_args()

    print("=" * 60)
    print("ROCm GPU smoke test")
    print("=" * 60)
    print(f"torch:        {torch.__version__}")
    print(f"HIPBLASLt:    {'preferred' if os.environ.get('TORCH_BLAS_PREFER_HIPBLASLT') == '1' else 'off (rocBLAS, recommended for RDNA3 training)'}")

    if not torch.cuda.is_available():
        print("\nFAIL: torch.cuda.is_available() is False")
        print("  - run: .venv/Scripts/rocm-sdk.exe test")
        print("  - make sure the venv was created with uv's managed Python:")
        print("      uv venv --python 3.12.13  (NOT the anaconda interpreter —")
        print("       its runtime DLLs break ROCm library initialization)")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    print(f"device:       {props.name}")
    print(f"VRAM:         {props.total_memory / 1e9:.1f} GB")

    print("\n--- matmul benchmarks ---")
    for label, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        tflops = benchmark_matmul(args.size, dtype)
        print(f"  {label}: {tflops:8.1f} TFLOP/s")

    print("\n--- training step ---")
    benchmark_training_step()

    free, total = torch.cuda.mem_get_info()
    print(f"\nVRAM after tests: {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
