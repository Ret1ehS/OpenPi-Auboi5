#!/usr/bin/env python3
"""JAX GPU micro-benchmark for Jetson/niic (run via run_niic_jax_gpu.sh + openpi-py311 python)."""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def _sync_jax() -> None:
    try:
        import jax

        for d in jax.local_devices():
            d.synchronize_all_activity()
    except Exception:
        pass


def main() -> int:
    print("=== PYTHON ===")
    print(sys.version)
    print()

    print("=== JAX / XLA ===")
    try:
        import jax
        import jax.numpy as jnp
        from jax import lax

        print("jax", jax.__version__)
        print("jaxlib", __import__("jaxlib").__version__)
        print("devices", jax.devices())
        print("default_backend", jax.default_backend())
    except Exception as exc:
        print("JAX_IMPORT_FAIL", type(exc).__name__, exc)
        return 1

    backend = jax.default_backend()
    rng = jax.random.PRNGKey(0)

    def bench(name: str, fn, *, repeats: int = 20, warmup: int = 3) -> None:
        for _ in range(warmup):
            _ = fn()
            _.block_until_ready()
        _sync_jax()
        t0 = time.perf_counter()
        for _ in range(repeats):
            out = fn()
            out.block_until_ready()
        _sync_jax()
        dt = (time.perf_counter() - t0) / repeats
        print(f"{name}: {dt * 1000:.2f} ms/it  (mean over {repeats} after {warmup} warmup)")

    print()
    print("=== MICROBENCH (GPU if backend==gpu) ===")

    @jax.jit
    def matmul_large(x, y):
        return x @ y

    k = 4096
    a = jax.random.normal(rng, (k, k), jnp.float32)
    b = jax.random.normal(jax.random.split(rng)[0], (k, k), jnp.float32)
    matmul_large(a, b).block_until_ready()
    bench(f"matmul fp32 [{k}x{k}]^2", lambda: matmul_large(a, b))

    @jax.jit
    def conv_tiny(img, ker):
        return lax.conv_general_dilated(
            img,
            ker,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    img = jnp.ones((2, 256, 256, 3), jnp.float32)
    ker = jnp.ones((3, 3, 3, 64), jnp.float32)
    conv_tiny(img, ker).block_until_ready()
    bench("conv NHWC 2x256x256x3 k3x3 out64", lambda: conv_tiny(img, ker))

    @jax.jit
    def reduce_sum(x):
        return jnp.sum(x * x)

    v = jax.random.normal(rng, (8 * 1024 * 1024,), jnp.float32)
    reduce_sum(v).block_until_ready()
    bench("reduce sum 32M fp32 elems", lambda: reduce_sum(v))

    print()
    print("=== INTERPRET ===")
    print(f"backend={backend}")
    if backend != "gpu":
        print("WARNING: default backend is not gpu.")
    print("If microbench is fast but OpenPI infer ~1s+, bottleneck is likely model (steps, size).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
