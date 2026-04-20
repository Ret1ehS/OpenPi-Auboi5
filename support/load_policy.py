#!/usr/bin/env python3
"""
Unified OpenPI policy loader for local and remote inference.

Local mode loads an OpenPI checkpoint in-process.
Remote mode starts kubectl port-forward and forwards inference over WebSocket.
"""

from __future__ import annotations

import atexit
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    _PARENT = Path(__file__).resolve().parent.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

from utils.env_utils import load_default_env
from utils.path_utils import get_openpi_root, get_repo_root
from utils.runtime_config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONFIG_NAME,
    DEFAULT_LOCAL_PORT,
    DEFAULT_NAMESPACE,
    DEFAULT_REMOTE_PORT,
    KUBECONFIG_PATH,
)

load_default_env()

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_ROOT = get_openpi_root()
DEFAULT_REPO_ROOT = get_repo_root()


def _ensure_openpi_repo_paths(repo_root: Path = DEFAULT_REPO_ROOT) -> None:
    paths = [
        repo_root / "src",
        repo_root / "packages" / "openpi-client" / "src",
    ]
    for path in paths:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


_ensure_openpi_repo_paths()


_JAX_GPU_PROBE = r"""
import jax
import jax.numpy as jnp
from jax import lax

print("devices", jax.devices())
if jax.default_backend() != "gpu":
    raise SystemExit("default backend is not gpu")
x = jnp.arange(1024 * 1024, dtype=jnp.float32).reshape(1024, 1024)
y = x.T @ x
z = jnp.repeat(y[:8, :8], 2, axis=0)
_ = z.block_until_ready()
# Exercise a tiny conv so cuDNN failures are detected before policy load.
image = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
kernel = jnp.ones((3, 3, 3, 8), dtype=jnp.float32)
conv = lax.conv_general_dilated(
    image,
    kernel,
    window_strides=(1, 1),
    padding="SAME",
    dimension_numbers=("NHWC", "HWIO", "NHWC"),
)
_ = conv.block_until_ready()
print("gpu probe ok", z.shape)
"""

_SAFE_GPU_XLA_FLAGS = (
    "--xla_gpu_autotune_level=0",
)
_SYSTEM_CUDA_LIBRARY_PATHS = (
    "/usr/lib/aarch64-linux-gnu",
    "/lib/aarch64-linux-gnu",
    "/usr/local/cuda/targets/aarch64-linux/lib",
    "/usr/local/cuda/lib64",
)


def _merge_xla_flags(value: str | None, flags: tuple[str, ...] = _SAFE_GPU_XLA_FLAGS) -> str:
    parts = shlex.split(value or "")
    for flag in flags:
        if flag not in parts:
            parts.append(flag)
    return " ".join(parts).strip()


def _prepend_library_paths(value: str | None, paths: tuple[str, ...] = _SYSTEM_CUDA_LIBRARY_PATHS) -> str:
    parts = [part for part in (value or "").split(":") if part]
    merged: list[str] = []
    for part in (*paths, *parts):
        if part and part not in merged:
            merged.append(part)
    return ":".join(merged)


def _apply_local_jax_runtime_defaults(env: dict[str, str] | None = None) -> dict[str, str] | None:
    target = os.environ if env is None else env

    if os.environ.get("OPENPI_DISABLE_JAX_GPU_FIX", "").strip().lower() not in {"1", "true", "yes", "on"}:
        merged_flags = _merge_xla_flags(target.get("XLA_FLAGS"))
        if merged_flags:
            target["XLA_FLAGS"] = merged_flags
        merged_library_path = _prepend_library_paths(target.get("LD_LIBRARY_PATH"))
        if merged_library_path:
            target["LD_LIBRARY_PATH"] = merged_library_path

    if not target.get("JAX_PLATFORMS"):
        target["JAX_PLATFORMS"] = "cuda"
    target.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    return target


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _checkpoint_has_safetensors_weights(checkpoint_dir: Path) -> bool:
    """Return True if the checkpoint dir already contains exported safetensors weights."""
    if not checkpoint_dir.is_dir():
        return False
    if (checkpoint_dir / "model.safetensors").exists():
        return True
    # LoRA / alternate exports often use other *.safetensors names at the checkpoint root.
    return any(checkpoint_dir.glob("*.safetensors"))


def _choose_local_jax_platform(checkpoint_dir: Path) -> str | None:
    requested = os.environ.get("OPENPI_JAX_PLATFORM")
    if requested:
        requested = requested.strip().lower()
        if requested == "gpu":
            return None
        return requested

    # Skip the subprocess probe (matmul + cuDNN conv). Use when OPENPI_JAX_PLATFORM=gpu is not set
    # but the machine is known-good on GPU (e.g. same env as run_niic_jax_gpu.sh).
    if _env_truthy("OPENPI_SKIP_JAX_GPU_PROBE"):
        return None

    if _checkpoint_has_safetensors_weights(checkpoint_dir):
        return None

    probe_env = dict(os.environ)
    probe_env["JAX_PLATFORMS"] = ""
    probe_env["JAX_PLATFORM_NAME"] = ""
    _apply_local_jax_runtime_defaults(probe_env)
    try:
        result = subprocess.run(
            [sys.executable, "-c", _JAX_GPU_PROBE],
            capture_output=True,
            text=True,
            timeout=90,
            env=probe_env,
        )
    except Exception as exc:
        print(f"  [local] JAX GPU probe error ({type(exc).__name__}: {exc}), forcing CPU backend.")
        return "cpu"

    if result.returncode == 0:
        return None

    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or f"rc={result.returncode}"
    print(f"  [local] JAX GPU probe failed ({detail}), forcing CPU backend.")
    return "cpu"


def _apply_jax_platform(platform: str | None) -> None:
    if not platform:
        return
    os.environ["JAX_PLATFORMS"] = platform
    os.environ["JAX_PLATFORM_NAME"] = platform


@dataclass(frozen=True)
class PolicyLoadSpec:
    remote: bool = False
    config_name: str = DEFAULT_CONFIG_NAME
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    default_prompt: str | None = None
    action_horizon: int | None = None
    pytorch_device: str | None = None
    #: Passed to OpenPI `create_trained_policy(..., sample_kwargs=...)`, e.g. `{"num_steps": 5}` for
    #: JAX Pi0 flow-matching (upstream default `num_steps` is 10 in `Pi0.sample_actions`).
    sample_kwargs: dict[str, Any] | None = None
    kubeconfig: Path = KUBECONFIG_PATH
    namespace: str = DEFAULT_NAMESPACE
    pod_label: str = "app=openpi"
    local_port: int = DEFAULT_LOCAL_PORT
    remote_port: int = DEFAULT_REMOTE_PORT


def _effective_sample_kwargs(spec: PolicyLoadSpec) -> dict[str, Any] | None:
    """Merge PolicyLoadSpec.sample_kwargs with OPENPI_SAMPLE_NUM_STEPS (env overrides num_steps)."""
    merged: dict[str, Any] = {}
    if spec.sample_kwargs:
        merged.update(spec.sample_kwargs)
    env_steps = os.environ.get("OPENPI_SAMPLE_NUM_STEPS", "").strip()
    if env_steps:
        merged["num_steps"] = int(env_steps)
    return merged if merged else None


class LocalPolicyRunner:
    def __init__(self, policy: Any, *, metadata: dict[str, Any] | None = None) -> None:
        self._policy = policy
        self._metadata = metadata or {}

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        return self._policy.infer(obs)

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class RemotePolicyRunner:
    """Policy runner that forwards inference to a remote server."""

    def __init__(
        self,
        *,
        kubeconfig: Path = KUBECONFIG_PATH,
        namespace: str = DEFAULT_NAMESPACE,
        pod_label: str = "app=openpi",
        local_port: int = DEFAULT_LOCAL_PORT,
        remote_port: int = DEFAULT_REMOTE_PORT,
    ) -> None:
        self._kubeconfig = Path(kubeconfig)
        self._namespace = namespace
        self._pod_label = pod_label
        self._local_port = int(local_port)
        self._remote_port = int(remote_port)
        self._port_forward_proc: subprocess.Popen | None = None
        self._client: Any = None
        self._metadata: dict[str, Any] = {}

    def connect(self, timeout: float = 30.0) -> None:
        self._start_port_forward()
        self._connect_ws(timeout=timeout)

    def close(self) -> None:
        self._client = None
        self._stop_port_forward()

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._client is None:
            raise RuntimeError("RemotePolicyRunner not connected. Call connect() first.")
        try:
            return self._client.infer(obs)
        except Exception as exc:
            print(f"  [remote] infer failed ({type(exc).__name__}: {exc}), reconnecting...")
            self._reconnect_ws()
            return self._client.infer(obs)

    def reset(self) -> None:
        if self._client is not None:
            try:
                self._client.reset()
            except Exception:
                pass

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _start_port_forward(self) -> None:
        if self._port_forward_proc is not None and self._port_forward_proc.poll() is None:
            return
        if not self._kubeconfig.exists():
            raise FileNotFoundError(f"kubeconfig not found: {self._kubeconfig}")

        pod_name = self._find_pod()
        print(f"  Port-forwarding to pod {pod_name} ({self._local_port}:{self._remote_port})...")
        cmd = [
            "kubectl",
            "--kubeconfig",
            str(self._kubeconfig),
            "-n",
            self._namespace,
            "port-forward",
            pod_name,
            f"{self._local_port}:{self._remote_port}",
        ]
        self._port_forward_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        atexit.register(self._stop_port_forward)
        time.sleep(2.0)
        if self._port_forward_proc.poll() is not None:
            stderr = self._port_forward_proc.stderr.read().decode() if self._port_forward_proc.stderr else ""
            raise RuntimeError(f"kubectl port-forward exited immediately: {stderr}")
        print("  Port-forward established.")

    def _find_pod(self) -> str:
        cmd = [
            "kubectl",
            "--kubeconfig",
            str(self._kubeconfig),
            "-n",
            self._namespace,
            "get",
            "pods",
            "-l",
            self._pod_label,
            "--field-selector=status.phase=Running",
            "-o",
            "jsonpath={.items[0].metadata.name}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                f"Cannot find running pod with label '{self._pod_label}' "
                f"in namespace '{self._namespace}'.\n"
                f"kubectl stderr: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    def _stop_port_forward(self) -> None:
        if self._port_forward_proc is not None and self._port_forward_proc.poll() is None:
            self._port_forward_proc.terminate()
            try:
                self._port_forward_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._port_forward_proc.kill()
        self._port_forward_proc = None

    def _connect_ws(self, timeout: float = 30.0) -> None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        print(f"  Connecting to ws://localhost:{self._local_port} ...")
        previous_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(timeout)
        try:
            self._client = WebsocketClientPolicy(host="localhost", port=self._local_port)
            self._metadata = self._client.get_server_metadata()
        finally:
            socket.setdefaulttimeout(previous_timeout)
        print(f"  Connected. Server metadata keys: {list(self._metadata.keys())}")

    def _reconnect_ws(self) -> None:
        if self._port_forward_proc is None or self._port_forward_proc.poll() is not None:
            print("  [remote] port-forward is dead, restarting...")
            self._stop_port_forward()
            self._start_port_forward()
        self._connect_ws()


def create_local_policy(spec: PolicyLoadSpec | None = None) -> LocalPolicyRunner:
    spec = spec or PolicyLoadSpec()
    checkpoint_dir = Path(spec.checkpoint_dir).resolve()
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    _apply_local_jax_runtime_defaults()
    _apply_jax_platform(_choose_local_jax_platform(checkpoint_dir))

    try:
        from openpi.training.config import get_config
        from openpi.policies.policy_config import create_trained_policy
        from openpi_client.action_chunk_broker import ActionChunkBroker
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenPI local inference dependencies are not available in the current Python environment. "
            "Please run under an environment with the OpenPI repo dependencies installed."
        ) from exc

    train_cfg = get_config(spec.config_name)
    sample_kwargs = _effective_sample_kwargs(spec)
    if sample_kwargs:
        print(f"  [local] OpenPI sample_kwargs: {sample_kwargs}")
    policy = create_trained_policy(
        train_cfg,
        checkpoint_dir,
        default_prompt=spec.default_prompt,
        pytorch_device=spec.pytorch_device,
        sample_kwargs=sample_kwargs,
    )
    if spec.action_horizon is not None:
        policy = ActionChunkBroker(policy, action_horizon=int(spec.action_horizon))
    metadata = getattr(policy, "metadata", {})
    return LocalPolicyRunner(policy, metadata=metadata)


def create_remote_policy(spec: PolicyLoadSpec | None = None) -> RemotePolicyRunner:
    spec = spec or PolicyLoadSpec(remote=True)
    runner = RemotePolicyRunner(
        kubeconfig=spec.kubeconfig,
        namespace=spec.namespace,
        pod_label=spec.pod_label,
        local_port=spec.local_port,
        remote_port=spec.remote_port,
    )
    runner.connect()
    return runner


_cached_policy: LocalPolicyRunner | RemotePolicyRunner | None = None
_cached_spec: PolicyLoadSpec | None = None


def load_policy(spec: PolicyLoadSpec | None = None, *, force_reload: bool = False) -> LocalPolicyRunner | RemotePolicyRunner:
    global _cached_policy, _cached_spec
    spec = spec or PolicyLoadSpec()
    if _cached_policy is not None and _cached_spec == spec and not force_reload:
        return _cached_policy

    if _cached_policy is not None and hasattr(_cached_policy, "close"):
        try:
            _cached_policy.close()
        except Exception:
            pass

    _cached_policy = create_remote_policy(spec) if spec.remote else create_local_policy(spec)
    _cached_spec = spec
    return _cached_policy


def close_policy() -> None:
    global _cached_policy, _cached_spec
    if _cached_policy is not None and hasattr(_cached_policy, "close"):
        try:
            _cached_policy.close()
        except Exception:
            pass
    _cached_policy = None
    _cached_spec = None


atexit.register(close_policy)


__all__ = [
    "DEFAULT_REPO_ROOT",
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_CHECKPOINT_DIR",
    "KUBECONFIG_PATH",
    "DEFAULT_NAMESPACE",
    "DEFAULT_LOCAL_PORT",
    "DEFAULT_REMOTE_PORT",
    "PolicyLoadSpec",
    "LocalPolicyRunner",
    "RemotePolicyRunner",
    "create_local_policy",
    "create_remote_policy",
    "load_policy",
    "close_policy",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load OpenPI policy locally or via remote websocket.")
    parser.add_argument("--remote", action="store_true")
    parser.add_argument("--pod-label", type=str, default="app=openpi")
    args = parser.parse_args()

    spec = PolicyLoadSpec(remote=args.remote, pod_label=args.pod_label)
    if spec.remote:
        print(f"Kubeconfig: {KUBECONFIG_PATH}")
        print("Connecting to remote inference server...")
    else:
        print(f"PYTHON={sys.executable}")
        print(f"CHECKPOINT_DIR={DEFAULT_CHECKPOINT_DIR}")
        print("Loading policy ...")

    t0 = time.monotonic()
    runner = load_policy(spec)
    elapsed = time.monotonic() - t0
    print(f"POLICY_READY  class={type(runner).__name__}  load_time={elapsed:.3f}s")
