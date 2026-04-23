#!/usr/bin/env python3
"""
Unified OpenPI policy loader for local and remote inference.

Local mode loads an OpenPI checkpoint in-process.
Remote mode starts kubectl port-forward and forwards inference over WebSocket.
"""

from __future__ import annotations

import atexit
import os
import pickle
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

if __package__ in (None, ""):
    _PARENT = Path(__file__).resolve().parent.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

from support.openpi_pytorch_policy import decode_worker_value, encode_worker_value
from utils.env_utils import load_default_env
from utils.path_utils import get_openpi_root, get_repo_root
from utils.runtime_config import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONFIG_NAME,
    DEFAULT_LOCAL_PORT,
    DEFAULT_NAMESPACE,
    DEFAULT_POLICY_BACKEND,
    DEFAULT_PYTORCH_CHECKPOINT_DIR,
    DEFAULT_PYTORCH_DEVICE,
    DEFAULT_REMOTE_PORT,
    KUBECONFIG_PATH,
)

load_default_env()

SCRIPT_DIR = Path(__file__).resolve().parent
OPENPI_ROOT = get_openpi_root()
DEFAULT_REPO_ROOT = get_repo_root()
DEFAULT_PYTORCH_RUNTIME_PYTHON = OPENPI_ROOT / "miniforge3" / "envs" / "openpi-py310-torch" / "bin" / "python"
PYTORCH_WORKER_SCRIPT = SCRIPT_DIR / "openpi_pytorch_policy.py"


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


def _select_pytorch_runtime_python() -> Path:
    explicit = os.environ.get("OPENPI_PYTORCH_RUNTIME_PYTHON", "").strip()
    candidate = Path(explicit).expanduser() if explicit else DEFAULT_PYTORCH_RUNTIME_PYTHON
    candidate = candidate.resolve()
    if not candidate.exists():
        raise FileNotFoundError(
            f"PyTorch runtime python not found: {candidate}. "
            "Set OPENPI_PYTORCH_RUNTIME_PYTHON to the py310 CUDA torch environment."
        )
    return candidate


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


def _build_pytorch_worker_env(runtime_python: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("PYTHONHOME", None)

    pythonpath = env.get("PYTHONPATH", "")
    if pythonpath:
        runtime_prefix = runtime_python.resolve().parent.parent
        filtered: list[str] = []
        for entry in pythonpath.split(os.pathsep):
            if not entry:
                continue
            normalized = entry.replace("\\", "/")
            if "/.local/lib/python" in normalized:
                continue
            try:
                resolved = Path(entry).expanduser().resolve()
            except OSError:
                filtered.append(entry)
                continue
            if "site-packages" in resolved.parts and runtime_prefix not in resolved.parents:
                continue
            filtered.append(entry)
        if filtered:
            env["PYTHONPATH"] = os.pathsep.join(filtered)
        else:
            env.pop("PYTHONPATH", None)
    return env


@dataclass(frozen=True)
class PolicyLoadSpec:
    remote: bool = False
    config_name: str = DEFAULT_CONFIG_NAME
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR
    backend: str = DEFAULT_POLICY_BACKEND  # "auto" | "jax" | "pytorch"
    pytorch_checkpoint_dir: Path = DEFAULT_PYTORCH_CHECKPOINT_DIR
    default_prompt: str | None = None
    action_horizon: int | None = None
    pytorch_device: str | None = DEFAULT_PYTORCH_DEVICE
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


def _resolve_local_backend_and_checkpoint(spec: PolicyLoadSpec) -> tuple[str, Path]:
    backend = (spec.backend or "auto").strip().lower()
    if backend not in {"auto", "jax", "pytorch"}:
        raise ValueError(f"Unsupported OPENPI_POLICY_BACKEND={spec.backend!r}; expected auto|jax|pytorch")

    jax_checkpoint_dir = Path(spec.checkpoint_dir).resolve()
    pytorch_checkpoint_dir = Path(spec.pytorch_checkpoint_dir).resolve()

    if backend == "jax":
        return "jax", jax_checkpoint_dir
    if backend == "pytorch":
        return "pytorch", pytorch_checkpoint_dir

    if _checkpoint_has_safetensors_weights(pytorch_checkpoint_dir):
        return "pytorch", pytorch_checkpoint_dir
    return "jax", jax_checkpoint_dir


class LocalPolicyRunner:
    def __init__(self, policy: Any, *, metadata: dict[str, Any] | None = None) -> None:
        self._policy = policy
        self._metadata = metadata or {}

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        if noise is None:
            return self._policy.infer(obs)
        return self._policy.infer(obs, noise=noise)

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def close(self) -> None:
        if hasattr(self._policy, "close"):
            self._policy.close()

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class SubprocessPyTorchPolicy:
    def __init__(
        self,
        *,
        repo_root: Path,
        checkpoint_dir: Path,
        pytorch_device: str | None,
        default_prompt: str | None,
        sample_kwargs: dict[str, Any] | None,
    ) -> None:
        self._lock = threading.Lock()
        self._runtime_python = _select_pytorch_runtime_python()
        num_steps = int((sample_kwargs or {}).get("num_steps", 10))
        compile_mode = os.environ.get("OPENPI_PYTORCH_COMPILE_MODE", "").strip()
        cmd = [
            str(self._runtime_python),
            "-u",
            "-s",
            str(PYTORCH_WORKER_SCRIPT),
            "--repo-root",
            str(repo_root),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--device",
            str(pytorch_device or DEFAULT_PYTORCH_DEVICE or "cuda"),
            "--num-steps",
            str(num_steps),
        ]
        if default_prompt:
            cmd.extend(["--default-prompt", str(default_prompt)])
        if compile_mode:
            cmd.extend(["--compile-mode", compile_mode])
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
            env=_build_pytorch_worker_env(self._runtime_python),
        )
        ready = self._recv()
        if not ready.get("ok"):
            self.close()
            raise RuntimeError(
                "PyTorch policy worker failed to start: "
                f"{ready.get('error', 'unknown error')}\n{ready.get('traceback', '')}".rstrip()
            )
        self._metadata = dict(ready.get("metadata", {}) or {})
        self._metadata.setdefault("policy_backend", "pytorch")
        self._metadata.setdefault("checkpoint_dir", str(checkpoint_dir))
        self._metadata.setdefault("runtime_python", str(self._runtime_python))

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def infer(self, obs: dict[str, Any], *, noise: np.ndarray | None = None) -> dict[str, Any]:
        with self._lock:
            self._send({"op": "infer", "obs": obs, "noise": noise})
            response = self._recv()
        if not response.get("ok"):
            raise RuntimeError(
                "PyTorch policy worker infer failed: "
                f"{response.get('error', 'unknown error')}\n{response.get('traceback', '')}".rstrip()
            )
        return response["result"]

    def reset(self) -> None:
        with self._lock:
            self._send({"op": "reset"})
            response = self._recv()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "PyTorch policy worker reset failed"))

    def close(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            if proc.poll() is None:
                try:
                    with self._lock:
                        self._send({"op": "close"})
                        self._recv()
                except Exception:
                    pass
        finally:
            if proc.stdin:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
            if proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            self._proc = None

    def _send(self, message: dict[str, Any]) -> None:
        if self._proc.stdin is None:
            raise RuntimeError("PyTorch policy worker stdin is not available.")
        payload = pickle.dumps(encode_worker_value(message), protocol=pickle.HIGHEST_PROTOCOL)
        self._proc.stdin.write(len(payload).to_bytes(8, "little", signed=False))
        self._proc.stdin.write(payload)
        self._proc.stdin.flush()

    def _recv(self) -> dict[str, Any]:
        if self._proc.stdout is None:
            raise RuntimeError("PyTorch policy worker stdout is not available.")
        header = self._proc.stdout.read(8)
        if not header:
            rc = self._proc.poll()
            raise RuntimeError(f"PyTorch policy worker exited unexpectedly (returncode={rc}).")
        size = int.from_bytes(header, "little", signed=False)
        payload = self._proc.stdout.read(size)
        while len(payload) < size:
            chunk = self._proc.stdout.read(size - len(payload))
            if not chunk:
                rc = self._proc.poll()
                raise RuntimeError(f"PyTorch policy worker stream closed unexpectedly (returncode={rc}).")
            payload += chunk
        return decode_worker_value(pickle.loads(payload))


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
    backend, checkpoint_dir = _resolve_local_backend_and_checkpoint(spec)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {checkpoint_dir}")

    sample_kwargs = _effective_sample_kwargs(spec)
    if sample_kwargs:
        print(f"  [local] OpenPI sample_kwargs: {sample_kwargs}")
    print(
        "  [local] OpenPI backend="
        f"{backend} checkpoint={checkpoint_dir} "
        f"pytorch_device={spec.pytorch_device if backend == 'pytorch' else 'n/a'}"
    )

    try:
        from openpi_client.action_chunk_broker import ActionChunkBroker
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "OpenPI client dependencies are not available in the current Python environment."
        ) from exc

    if backend == "jax":
        _apply_local_jax_runtime_defaults()
        _apply_jax_platform(_choose_local_jax_platform(checkpoint_dir))
        try:
            from openpi.training.config import get_config
            from openpi.policies.policy_config import create_trained_policy
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "OpenPI JAX local inference dependencies are not available in the current Python environment. "
                "Please run under an environment with the OpenPI repo dependencies installed."
            ) from exc
        train_cfg = get_config(spec.config_name)
        policy = create_trained_policy(
            train_cfg,
            checkpoint_dir,
            default_prompt=spec.default_prompt,
            pytorch_device=spec.pytorch_device,
            sample_kwargs=sample_kwargs,
        )
    else:
        policy = SubprocessPyTorchPolicy(
            repo_root=DEFAULT_REPO_ROOT,
            checkpoint_dir=checkpoint_dir,
            pytorch_device=spec.pytorch_device,
            default_prompt=spec.default_prompt,
            sample_kwargs=sample_kwargs,
        )

    metadata = dict(getattr(policy, "metadata", {}) or {})
    if spec.action_horizon is not None:
        policy = ActionChunkBroker(policy, action_horizon=int(spec.action_horizon))
    metadata.setdefault("policy_backend", backend)
    metadata.setdefault("checkpoint_dir", str(checkpoint_dir))
    if backend == "pytorch" and spec.pytorch_device:
        metadata.setdefault("pytorch_device", str(spec.pytorch_device))
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
