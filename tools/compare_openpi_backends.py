#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from support.get_obs import RealRobotOpenPIObservationBuilder
from support.load_policy import (
    DEFAULT_REPO_ROOT,
    SubprocessPyTorchPolicy,
    _apply_jax_platform,
    _apply_local_jax_runtime_defaults,
    _choose_local_jax_platform,
)
from support.pose_align import set_runtime_alignment
from support.tcp_control import get_robot_snapshot
from utils.runtime_config import DEFAULT_CHECKPOINT_DIR, DEFAULT_CONFIG_NAME, DEFAULT_PYTORCH_CHECKPOINT_DIR


@contextlib.contextmanager
def temporary_env(overrides: dict[str, str | None]) -> Iterator[None]:
    previous: dict[str, str | None] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _capture_observation(prompt: str, *, pose_frame: str) -> dict[str, Any]:
    snap = get_robot_snapshot()
    set_runtime_alignment(snap.tcp_pose, frame_mode=pose_frame)
    with RealRobotOpenPIObservationBuilder() as builder:
        aligned = builder.build_observation(prompt)
    obs = copy.deepcopy(aligned.obs)
    state = np.asarray(obs["observation/state"], dtype=np.float32)
    main_image = np.asarray(obs["observation/image"], dtype=np.uint8)
    wrist_image = np.asarray(obs["observation/wrist_image"], dtype=np.uint8)
    print(
        json.dumps(
            {
                "captured_state": [round(float(v), 6) for v in state.tolist()],
                "main_image_shape": list(main_image.shape),
                "wrist_image_shape": list(wrist_image.shape),
            },
            ensure_ascii=False,
        )
    )
    return obs


def _load_saved_observation(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as data:
        state = np.asarray(data["observation_state"], dtype=np.float32)
        image = np.asarray(data["observation_image"], dtype=np.uint8)
        wrist = np.asarray(data["observation_wrist_image"], dtype=np.uint8)
        prompt_raw = data["prompt"]
        noise = np.asarray(data["noise"], dtype=np.float32)
    prompt = str(prompt_raw.item() if isinstance(prompt_raw, np.ndarray) and prompt_raw.shape == () else prompt_raw)
    obs = {
        "observation/state": state,
        "observation/image": image,
        "observation/wrist_image": wrist,
        "prompt": prompt,
    }
    print(
        json.dumps(
            {
                "loaded_observation": str(path),
                "loaded_state": [round(float(v), 6) for v in state.tolist()],
                "main_image_shape": list(image.shape),
                "wrist_image_shape": list(wrist.shape),
                "noise_shape": list(noise.shape),
            },
            ensure_ascii=False,
        )
    )
    return obs, noise


def _load_action_shape(checkpoint_dir: Path) -> tuple[int, int]:
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        return int(config_data["action_horizon"]), int(config_data["action_dim"])
    return 30, 32


def _load_jax_policy(*, config_name: str, checkpoint_dir: Path, default_prompt: str | None, num_steps: int):
    _apply_local_jax_runtime_defaults()
    _apply_jax_platform(_choose_local_jax_platform(checkpoint_dir))
    from openpi.policies.policy_config import create_trained_policy
    from openpi.training.config import get_config

    train_cfg = get_config(config_name)
    return create_trained_policy(
        train_cfg,
        checkpoint_dir,
        default_prompt=default_prompt,
        sample_kwargs={"num_steps": int(num_steps)},
    )


def _run_policy(policy: Any, obs: dict[str, Any], noise: np.ndarray) -> dict[str, Any]:
    started = time.perf_counter()
    result = policy.infer(obs, noise=noise)
    wall_ms = (time.perf_counter() - started) * 1000.0
    actions = np.asarray(result["actions"], dtype=np.float32)
    policy_ms = float(result.get("policy_timing", {}).get("infer_ms", wall_ms))
    return {
        "actions": actions,
        "policy_infer_ms": policy_ms,
        "wall_ms": wall_ms,
    }


def _summarize_diff(reference: np.ndarray, other: np.ndarray) -> dict[str, Any]:
    delta = other - reference
    flat = delta.reshape(-1)
    per_step_l2 = np.linalg.norm(delta, axis=1)
    return {
        "max_abs": float(np.max(np.abs(flat))),
        "mean_abs": float(np.mean(np.abs(flat))),
        "l2": float(np.linalg.norm(flat)),
        "first_step_l2": float(per_step_l2[0]),
        "mean_step_l2": float(np.mean(per_step_l2)),
        "first_step_delta": [float(v) for v in delta[0].tolist()],
    }


def _run_pytorch_variant(
    *,
    obs: dict[str, Any],
    noise: np.ndarray,
    checkpoint_dir: Path,
    num_steps: int,
    trt_denoise: bool,
    trt_vision: bool,
    attn_backend: str,
) -> dict[str, Any]:
    env = {
        "OPENPI_PYTORCH_TRT_DENOISE": "1" if trt_denoise else "0",
        "OPENPI_PYTORCH_TRT_VISION": "1" if trt_vision else "0",
        "OPENPI_PYTORCH_ATTN_BACKEND": attn_backend,
        "OPENPI_PYTORCH_COMPILE_MODE": "none",
    }
    with temporary_env(env):
        policy = SubprocessPyTorchPolicy(
            repo_root=DEFAULT_REPO_ROOT,
            checkpoint_dir=checkpoint_dir,
            pytorch_device="cuda",
            default_prompt=str(obs.get("prompt") or ""),
            sample_kwargs={"num_steps": int(num_steps)},
        )
        try:
            result = _run_policy(policy, obs, noise)
            result["metadata"] = policy.metadata
            return result
        finally:
            policy.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare JAX / PyTorch / PyTorch+TRT outputs on the same observation.")
    parser.add_argument("--prompt", type=str)
    parser.add_argument("--observation-npz", type=Path)
    parser.add_argument("--config-name", type=str, default=str(DEFAULT_CONFIG_NAME))
    parser.add_argument("--jax-checkpoint-dir", type=Path, default=Path(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--pytorch-checkpoint-dir", type=Path, default=Path(DEFAULT_PYTORCH_CHECKPOINT_DIR))
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pose-frame", type=str, choices=("sim", "real"), default="sim")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "backend_compare")
    parser.add_argument("--disable-trt-vision", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_dir = output_dir / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    if args.observation_npz is not None:
        obs, noise = _load_saved_observation(args.observation_npz)
    else:
        if not args.prompt:
            parser.error("--prompt is required when --observation-npz is not provided")
        obs = _capture_observation(args.prompt, pose_frame=args.pose_frame)
        rng = np.random.default_rng(int(args.seed))
        action_horizon, action_dim = _load_action_shape(args.pytorch_checkpoint_dir.expanduser().resolve())
        noise = rng.standard_normal((action_horizon, action_dim), dtype=np.float32)

    state = np.asarray(obs["observation/state"], dtype=np.float32)

    np.savez_compressed(
        run_dir / "observation.npz",
        observation_state=state,
        observation_image=np.asarray(obs["observation/image"], dtype=np.uint8),
        observation_wrist_image=np.asarray(obs["observation/wrist_image"], dtype=np.uint8),
        prompt=np.asarray(str(obs["prompt"])),
        noise=noise,
    )

    print("Running JAX baseline...")
    jax_policy = _load_jax_policy(
        config_name=args.config_name,
        checkpoint_dir=args.jax_checkpoint_dir.expanduser().resolve(),
        default_prompt=str(obs["prompt"]),
        num_steps=int(args.num_steps),
    )
    try:
        jax_result = _run_policy(jax_policy, obs, noise)
        jax_metadata = dict(getattr(jax_policy, "metadata", {}) or {})
    finally:
        close_fn = getattr(jax_policy, "close", None)
        if callable(close_fn):
            close_fn()

    print("Running PyTorch eager baseline...")
    pytorch_result = _run_pytorch_variant(
        obs=obs,
        noise=noise,
        checkpoint_dir=args.pytorch_checkpoint_dir.expanduser().resolve(),
        num_steps=int(args.num_steps),
        trt_denoise=False,
        trt_vision=False,
        attn_backend="eager",
    )

    print("Running PyTorch + TRT fast path...")
    pytorch_trt_result = _run_pytorch_variant(
        obs=obs,
        noise=noise,
        checkpoint_dir=args.pytorch_checkpoint_dir.expanduser().resolve(),
        num_steps=int(args.num_steps),
        trt_denoise=True,
        trt_vision=not args.disable_trt_vision,
        attn_backend="eager",
    )

    report = {
        "prompt": str(obs["prompt"]),
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "observation_state": [float(v) for v in state.tolist()],
        "jax": {
            "policy_infer_ms": jax_result["policy_infer_ms"],
            "wall_ms": jax_result["wall_ms"],
            "first_action": [float(v) for v in jax_result["actions"][0].tolist()],
            "metadata": jax_metadata,
        },
        "pytorch": {
            "policy_infer_ms": pytorch_result["policy_infer_ms"],
            "wall_ms": pytorch_result["wall_ms"],
            "first_action": [float(v) for v in pytorch_result["actions"][0].tolist()],
            "metadata": pytorch_result["metadata"],
            "diff_vs_jax": _summarize_diff(jax_result["actions"], pytorch_result["actions"]),
        },
        "pytorch_trt": {
            "policy_infer_ms": pytorch_trt_result["policy_infer_ms"],
            "wall_ms": pytorch_trt_result["wall_ms"],
            "first_action": [float(v) for v in pytorch_trt_result["actions"][0].tolist()],
            "metadata": pytorch_trt_result["metadata"],
            "diff_vs_jax": _summarize_diff(jax_result["actions"], pytorch_trt_result["actions"]),
            "diff_vs_pytorch": _summarize_diff(pytorch_result["actions"], pytorch_trt_result["actions"]),
        },
    }

    (run_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    print(f"Saved artifacts to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
