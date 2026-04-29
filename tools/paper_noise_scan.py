#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import contextlib
import csv
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

from utils.env_utils import load_default_env
from utils.runtime_config import DEFAULT_PYTORCH_CHECKPOINT_DIR

load_default_env()

PAPER_NOISE_SCAN_SCHEMA = "paper_noise_scan_v1"


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


def _json_safe(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return str(value)


def compute_action_jerks(actions: np.ndarray, *, action_dims: int = 6) -> np.ndarray:
    arr = np.asarray(actions, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"actions must be 2D, got shape {arr.shape}")
    usable_dims = min(int(action_dims), arr.shape[1])
    if usable_dims <= 0:
        raise ValueError("action_dims must be positive")
    if arr.shape[0] < 3:
        return np.zeros((0,), dtype=np.float64)
    deltas = arr[2:, :usable_dims] - 2.0 * arr[1:-1, :usable_dims] + arr[:-2, :usable_dims]
    return np.linalg.norm(deltas, axis=1)


def compute_boundary_artifact_metrics(
    actions: np.ndarray,
    *,
    chunk_k: int = 5,
    action_dims: int = 6,
) -> dict[str, object]:
    chunk = int(chunk_k)
    if chunk <= 0:
        raise ValueError("chunk_k must be positive")
    jerks = compute_action_jerks(actions, action_dims=action_dims)
    action_indices = np.arange(2, 2 + len(jerks), dtype=np.int64)
    phases = action_indices % chunk
    boundary_mask = np.isin(phases, [0, 1])
    interior_mask = np.isin(phases, [2, 3, 4])

    boundary_mean = float(np.mean(jerks[boundary_mask])) if np.any(boundary_mask) else None
    interior_mean = float(np.mean(jerks[interior_mask])) if np.any(interior_mask) else None
    boundary_gap = (
        None
        if boundary_mean is None or interior_mean is None
        else float(boundary_mean - interior_mean)
    )

    first_transition = None
    if chunk >= 2 and chunk - 2 < len(jerks):
        first_transition = float(jerks[chunk - 2])

    first_gap = None
    local_boundary = jerks[max(0, chunk - 2) : max(0, chunk - 2) + 2]
    local_interior = jerks[max(0, chunk) : max(0, chunk) + 3]
    if len(local_boundary) > 0 and len(local_interior) > 0:
        first_gap = float(np.mean(local_boundary) - np.mean(local_interior))

    return {
        "action_count": int(np.asarray(actions).shape[0]),
        "jerk_count": int(len(jerks)),
        "chunk_k": int(chunk),
        "action_dims": int(min(int(action_dims), np.asarray(actions).shape[1])),
        "jerks": [float(v) for v in jerks.tolist()],
        "boundary_mean_jerk": boundary_mean,
        "interior_mean_jerk": interior_mean,
        "boundary_interior_gap": boundary_gap,
        "first_boundary_transition_jerk": first_transition,
        "first_boundary_gap": first_gap,
    }


def iter_noise_plan(
    *,
    base_noise: np.ndarray | None = None,
    direction: np.ndarray | None = None,
    alphas: list[float] | None = None,
    noise_bank: np.ndarray | None = None,
    samples: int = 24,
    seed: int = 0,
    action_shape: tuple[int, int] | None = None,
) -> Iterator[dict[str, object]]:
    if noise_bank is not None:
        bank = np.asarray(noise_bank, dtype=np.float32)
        if bank.ndim == 2:
            bank = bank[None, ...]
        if bank.ndim != 3:
            raise ValueError(f"noise bank must have shape (N,H,D), got {bank.shape}")
        for idx, noise in enumerate(bank):
            yield {"noise_id": f"bank_{idx:04d}", "sample_idx": idx, "alpha": None, "noise": noise.copy()}
        return

    if direction is not None:
        if base_noise is None:
            raise ValueError("base_noise is required when direction is provided")
        base = np.asarray(base_noise, dtype=np.float32)
        direc = np.asarray(direction, dtype=np.float32)
        if base.shape != direc.shape:
            raise ValueError(f"base noise shape {base.shape} != direction shape {direc.shape}")
        for idx, alpha in enumerate(alphas or [-1.0, 0.0, 1.0]):
            alpha_f = float(alpha)
            yield {
                "noise_id": f"alpha_{alpha_f:g}",
                "sample_idx": idx,
                "alpha": alpha_f,
                "noise": (base + alpha_f * direc).astype(np.float32),
            }
        return

    if action_shape is None:
        raise ValueError("action_shape is required when generating random noises")
    rng = np.random.default_rng(int(seed))
    for idx in range(int(samples)):
        noise = rng.standard_normal(action_shape, dtype=np.float32)
        yield {"noise_id": f"seed{int(seed)}_{idx:04d}", "sample_idx": idx, "alpha": None, "noise": noise}


def _load_action_shape(checkpoint_dir: Path) -> tuple[int, int]:
    config_path = checkpoint_dir / "config.json"
    if config_path.exists():
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        return int(config_data["action_horizon"]), int(config_data["action_dim"])
    return 30, 32


def _capture_observation(prompt: str, *, pose_frame: str) -> dict[str, Any]:
    from support.get_obs import RealRobotOpenPIObservationBuilder
    from support.pose_align import set_runtime_alignment
    from support.tcp_control import get_robot_snapshot
    from utils.run_lock import acquire_camera_runtime_lock

    runtime_lock = acquire_camera_runtime_lock("tools/paper_noise_scan.py")
    atexit.register(runtime_lock.release)
    snap = get_robot_snapshot()
    set_runtime_alignment(snap.tcp_pose, frame_mode=pose_frame)
    with RealRobotOpenPIObservationBuilder() as builder:
        aligned = builder.build_observation(prompt)
    return {
        "observation/state": np.asarray(aligned.obs["observation/state"], dtype=np.float32).copy(),
        "observation/image": np.asarray(aligned.obs["observation/image"], dtype=np.uint8).copy(),
        "observation/wrist_image": np.asarray(aligned.obs["observation/wrist_image"], dtype=np.uint8).copy(),
        "prompt": str(prompt),
    }


def _load_observation_npz(path: Path) -> dict[str, Any]:
    with np.load(path.expanduser().resolve(), allow_pickle=False) as data:
        prompt_raw = data["prompt"]
        prompt = str(prompt_raw.item() if isinstance(prompt_raw, np.ndarray) and prompt_raw.shape == () else prompt_raw)
        return {
            "observation/state": np.asarray(data["observation_state"], dtype=np.float32),
            "observation/image": np.asarray(data["observation_image"], dtype=np.uint8),
            "observation/wrist_image": np.asarray(data["observation_wrist_image"], dtype=np.uint8),
            "prompt": prompt,
        }


def _save_observation_npz(path: Path, obs: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        observation_state=np.asarray(obs["observation/state"], dtype=np.float32),
        observation_image=np.asarray(obs["observation/image"], dtype=np.uint8),
        observation_wrist_image=np.asarray(obs["observation/wrist_image"], dtype=np.uint8),
        prompt=np.asarray(str(obs["prompt"])),
    )


def _load_policy(*, checkpoint_dir: Path, device: str, num_steps: int, prompt: str):
    from support.load_policy import DEFAULT_REPO_ROOT, SubprocessPyTorchPolicy

    env = {
        "OPENPI_PYTORCH_COMPILE_MODE": os.environ.get("OPENPI_PYTORCH_COMPILE_MODE", "none"),
        "OPENPI_PYTORCH_ATTN_BACKEND": os.environ.get("OPENPI_PYTORCH_ATTN_BACKEND", "eager"),
    }
    with temporary_env(env):
        return SubprocessPyTorchPolicy(
            repo_root=DEFAULT_REPO_ROOT,
            checkpoint_dir=checkpoint_dir,
            pytorch_device=device,
            default_prompt=prompt,
            sample_kwargs={"num_steps": int(num_steps)},
        )


def _parse_alphas(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture one observation and scan OpenPI artifact metrics over noises.")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--observation-npz", type=Path)
    parser.add_argument("--pytorch-checkpoint-dir", type=Path, default=Path(DEFAULT_PYTORCH_CHECKPOINT_DIR))
    parser.add_argument("--device", type=str, default=os.environ.get("OPENPI_PYTORCH_DEVICE", "cuda"))
    parser.add_argument("--num-steps", type=int, default=int(os.environ.get("OPENPI_SAMPLE_NUM_STEPS", "10")))
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-k", type=int, default=5)
    parser.add_argument("--condition", type=str, default="noise-scan")
    parser.add_argument("--pose-frame", type=str, choices=("sim", "real"), default="sim")
    parser.add_argument("--noise-bank-npy", type=Path)
    parser.add_argument("--base-noise-npy", type=Path)
    parser.add_argument("--direction-npy", type=Path)
    parser.add_argument("--alphas", type=str, default="-1,-0.5,0,0.5,1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "paper_noise_scan")
    parser.add_argument("--no-actions", action="store_true", help="Omit full action arrays from JSONL records.")
    args = parser.parse_args()

    checkpoint_dir = args.pytorch_checkpoint_dir.expanduser().resolve()
    output_root = args.output_dir.expanduser().resolve()
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    if args.observation_npz is not None:
        obs = _load_observation_npz(args.observation_npz)
    else:
        if not args.prompt:
            parser.error("--prompt is required when --observation-npz is not provided")
        obs = _capture_observation(args.prompt, pose_frame=args.pose_frame)
    _save_observation_npz(run_dir / "observation.npz", obs)

    action_shape = _load_action_shape(checkpoint_dir)
    noise_bank = None if args.noise_bank_npy is None else np.load(args.noise_bank_npy.expanduser().resolve()).astype(np.float32)
    base_noise = None if args.base_noise_npy is None else np.load(args.base_noise_npy.expanduser().resolve()).astype(np.float32)
    direction = None if args.direction_npy is None else np.load(args.direction_npy.expanduser().resolve()).astype(np.float32)
    if direction is not None and base_noise is None:
        rng = np.random.default_rng(int(args.seed))
        base_noise = rng.standard_normal(action_shape, dtype=np.float32)

    noise_items = list(
        iter_noise_plan(
            base_noise=base_noise,
            direction=direction,
            alphas=_parse_alphas(args.alphas),
            noise_bank=noise_bank,
            samples=int(args.samples),
            seed=int(args.seed),
            action_shape=action_shape,
        )
    )
    np.save(run_dir / "noise_bank.npy", np.stack([np.asarray(item["noise"], dtype=np.float32) for item in noise_items], axis=0))

    policy = _load_policy(
        checkpoint_dir=checkpoint_dir,
        device=str(args.device),
        num_steps=int(args.num_steps),
        prompt=str(obs["prompt"]),
    )
    records: list[dict[str, object]] = []
    jsonl_path = run_dir / "noise_scan.jsonl"
    try:
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for item in noise_items:
                started = time.perf_counter()
                result = policy.infer(obs, noise=np.asarray(item["noise"], dtype=np.float32))
                wall_ms = (time.perf_counter() - started) * 1000.0
                actions = np.asarray(result["actions"], dtype=np.float32)
                metrics = compute_boundary_artifact_metrics(actions, chunk_k=int(args.chunk_k), action_dims=6)
                record = {
                    "event": "noise_scan_sample",
                    "schema": PAPER_NOISE_SCAN_SCHEMA,
                    "condition": str(args.condition),
                    "prompt": str(obs["prompt"]),
                    "sample_idx": int(item["sample_idx"]),
                    "noise_id": str(item["noise_id"]),
                    "alpha": item["alpha"],
                    "seed": int(args.seed),
                    "chunk_k": int(args.chunk_k),
                    "noise_shape": [int(v) for v in np.asarray(item["noise"]).shape],
                    "policy_infer_ms": float(result.get("policy_timing", {}).get("infer_ms", wall_ms)),
                    "wall_ms": float(wall_ms),
                    "metrics": metrics,
                }
                if not args.no_actions:
                    record["actions"] = actions.tolist()
                records.append(record)
                fh.write(json.dumps(_json_safe(record), ensure_ascii=False) + "\n")
                print(
                    f"[{int(item['sample_idx']):04d}] {item['noise_id']} "
                    f"gap={metrics['first_boundary_gap']} btj={metrics['first_boundary_transition_jerk']}"
                )
    finally:
        policy.close()

    csv_path = run_dir / "noise_scan_metrics.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_idx",
                "noise_id",
                "alpha",
                "condition",
                "first_boundary_transition_jerk",
                "first_boundary_gap",
                "boundary_interior_gap",
                "policy_infer_ms",
                "wall_ms",
            ],
        )
        writer.writeheader()
        for record in records:
            metrics = record["metrics"]
            assert isinstance(metrics, dict)
            writer.writerow(
                {
                    "sample_idx": record["sample_idx"],
                    "noise_id": record["noise_id"],
                    "alpha": record["alpha"],
                    "condition": record["condition"],
                    "first_boundary_transition_jerk": metrics["first_boundary_transition_jerk"],
                    "first_boundary_gap": metrics["first_boundary_gap"],
                    "boundary_interior_gap": metrics["boundary_interior_gap"],
                    "policy_infer_ms": record["policy_infer_ms"],
                    "wall_ms": record["wall_ms"],
                }
            )

    summary = {
        "schema": PAPER_NOISE_SCAN_SCHEMA,
        "run_dir": str(run_dir),
        "jsonl": str(jsonl_path),
        "csv": str(csv_path),
        "observation_npz": str(run_dir / "observation.npz"),
        "noise_bank_npy": str(run_dir / "noise_bank.npy"),
        "samples": len(records),
        "condition": str(args.condition),
        "chunk_k": int(args.chunk_k),
        "prompt": str(obs["prompt"]),
        "checkpoint_dir": str(checkpoint_dir),
    }
    (run_dir / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_json_safe(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
