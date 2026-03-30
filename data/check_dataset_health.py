#!/usr/bin/env python3
"""
Dataset health checker for collected OpenPI real-robot episodes.

Checks:
- required files and array shapes
- metadata consistency
- NaN / Inf
- quaternion normalization
- action/state consistency
- timestamp / env_step consistency
- image corruption signals: black frames, duplicate runs, low unique-frame count
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


STATE_DIM = 8
ACTION_DIM = 7
DEFAULT_RAW_CAPTURE_FPS = 30.0
UNIQUE_RATIO_WARN = 0.90
UNIQUE_RATIO_FAIL = 0.75
DUPLICATE_RUN_WARN = 5
DUPLICATE_RUN_FAIL = 10
TIMESTAMP_GRID_WARN_S = 1e-4
TIMESTAMP_GRID_FAIL_S = 1e-3
QUAT_NORM_WARN = 1e-3
QUAT_NORM_FAIL = 1e-2
ACTION_ERR_WARN = 1e-4
ACTION_ERR_FAIL = 1e-3


@dataclass
class EpisodeReport:
    episode: str
    status: str
    task: str | None
    n_frames: int
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


def quat_to_euler_wxyz(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64).reshape(4)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.copysign(np.pi / 2.0, sinp)
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return np.array([roll, pitch, yaw], dtype=np.float64)


def expected_actions_from_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    actions = np.zeros((states.shape[0], ACTION_DIM), dtype=np.float64)
    if len(states) == 0:
        return actions

    pos = states[:, :3]
    quat = states[:, 3:7]
    grip = states[:, 7]
    eulers = np.stack([quat_to_euler_wxyz(q) for q in quat], axis=0)

    actions[:-1, :3] = pos[1:] - pos[:-1]
    actions[:-1, 3:6] = np.arctan2(np.sin(eulers[1:] - eulers[:-1]), np.cos(eulers[1:] - eulers[:-1]))
    actions[:-1, 6] = grip[1:]
    actions[-1, 6] = grip[-1]
    return actions


def image_stream_metrics(arr: np.ndarray, *, saved_fps: float, raw_capture_fps: float) -> dict[str, Any]:
    frames = int(arr.shape[0])
    if frames == 0:
        return {
            "frames": 0,
            "expected_unique": 0,
            "unique_frames": 0,
            "unique_ratio": 0.0,
            "zero_frames": 0,
            "near_zero_frames": 0,
            "same_pairs": 0,
            "longest_duplicate_run": 0,
        }

    hashes = [hashlib.sha1(arr[idx].tobytes()).hexdigest() for idx in range(frames)]
    unique_frames = len(set(hashes))
    expected_unique = max(1, int(round((frames - 1) * raw_capture_fps / max(saved_fps, 1e-6))) + 1)
    unique_ratio = float(unique_frames) / float(expected_unique)

    flat = arr.reshape(frames, -1)
    zero_frames = int(np.sum(flat.max(axis=1) == 0))
    near_zero_frames = int(np.sum(flat.mean(axis=1) < 1.0))

    same = np.all(arr[1:] == arr[:-1], axis=(1, 2, 3)) if frames > 1 else np.array([], dtype=bool)
    longest = 0
    current = 0
    for flag in same:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    longest_run = longest + (1 if longest else 0)

    return {
        "frames": frames,
        "expected_unique": expected_unique,
        "unique_frames": unique_frames,
        "unique_ratio": unique_ratio,
        "zero_frames": zero_frames,
        "near_zero_frames": near_zero_frames,
        "same_pairs": int(np.sum(same)),
        "longest_duplicate_run": longest_run,
    }


def inspect_episode(ep_dir: Path, *, raw_capture_fps: float) -> EpisodeReport:
    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {}

    required = [
        "states.npy",
        "actions.npy",
        "timestamps.npy",
        "env_steps.npy",
        "images.npz",
        "metadata.json",
    ]
    missing = [name for name in required if not (ep_dir / name).exists()]
    if missing:
        return EpisodeReport(
            episode=ep_dir.name,
            status="fail",
            task=None,
            n_frames=0,
            errors=[f"missing files: {missing}"],
            warnings=[],
            metrics={},
        )

    metadata = json.loads((ep_dir / "metadata.json").read_text(encoding="utf-8"))
    states = np.load(ep_dir / "states.npy")
    actions = np.load(ep_dir / "actions.npy")
    timestamps = np.load(ep_dir / "timestamps.npy")
    env_steps = np.load(ep_dir / "env_steps.npy")
    images = np.load(ep_dir / "images.npz")
    main_images = images["main_images"]
    wrist_images = images["wrist_images"]

    n_frames = int(states.shape[0])
    task = metadata.get("task")
    metrics["task"] = task
    metrics["n_frames"] = n_frames

    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        errors.append(f"states shape invalid: {states.shape}")
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        errors.append(f"actions shape invalid: {actions.shape}")
    if timestamps.shape != (n_frames,):
        errors.append(f"timestamps shape invalid: {timestamps.shape} vs ({n_frames},)")
    if env_steps.shape != (n_frames,):
        errors.append(f"env_steps shape invalid: {env_steps.shape} vs ({n_frames},)")
    if main_images.shape[:1] != (n_frames,):
        errors.append(f"main_images first dim invalid: {main_images.shape}")
    if wrist_images.shape[:1] != (n_frames,):
        errors.append(f"wrist_images first dim invalid: {wrist_images.shape}")
    if main_images.dtype != np.uint8:
        warnings.append(f"main_images dtype is {main_images.dtype}, expected uint8")
    if wrist_images.dtype != np.uint8:
        warnings.append(f"wrist_images dtype is {wrist_images.dtype}, expected uint8")

    for name, arr in [
        ("states", states),
        ("actions", actions),
        ("timestamps", timestamps),
    ]:
        if np.issubdtype(arr.dtype, np.floating):
            nan_count = int(np.isnan(arr).sum())
            inf_count = int(np.isinf(arr).sum())
            metrics[f"{name}_nan"] = nan_count
            metrics[f"{name}_inf"] = inf_count
            if nan_count or inf_count:
                errors.append(f"{name} contains nan/inf: nan={nan_count}, inf={inf_count}")

    if states.ndim == 2 and states.shape[1] == STATE_DIM:
        quat = states[:, 3:7].astype(np.float64)
        quat_norm = np.linalg.norm(quat, axis=1)
        max_quat_err = float(np.max(np.abs(quat_norm - 1.0))) if len(quat_norm) else 0.0
        metrics["quat_norm_err_max"] = max_quat_err
        if max_quat_err > QUAT_NORM_FAIL:
            errors.append(f"quaternion norm error too large: {max_quat_err:.6g}")
        elif max_quat_err > QUAT_NORM_WARN:
            warnings.append(f"quaternion norm error elevated: {max_quat_err:.6g}")

    saved_fps = float(metadata.get("fps", 0.0) or 0.0)
    if saved_fps <= 0:
        warnings.append("metadata fps missing or invalid")
        saved_fps = 50.0

    if len(env_steps) == n_frames and len(timestamps) == n_frames:
        step_diff = np.diff(env_steps)
        unique_step_diff = np.unique(step_diff) if len(step_diff) else np.array([], dtype=np.int64)
        metrics["env_step_diff_unique"] = unique_step_diff.tolist()
        if len(step_diff) and not np.all(step_diff == 1):
            errors.append(f"env_steps are not strictly incremental by 1: {unique_step_diff.tolist()}")

        expected_timestamps = env_steps.astype(np.float64) / saved_fps
        grid_err = float(np.max(np.abs(timestamps.astype(np.float64) - expected_timestamps))) if len(timestamps) else 0.0
        metrics["timestamp_grid_err_max_s"] = grid_err
        if grid_err > TIMESTAMP_GRID_FAIL_S:
            errors.append(f"timestamps deviate from env_step/fps grid: {grid_err:.6g}s")
        elif grid_err > TIMESTAMP_GRID_WARN_S:
            warnings.append(f"timestamps slightly deviate from grid: {grid_err:.6g}s")

    if states.shape == (n_frames, STATE_DIM) and actions.shape == (n_frames, ACTION_DIM):
        expected_actions = expected_actions_from_states(states)
        action_err = np.abs(actions.astype(np.float64) - expected_actions)
        max_action_err = np.max(action_err, axis=0)
        metrics["action_err_max"] = max_action_err.tolist()
        if float(np.max(max_action_err)) > ACTION_ERR_FAIL:
            errors.append(f"actions inconsistent with states: max_err={max_action_err.tolist()}")
        elif float(np.max(max_action_err)) > ACTION_ERR_WARN:
            warnings.append(f"actions slightly inconsistent with states: max_err={max_action_err.tolist()}")

    for key, arr in [("main_images", main_images), ("wrist_images", wrist_images)]:
        m = image_stream_metrics(arr, saved_fps=saved_fps, raw_capture_fps=raw_capture_fps)
        metrics[key] = m

        if m["zero_frames"] > 0:
            errors.append(f"{key} contains {m['zero_frames']} all-zero frames")
        if m["near_zero_frames"] > 0 and m["zero_frames"] == 0:
            warnings.append(f"{key} contains {m['near_zero_frames']} near-black frames")

        if m["unique_ratio"] < UNIQUE_RATIO_FAIL:
            errors.append(
                f"{key} unique frame ratio too low: {m['unique_frames']}/{m['expected_unique']} "
                f"({m['unique_ratio']:.3f})"
            )
        elif m["unique_ratio"] < UNIQUE_RATIO_WARN:
            warnings.append(
                f"{key} unique frame ratio is low: {m['unique_frames']}/{m['expected_unique']} "
                f"({m['unique_ratio']:.3f})"
            )

        if m["longest_duplicate_run"] > DUPLICATE_RUN_FAIL:
            errors.append(f"{key} longest exact duplicate run too long: {m['longest_duplicate_run']}")
        elif m["longest_duplicate_run"] > DUPLICATE_RUN_WARN:
            warnings.append(f"{key} longest exact duplicate run is elevated: {m['longest_duplicate_run']}")

    status = "ok"
    if errors:
        status = "fail"
    elif warnings:
        status = "warn"

    return EpisodeReport(
        episode=ep_dir.name,
        status=status,
        task=task,
        n_frames=n_frames,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check collected episode integrity and camera health.")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent))
    parser.add_argument("--episode", type=str, default="", help="Optional single episode directory name, e.g. episode_0003")
    parser.add_argument("--raw-capture-fps", type=float, default=DEFAULT_RAW_CAPTURE_FPS)
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--keep-bad", action="store_true", help="Only report bad episodes, do not delete or renumber.")
    return parser.parse_args()


def collect_episode_dirs(data_dir: Path) -> list[Path]:
    return sorted(
        (p for p in data_dir.glob("episode_*") if p.is_dir()),
        key=lambda p: p.name,
    )


def delete_bad_episodes(reports: list[EpisodeReport], data_dir: Path) -> list[str]:
    deleted: list[str] = []
    for report in reports:
        if report.status != "fail":
            continue
        ep_dir = data_dir / report.episode
        if ep_dir.exists():
            shutil.rmtree(ep_dir)
            deleted.append(report.episode)
    return deleted


def renumber_episode_dirs(data_dir: Path) -> list[dict[str, str]]:
    episode_dirs = collect_episode_dirs(data_dir)
    rename_pairs: list[tuple[Path, Path]] = []
    for idx, ep_dir in enumerate(episode_dirs):
        target = data_dir / f"episode_{idx:04d}"
        if ep_dir != target:
            rename_pairs.append((ep_dir, target))

    if not rename_pairs:
        return []

    staged_pairs: list[tuple[Path, Path, str]] = []
    for idx, (src, dst) in enumerate(rename_pairs):
        tmp = data_dir / f"__renumber_tmp_{idx:04d}"
        original_name = src.name
        src.rename(tmp)
        staged_pairs.append((tmp, dst, original_name))

    rename_log: list[dict[str, str]] = []
    for tmp, dst, original_name in staged_pairs:
        tmp.rename(dst)
        rename_log.append({"from": original_name, "to": dst.name})
    return rename_log


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    if args.episode:
        episode_dirs = [data_dir / args.episode]
    else:
        episode_dirs = collect_episode_dirs(data_dir)

    reports = [inspect_episode(ep_dir, raw_capture_fps=float(args.raw_capture_fps)) for ep_dir in episode_dirs if ep_dir.exists()]
    deleted_episodes: list[str] = []
    rename_log: list[dict[str, str]] = []

    if not args.keep_bad:
        deleted_episodes = delete_bad_episodes(reports, data_dir)
        if deleted_episodes:
            print(f"Deleted bad episodes: {deleted_episodes}")
            rename_log = renumber_episode_dirs(data_dir)
            if rename_log:
                print("Renumbered remaining episodes:")
                for item in rename_log:
                    print(f"  {item['from']} -> {item['to']}")
            episode_dirs = collect_episode_dirs(data_dir)
            reports = [inspect_episode(ep_dir, raw_capture_fps=float(args.raw_capture_fps)) for ep_dir in episode_dirs if ep_dir.exists()]

    summary = {
        "data_dir": str(data_dir),
        "raw_capture_fps": float(args.raw_capture_fps),
        "deleted_episodes": deleted_episodes,
        "renamed_episodes": rename_log,
        "episodes": [
            {
                "episode": r.episode,
                "status": r.status,
                "task": r.task,
                "n_frames": r.n_frames,
                "errors": r.errors,
                "warnings": r.warnings,
                "metrics": r.metrics,
            }
            for r in reports
        ],
    }

    ok = warn = fail = 0
    for report in reports:
        if report.status == "ok":
            ok += 1
        elif report.status == "warn":
            warn += 1
        else:
            fail += 1

        print(f"[{report.status.upper()}] {report.episode}  task={report.task!r}  frames={report.n_frames}")
        if report.errors:
            for item in report.errors:
                print(f"  ERROR: {item}")
        if report.warnings:
            for item in report.warnings:
                print(f"  WARN: {item}")
        for key in ("main_images", "wrist_images"):
            if key in report.metrics:
                metric = report.metrics[key]
                print(
                    f"  {key}: unique={metric['unique_frames']}/{metric['expected_unique']} "
                    f"ratio={metric['unique_ratio']:.3f} zero={metric['zero_frames']} "
                    f"dup_run={metric['longest_duplicate_run']}"
                )

    print(f"\nSummary: ok={ok} warn={warn} fail={fail} total={len(reports)}")

    json_out = Path(args.json_out).resolve() if args.json_out else data_dir / "dataset_health_report.json"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Report written to: {json_out}")

    return 1 if fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
