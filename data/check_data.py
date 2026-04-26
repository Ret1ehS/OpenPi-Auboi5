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
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


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
ACTION_ERR_WARN_STRICT = 1e-4
ACTION_ERR_WARN_ORIENTATION = 1e-3
ACTION_ERR_FAIL = 1e-3
STATE_MODE_YAW = "yaw"
CHECK_PASSED_KEY = "check_data_passed"
CHECKER_VERSION_KEY = "check_data_checker_version"
CHECKER_VERSION = 1


@dataclass
class EpisodeReport:
    episode: str
    status: str
    task: str | None
    n_frames: int
    errors: list[str]
    warnings: list[str]
    metrics: dict[str, Any]


def read_episode_metadata(ep_dir: Path) -> dict[str, Any]:
    return json.loads((ep_dir / "metadata.json").read_text(encoding="utf-8"))


def write_episode_metadata(ep_dir: Path, metadata: dict[str, Any]) -> None:
    (ep_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


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


def axis_angle_to_quat_wxyz(axis_angle: np.ndarray) -> np.ndarray:
    aa = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(aa))
    if angle <= 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    axis = aa / angle
    half = 0.5 * angle
    return np.array(
        [np.cos(half), axis[0] * np.sin(half), axis[1] * np.sin(half), axis[2] * np.sin(half)],
        dtype=np.float64,
    )


def expected_actions_from_states(states: np.ndarray) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    actions = np.zeros((states.shape[0], ACTION_DIM), dtype=np.float64)
    if len(states) == 0:
        return actions

    pos = states[:, :3]
    axis_angle = states[:, 3:6]
    grip = states[:, 6]
    eulers = np.stack([quat_to_euler_wxyz(axis_angle_to_quat_wxyz(aa)) for aa in axis_angle], axis=0)

    actions[:-1, :3] = pos[1:] - pos[:-1]
    deulers = np.arctan2(np.sin(eulers[1:] - eulers[:-1]), np.cos(eulers[1:] - eulers[:-1]))
    actions[:-1, 3:5] = deulers[:, :2]
    actions[:-1, 5] = deulers[:, 2]
    actions[:-1, 6] = grip[1:]
    actions[-1, 6] = grip[-1]
    return actions


def action_err_warn_thresholds() -> np.ndarray:
    thresholds = np.full(ACTION_DIM, ACTION_ERR_WARN_STRICT, dtype=np.float64)
    thresholds[3:6] = ACTION_ERR_WARN_ORIENTATION
    return thresholds


def validate_state_mode(metadata: dict[str, Any], *, state_dim: int) -> str:
    state_mode = str(metadata.get("state_mode", STATE_MODE_YAW) or STATE_MODE_YAW).strip().lower()
    if state_mode != STATE_MODE_YAW:
        raise ValueError(f"only yaw state_mode is supported, got {state_mode!r}")
    if int(state_dim) != 7:
        raise ValueError(f"only yaw state_mode is supported; expected 7-column states, got state_dim={state_dim}")
    return STATE_MODE_YAW


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


def inspect_episode(ep_dir: Path, *, raw_capture_fps: float, force_recheck: bool = False) -> EpisodeReport:
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

    metadata = read_episode_metadata(ep_dir)
    marker_version = metadata.get(CHECKER_VERSION_KEY)
    if (
        not force_recheck
        and bool(metadata.get(CHECK_PASSED_KEY, False))
        and marker_version == CHECKER_VERSION
    ):
        n_frames = int(metadata.get("n_frames", 0) or 0)
        task = metadata.get("task")
        metrics["task"] = task
        metrics["n_frames"] = n_frames
        metrics["skipped_prechecked"] = True
        return EpisodeReport(
            episode=ep_dir.name,
            status="skip",
            task=task,
            n_frames=n_frames,
            errors=[],
            warnings=["skipped: metadata already marked as passed"],
            metrics=metrics,
        )

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

    state_dim = int(metadata.get("state_dim", states.shape[1] if states.ndim == 2 else 0))
    action_dim = int(metadata.get("action_dim", ACTION_DIM))
    state_mode = validate_state_mode(metadata, state_dim=state_dim)
    metrics["state_dim"] = state_dim
    metrics["action_dim"] = action_dim
    metrics["state_mode"] = state_mode

    if states.ndim != 2 or states.shape[1] != state_dim:
        errors.append(f"states shape invalid: {states.shape}, expected (*, {state_dim})")
    if actions.ndim != 2 or actions.shape[1] != action_dim:
        errors.append(f"actions shape invalid: {actions.shape}, expected (*, {action_dim})")
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

    if states.ndim == 2 and states.shape[1] >= 6:
        axis_angle_norm = np.linalg.norm(states[:, 3:6].astype(np.float64), axis=1)
        metrics["axis_angle_norm_max"] = float(np.max(axis_angle_norm)) if len(axis_angle_norm) else 0.0

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

    if states.shape == (n_frames, state_dim) and actions.shape == (n_frames, action_dim) and action_dim == ACTION_DIM:
        expected_actions = expected_actions_from_states(states)
        action_err = np.abs(actions.astype(np.float64) - expected_actions)
        max_action_err = np.max(action_err, axis=0)
        warn_thresholds = action_err_warn_thresholds()
        metrics["action_err_max"] = max_action_err.tolist()
        metrics["action_err_warn_thresholds"] = warn_thresholds.tolist()
        if float(np.max(max_action_err)) > ACTION_ERR_FAIL:
            errors.append(f"actions inconsistent with states: max_err={max_action_err.tolist()}")
        elif np.any(max_action_err > warn_thresholds):
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
    parser.add_argument(
        "--delete-bad",
        action="store_true",
        help="Delete failed episodes without interactive confirmation and renumber the remainder.",
    )
    parser.add_argument(
        "--force-recheck",
        action="store_true",
        help="Ignore metadata pass markers and re-run checks for every episode.",
    )
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


def print_reports(reports: list[EpisodeReport]) -> tuple[int, int, int]:
    ok = warn = fail = 0
    for report in reports:
        if report.status == "ok":
            ok += 1
        elif report.status == "warn":
            warn += 1
        elif report.status == "skip":
            pass
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
    return ok, warn, fail


def update_metadata_pass_markers(reports: list[EpisodeReport], data_dir: Path) -> None:
    for report in reports:
        ep_dir = data_dir / report.episode
        meta_path = ep_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            metadata = read_episode_metadata(ep_dir)
        except Exception:
            continue

        if report.status == "ok":
            metadata[CHECK_PASSED_KEY] = True
            metadata[CHECKER_VERSION_KEY] = CHECKER_VERSION
        elif report.status in {"warn", "fail"}:
            metadata.pop(CHECK_PASSED_KEY, None)
            metadata.pop(CHECKER_VERSION_KEY, None)
        else:
            continue
        write_episode_metadata(ep_dir, metadata)


def should_delete_bad_episodes(*, args: argparse.Namespace, failed_reports: list[EpisodeReport]) -> bool:
    if not failed_reports:
        return False
    if args.keep_bad:
        return False
    if args.delete_bad:
        return True
    if not sys.stdin.isatty():
        print("\nNon-interactive session detected. Skipping deletion.")
        print("Re-run with --delete-bad to delete failed episodes without a prompt.")
        return False

    print("\nFailed episodes:")
    for report in failed_reports:
        print(f"  {report.episode}: {', '.join(report.errors)}")
    answer = input("\nDelete failed episodes and renumber the remainder? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"data dir not found: {data_dir}")

    if args.episode:
        episode_dirs = [data_dir / args.episode]
    else:
        episode_dirs = collect_episode_dirs(data_dir)

    reports = [
        inspect_episode(ep_dir, raw_capture_fps=float(args.raw_capture_fps), force_recheck=bool(args.force_recheck))
        for ep_dir in episode_dirs
        if ep_dir.exists()
    ]
    deleted_episodes: list[str] = []
    rename_log: list[dict[str, str]] = []

    ok, warn, fail = print_reports(reports)
    print(f"\nSummary: ok={ok} warn={warn} fail={fail} total={len(reports)}")
    update_metadata_pass_markers(reports, data_dir)

    failed_reports = [report for report in reports if report.status == "fail"]
    if should_delete_bad_episodes(args=args, failed_reports=failed_reports):
        deleted_episodes = delete_bad_episodes(reports, data_dir)
        if deleted_episodes:
            print(f"Deleted bad episodes: {deleted_episodes}")
            rename_log = renumber_episode_dirs(data_dir)
            if rename_log:
                print("Renumbered remaining episodes:")
                for item in rename_log:
                    print(f"  {item['from']} -> {item['to']}")
            episode_dirs = collect_episode_dirs(data_dir)
            reports = [
                inspect_episode(ep_dir, raw_capture_fps=float(args.raw_capture_fps), force_recheck=bool(args.force_recheck))
                for ep_dir in episode_dirs
                if ep_dir.exists()
            ]
            print("\nPost-delete check:")
            ok, warn, fail = print_reports(reports)
            print(f"\nSummary: ok={ok} warn={warn} fail={fail} total={len(reports)}")
            update_metadata_pass_markers(reports, data_dir)

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

    json_out = Path(args.json_out).resolve() if args.json_out else data_dir / "dataset_health_report.json"
    json_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Report written to: {json_out}")

    return 1 if fail > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
