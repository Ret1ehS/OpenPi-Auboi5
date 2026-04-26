"""
Convert Mujoco collected data -> LeRobot local dataset layout (HF cache style), robust for OpenPI/LeRobot.

Usage:
  python convert_data.py \
    --data-dir path/to/your/data \
    --repo-id <repo-id>/mujoco_aubo_data \
    --robot-type aubo_i5 \
    --fps 50 \
    --image-size 224 \
    --chunk-size 1000 \
    --overwrite
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro
import imageio.v2 as imageio

# LeRobot cache root (same place you看到的 ~/.cache/huggingface/lerobot/...)
try:
    from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
except Exception:
    # fallback: typical default
    HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"


# -------------------------
# state helpers
# -------------------------
STATE_MODE_YAW = "yaw"
LEROBOT_STATE_DIM = 8
ACTION_DIM = 7


def infer_state_mode_from_metadata(metadata: dict) -> str | None:
    state_mode = metadata.get("state_mode")
    if state_mode is None:
        return None

    state_mode = str(state_mode).strip().lower()
    if state_mode == STATE_MODE_YAW:
        return STATE_MODE_YAW
    raise ValueError(f"only yaw state_mode is supported, got metadata['state_mode']={state_mode!r}")


def infer_episode_state_format(episode_dir: Path, states: np.ndarray, metadata: dict) -> str:
    """
    Supported rules:
      - metadata['state_mode'] present: only 'yaw' is accepted
      - metadata['state_mode'] missing: only 7-column yaw states are accepted
    """
    if len(states.shape) != 2:
        raise ValueError(f"{episode_dir}: states must be rank-2, got {states.shape}")

    state_dim = int(states.shape[1])
    metadata_state_mode = infer_state_mode_from_metadata(metadata)
    if metadata_state_mode is not None and metadata_state_mode != STATE_MODE_YAW:
        raise ValueError(f"{episode_dir}: only yaw state_mode is supported")
    if state_dim != 7:
        raise ValueError(
            f"{episode_dir}: only yaw state_mode is supported; expected states shape (N,7), got {states.shape}"
        )
    return STATE_MODE_YAW


def inspect_episode_state_mode(episode_dir: Path) -> str:
    meta_path = episode_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    states = np.load(episode_dir / "states.npy", mmap_mode="r")
    return infer_episode_state_format(episode_dir, states, metadata)


def ensure_dataset_state_mode(episode_dirs: List[Path]) -> str | None:
    mode_to_episodes: Dict[str, List[str]] = {}

    for ep_dir in episode_dirs:
        try:
            state_mode = inspect_episode_state_mode(ep_dir)
        except Exception:
            continue
        mode_to_episodes.setdefault(state_mode, []).append(ep_dir.name)

    if not mode_to_episodes:
        return None

    return next(iter(mode_to_episodes))


def convert_state_to_lerobot(state: np.ndarray) -> np.ndarray:
    """
    yaw input: [x,y,z, aa_x,aa_y,aa_z, gripper] -> [x,y,z, aa_x,aa_y,aa_z, gripper,gripper]
    """
    s = np.asarray(state, dtype=np.float32)
    if s.shape[0] != 7:
        raise ValueError(f"only yaw state_mode is supported; expected 7-column state, got shape {s.shape}")
    pos = s[:3]
    aa = s[3:6]
    grip = float(s[6])
    tail = np.array([grip, grip], dtype=np.float32)
    out = np.concatenate([pos.astype(np.float32), aa.astype(np.float32), tail], axis=0)
    assert out.shape == (LEROBOT_STATE_DIM,)
    return out


def ensure_uint8_rgb(img: np.ndarray, image_size: int) -> np.ndarray:
    """Ensure (H,W,3) uint8"""
    arr = np.asarray(img)
    assert arr.shape == (image_size, image_size, 3), f"bad image shape: {arr.shape}"
    if arr.dtype != np.uint8:
        # if float [0,1] -> uint8
        if np.issubdtype(arr.dtype, np.floating):
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return arr


def compute_stats_1d(x: np.ndarray) -> Dict[str, object]:
    """
    Return stats dict with keys: min/max/mean/std/count/q01/q10/q50/q90/q99
    Values use python scalars / lists for json.
    """
    x = np.asarray(x)
    x = x.astype(np.float64)

    def q(p: float):
        return np.quantile(x, p, axis=0)

    stats = {
        "min": np.min(x, axis=0),
        "max": np.max(x, axis=0),
        "mean": np.mean(x, axis=0),
        "std": np.std(x, axis=0),
        "count": np.array([x.shape[0]], dtype=np.int64),
        "q01": q(0.01),
        "q10": q(0.10),
        "q50": q(0.50),
        "q90": q(0.90),
        "q99": q(0.99),
    }

    def to_jsonable(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v

    return {k: to_jsonable(v) for k, v in stats.items()}


def resolve_episode_fps(metadata: dict, fallback_fps: float) -> float:
    value = metadata.get("nominal_fps")
    if value is not None:
        fps = float(value)
        if fps > 0:
            return fps

    base_fps = metadata.get("base_fps")
    record_every = metadata.get("record_every")
    if base_fps is not None and record_every is not None:
        record_every = float(record_every)
        if record_every > 0:
            fps = float(base_fps) / record_every
            if fps > 0:
                return fps

    value = metadata.get("fps")
    if value is not None:
        fps = float(value)
        if fps > 0:
            return fps

    return float(fallback_fps)


def build_uniform_timestamps(n_frames: int, fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    return np.arange(n_frames, dtype=np.float32) / float(fps)


def max_timestamp_grid_deviation(raw_timestamps: np.ndarray, fps: float) -> float:
    expected = build_uniform_timestamps(len(raw_timestamps), fps).astype(np.float64)
    actual = np.asarray(raw_timestamps, dtype=np.float64)
    return float(np.max(np.abs(actual - expected)))


def select_dataset_fps(episode_dirs: List[Path], fallback_fps: float) -> float:
    rounded_counts: Dict[float, int] = {}

    for ep_dir in episode_dirs:
        meta_path = ep_dir / "metadata.json"
        metadata = {}
        if meta_path.exists():
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        fps = resolve_episode_fps(metadata, 0.0)
        if fps <= 0:
            continue
        key = round(float(fps), 6)
        rounded_counts[key] = rounded_counts.get(key, 0) + 1

    if rounded_counts:
        dataset_fps = max(rounded_counts.items(), key=lambda item: (item[1], item[0]))[0]
        if len(rounded_counts) > 1:
            stats = ", ".join(f"{fps:g}Hz x{count}" for fps, count in sorted(rounded_counts.items()))
            print(f"[warn] mixed episode fps detected; using dominant dataset fps {dataset_fps:g}Hz ({stats})")
        else:
            print(f"[info] using dataset fps {dataset_fps:g}Hz from episode metadata")
        return float(dataset_fps)

    print(f"[warn] could not infer fps from episode metadata; falling back to args.fps={fallback_fps:g}Hz")
    return float(fallback_fps)


# -------------------------
# IO helpers
# -------------------------
def load_episode(
    episode_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, np.ndarray | None, np.ndarray | None]:
    """
    episode_dir contains:
      states.npy (N,7) yaw mode
      actions.npy (N,7)
      images.npz (optional; contains main_images, wrist_images)
      OR main_images.npy + wrist_images.npy (legacy)
      metadata.json (optional)
    """
    states = np.load(episode_dir / "states.npy")
    actions = np.load(episode_dir / "actions.npy")
    timestamps = None
    env_steps = None

    meta_path = episode_dir / "metadata.json"
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    timestamps_path = episode_dir / "timestamps.npy"
    if timestamps_path.exists():
        timestamps = np.load(timestamps_path)

    env_steps_path = episode_dir / "env_steps.npy"
    if env_steps_path.exists():
        env_steps = np.load(env_steps_path)

    # Prefer compressed images.npz if present
    npz_path = episode_dir / "images.npz"
    if npz_path.exists():
        d = np.load(npz_path)
        # your collector used keys: main_images / wrist_images
        if "main_images" in d and "wrist_images" in d:
            main_images = d["main_images"]
            wrist_images = d["wrist_images"]
        else:
            raise ValueError(f"{episode_dir}: images.npz missing keys main_images/wrist_images. keys={list(d.keys())}")
        return states, actions, main_images, wrist_images, metadata, timestamps, env_steps

    # Legacy fallback
    main_images_path = episode_dir / "main_images.npy"
    wrist_images_path = episode_dir / "wrist_images.npy"
    if main_images_path.exists() and wrist_images_path.exists():
        main_images = np.load(main_images_path)
        wrist_images = np.load(wrist_images_path)
        return states, actions, main_images, wrist_images, metadata, timestamps, env_steps

    raise FileNotFoundError(f"{episode_dir}: no images found (expected images.npz or main_images.npy+wrist_images.npy)")


def validate_episode_data(
    episode_dir: Path,
    states: np.ndarray,
    actions: np.ndarray,
    main_imgs: np.ndarray,
    wrist_imgs: np.ndarray,
    metadata: dict,
    timestamps: np.ndarray | None = None,
    env_steps: np.ndarray | None = None,
) -> Tuple[str, int, List[str]]:
    """Validate one episode and return `(state_mode, image_size, warnings)`."""
    warnings: List[str] = []

    state_mode = infer_episode_state_format(episode_dir, states, metadata)

    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        raise ValueError(f"{episode_dir}: actions shape {actions.shape} != (N,{ACTION_DIM})")

    n_frames = int(states.shape[0])
    if actions.shape[0] != n_frames:
        raise ValueError(f"{episode_dir}: actions length {actions.shape[0]} != states length {n_frames}")

    if main_imgs.ndim != 4 or main_imgs.shape[-1] != 3:
        raise ValueError(f"{episode_dir}: bad main_images shape {main_imgs.shape}")
    if wrist_imgs.ndim != 4 or wrist_imgs.shape[-1] != 3:
        raise ValueError(f"{episode_dir}: bad wrist_images shape {wrist_imgs.shape}")
    if main_imgs.shape[0] != n_frames:
        raise ValueError(f"{episode_dir}: main_images length {main_imgs.shape[0]} != {n_frames}")
    if wrist_imgs.shape[0] != n_frames:
        raise ValueError(f"{episode_dir}: wrist_images length {wrist_imgs.shape[0]} != {n_frames}")
    if main_imgs.shape[1:] != wrist_imgs.shape[1:]:
        raise ValueError(
            f"{episode_dir}: main_images shape {main_imgs.shape} != wrist_images shape {wrist_imgs.shape}"
        )
    if main_imgs.shape[1] != main_imgs.shape[2]:
        raise ValueError(f"{episode_dir}: image size must be square, got {main_imgs.shape[1:3]}")

    image_size = int(main_imgs.shape[1])

    if not np.isfinite(states).all():
        raise ValueError(f"{episode_dir}: states contain NaN/Inf")
    if not np.isfinite(actions).all():
        raise ValueError(f"{episode_dir}: actions contain NaN/Inf")

    gripper_values = states[:, 6]
    if np.any(gripper_values < -0.1) or np.any(gripper_values > 1.1):
        raise ValueError(f"{episode_dir}: gripper state out of expected range [-0.1, 1.1]")

    meta_n_frames = metadata.get("n_frames")
    if meta_n_frames is not None and int(meta_n_frames) != n_frames:
        raise ValueError(f"{episode_dir}: metadata n_frames={meta_n_frames} != {n_frames}")

    meta_image_size = metadata.get("image_size")
    if isinstance(meta_image_size, list) and len(meta_image_size) >= 2:
        if [image_size, image_size] != [int(meta_image_size[0]), int(meta_image_size[1])]:
            warnings.append(
                f"metadata image_size={meta_image_size} differs from actual {[image_size, image_size]}"
            )

    meta_state_dim = metadata.get("state_dim")
    expected_state_dim = 7
    if meta_state_dim is not None and int(meta_state_dim) != expected_state_dim:
        warnings.append(f"metadata state_dim={meta_state_dim} (expected {expected_state_dim})")

    meta_action_dim = metadata.get("action_dim")
    if meta_action_dim is not None and int(meta_action_dim) != ACTION_DIM:
        warnings.append(f"metadata action_dim={meta_action_dim} (expected {ACTION_DIM})")

    if timestamps is not None:
        timestamps = np.asarray(timestamps)
        if timestamps.ndim != 1 or timestamps.shape[0] != n_frames:
            raise ValueError(f"{episode_dir}: timestamps shape {timestamps.shape} incompatible with {n_frames} frames")
        if not np.all(np.diff(timestamps) >= 0.0):
            raise ValueError(f"{episode_dir}: timestamps must be monotonic non-decreasing")

    if env_steps is not None:
        env_steps = np.asarray(env_steps)
        if env_steps.ndim != 1 or env_steps.shape[0] != n_frames:
            raise ValueError(f"{episode_dir}: env_steps shape {env_steps.shape} incompatible with {n_frames} frames")
        if not np.all(np.diff(env_steps) >= 0):
            raise ValueError(f"{episode_dir}: env_steps must be monotonic non-decreasing")

    return state_mode, image_size, warnings


def write_images(
    root: Path,
    episode_index: int,
    main_images: np.ndarray,
    wrist_images: np.ndarray,
    image_size: int,
) -> Tuple[List[str], List[str]]:
    """
    Write frames to:
      root/images/image/frame-XXXXXX.png
      root/images/wrist_image/frame-XXXXXX.png
    Return absolute paths lists (len N).
    """
    out_image = root / "images" / "image"
    out_wrist = root / "images" / "wrist_image"
    out_image.mkdir(parents=True, exist_ok=True)
    out_wrist.mkdir(parents=True, exist_ok=True)

    N = main_images.shape[0]
    image_paths: List[str] = []
    wrist_paths: List[str] = []

    # Use global naming frame-000000.png ... (same as你后面 patch 的逻辑)
    # If you want per-episode prefix, change here, but then parquet paths也要一致。
    for i in range(N):
        img = ensure_uint8_rgb(main_images[i], image_size)
        wimg = ensure_uint8_rgb(wrist_images[i], image_size)

        p1 = out_image / f"frame-{(episode_index*1_000_000 + i):06d}.png"
        p2 = out_wrist / f"frame-{(episode_index*1_000_000 + i):06d}.png"

        imageio.imwrite(p1, img)
        imageio.imwrite(p2, wimg)

        image_paths.append(str(p1))
        wrist_paths.append(str(p2))

    return image_paths, wrist_paths


def write_parquet_no_metadata(path: Path, table: pa.Table) -> None:
    """Write parquet stripping schema metadata (avoid datasets feature parsing issues)."""
    table = table.replace_schema_metadata(None)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(path)


# -------------------------
# main conversion
# -------------------------
@dataclass
class Args:
    data_dir: str
    repo_id: str = "wangrui/mujoco_aubo_data"
    robot_type: str = "aubo_i5"
    fps: float = 200.0
    image_size: int = 512
    chunk_size: int = 1000
    overwrite: bool = False
    preserve_raw_timestamps: bool = False


def main(args: Args) -> None:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir not found: {data_dir}")

    # discover episodes
    episode_dirs = sorted([p for p in data_dir.glob("episode_*") if p.is_dir()])
    if not episode_dirs:
        raise RuntimeError(f"No episode_* folders under: {data_dir}")

    dataset_state_mode = ensure_dataset_state_mode(episode_dirs)
    if dataset_state_mode is not None:
        print(f"[info] detected dataset state mode: {dataset_state_mode}")

    # output root: ~/.cache/huggingface/lerobot/<repo_id>
    out_root = Path(HF_LEROBOT_HOME) / args.repo_id
    meta_dir = out_root / "meta"
    data_out_dir = out_root / "data"

    if out_root.exists():
        if not args.overwrite:
            raise RuntimeError(
                f"Output exists: {out_root}\n"
                f"Use --overwrite to remove and recreate."
            )
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    dataset_fps = select_dataset_fps(episode_dirs, float(args.fps))

    # task registry: task string -> task_index
    task_to_index: Dict[str, int] = {}
    tasks_list: List[Dict[str, object]] = []

    episodes_jsonl: List[Dict[str, object]] = []
    episodes_stats_jsonl: List[Dict[str, object]] = []
    skipped_episodes: List[Dict[str, object]] = []

    global_frame_index = 0
    total_frames = 0
    converted_episodes = 0

    for source_ep_i, ep_dir in enumerate(episode_dirs):
        try:
            states, actions, main_imgs, wrist_imgs, metadata, timestamps, env_steps = load_episode(ep_dir)
            N = int(states.shape[0])
            ep_state_mode, ep_image_size, validation_warnings = validate_episode_data(
                ep_dir, states, actions, main_imgs, wrist_imgs, metadata, timestamps=timestamps, env_steps=env_steps
            )

        except Exception as e:
            print(f"Skipping episode {ep_dir.name} due to error: {e}")
            skipped_episodes.append(
                {
                    "source_episode_index": source_ep_i,
                    "source_episode_name": ep_dir.name,
                    "error": str(e),
                }
            )
            continue

        if validation_warnings:
            print(f"[warn] {ep_dir.name}: " + " | ".join(validation_warnings))

        task = metadata.get("task", "default task")
        if not isinstance(task, str):
            task = str(task)

        if task not in task_to_index:
            tid = len(task_to_index)
            task_to_index[task] = tid
            tasks_list.append({"task_index": tid, "task": task})
        task_index = task_to_index[task]

        # write images to disk, store absolute paths
        img_paths, wrist_paths = write_images(out_root, converted_episodes, main_imgs, wrist_imgs, ep_image_size)

        # build columns
        # state convert
        lerobot_states = np.stack(
            [convert_state_to_lerobot(states[i]) for i in range(N)],
            axis=0,
        ).astype(np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        ep_fps = resolve_episode_fps(metadata, dataset_fps)

        if timestamps is not None and args.preserve_raw_timestamps:
            timestamp = np.asarray(timestamps, dtype=np.float32)
        else:
            if timestamps is not None:
                max_dev = max_timestamp_grid_deviation(timestamps, dataset_fps)
                if max_dev > (0.5 / float(dataset_fps)):
                    print(
                        f"[warn] {ep_dir.name}: raw timestamps deviate from a uniform {dataset_fps:.6g} Hz grid "
                        f"(max_dev={max_dev:.4f}s); using normalized LeRobot timestamps instead"
                    )
            if abs(ep_fps - dataset_fps) > 1e-6:
                print(
                    f"[warn] {ep_dir.name}: episode fps={ep_fps:.6g}Hz differs from dataset fps={dataset_fps:.6g}Hz; "
                    "timestamps will be normalized to the dataset fps"
                )
            timestamp = build_uniform_timestamps(N, dataset_fps)

        frame_index = np.arange(N, dtype=np.int64)
        episode_index = np.full((N,), converted_episodes, dtype=np.int64)
        task_index_col = np.full((N,), task_index, dtype=np.int64)
        index_col = np.arange(global_frame_index, global_frame_index + N, dtype=np.int64)

        # arrow schema: use string for images; fixed_size_list for vectors
        # fixed_size_list in pyarrow: pa.list_(value_type, list_size) isn't fixed; use pa.fixed_size_list
        state_type = pa.list_(pa.float32(), 8)
        action_type = pa.list_(pa.float32(), 7)

        table = pa.table(
            {
                "image": pa.array(img_paths, type=pa.string()),
                "wrist_image": pa.array(wrist_paths, type=pa.string()),
                "state": pa.array(lerobot_states.tolist(), type=state_type),
                "actions": pa.array(actions.tolist(), type=action_type),
                "timestamp": pa.array(timestamp.tolist(), type=pa.float32()),
                "frame_index": pa.array(frame_index.tolist(), type=pa.int64()),
                "episode_index": pa.array(episode_index.tolist(), type=pa.int64()),
                "index": pa.array(index_col.tolist(), type=pa.int64()),
                "task_index": pa.array(task_index_col.tolist(), type=pa.int64()),
            }
        )

        # decide parquet path by chunking episodes (not frames).
        # Keep file_index as the global episode index for compatibility with
        # older LeRobot loaders that derive data paths from episode_index.
        chunk_index = converted_episodes // args.chunk_size
        file_index = converted_episodes
        parquet_path = data_out_dir / f"chunk-{chunk_index:03d}" / f"file-{file_index:03d}.parquet"
        write_parquet_no_metadata(parquet_path, table)

        # episodes.jsonl record
        ep_from = global_frame_index
        ep_to = global_frame_index + N
        episodes_jsonl.append(
            {
                "episode_index": converted_episodes,
                "tasks": [task],
                "length": N,
                "chunk_index": int(chunk_index),
                "file_index": int(file_index),
                "data/chunk_index": int(chunk_index),
                "data/file_index": int(file_index),
                "dataset_from_index": int(ep_from),
                "dataset_to_index": int(ep_to),
            }
        )

        # episodes_stats.jsonl (must have {"episode_index":..., "stats": {...}})
        # Only compute numeric stats (no image stats)
        stats_dict: Dict[str, object] = {}

        # state stats -> keys like "state/min", etc.
        st = compute_stats_1d(lerobot_states)  # returns min/max/...
        for k, v in st.items():
            stats_dict[f"state/{k}"] = v

        ac = compute_stats_1d(actions)
        for k, v in ac.items():
            stats_dict[f"actions/{k}"] = v

        ts = compute_stats_1d(timestamp.reshape(-1, 1))
        for k, v in ts.items():
            stats_dict[f"timestamp/{k}"] = v

        fi = compute_stats_1d(frame_index.reshape(-1, 1))
        for k, v in fi.items():
            stats_dict[f"frame_index/{k}"] = v

        ii = compute_stats_1d(index_col.reshape(-1, 1))
        for k, v in ii.items():
            stats_dict[f"index/{k}"] = v

        ti = compute_stats_1d(task_index_col.reshape(-1, 1))
        for k, v in ti.items():
            stats_dict[f"task_index/{k}"] = v

        ei = compute_stats_1d(episode_index.reshape(-1, 1))
        for k, v in ei.items():
            stats_dict[f"episode_index/{k}"] = v

        episodes_stats_jsonl.append({"episode_index": converted_episodes, "stats": stats_dict})

        global_frame_index += N
        total_frames += N
        converted_episodes += 1

        print(
            f"[{source_ep_i+1}/{len(episode_dirs)}] wrote episode {ep_dir.name} "
            f"-> {parquet_path} | frames={N} | task_index={task_index}"
        )

    if converted_episodes == 0:
        shutil.rmtree(out_root, ignore_errors=True)
        raise RuntimeError("No valid episodes were converted.")

    # write tasks.jsonl
    tasks_jsonl_path = meta_dir / "tasks.jsonl"
    with tasks_jsonl_path.open("w", encoding="utf-8") as f:
        for item in tasks_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # write tasks.parquet (for compatibility)
    tasks_parquet_path = meta_dir / "tasks.parquet"
    tasks_table = pa.table(
        {
            "task_index": pa.array([int(x["task_index"]) for x in tasks_list], type=pa.int64()),
            "task": pa.array([str(x["task"]) for x in tasks_list], type=pa.string()),
        }
    )
    write_parquet_no_metadata(tasks_parquet_path, tasks_table)

    # write episodes.jsonl
    episodes_jsonl_path = meta_dir / "episodes.jsonl"
    with episodes_jsonl_path.open("w", encoding="utf-8") as f:
        for item in episodes_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # write episodes_stats.jsonl
    episodes_stats_path = meta_dir / "episodes_stats.jsonl"
    with episodes_stats_path.open("w", encoding="utf-8") as f:
        for item in episodes_stats_jsonl:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    skipped_path = meta_dir / "skipped_episodes.jsonl"
    with skipped_path.open("w", encoding="utf-8") as f:
        for item in skipped_episodes:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # write global stats.json (optional but nice)
    # aggregate from episodes_stats by just taking overall frame-level stats is more work;
    # keep a minimal file to satisfy "meta/stats.json" presence like你之前的目录。
    (meta_dir / "stats.json").write_text(json.dumps({"note": "global stats not precomputed; use compute_norm_stats.py"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # write info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": args.robot_type,
        "total_episodes": converted_episodes,
        "source_total_episodes": len(episode_dirs),
        "skipped_episodes": len(skipped_episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks_list),
        "chunks_size": args.chunk_size,
        "data_files_size_in_mb": None,
        "video_files_size_in_mb": None,
        "fps": dataset_fps,
        "splits": {"train": f"0:{converted_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/file-{episode_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/file-{episode_index:03d}.mp4",
        # IMPORTANT: images stored as string paths (not HF Image struct)
        "features": {
            "image": {"dtype": "string", "shape": [1], "names": None},
            "wrist_image": {"dtype": "string", "shape": [1], "names": None},
            "state": {"dtype": "float32", "shape": [8], "names": ["state"]},
            "actions": {"dtype": "float32", "shape": [7], "names": ["actions"]},
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    (meta_dir / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")

    print("\nDONE")
    print("out_root:", out_root)
    print("meta files:", meta_dir)
    print("sample:")
    print("  ", tasks_jsonl_path)
    print("  ", episodes_jsonl_path)
    print("  ", episodes_stats_path)
    print("  ", meta_dir / "info.json")


if __name__ == "__main__":
    main(tyro.cli(Args))
