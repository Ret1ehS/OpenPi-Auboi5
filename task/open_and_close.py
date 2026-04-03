from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from task.pick_and_place import TaskStep


OPEN_PROMPT = "open the storage box"
CLOSE_PROMPT = "close the storage box"

# Obstacle constants
NUM_OBSTACLES = 5
OBSTACLE_NAMES = tuple(f"obj{i}" for i in range(1, NUM_OBSTACLES + 1))
CLEAR_BAND_M = 0.12
CLEAR_MIN_SPACING_M = 0.10
BAND_OBJECT_COUNT_MIN = 2
BAND_OBJECT_COUNT_MAX = 4
STACK_PROBABILITY = 0.70
OBSTACLE_LAYOUT_X_MAX_M = 0.60
OBSTACLE_ROTATE_PROB = 0.70
OBSTACLE_ROTATE_DEG_MIN = float(np.degrees(np.pi / 15.0))
OBSTACLE_ROTATE_DEG_MAX = float(np.degrees(np.pi / 8.0))


def _bounded_layout_x_max(workspace_x_min: float, workspace_x_max: float) -> float:
    bounded_x_max = min(float(workspace_x_max), OBSTACLE_LAYOUT_X_MAX_M)
    if bounded_x_max < float(workspace_x_min):
        raise RuntimeError(
            f"invalid layout x bounds: x_min={workspace_x_min}, x_max={bounded_x_max}"
        )
    return bounded_x_max


@dataclass(frozen=True)
class OpenCloseReference:
    x_start: float
    y_start: float
    reference_pose6: np.ndarray


@dataclass(frozen=True)
class MoveStep:
    x: float
    y: float
    z: float
    note: str


@dataclass(frozen=True)
class OpenCloseEpisodePlan:
    task_kind: str  # "open" | "close"
    prompt: str
    recorded_steps: list[MoveStep]


@dataclass
class ObstacleState:
    name: str
    xy: tuple[float, float]
    is_rotate: bool
    deg: float
    upper: str | None = None
    lower: str | None = None


def build_reference_from_tcp_pose(tcp_pose6: np.ndarray) -> OpenCloseReference:
    pose = np.asarray(tcp_pose6, dtype=np.float64).reshape(6).copy()
    return OpenCloseReference(
        x_start=float(pose[0]),
        y_start=float(pose[1]),
        reference_pose6=pose,
    )


def build_open_episode_plan(
    reference: OpenCloseReference,
    *,
    target_z: float,
    press_z: float,
) -> OpenCloseEpisodePlan:
    x_start = float(reference.x_start)
    y_start = float(reference.y_start)
    return OpenCloseEpisodePlan(
        task_kind="open",
        prompt=OPEN_PROMPT,
        recorded_steps=[
            MoveStep(x=x_start, y=y_start, z=float(target_z), note="open step 1 approach"),
            MoveStep(x=x_start, y=y_start, z=float(press_z), note="open step 2 lower"),
            MoveStep(x=x_start - 0.20, y=y_start, z=float(press_z), note="open step 3 pull"),
            MoveStep(x=x_start - 0.20, y=y_start, z=float(target_z), note="open step 4 lift"),
        ],
    )


def build_close_episode_plan(
    reference: OpenCloseReference,
    *,
    target_z: float,
    press_z: float,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
) -> OpenCloseEpisodePlan:
    x_rand = float(np.random.uniform(workspace_x_min, workspace_x_max))
    y_rand = float(np.random.uniform(workspace_y_min, workspace_y_max))
    x_start = float(reference.x_start)
    y_start = float(reference.y_start)
    return OpenCloseEpisodePlan(
        task_kind="close",
        prompt=CLOSE_PROMPT,
        recorded_steps=[
            MoveStep(x=x_rand, y=y_rand, z=float(target_z), note="close step 1 random approach"),
            MoveStep(x=x_start - 0.26, y=y_start, z=float(target_z), note="close step 2 align high"),
            MoveStep(x=x_start - 0.26, y=y_start, z=float(press_z - 0.02), note="close step 3 lower"),
            MoveStep(x=x_start + 0.01, y=y_start, z=float(press_z - 0.02), note="close step 4 push"),
        ],
    )


class ObstacleScene:
    def __init__(self, obstacles: dict[str, ObstacleState]) -> None:
        self.obstacles = obstacles

    @classmethod
    def empty(cls) -> "ObstacleScene":
        return cls({})

    @classmethod
    def from_serialized(cls, payload: dict[str, Any]) -> "ObstacleScene | None":
        if not isinstance(payload, dict):
            return None
        obstacles: dict[str, ObstacleState] = {}
        for name in OBSTACLE_NAMES:
            raw = payload.get(name)
            if raw is None or not isinstance(raw, dict):
                return None
            xy = raw.get("xy")
            if xy is None:
                return None
            arr = np.asarray(xy, dtype=np.float64).reshape(2)
            obstacles[name] = ObstacleState(
                name=name,
                xy=(float(arr[0]), float(arr[1])),
                is_rotate=bool(raw.get("is_rotate", False)),
                deg=float(raw.get("deg", 0.0)),
                upper=_norm_link(raw.get("upper")),
                lower=_norm_link(raw.get("lower")),
            )
        return cls(obstacles)

    def copy(self) -> "ObstacleScene":
        return ObstacleScene(
            {
                n: ObstacleState(
                    name=o.name,
                    xy=tuple(o.xy),
                    is_rotate=o.is_rotate,
                    deg=o.deg,
                    upper=o.upper,
                    lower=o.lower,
                )
                for n, o in self.obstacles.items()
            }
        )

    def to_serializable(self) -> dict[str, dict[str, Any]]:
        return {
            n: {
                "xy": [o.xy[0], o.xy[1]],
                "is_rotate": o.is_rotate,
                "deg": o.deg,
                "upper": o.upper,
                "lower": o.lower,
            }
            for n, o in self.obstacles.items()
        }

    def occupied_positions(self, *, exclude: set[str] | None = None) -> list[np.ndarray]:
        exc = exclude or set()
        return [np.asarray(o.xy) for n, o in self.obstacles.items() if n not in exc]

    def top_of(self, name: str) -> str:
        current = name
        seen: set[str] = set()
        while self.obstacles[current].upper is not None:
            if current in seen:
                raise RuntimeError(f"cycle in upper chain from {name}")
            seen.add(current)
            current = self.obstacles[current].upper  # type: ignore[assignment]
        return current

    def detach_top(self, name: str) -> None:
        obj = self.obstacles[name]
        if obj.upper is not None:
            raise RuntimeError(f"cannot detach {name}: upper={obj.upper}")
        if obj.lower is not None:
            self.obstacles[obj.lower].upper = None
        obj.lower = None

    def place_on_table(self, name: str, xy: tuple[float, float], *, is_rotate: bool, deg: float) -> None:
        obj = self.obstacles[name]
        obj.xy = (float(xy[0]), float(xy[1]))
        obj.is_rotate = bool(is_rotate)
        obj.deg = float(deg) if is_rotate else 0.0
        obj.lower = None

    def place_on_object(self, name: str, target: str) -> None:
        actual = self.top_of(target)
        obj = self.obstacles[name]
        obj.xy = tuple(self.obstacles[actual].xy)
        obj.is_rotate = self.obstacles[actual].is_rotate
        obj.deg = self.obstacles[actual].deg
        obj.lower = actual
        self.obstacles[actual].upper = name


def sample_obstacle_orientation(
    *,
    rotate_prob: float = OBSTACLE_ROTATE_PROB,
    rotate_deg_min: float = OBSTACLE_ROTATE_DEG_MIN,
    rotate_deg_max: float = OBSTACLE_ROTATE_DEG_MAX,
) -> tuple[bool, float]:
    if np.random.random() < rotate_prob:
        abs_deg = float(np.random.uniform(rotate_deg_min, rotate_deg_max))
        sign = -1.0 if np.random.random() < 0.5 else 1.0
        return True, sign * abs_deg
    return False, 0.0


def sample_obstacle_xy(
    scene: ObstacleScene,
    *,
    object_name: str,
    workspace_x_min: float,
    workspace_x_max: float,
    y_min: float,
    y_max: float,
    min_spacing: float = CLEAR_MIN_SPACING_M,
) -> tuple[float, float]:
    if float(y_max) < float(y_min):
        raise RuntimeError(f"invalid layout y bounds: y_min={y_min}, y_max={y_max}")

    layout_x_max = _bounded_layout_x_max(workspace_x_min, workspace_x_max)
    occupied = scene.occupied_positions(exclude={object_name})
    for _ in range(512):
        x = float(np.random.uniform(workspace_x_min, layout_x_max))
        y = float(np.random.uniform(y_min, y_max))
        candidate = np.array([x, y], dtype=np.float64)
        if all(float(np.linalg.norm(candidate - p)) >= min_spacing for p in occupied):
            return float(candidate[0]), float(candidate[1])
    raise RuntimeError(f"failed to sample obstacle xy for {object_name}")


def build_clearing_steps(
    scene: ObstacleScene,
    *,
    y_target: float,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
    min_spacing: float = CLEAR_MIN_SPACING_M,
) -> tuple[list[TaskStep], ObstacleScene]:
    """Build pick/place steps to clear all obstacles inside the y-target band."""
    result = scene.copy()
    band_lo = y_target - CLEAR_BAND_M
    band_hi = y_target + CLEAR_BAND_M
    layout_x_max = _bounded_layout_x_max(workspace_x_min, workspace_x_max)

    can_go_upper = (workspace_y_max - band_hi) >= min_spacing
    can_go_lower = (band_lo - workspace_y_min) >= min_spacing

    def _objects_in_band() -> list[str]:
        in_band: list[str] = []
        for name, obj in result.obstacles.items():
            if band_lo <= obj.xy[1] <= band_hi:
                in_band.append(name)
        return in_band

    def _pick_clear_y(obj_y: float) -> tuple[float, float]:
        wants_upper = obj_y >= y_target
        if wants_upper and can_go_upper:
            return band_hi + min_spacing, workspace_y_max
        if (not wants_upper) and can_go_lower:
            return workspace_y_min, band_lo - min_spacing
        if can_go_upper:
            return band_hi + min_spacing, workspace_y_max
        if can_go_lower:
            return workspace_y_min, band_lo - min_spacing
        return workspace_y_min, workspace_y_max

    def _sample_clear_xy(obj_name: str, obj_y: float) -> np.ndarray:
        y_lo, y_hi = _pick_clear_y(obj_y)
        occupied = result.occupied_positions(exclude={obj_name})
        candidate_ranges = [
            (float(y_lo), float(y_hi)),
            (float(workspace_y_min), float(workspace_y_max)),
        ]
        for y_range_lo, y_range_hi in candidate_ranges:
            for _ in range(512):
                x = float(np.random.uniform(workspace_x_min, layout_x_max))
                y = float(np.random.uniform(y_range_lo, y_range_hi))
                candidate = np.array([x, y], dtype=np.float64)
                if all(float(np.linalg.norm(candidate - p)) >= min_spacing for p in occupied):
                    return candidate
        raise RuntimeError(f"failed to sample clearing xy for {obj_name}")

    steps: list[TaskStep] = []
    processed: set[str] = set()

    def _clear_object(name: str) -> None:
        if name in processed:
            return
        obj = result.obstacles[name]
        if obj.upper is not None:
            _clear_object(obj.upper)

        if name not in _objects_in_band():
            return

        obj = result.obstacles[name]
        level = 1 if obj.lower is not None else 0
        steps.append(
            TaskStep(
                kind="pick",
                object_name=name,
                xy=tuple(obj.xy),
                level=level,
                is_rotate=obj.is_rotate,
                deg=obj.deg,
                note=f"clear pick {name}",
                align_j6=True,
            )
        )
        result.detach_top(name)

        clear_xy = _sample_clear_xy(name, obj.xy[1])
        steps.append(
            TaskStep(
                kind="place",
                object_name=name,
                xy=(float(clear_xy[0]), float(clear_xy[1])),
                level=0,
                is_rotate=False,
                deg=0.0,
                note=f"clear place {name}",
                align_j6=True,
            )
        )
        result.place_on_table(name, (float(clear_xy[0]), float(clear_xy[1])), is_rotate=False, deg=0.0)
        processed.add(name)

    for name in _objects_in_band():
        _clear_object(name)

    return steps, result


def _norm_link(v: Any) -> str | None:
    if v is None:
        return None
    text = str(v).strip()
    return text if text else None
