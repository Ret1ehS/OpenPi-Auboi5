from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


OPEN_PROMPT = "open the storage box"
CLOSE_PROMPT = "close the storage box"

# Obstacle constants
NUM_OBSTACLES = 5
OBSTACLE_NAMES = tuple(f"obj{i}" for i in range(1, NUM_OBSTACLES + 1))
CLEAR_BAND_M = 0.12  # ±12 cm around target y
CLEAR_MIN_SPACING_M = 0.10  # obstacles must be ≥10 cm apart after clearing
BAND_OBJECT_COUNT_MIN = 2
BAND_OBJECT_COUNT_MAX = 4
STACK_PROBABILITY = 0.70


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


# ---- TaskStep for obstacle pick/place (reuses pick_and_place's step structure) ----

@dataclass
class ObstacleTaskStep:
    kind: str  # "pick" | "place"
    object_name: str
    xy: tuple[float, float]
    level: int
    is_rotate: bool
    deg: float
    note: str
    align_j6: bool = True


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------

def build_reference_from_tcp_pose(tcp_pose6: np.ndarray) -> OpenCloseReference:
    pose = np.asarray(tcp_pose6, dtype=np.float64).reshape(6).copy()
    return OpenCloseReference(
        x_start=float(pose[0]),
        y_start=float(pose[1]),
        reference_pose6=pose,
    )


# ---------------------------------------------------------------------------
# Episode plans (open / close)
# ---------------------------------------------------------------------------

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
            MoveStep(x=x_start + 0.02, y=y_start, z=float(press_z - 0.02), note="close step 4 push"),
        ],
    )


# ---------------------------------------------------------------------------
# Obstacle scene state
# ---------------------------------------------------------------------------

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
        return ObstacleScene({
            n: ObstacleState(
                name=o.name, xy=tuple(o.xy), is_rotate=o.is_rotate, deg=o.deg,
                upper=o.upper, lower=o.lower,
            )
            for n, o in self.obstacles.items()
        })

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
            current = self.obstacles[current].upper  # type: ignore
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


# ---------------------------------------------------------------------------
# Layout generation: random placement for each cycle
# ---------------------------------------------------------------------------

def generate_random_layout(
    scene: ObstacleScene,
    *,
    y_target: float,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
    min_spacing: float = CLEAR_MIN_SPACING_M,
    rotate_prob: float = 0.5,
    rotate_deg_min: float = 12.0,
    rotate_deg_max: float = 22.5,
) -> ObstacleScene:
    """Generate a random layout for all 5 obstacles.

    - Pick 2~4 to fall inside y_target ± CLEAR_BAND_M
    - 70% chance to stack 2 of those band objects
    - Rest go outside the band
    """
    result = scene.copy()
    names = list(OBSTACLE_NAMES)
    np.random.shuffle(names)

    # Clear all links
    for n in names:
        result.obstacles[n].upper = None
        result.obstacles[n].lower = None

    # Decide band membership
    n_in_band = int(np.random.randint(BAND_OBJECT_COUNT_MIN, BAND_OBJECT_COUNT_MAX + 1))
    band_names = names[:n_in_band]
    outside_names = names[n_in_band:]

    band_y_lo = y_target - CLEAR_BAND_M
    band_y_hi = y_target + CLEAR_BAND_M
    # Clamp band to workspace
    band_y_lo = max(band_y_lo, workspace_y_min)
    band_y_hi = min(band_y_hi, workspace_y_max)

    occupied: list[np.ndarray] = []

    def _sample_xy(y_lo: float, y_hi: float) -> np.ndarray:
        for _ in range(512):
            x = float(np.random.uniform(workspace_x_min, workspace_x_max))
            y = float(np.random.uniform(y_lo, y_hi))
            c = np.array([x, y], dtype=np.float64)
            if all(float(np.linalg.norm(c - p)) >= min_spacing for p in occupied):
                return c
        raise RuntimeError("failed to sample obstacle xy")

    def _sample_orient() -> tuple[bool, float]:
        if np.random.random() < rotate_prob:
            abs_deg = float(np.random.uniform(rotate_deg_min, rotate_deg_max))
            sign = -1.0 if np.random.random() < 0.5 else 1.0
            return True, sign * abs_deg
        return False, 0.0

    # Place band objects on table first
    for name in band_names:
        xy = _sample_xy(band_y_lo, band_y_hi)
        occupied.append(xy)
        is_rot, deg = _sample_orient()
        result.place_on_table(name, (float(xy[0]), float(xy[1])), is_rotate=is_rot, deg=deg)

    # 70% chance: stack 2 of the band objects
    if len(band_names) >= 2 and np.random.random() < STACK_PROBABILITY:
        stack_pair = list(np.random.choice(band_names, size=2, replace=False))
        bottom, top = stack_pair[0], stack_pair[1]
        result.place_on_object(top, bottom)

    # Place outside objects
    for name in outside_names:
        # Sample outside the band
        outside_y_ranges = []
        if workspace_y_min < band_y_lo - min_spacing:
            outside_y_ranges.append((workspace_y_min, band_y_lo - min_spacing))
        if band_y_hi + min_spacing < workspace_y_max:
            outside_y_ranges.append((band_y_hi + min_spacing, workspace_y_max))
        if not outside_y_ranges:
            # Fallback: full workspace
            outside_y_ranges = [(workspace_y_min, workspace_y_max)]
        chosen_range = outside_y_ranges[int(np.random.randint(len(outside_y_ranges)))]
        xy = _sample_xy(chosen_range[0], chosen_range[1])
        occupied.append(xy)
        is_rot, deg = _sample_orient()
        result.place_on_table(name, (float(xy[0]), float(xy[1])), is_rotate=is_rot, deg=deg)

    return result


# ---------------------------------------------------------------------------
# Clearing plan: move band objects outside y_target ± 12cm
# ---------------------------------------------------------------------------

def build_clearing_steps(
    scene: ObstacleScene,
    *,
    y_target: float,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
    min_spacing: float = CLEAR_MIN_SPACING_M,
) -> tuple[list[ObstacleTaskStep], ObstacleScene]:
    """Build pick/place steps to clear all obstacles inside y_target ± CLEAR_BAND_M.

    Returns (steps, updated_scene). Steps are recorded into the open episode.
    If stacked, clears upper first then lower.
    Clearing places use is_rotate=False, deg=0 (default angle).
    """
    result = scene.copy()
    band_lo = y_target - CLEAR_BAND_M
    band_hi = y_target + CLEAR_BAND_M

    # Determine which side to clear to
    # Check if each side has room (at least min_spacing from workspace edge)
    can_go_upper = (workspace_y_max - band_hi) >= min_spacing
    can_go_lower = (band_lo - workspace_y_min) >= min_spacing

    def _objects_in_band() -> list[str]:
        """Names of objects whose y is within [band_lo, band_hi] (inclusive)."""
        in_band = []
        for name, obj in result.obstacles.items():
            if band_lo <= obj.xy[1] <= band_hi:
                in_band.append(name)
        return in_band

    def _pick_clear_y(obj_y: float) -> tuple[float, float]:
        """Choose the target y range for clearing based on which side the object is on."""
        wants_upper = obj_y >= y_target
        if wants_upper and can_go_upper:
            return band_hi + min_spacing, workspace_y_max
        elif not wants_upper and can_go_lower:
            return workspace_y_min, band_lo - min_spacing
        elif can_go_upper:
            return band_hi + min_spacing, workspace_y_max
        elif can_go_lower:
            return workspace_y_min, band_lo - min_spacing
        else:
            # Fallback: just use full workspace
            return workspace_y_min, workspace_y_max

    def _sample_clear_xy(obj_name: str, obj_y: float) -> np.ndarray:
        y_lo, y_hi = _pick_clear_y(obj_y)
        occupied = result.occupied_positions(exclude={obj_name})
        for _ in range(512):
            x = float(np.random.uniform(workspace_x_min, workspace_x_max))
            y = float(np.random.uniform(y_lo, y_hi))
            c = np.array([x, y], dtype=np.float64)
            if all(float(np.linalg.norm(c - p)) >= min_spacing for p in occupied):
                return c
        raise RuntimeError(f"failed to sample clearing xy for {obj_name}")

    steps: list[ObstacleTaskStep] = []

    # Collect objects in band, process stacked ones first (upper before lower)
    processed: set[str] = set()

    def _clear_object(name: str) -> None:
        if name in processed:
            return
        obj = result.obstacles[name]
        # If it has something on top, clear that first
        if obj.upper is not None:
            _clear_object(obj.upper)

        if name not in _objects_in_band():
            return

        obj = result.obstacles[name]
        level = 0
        lower = obj.lower
        if lower is not None:
            level = 1  # it's stacked

        # Pick step
        steps.append(ObstacleTaskStep(
            kind="pick",
            object_name=name,
            xy=tuple(obj.xy),
            level=level,
            is_rotate=obj.is_rotate,
            deg=obj.deg,
            note=f"clear pick {name}",
        ))
        result.detach_top(name)

        # Place step: outside band, no rotation
        clear_xy = _sample_clear_xy(name, obj.xy[1])
        steps.append(ObstacleTaskStep(
            kind="place",
            object_name=name,
            xy=(float(clear_xy[0]), float(clear_xy[1])),
            level=0,
            is_rotate=False,
            deg=0.0,
            note=f"clear place {name}",
        ))
        result.place_on_table(name, (float(clear_xy[0]), float(clear_xy[1])), is_rotate=False, deg=0.0)
        processed.add(name)

    for name in _objects_in_band():
        _clear_object(name)

    return steps, result


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _norm_link(v: Any) -> str | None:
    if v is None:
        return None
    t = str(v).strip()
    return t if t else None
