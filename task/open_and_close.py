from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OPEN_PROMPT = "open the storage box"
CLOSE_PROMPT = "close the storage box"


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
            MoveStep(x=x_start - 0.24, y=y_start, z=float(target_z), note="close step 2 align high"),
            MoveStep(x=x_start - 0.24, y=y_start, z=float(press_z), note="close step 3 lower"),
            MoveStep(x=x_start + 0.02, y=y_start, z=float(press_z), note="close step 4 push"),
        ],
    )
