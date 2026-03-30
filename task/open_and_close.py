from __future__ import annotations

from dataclasses import dataclass

import numpy as np


OPEN_PROMPT = "open the storage box"
CLOSE_PROMPT = "close the storage box"


@dataclass(frozen=True)
class OpenCloseReference:
    x_start: float
    y_start: float
    z_base: float
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
        z_base=float(pose[2]),
        reference_pose6=pose,
    )


def build_open_episode_plan(reference: OpenCloseReference) -> OpenCloseEpisodePlan:
    x_start = float(reference.x_start)
    y_start = float(reference.y_start)
    z_base = float(reference.z_base)
    return OpenCloseEpisodePlan(
        task_kind="open",
        prompt=OPEN_PROMPT,
        recorded_steps=[
            MoveStep(x=x_start, y=y_start, z=z_base + 0.15, note="open step 1 approach"),
            MoveStep(x=x_start, y=y_start, z=z_base + 0.05, note="open step 2 lower"),
            MoveStep(x=x_start - 0.20, y=y_start, z=z_base + 0.05, note="open step 3 pull"),
            MoveStep(x=x_start - 0.20, y=y_start, z=z_base + 0.15, note="open step 4 lift"),
        ],
    )


def build_close_episode_plan(
    reference: OpenCloseReference,
    *,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
) -> OpenCloseEpisodePlan:
    x_rand = float(np.random.uniform(workspace_x_min, workspace_x_max))
    y_rand = float(np.random.uniform(workspace_y_min, workspace_y_max))
    x_start = float(reference.x_start)
    y_start = float(reference.y_start)
    z_base = float(reference.z_base)
    return OpenCloseEpisodePlan(
        task_kind="close",
        prompt=CLOSE_PROMPT,
        recorded_steps=[
            MoveStep(x=x_rand, y=y_rand, z=z_base + 0.15, note="close step 1 random approach"),
            MoveStep(x=x_start - 0.24, y=y_start, z=z_base + 0.05, note="close step 2 touch"),
            MoveStep(x=x_start + 0.02, y=y_start, z=z_base + 0.05, note="close step 3 push"),
            MoveStep(x=x_start + 0.02, y=y_start, z=z_base + 0.15, note="close step 4 lift"),
        ],
    )


def build_open_and_close_episode_plan(
    *,
    reference: OpenCloseReference,
    episode_index: int,
    workspace_x_min: float,
    workspace_x_max: float,
    workspace_y_min: float,
    workspace_y_max: float,
) -> OpenCloseEpisodePlan:
    if int(episode_index) % 2 == 0:
        return build_open_episode_plan(reference)
    return build_close_episode_plan(
        reference,
        workspace_x_min=workspace_x_min,
        workspace_x_max=workspace_x_max,
        workspace_y_min=workspace_y_min,
        workspace_y_max=workspace_y_max,
    )
