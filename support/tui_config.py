#!/usr/bin/env python3
"""
Interactive TUI configuration menu for main.py.

Renders a keyboard-navigable settings panel in the terminal:
  - Up/Down arrows to move between rows
  - Left/Right arrows or Enter to toggle option values
  - Enter on action buttons executes the action
  - Enter on the "Start" row to finalize and proceed
  - q to quit

Works on Linux terminals with termios (Orin / SSH).
"""

from __future__ import annotations

import sys
import tty
import termios
import os
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

_ESC = "\033["
CLEAR_SCREEN = f"{_ESC}2J{_ESC}H"
HIDE_CURSOR = f"{_ESC}?25l"
SHOW_CURSOR = f"{_ESC}?25h"
BOLD = f"{_ESC}1m"
DIM = f"{_ESC}2m"
RESET = f"{_ESC}0m"
UNDERLINE = f"{_ESC}4m"
# Colors
FG_CYAN = f"{_ESC}36m"
FG_GREEN = f"{_ESC}32m"
FG_YELLOW = f"{_ESC}33m"
FG_WHITE = f"{_ESC}37m"
FG_RED = f"{_ESC}31m"
FG_BLACK = f"{_ESC}30m"
BG_BLUE = f"{_ESC}44m"
BG_DEFAULT = f"{_ESC}49m"


def _move_to(row: int, col: int) -> str:
    return f"{_ESC}{row};{col}H"


# ---------------------------------------------------------------------------
# Menu item types
# ---------------------------------------------------------------------------

@dataclass
class ToggleItem:
    """A row with two or more choices, toggled by Left/Right/Enter."""
    label: str
    choices: list[str]
    selected: int = 0

    @property
    def value(self) -> str:
        return self.choices[self.selected]


@dataclass
class TextItem:
    """A row with an editable text value (edited inline on Enter)."""
    label: str
    value: str = ""
    editing: bool = False
    _edit_buf: str = ""  # buffer used during inline editing


@dataclass
class ActionItem:
    """A button row that triggers a callback on Enter."""
    label: str
    key: str  # unique key for dispatch


@dataclass
class SeparatorItem:
    """A visual separator line."""
    pass


@dataclass
class StartItem:
    """The 'Start' row: Enter twice to proceed."""
    label: str = ">>> Start Inference <<<"


MenuRow = ToggleItem | TextItem | ActionItem | SeparatorItem | StartItem


# ---------------------------------------------------------------------------
# TUI Config result
# ---------------------------------------------------------------------------

@dataclass
class TUIConfig:
    policy_location: str  # "remote" | "local"
    pose_frame: str       # "real" | "sim"
    obs_state_mode: str   # "yaw"
    dry_run: bool
    exec_speed_mps: float  # max TCP linear speed (m/s) for servo execution
    speed_mode: str = "limited"  # "limited" | "native"
    record: bool = False   # True to record video of inference session
    quit: bool = False     # True if user pressed q


@dataclass
class CollectTUIConfig:
    mode: str              # "manual" | "auto"
    auto_episodes: int
    resume_mode: str       # "continue" | "reset"
    task: str              # "pick_and_place" | "open_and_close" | "keyboard_teleop" | "storage"
    save_fps: int          # 30 | 50
    state_mode: str        # "yaw"
    quit: bool = False


# ---------------------------------------------------------------------------
# Key reading
# ---------------------------------------------------------------------------

def _read_key(fd: int) -> str:
    """Read a single keypress (handles arrow key escape sequences)."""
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        seq = sys.stdin.read(1)
        if seq == "[":
            code = sys.stdin.read(1)
            return {
                "A": "UP",
                "B": "DOWN",
                "C": "RIGHT",
                "D": "LEFT",
            }.get(code, "")
        return "ESC"
    if ch == "\r" or ch == "\n":
        return "ENTER"
    if ch == "\x7f" or ch == "\x08":
        return "BACKSPACE"
    if ch == "\x03":
        return "CTRL_C"
    return ch


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def _render(
    rows: list[MenuRow],
    cursor: int,
    status_line: str = "",
    *,
    title: str = "=== OpenPI Robot Controller ===",
    help_line: str = "Up/Down: navigate  Left/Right/Enter: toggle  Enter on Start: go  q: quit",
    disabled_labels: set[str] | None = None,
    nonselectable_labels: set[str] | None = None,
) -> str:
    """Build the full screen content string."""
    disabled = disabled_labels or set()
    nonselectable = nonselectable_labels or set()
    lines: list[str] = []
    lines.append(f"{BOLD}{FG_CYAN}{title}{RESET}")
    lines.append("")

    selectable_idx = 0
    for row in rows:
        if isinstance(row, SeparatorItem):
            lines.append(f"  {DIM}{'-' * 50}{RESET}")
            continue

        row_label = row.label if hasattr(row, "label") else ""
        is_disabled_row = bool(row_label and row_label in nonselectable)
        is_focused = (not is_disabled_row) and selectable_idx == cursor
        prefix = f"  {FG_GREEN}>{RESET} " if is_focused else "    "

        if isinstance(row, ToggleItem):
            label = f"{row.label:<20s}"
            if row.label == "Task":
                current = row.choices[row.selected]
                if is_focused:
                    value = f"{BOLD}{BG_BLUE}{FG_WHITE}< {current} >{RESET}"
                else:
                    value = f"{FG_WHITE}< {current} >{RESET}"
                line = f"{prefix}{label}{value}"
            else:
                parts = []
                for i, choice in enumerate(row.choices):
                    if i == row.selected:
                        parts.append(f"{BOLD}{BG_BLUE}{FG_WHITE} {choice} {RESET}")
                    else:
                        parts.append(f"{DIM} {choice} {RESET}")
                line = f"{prefix}{label}{''.join(parts)}"
            lines.append(line)

        elif isinstance(row, TextItem):
            label = f"{row.label:<20s}"
            is_disabled = row.label in disabled
            if row.editing:
                # Show edit buffer with a block cursor
                val_display = f"{BOLD}{FG_YELLOW}{row._edit_buf}{RESET}{BG_BLUE} {RESET}"
            elif is_disabled:
                val_display = f"{DIM}{row.value}{RESET}" if row.value else f"{DIM}(empty){RESET}"
                line = f"{prefix}{DIM}{label}{RESET}{val_display}"
                lines.append(line)
                continue
            elif is_focused:
                val_display = f"{UNDERLINE}{row.value}{RESET}" if row.value else f"{DIM}(empty){RESET}"
            else:
                val_display = row.value if row.value else f"{DIM}(empty){RESET}"
            line = f"{prefix}{label}{val_display}"
            lines.append(line)

        elif isinstance(row, ActionItem):
            if is_focused:
                line = f"{prefix}{BOLD}{FG_YELLOW}[ {row.label} ]{RESET}"
            else:
                line = f"{prefix}{FG_WHITE}[ {row.label} ]{RESET}"
            lines.append(line)

        elif isinstance(row, StartItem):
            if is_focused:
                line = f"{prefix}{BOLD}{FG_GREEN}{row.label}{RESET}"
            else:
                line = f"{prefix}{FG_GREEN}{row.label}{RESET}"
            lines.append(line)

        if not is_disabled_row:
            selectable_idx += 1

    lines.append("")
    lines.append(f"  {DIM}{help_line}{RESET}")
    if status_line:
        lines.append(f"  {FG_YELLOW}{status_line}{RESET}")

    return CLEAR_SCREEN + "\r\n".join(lines)


# ---------------------------------------------------------------------------
# Main TUI loop
# ---------------------------------------------------------------------------

def run_tui_config(
    *,
    action_callback=None,
) -> TUIConfig:
    """
    Display the interactive configuration menu and return the user's choices.

    action_callback(key: str, cfg: TUIConfig) -> str | None
        Called when user presses Enter on an ActionItem.
        Return a status string to display, or None.
    """
    rows: list[MenuRow] = [
        ToggleItem("Policy", ["remote", "local"], selected=0),
        ToggleItem("Frame", ["sim", "real"], selected=0),
        ToggleItem("Record", ["false", "true"], selected=0),
        ToggleItem("Speed Mode", ["limited", "native"], selected=0),
        TextItem("Exec Speed (m/s)", "0.05"),
        SeparatorItem(),
        ActionItem("Align Joints", "align"),
        ActionItem("Open Gripper", "grip_open"),
        ActionItem("Close Gripper", "grip_close"),
        SeparatorItem(),
        StartItem(),
    ]

    def _infer_selectable_rows() -> list[MenuRow]:
        nonselectable = _infer_nonselectable_labels(rows)
        return [r for r in rows
                if not isinstance(r, SeparatorItem)
                and not (hasattr(r, "label") and r.label in nonselectable)]

    selectable_rows = _infer_selectable_rows()
    n_selectable = len(selectable_rows)
    cursor = 0
    status_line = ""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    def _render_infer(status: str) -> str:
        return _render(
            rows,
            cursor,
            status,
            disabled_labels=_infer_disabled_labels(rows),
            nonselectable_labels=_infer_nonselectable_labels(rows),
        )

    try:
        tty.setraw(fd)
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.write(_render_infer(status_line))
        sys.stdout.flush()

        while True:
            selectable_rows = _infer_selectable_rows()
            n_selectable = len(selectable_rows)
            cursor = min(cursor, max(0, n_selectable - 1))

            key = _read_key(fd)
            if not key:
                continue

            current_row = selectable_rows[cursor]

            if key == "CTRL_C" or key == "q":
                cfg = _snapshot_config(rows)
                cfg.quit = True
                return cfg

            if key == "UP":
                cursor = (cursor - 1) % n_selectable
                status_line = ""
            elif key == "DOWN":
                cursor = (cursor + 1) % n_selectable
                status_line = ""
            elif key in ("LEFT", "RIGHT"):
                if isinstance(current_row, ToggleItem):
                    if key == "RIGHT":
                        current_row.selected = (current_row.selected + 1) % len(current_row.choices)
                    else:
                        current_row.selected = (current_row.selected - 1) % len(current_row.choices)
                    status_line = ""
            elif key == "ENTER":
                if isinstance(current_row, ToggleItem):
                    current_row.selected = (current_row.selected + 1) % len(current_row.choices)
                    status_line = ""
                elif isinstance(current_row, TextItem):
                    # Enter inline edit mode on the same row
                    current_row.editing = True
                    current_row._edit_buf = current_row.value
                    status_line = "Type new value, Enter to confirm, Esc to cancel"
                    sys.stdout.write(_render_infer(status_line))
                    sys.stdout.flush()
                    while True:
                        ekey = _read_key(fd)
                        if ekey == "ENTER":
                            current_row.value = current_row._edit_buf
                            break
                        elif ekey in ("CTRL_C", "ESC"):
                            break  # discard changes
                        elif ekey == "BACKSPACE":
                            if current_row._edit_buf:
                                current_row._edit_buf = current_row._edit_buf[:-1]
                        elif len(ekey) == 1 and ekey.isprintable():
                            current_row._edit_buf += ekey
                        else:
                            continue
                        sys.stdout.write(_render_infer(status_line))
                        sys.stdout.flush()
                    current_row.editing = False
                    status_line = ""
                elif isinstance(current_row, ActionItem):
                    if action_callback:
                        current_cfg = _snapshot_config(rows)
                        # Temporarily restore terminal for callback output
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        sys.stdout.write(SHOW_CURSOR)
                        sys.stdout.write(CLEAR_SCREEN)
                        sys.stdout.flush()
                        try:
                            result_msg = action_callback(current_row.key, current_cfg)
                        except TypeError:
                            result_msg = action_callback(current_row.key)
                        status_line = result_msg or f"{current_row.label}: done"
                        input("\nPress Enter to return to menu...")
                        tty.setraw(fd)
                        sys.stdout.write(HIDE_CURSOR)
                    else:
                        status_line = f"{current_row.label}: no handler"
                elif isinstance(current_row, StartItem):
                    cfg = _snapshot_config(rows)
                    if cfg.speed_mode == "limited" and cfg.exec_speed_mps <= 0.0:
                        status_line = "Exec Speed must be > 0"
                    else:
                        return cfg

            selectable_rows = _infer_selectable_rows()
            n_selectable = len(selectable_rows)
            cursor = min(cursor, max(0, n_selectable - 1))
            sys.stdout.write(_render_infer(status_line))
            sys.stdout.flush()

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.write(CLEAR_SCREEN)
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Value extraction helpers
# ---------------------------------------------------------------------------

def _get_toggle(rows: list[MenuRow], label: str) -> str:
    for r in rows:
        if isinstance(r, ToggleItem) and r.label == label:
            return r.value
    return ""


def _get_text(rows: list[MenuRow], label: str) -> str:
    for r in rows:
        if isinstance(r, TextItem) and r.label == label:
            return r.value
    return ""


def _get_float(rows: list[MenuRow], label: str, default: float = 0.0) -> float:
    val = _get_text(rows, label)
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _get_int(rows: list[MenuRow], label: str, default: int = 0) -> int:
    val = _get_text(rows, label)
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _get_env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _infer_nonselectable_labels(rows: list[MenuRow]) -> set[str]:
    """When Speed Mode is 'native', hide the Exec Speed text input."""
    labels: set[str] = set()
    if _get_toggle(rows, "Speed Mode") == "native":
        labels.add("Exec Speed (m/s)")
    return labels


def _infer_disabled_labels(rows: list[MenuRow]) -> set[str]:
    """Disabled (greyed-out) labels for inference TUI."""
    return _infer_nonselectable_labels(rows)


def _snapshot_config(rows: list[MenuRow]) -> TUIConfig:
    speed_mode = _get_toggle(rows, "Speed Mode")
    return TUIConfig(
        policy_location=_get_toggle(rows, "Policy"),
        pose_frame=_get_toggle(rows, "Frame"),
        obs_state_mode="yaw",
        dry_run=_get_env_bool("OPENPI_DEBUG_DRY_RUN", False),
        exec_speed_mps=_get_float(rows, "Exec Speed (m/s)", 0.05),
        speed_mode=speed_mode,
        record=_get_toggle(rows, "Record") == "true",
    )


def _choice_index(choices: list[str], value: str, default: int = 0) -> int:
    try:
        return choices.index(value)
    except ValueError:
        return default


def _snapshot_collect_config(rows: list[MenuRow]) -> CollectTUIConfig:
    task = _get_toggle(rows, "Task")
    mode = _get_toggle(rows, "Mode")
    auto_episodes = _get_int(rows, "Auto Episodes", 10)
    if _is_manual_locked_task(rows):
        mode = "manual"
        auto_episodes = 1
    return CollectTUIConfig(
        mode=mode,
        auto_episodes=auto_episodes,
        resume_mode=_get_toggle(rows, "Resume"),
        task=task,
        save_fps=_get_int(rows, "Save FPS", 30),
        state_mode="yaw",
    )


def _collect_disabled_labels(rows: list[MenuRow]) -> set[str]:
    disabled: set[str] = set()
    task = _get_toggle(rows, "Task")
    if _get_toggle(rows, "Mode") != "auto" or task in {"open_and_close", "keyboard_teleop", "storage"}:
        disabled.add("Auto Episodes")
    return disabled


def _is_manual_locked_task(rows: list[MenuRow]) -> bool:
    return _get_toggle(rows, "Task") in {"open_and_close", "keyboard_teleop", "storage"}


def _enforce_collect_task_constraints(rows: list[MenuRow]) -> None:
    if not _is_manual_locked_task(rows):
        return
    for row in rows:
        if isinstance(row, ToggleItem) and row.label == "Mode":
            row.selected = _choice_index(row.choices, "manual", 0)
        elif isinstance(row, TextItem) and row.label == "Auto Episodes":
            row.value = "1"


def _collect_selectable_rows(rows: list[MenuRow]) -> list[MenuRow]:
    nonselectable = _collect_nonselectable_labels(rows)
    selectable: list[MenuRow] = []
    for row in rows:
        if isinstance(row, SeparatorItem):
            continue
        if isinstance(row, (ToggleItem, TextItem)) and row.label in nonselectable:
            continue
        selectable.append(row)
    return selectable


def _collect_nonselectable_labels(rows: list[MenuRow]) -> set[str]:
    labels = _collect_disabled_labels(rows)
    if _is_manual_locked_task(rows):
        labels.add("Mode")
    return labels


def run_collect_tui_config(
    *,
    default_mode: str = "auto",
    default_auto_episodes: int = 10,
    default_resume_mode: str = "continue",
    default_task: str = "pick_and_place",
    default_save_fps: int = 30,
    default_state_mode: str = "yaw",
) -> CollectTUIConfig:
    rows: list[MenuRow] = [
        ToggleItem("Mode", ["auto", "manual"], selected=_choice_index(["auto", "manual"], default_mode, 0)),
        ToggleItem("Resume", ["continue", "reset"], selected=_choice_index(["continue", "reset"], default_resume_mode, 0)),
        ToggleItem("Save FPS", ["30", "50"], selected=_choice_index(["30", "50"], str(default_save_fps), 0)),
        ToggleItem(
            "Task",
            ["pick_and_place", "open_and_close", "keyboard_teleop", "storage"],
            selected=_choice_index(["pick_and_place", "open_and_close", "keyboard_teleop", "storage"], default_task, 0),
        ),
        TextItem("Auto Episodes", str(max(1, int(default_auto_episodes)))),
        SeparatorItem(),
        StartItem(">>> Start Collection <<<"),
    ]
    _enforce_collect_task_constraints(rows)

    cursor = 0
    status_line = ""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def _render_collect(status: str) -> str:
        return _render(
            rows,
            cursor,
            status,
            title="=== OpenPI Data Collector ===",
            help_line="Up/Down: navigate  Left/Right/Enter: toggle  Enter on Start: go  q: quit",
            disabled_labels=_collect_disabled_labels(rows),
            nonselectable_labels=_collect_nonselectable_labels(rows),
        )

    try:
        tty.setraw(fd)
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.write(_render_collect(status_line))
        sys.stdout.flush()

        while True:
            selectable_rows = _collect_selectable_rows(rows)
            n_selectable = len(selectable_rows)
            cursor = min(cursor, max(0, n_selectable - 1))
            key = _read_key(fd)
            if not key:
                continue

            current_row = selectable_rows[cursor]

            if key == "CTRL_C" or key == "q":
                cfg = _snapshot_collect_config(rows)
                cfg.quit = True
                return cfg

            if key == "UP":
                cursor = (cursor - 1) % n_selectable
                status_line = ""
            elif key == "DOWN":
                cursor = (cursor + 1) % n_selectable
                status_line = ""
            elif key in ("LEFT", "RIGHT"):
                if isinstance(current_row, ToggleItem):
                    if key == "RIGHT":
                        current_row.selected = (current_row.selected + 1) % len(current_row.choices)
                    else:
                        current_row.selected = (current_row.selected - 1) % len(current_row.choices)
                    _enforce_collect_task_constraints(rows)
                    status_line = ""
            elif key == "ENTER":
                if isinstance(current_row, ToggleItem):
                    current_row.selected = (current_row.selected + 1) % len(current_row.choices)
                    _enforce_collect_task_constraints(rows)
                    status_line = ""
                elif isinstance(current_row, TextItem):
                    current_row.editing = True
                    current_row._edit_buf = current_row.value
                    status_line = "Type new value, Enter to confirm, Esc to cancel"
                    sys.stdout.write(_render_collect(status_line))
                    sys.stdout.flush()
                    while True:
                        ekey = _read_key(fd)
                        if ekey == "ENTER":
                            current_row.value = current_row._edit_buf
                            break
                        elif ekey in ("CTRL_C", "ESC"):
                            break
                        elif ekey == "BACKSPACE":
                            if current_row._edit_buf:
                                current_row._edit_buf = current_row._edit_buf[:-1]
                        elif len(ekey) == 1 and ekey.isprintable():
                            current_row._edit_buf += ekey
                        else:
                            continue
                        sys.stdout.write(_render_collect(status_line))
                        sys.stdout.flush()
                    current_row.editing = False
                    status_line = ""
                elif isinstance(current_row, StartItem):
                    cfg = _snapshot_collect_config(rows)
                    if cfg.mode == "auto" and cfg.auto_episodes <= 0:
                        status_line = "Auto Episodes must be > 0"
                    else:
                        return cfg

            selectable_rows = _collect_selectable_rows(rows)
            n_selectable = len(selectable_rows)
            cursor = min(cursor, max(0, n_selectable - 1))
            sys.stdout.write(_render_collect(status_line))
            sys.stdout.flush()

    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.write(CLEAR_SCREEN)
        sys.stdout.flush()
