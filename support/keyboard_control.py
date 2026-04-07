#!/usr/bin/env python3
from __future__ import annotations

import re
import select
import sys
import termios
import tty
from dataclasses import dataclass


_ESC = "\033["
CLEAR_SCREEN = f"{_ESC}2J{_ESC}H"
HIDE_CURSOR = f"{_ESC}?25l"
SHOW_CURSOR = f"{_ESC}?25h"
BOLD = f"{_ESC}1m"
DIM = f"{_ESC}2m"
RESET = f"{_ESC}0m"
FG_CYAN = f"{_ESC}36m"
FG_GREEN = f"{_ESC}32m"
FG_YELLOW = f"{_ESC}33m"
FG_WHITE = f"{_ESC}37m"
FG_RED = f"{_ESC}31m"
BG_BLUE = f"{_ESC}44m"


KEY_UP = "UP"
KEY_DOWN = "DOWN"
KEY_LEFT = "LEFT"
KEY_RIGHT = "RIGHT"
KEY_CTRL_UP = "CTRL_UP"
KEY_CTRL_DOWN = "CTRL_DOWN"
KEY_CTRL_LEFT = "CTRL_LEFT"
KEY_CTRL_RIGHT = "CTRL_RIGHT"
KEY_ENTER = "ENTER"
KEY_SPACE = "SPACE"
KEY_QUIT = "QUIT"
KEY_CTRL_C = "CTRL_C"


def _parse_csi_sequence(seq: str) -> str:
    if seq in ("A", "B", "C", "D"):
        return {
            "A": KEY_UP,
            "B": KEY_DOWN,
            "C": KEY_RIGHT,
            "D": KEY_LEFT,
        }[seq]

    match = re.fullmatch(r"(?:(?:1|5|6);)?([0-9]+)([ABCD])", seq)
    if match:
        modifier = int(match.group(1))
        suffix = match.group(2)
        if modifier >= 5:
            return {
                "A": KEY_CTRL_UP,
                "B": KEY_CTRL_DOWN,
                "C": KEY_CTRL_RIGHT,
                "D": KEY_CTRL_LEFT,
            }[suffix]

    match = re.fullmatch(r"1;([0-9]+)([ABCD])", seq)
    if match:
        modifier = int(match.group(1))
        suffix = match.group(2)
        if modifier >= 5:
            return {
                "A": KEY_CTRL_UP,
                "B": KEY_CTRL_DOWN,
                "C": KEY_CTRL_RIGHT,
                "D": KEY_CTRL_LEFT,
            }[suffix]

    return ""


def read_key(fd: int) -> str:
    ch = sys.stdin.read(1)
    if ch == "\x03":
        return KEY_CTRL_C
    if ch in ("\r", "\n"):
        return KEY_ENTER
    if ch == " ":
        return KEY_SPACE
    if ch.lower() == "q":
        return KEY_QUIT
    if ch != "\x1b":
        return ch

    seq0 = sys.stdin.read(1)
    if seq0 == "[":
        buf = ""
        while True:
            part = sys.stdin.read(1)
            if not part:
                break
            buf += part
            if part.isalpha() or part == "~":
                break
        return _parse_csi_sequence(buf)
    if seq0 == "O":
        suffix = sys.stdin.read(1)
        return _parse_csi_sequence(suffix)
    return ""


def read_key_nonblocking(fd: int, timeout_s: float = 0.0) -> str | None:
    ready, _, _ = select.select([fd], [], [], max(0.0, float(timeout_s)))
    if not ready:
        return None
    return read_key(fd)


def drain_keys(fd: int) -> list[str]:
    keys: list[str] = []
    while True:
        key = read_key_nonblocking(fd, 0.0)
        if key is None:
            break
        if key:
            keys.append(key)
    return keys


def render_keyboard_ui(
    *,
    prompt: str,
    recording: bool,
    gripper_open: bool,
    state_mode: str,
    move_step_mm: float,
    rotate_step_deg: float,
    status_line: str = "",
) -> str:
    lines: list[str] = []
    lines.append(f"{BOLD}{FG_CYAN}=== OpenPI Keyboard Collect ==={RESET}")
    lines.append("")
    lines.append(f"  Prompt:      {BOLD}{FG_WHITE}{prompt}{RESET}")
    lines.append(
        f"  Recording:   "
        f"{BOLD}{(FG_RED if recording else FG_GREEN)}{'ON' if recording else 'OFF'}{RESET}"
    )
    lines.append(f"  Gripper:     {BOLD}{FG_WHITE}{'open' if gripper_open else 'closed'}{RESET}")
    lines.append(f"  State Mode:  {BOLD}{FG_WHITE}{state_mode}{RESET}")
    lines.append(
        f"  Step:        {BOLD}{FG_WHITE}{move_step_mm:.1f} mm{RESET}  /  "
        f"{BOLD}{FG_WHITE}{rotate_step_deg:.1f} deg{RESET}"
    )
    lines.append("")
    lines.append(f"  {BOLD}Arrow keys{RESET}")
    lines.append("    Up: x+    Down: x-    Left: y+    Right: y-")
    lines.append("    Ctrl+Up: z+    Ctrl+Down: z-")
    lines.append("    Ctrl+Left: CCW rotate    Ctrl+Right: CW rotate")
    lines.append("")
    lines.append(f"  {BOLD}Session{RESET}")
    lines.append("    Enter: start/stop recording")
    lines.append("    Space: toggle gripper")
    lines.append("    q / Ctrl+C: quit")
    lines.append("")
    if status_line:
        lines.append(f"  {FG_YELLOW}{status_line}{RESET}")
    else:
        lines.append(f"  {DIM}Idle. No frames are recorded until you press Enter.{RESET}")
    return CLEAR_SCREEN + "\r\n".join(lines)


@dataclass
class RawTerminal:
    fd: int
    old_settings: object

    @classmethod
    def open(cls) -> "RawTerminal":
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        sys.stdout.write(HIDE_CURSOR)
        sys.stdout.flush()
        return cls(fd=fd, old_settings=old_settings)

    def close(self) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        sys.stdout.write(SHOW_CURSOR)
        sys.stdout.write(CLEAR_SCREEN)
        sys.stdout.flush()
