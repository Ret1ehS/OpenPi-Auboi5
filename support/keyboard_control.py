#!/usr/bin/env python3
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import http.server
import json
import os
from pathlib import Path
import re
import select
import socket
import sys
import termios
import threading
import time
import tty

try:
    from pynput import keyboard as pynput_keyboard
except (ImportError, Exception):
    pynput_keyboard = None


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
KEY_SHIFT = "SHIFT"
KEY_TAB = "TAB"
KEY_QUIT = "QUIT"
KEY_CTRL_C = "CTRL_C"

REMOTE_HELPER_PATH = Path(__file__).with_name("keyboard_remote_client.ps1")
REMOTE_STALE_TIMEOUT_S = 0.20
REMOTE_TCP_PORT = int(os.environ.get("OPENPI_REMOTE_KEYBOARD_TCP_PORT", "28731"))
REMOTE_HTTP_PORT = int(os.environ.get("OPENPI_REMOTE_KEYBOARD_HTTP_PORT", "28732"))


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
    if ch == "\t":
        return KEY_TAB
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


@dataclass
class ContinuousKeyState:
    _listener: object | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _backend: str = "none"
    _up: bool = False
    _down: bool = False
    _left: bool = False
    _right: bool = False
    _ctrl: bool = False
    _last_up_ts: float = -1.0
    _last_down_ts: float = -1.0
    _last_left_ts: float = -1.0
    _last_right_ts: float = -1.0
    _last_ctrl_up_ts: float = -1.0
    _last_ctrl_down_ts: float = -1.0
    _last_ctrl_left_ts: float = -1.0
    _last_ctrl_right_ts: float = -1.0
    _repeat_hold_s: float = 0.08

    @property
    def available(self) -> bool:
        return True

    @property
    def backend(self) -> str:
        return self._backend

    def start(self, *, fd: int | None = None, repeat_hold_s: float = 0.08) -> bool:
        self.stop()
        self._repeat_hold_s = float(repeat_hold_s)
        if pynput_keyboard is None:
            self._backend = "terminal_repeat" if fd is not None else "none"
            return self._backend != "none"
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()
        self._backend = "pynput"
        return True

    def stop(self) -> None:
        listener = self._listener
        self._listener = None
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        with self._lock:
            self._backend = "none"
            self._up = False
            self._down = False
            self._left = False
            self._right = False
            self._ctrl = False
            self._last_up_ts = -1.0
            self._last_down_ts = -1.0
            self._last_left_ts = -1.0
            self._last_right_ts = -1.0
            self._last_ctrl_up_ts = -1.0
            self._last_ctrl_down_ts = -1.0
            self._last_ctrl_LEFT_ts = -1.0
            self._last_ctrl_right_ts = -1.0

    def clear(self) -> None:
        with self._lock:
            self._up = False
            self._down = False
            self._left = False
            self._right = False
            self._ctrl = False
            self._last_up_ts = -1.0
            self._last_down_ts = -1.0
            self._last_left_ts = -1.0
            self._last_right_ts = -1.0
            self._last_ctrl_up_ts = -1.0
            self._last_ctrl_down_ts = -1.0
            self._last_ctrl_left_ts = -1.0
            self._last_ctrl_right_ts = -1.0

    def feed_terminal_keys(self, keys: list[str], now_ts: float) -> list[str]:
        if self._backend != "terminal_repeat":
            return [
                key for key in keys
                if key not in {
                    KEY_UP,
                    KEY_DOWN,
                    KEY_LEFT,
                    KEY_RIGHT,
                    KEY_CTRL_UP,
                    KEY_CTRL_DOWN,
                    KEY_CTRL_LEFT,
                    KEY_CTRL_RIGHT,
                }
            ]

        discrete: list[str] = []
        with self._lock:
            for key in keys:
                if key == KEY_UP:
                    self._last_up_ts = now_ts
                elif key == KEY_DOWN:
                    self._last_down_ts = now_ts
                elif key == KEY_LEFT:
                    self._last_left_ts = now_ts
                elif key == KEY_RIGHT:
                    self._last_right_ts = now_ts
                elif key == KEY_CTRL_UP:
                    self._last_ctrl_up_ts = now_ts
                elif key == KEY_CTRL_DOWN:
                    self._last_ctrl_down_ts = now_ts
                elif key == KEY_CTRL_LEFT:
                    self._last_ctrl_left_ts = now_ts
                elif key == KEY_CTRL_RIGHT:
                    self._last_ctrl_right_ts = now_ts
                else:
                    discrete.append(key)
        return discrete

    def axes(self, now_ts: float | None = None) -> tuple[float, float, float, float]:
        with self._lock:
            if self._backend == "terminal_repeat":
                ts = 0.0 if now_ts is None else float(now_ts)
                up = (ts - self._last_up_ts) <= self._repeat_hold_s
                down = (ts - self._last_down_ts) <= self._repeat_hold_s
                left = (ts - self._last_left_ts) <= self._repeat_hold_s
                right = (ts - self._last_right_ts) <= self._repeat_hold_s
                ctrl_up = (ts - self._last_ctrl_up_ts) <= self._repeat_hold_s
                ctrl_down = (ts - self._last_ctrl_down_ts) <= self._repeat_hold_s
                ctrl_left = (ts - self._last_ctrl_left_ts) <= self._repeat_hold_s
                ctrl_right = (ts - self._last_ctrl_right_ts) <= self._repeat_hold_s
                vertical = (1.0 if up else 0.0) - (1.0 if down else 0.0)
                horizontal = (1.0 if left else 0.0) - (1.0 if right else 0.0)
                z_axis = (1.0 if ctrl_up else 0.0) - (1.0 if ctrl_down else 0.0)
                yaw_axis = (1.0 if ctrl_right else 0.0) - (1.0 if ctrl_left else 0.0)
                return vertical, horizontal, z_axis, yaw_axis

            up = self._up
            down = self._down
            left = self._left
            right = self._right
            ctrl = self._ctrl
        vertical = (1.0 if up else 0.0) - (1.0 if down else 0.0)
        horizontal = (1.0 if left else 0.0) - (1.0 if right else 0.0)
        if ctrl:
            return 0.0, 0.0, vertical, -horizontal
        return vertical, horizontal, 0.0, 0.0

    def _on_press(self, key) -> None:
        with self._lock:
            if key in {pynput_keyboard.Key.ctrl, pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r}:
                self._ctrl = True
            elif key == pynput_keyboard.Key.up:
                self._up = True
            elif key == pynput_keyboard.Key.down:
                self._down = True
            elif key == pynput_keyboard.Key.left:
                self._left = True
            elif key == pynput_keyboard.Key.right:
                self._right = True

    def _on_release(self, key) -> None:
        with self._lock:
            if key in {pynput_keyboard.Key.ctrl, pynput_keyboard.Key.ctrl_l, pynput_keyboard.Key.ctrl_r}:
                self._ctrl = False
            elif key == pynput_keyboard.Key.up:
                self._up = False
            elif key == pynput_keyboard.Key.down:
                self._down = False
            elif key == pynput_keyboard.Key.left:
                self._left = False
            elif key == pynput_keyboard.Key.right:
                self._right = False


def render_keyboard_ui(
    *,
    prompt: str,
    recording: bool,
    saving: bool = False,
    gripper_open: bool,
    state_mode: str,
    move_step_mm: float,
    rotate_step_deg: float,
    input_source: str = "",
    helper_command: str = "",
    status_line: str = "",
) -> str:
    lines: list[str] = []
    record_color = FG_YELLOW if saving else (FG_RED if recording else FG_GREEN)
    record_text = "SAVING" if saving else ("ON" if recording else "OFF")
    lines.append(f"{BOLD}{FG_CYAN}=== OpenPI Keyboard Collect ==={RESET}")
    lines.append("")
    lines.append(f"  Prompt:      {BOLD}{FG_WHITE}{prompt}{RESET}")
    lines.append(
        f"  Recording:   "
        f"{BOLD}{record_color}{record_text}{RESET}"
    )
    lines.append(f"  Gripper:     {BOLD}{FG_WHITE}{'open' if gripper_open else 'closed'}{RESET}")
    lines.append(f"  State Mode:  {BOLD}{FG_WHITE}{state_mode}{RESET}")
    if input_source:
        lines.append(f"  Input:       {BOLD}{FG_WHITE}{input_source}{RESET}")
    lines.append(
        f"  Step:        {BOLD}{FG_WHITE}{move_step_mm:.1f} mm{RESET}  /  "
        f"{BOLD}{FG_WHITE}{rotate_step_deg:.1f} deg{RESET}"
    )
    if helper_command:
        lines.append(f"  Helper:      {DIM}{helper_command}{RESET}")
    lines.append("")
    lines.append(f"  {BOLD}Arrow keys{RESET}")
    lines.append("    Up: x+    Down: x-    Left: y+    Right: y-")
    lines.append("    Ctrl+Up: z+    Ctrl+Down: z-")
    lines.append("    Ctrl+Left: CCW rotate   Ctrl+Right: CW rotate")
    lines.append("")
    lines.append(f"  {BOLD}Session{RESET}")
    lines.append("    Enter: start/stop recording")
    lines.append("    Space: toggle gripper")
    lines.append("    Shift: edit prompt (idle only, helper)")
    lines.append("    Tab: return home (idle only)")
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


class RemoteKeyboardRelay:
    def __init__(
        self,
        *,
        advertised_host: str,
        allowed_client_ip: str | None = None,
        stale_timeout_s: float = REMOTE_STALE_TIMEOUT_S,
    ) -> None:
        self._advertised_host = str(advertised_host).strip()
        self._allowed_client_ip = str(allowed_client_ip).strip() if allowed_client_ip else None
        self._stale_timeout_s = float(stale_timeout_s)
        token_src = f"{self._advertised_host}|{self._allowed_client_ip or '*'}|openpi-keyboard-relay"
        self._token = hashlib.sha256(token_src.encode("utf-8")).hexdigest()[:24]

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._accept_thread: threading.Thread | None = None
        self._http_thread: threading.Thread | None = None
        self._tcp_sock: socket.socket | None = None
        self._client_sock: socket.socket | None = None
        self._http_server: http.server.ThreadingHTTPServer | None = None

        self._tcp_port = 0
        self._http_port = 0
        self._client_addr: tuple[str, int] | None = None
        self._connected_once = False
        self._last_packet_ts = 0.0
        self._up = False
        self._down = False
        self._left = False
        self._right = False
        self._ctrl = False
        self._discrete_keys: deque[str] = deque()

    @classmethod
    def from_ssh_session(cls) -> "RemoteKeyboardRelay" | None:
        ssh_connection = str(os.environ.get("SSH_CONNECTION", "")).strip()
        if not ssh_connection:
            return None
        parts = ssh_connection.split()
        if len(parts) < 4:
            return None
        client_ip, _, server_ip, _ = parts[:4]
        if not server_ip:
            return None
        return cls(advertised_host=server_ip, allowed_client_ip=client_ip)

    @property
    def launcher_command(self) -> str:
        if self._http_port <= 0:
            return ""
        return (
            "powershell -NoP -EP Bypass -Command "
            f"\"iwr http://{self._advertised_host}:{self._http_port}/keyboard.ps1 "
            "-UseBasicParsing | iex\""
        )

    @property
    def status_text(self) -> str:
        now_ts = time.monotonic()
        with self._lock:
            client_addr = self._client_addr
            connected_once = self._connected_once
            last_packet_ts = self._last_packet_ts
        if client_addr is not None and (now_ts - last_packet_ts) <= self._stale_timeout_s:
            return f"remote stream connected from {client_addr[0]}:{client_addr[1]}"
        if connected_once:
            return "remote stream waiting for reconnect"
        return "remote stream waiting for first connection"

    def has_active_connection(self, now_ts: float | None = None) -> bool:
        ts = time.monotonic() if now_ts is None else float(now_ts)
        with self._lock:
            return self._client_addr is not None and (ts - self._last_packet_ts) <= self._stale_timeout_s

    def start(self) -> bool:
        self.stop()
        self._stop.clear()

        tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            tcp_sock.bind(("0.0.0.0", int(REMOTE_TCP_PORT)))
        except OSError as exc:
            tcp_sock.close()
            raise RuntimeError(f"remote keyboard tcp port {REMOTE_TCP_PORT} busy: {exc}") from exc
        tcp_sock.listen(1)
        tcp_sock.settimeout(0.2)
        self._tcp_sock = tcp_sock
        self._tcp_port = int(tcp_sock.getsockname()[1])

        relay = self

        class _HelperHandler(http.server.BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path != "/keyboard.ps1":
                    self.send_error(404)
                    return
                if relay._allowed_client_ip and self.client_address[0] != relay._allowed_client_ip:
                    self.send_error(403)
                    return
                payload = relay._render_helper_script().encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args) -> None:
                return

        try:
            http_server = http.server.ThreadingHTTPServer(("0.0.0.0", int(REMOTE_HTTP_PORT)), _HelperHandler)
        except OSError as exc:
            tcp_sock.close()
            self._tcp_sock = None
            raise RuntimeError(f"remote keyboard http port {REMOTE_HTTP_PORT} busy: {exc}") from exc
        self._http_server = http_server
        self._http_port = int(http_server.server_address[1])

        self._accept_thread = threading.Thread(target=self._accept_loop, name="remote-keyboard-accept", daemon=True)
        self._accept_thread.start()
        self._http_thread = threading.Thread(target=http_server.serve_forever, name="remote-keyboard-http", daemon=True)
        self._http_thread.start()
        return True

    def stop(self) -> None:
        self._stop.set()
        client_sock = self._client_sock
        self._client_sock = None
        if client_sock is not None:
            try:
                client_sock.close()
            except Exception:
                pass
        tcp_sock = self._tcp_sock
        self._tcp_sock = None
        if tcp_sock is not None:
            try:
                tcp_sock.close()
            except Exception:
                pass
        http_server = self._http_server
        self._http_server = None
        if http_server is not None:
            try:
                http_server.shutdown()
            except Exception:
                pass
            try:
                http_server.server_close()
            except Exception:
                pass
        accept_thread = self._accept_thread
        self._accept_thread = None
        if accept_thread is not None:
            accept_thread.join(timeout=1.0)
        http_thread = self._http_thread
        self._http_thread = None
        if http_thread is not None:
            http_thread.join(timeout=1.0)
        with self._lock:
            self._client_addr = None
            self._last_packet_ts = 0.0
            self._up = False
            self._down = False
            self._left = False
            self._right = False
            self._ctrl = False
            self._discrete_keys.clear()

    def pop_discrete(self) -> list[str]:
        with self._lock:
            keys = list(self._discrete_keys)
            self._discrete_keys.clear()
            return keys

    def clear(self) -> None:
        with self._lock:
            self._up = False
            self._down = False
            self._left = False
            self._right = False
            self._ctrl = False
            self._discrete_keys.clear()

    def axes(self, now_ts: float | None = None) -> tuple[float, float, float, float]:
        ts = time.monotonic() if now_ts is None else float(now_ts)
        with self._lock:
            if self._client_addr is None or (ts - self._last_packet_ts) > self._stale_timeout_s:
                return 0.0, 0.0, 0.0, 0.0
            up = self._up
            down = self._down
            left = self._left
            right = self._right
            ctrl = self._ctrl
        vertical = (1.0 if up else 0.0) - (1.0 if down else 0.0)
        horizontal = (1.0 if left else 0.0) - (1.0 if right else 0.0)
        if ctrl:
            return 0.0, 0.0, vertical, -horizontal
        return vertical, horizontal, 0.0, 0.0

    def _render_helper_script(self) -> str:
        template = REMOTE_HELPER_PATH.read_text(encoding="utf-8")
        return (
            template.replace("__SERVER_HOST__", self._advertised_host)
            .replace("__SERVER_PORT__", str(self._tcp_port))
            .replace("__TOKEN__", self._token)
        )

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            sock = self._tcp_sock
            if sock is None:
                return
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            if self._allowed_client_ip and addr[0] != self._allowed_client_ip:
                try:
                    conn.close()
                except Exception:
                    pass
                continue
            conn.settimeout(0.2)
            with self._lock:
                prev = self._client_sock
                self._client_sock = conn
                self._client_addr = (str(addr[0]), int(addr[1]))
                self._connected_once = True
            if prev is not None and prev is not conn:
                try:
                    prev.close()
                except Exception:
                    pass
            self._read_client(conn, addr)

    def _read_client(self, conn: socket.socket, addr: tuple[str, int]) -> None:
        buffer = ""
        while not self._stop.is_set():
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="ignore")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                self._handle_line(line.strip())
        try:
            conn.close()
        except Exception:
            pass
        with self._lock:
            if self._client_addr == (str(addr[0]), int(addr[1])):
                self._client_addr = None
                self._up = False
                self._down = False
                self._left = False
                self._right = False
                self._ctrl = False

    def _handle_line(self, line: str) -> None:
        if not line:
            return
        try:
            payload = json.loads(line)
        except Exception:
            return
        if str(payload.get("token", "")) != self._token:
            return
        msg_type = str(payload.get("type", "")).strip().lower()
        now_ts = time.monotonic()
        with self._lock:
            self._last_packet_ts = now_ts
            if msg_type == "state":
                self._up = bool(payload.get("up", False))
                self._down = bool(payload.get("down", False))
                self._left = bool(payload.get("left", False))
                self._right = bool(payload.get("right", False))
                self._ctrl = bool(payload.get("ctrl", False))
                return
            if msg_type == "event":
                key = str(payload.get("key", "")).strip().upper()
                if key == "ENTER":
                    self._discrete_keys.append(KEY_ENTER)
                elif key == "SPACE":
                    self._discrete_keys.append(KEY_SPACE)
                elif key == "SHIFT":
                    self._discrete_keys.append(KEY_SHIFT)
                elif key == "TAB":
                    self._discrete_keys.append(KEY_TAB)
                elif key in {"QUIT", "ESC"}:
                    self._discrete_keys.append(KEY_QUIT)
