#!/usr/bin/env python3
from __future__ import annotations

import http.server
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
from collections import deque

from support.keyboard_control import KEY_ENTER, KEY_QUIT, KEY_SPACE


REMOTE_HELPER_PATH = Path(__file__).with_name("keyboard_remote_client.ps1")
REMOTE_STALE_TIMEOUT_S = 0.20
REMOTE_TCP_PORT = int(os.environ.get("OPENPI_REMOTE_KEYBOARD_TCP_PORT", "28731"))
REMOTE_HTTP_PORT = int(os.environ.get("OPENPI_REMOTE_KEYBOARD_HTTP_PORT", "28732"))


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
            return (
                self._client_addr is not None
                and (ts - self._last_packet_ts) <= self._stale_timeout_s
            )

    def start(self) -> bool:
        self.stop()
        self._stop.clear()

        tcp_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            tcp_sock.bind(("0.0.0.0", int(REMOTE_TCP_PORT)))
        except OSError as exc:
            tcp_sock.close()
            raise RuntimeError(
                f"remote keyboard tcp port {REMOTE_TCP_PORT} busy: {exc}"
            ) from exc
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
            raise RuntimeError(
                f"remote keyboard http port {REMOTE_HTTP_PORT} busy: {exc}"
            ) from exc
        self._http_server = http_server
        self._http_port = int(http_server.server_address[1])

        self._accept_thread = threading.Thread(
            target=self._accept_loop,
            name="remote-keyboard-accept",
            daemon=True,
        )
        self._accept_thread.start()
        self._http_thread = threading.Thread(
            target=http_server.serve_forever,
            name="remote-keyboard-http",
            daemon=True,
        )
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
                elif key in {"QUIT", "ESC"}:
                    self._discrete_keys.append(KEY_QUIT)
