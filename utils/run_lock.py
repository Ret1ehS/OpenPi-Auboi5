from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from utils.path_utils import get_log_dir


CAMERA_RUNTIME_LOCK = get_log_dir() / "camera_runtime.lock"
_LOCK_HEADER = "#"
_METADATA_OFFSET = 1


class RuntimeLockError(RuntimeError):
    """Raised when another camera-owning OpenPI entrypoint is already active."""


class RuntimeLock:
    def __init__(self, app_name: str, path: Path = CAMERA_RUNTIME_LOCK) -> None:
        self.app_name = app_name
        self.path = path
        self._fh = None

    def acquire(self) -> "RuntimeLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a+", encoding="utf-8")
        try:
            _lock_file(self._fh)
        except OSError as exc:
            owner = _read_lock_owner(self._fh)
            self._fh.close()
            self._fh = None
            raise RuntimeLockError(_format_lock_error(self.app_name, owner, self.path)) from exc

        metadata = {
            "app": self.app_name,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "cwd": os.getcwd(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write(_LOCK_HEADER)
        self._fh.seek(_METADATA_OFFSET)
        json.dump(metadata, self._fh, ensure_ascii=True, sort_keys=True)
        self._fh.write("\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return self

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            _unlock_file(self._fh)
        finally:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "RuntimeLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.release()


def acquire_camera_runtime_lock(app_name: str) -> RuntimeLock:
    return RuntimeLock(app_name).acquire()


def _read_lock_owner(fh: Any) -> dict[str, Any]:
    try:
        fh.seek(_METADATA_OFFSET)
        data = fh.read().strip()
        if not data:
            return {}
        owner = json.loads(data)
        return owner if isinstance(owner, dict) else {}
    except Exception:
        return {}


def _format_lock_error(app_name: str, owner: dict[str, Any], path: Path) -> str:
    owner_app = str(owner.get("app") or "unknown entrypoint")
    owner_pid = owner.get("pid")
    owner_host = str(owner.get("host") or "unknown host")
    owner_started_at = str(owner.get("started_at") or "unknown time")
    pid_text = f" pid={owner_pid}" if owner_pid is not None else ""
    return (
        f"Another OpenPI camera runtime is already active: {owner_app}{pid_text} "
        f"on {owner_host} since {owner_started_at}. "
        f"Stop it before starting {app_name}. Lock: {path}"
    )


if os.name == "nt":
    import msvcrt

    def _lock_file(fh: Any) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)

    def _unlock_file(fh: Any) -> None:
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_file(fh: Any) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock_file(fh: Any) -> None:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
