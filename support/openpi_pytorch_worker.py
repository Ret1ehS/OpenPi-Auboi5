#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys
import traceback

if __package__ in (None, ""):
    _PARENT = Path(__file__).resolve().parent.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

from support.openpi_pytorch_policy import OpenPIPyTorchPolicy
from support.openpi_worker_codec import decode_worker_value, encode_worker_value


def _read_message() -> dict | None:
    header = sys.stdin.buffer.read(8)
    if not header:
        return None
    size = int.from_bytes(header, "little", signed=False)
    payload = sys.stdin.buffer.read(size)
    while len(payload) < size:
        chunk = sys.stdin.buffer.read(size - len(payload))
        if not chunk:
            raise EOFError("Unexpected EOF while reading worker payload.")
        payload += chunk
    return decode_worker_value(pickle.loads(payload))


def _write_message(message: dict) -> None:
    payload = pickle.dumps(encode_worker_value(message), protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.write(len(payload).to_bytes(8, "little", signed=False))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="OpenPI PyTorch local policy worker.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--default-prompt", type=str, default="")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--compile-mode", type=str, default="")
    args = parser.parse_args()

    try:
        policy = OpenPIPyTorchPolicy(
            repo_root=args.repo_root,
            checkpoint_dir=args.checkpoint_dir,
            pytorch_device=args.device,
            default_prompt=args.default_prompt or None,
            sample_kwargs={"num_steps": int(args.num_steps)},
            compile_mode=args.compile_mode or None,
        )
        _write_message({"ok": True, "metadata": policy.metadata})
    except Exception as exc:
        _write_message(
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return 1

    while True:
        try:
            message = _read_message()
            if message is None:
                return 0
            op = message.get("op")
            if op == "infer":
                _write_message({"ok": True, "result": policy.infer(message["obs"])})
            elif op == "reset":
                policy.reset()
                _write_message({"ok": True})
            elif op == "close":
                policy.close()
                _write_message({"ok": True})
                return 0
            else:
                _write_message({"ok": False, "error": f"Unsupported worker op: {op!r}"})
        except Exception as exc:
            _write_message(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )


if __name__ == "__main__":
    raise SystemExit(main())
