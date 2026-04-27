#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ''):
    _PARENT = Path(__file__).resolve().parent.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

import numpy as np

from utils.env_utils import load_default_env
from utils.path_utils import SCRIPTS_ROOT, get_captures_dir, get_log_dir
from utils.runtime_config import DEFAULT_OBSERVER_MODEL, DEFAULT_OBSERVER_PYTHON

load_default_env()

_DEFAULT_PYTHON = DEFAULT_OBSERVER_PYTHON
_DEFAULT_MODEL = DEFAULT_OBSERVER_MODEL
_WORKER_SCRIPT = Path(__file__).resolve()


@dataclasses.dataclass(frozen=True)
class TaskObserverConfig:
    enabled: bool = False
    interval_s: float = 5.0
    python_bin: Path = _DEFAULT_PYTHON
    model_path: Path = _DEFAULT_MODEL
    worker_script: Path = _WORKER_SCRIPT
    captures_dir: Path = get_captures_dir() / 'task_observer'
    log_path: Path = get_log_dir() / 'task_observer.log'
    max_new_tokens: int = 96
    task_spec: str = ''

    @classmethod
    def from_env(cls) -> 'TaskObserverConfig':
        enabled = os.environ.get('OPENPI_TASK_OBSERVER_ENABLE', '').strip().lower() in {'1', 'true', 'yes', 'on'}
        interval_s = float(os.environ.get('OPENPI_TASK_OBSERVER_INTERVAL_S', '5.0'))
        max_new_tokens = int(os.environ.get('OPENPI_TASK_OBSERVER_MAX_NEW_TOKENS', '96'))
        task_spec = os.environ.get('OPENPI_TASK_OBSERVER_SPEC', '').strip()
        spec_file = os.environ.get('OPENPI_TASK_OBSERVER_SPEC_FILE', '').strip()
        if not task_spec and spec_file:
            spec_path = Path(spec_file).expanduser()
            if not spec_path.is_absolute():
                spec_path = SCRIPTS_ROOT / spec_path
            if spec_path.exists():
                task_spec = spec_path.read_text(encoding='utf-8').strip()
        python_bin = Path(os.environ.get('OPENPI_TASK_OBSERVER_PYTHON', str(_DEFAULT_PYTHON))).expanduser()
        model_path = Path(os.environ.get('OPENPI_TASK_OBSERVER_MODEL', str(_DEFAULT_MODEL))).expanduser()
        return cls(
            enabled=enabled,
            interval_s=max(0.5, interval_s),
            python_bin=python_bin,
            model_path=model_path,
            max_new_tokens=max(16, max_new_tokens),
            task_spec=task_spec,
        )


@dataclasses.dataclass
class TaskObserverResult:
    session_id: int
    request_id: int
    step: int
    complete: bool
    confidence: float
    reason: str
    raw_text: str = ''


@dataclasses.dataclass
class _Snapshot:
    step: int
    tcp_pose_sim: np.ndarray
    gripper_open_scalar: float
    yaw_readback_scalar: float | None
    main_bgr: np.ndarray
    wrist_bgr: np.ndarray


def _extract_json(text: str) -> dict[str, object] | None:
    import re

    text = text.strip()
    if not text:
        return None
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?', '', text).strip()
        text = re.sub(r'```$', '', text).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r'\{.*\}', text, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _normalize_worker_result(obj: dict[str, object] | None, raw_text: str) -> dict[str, object]:
    if obj is None:
        return {
            'complete': False,
            'confidence': 0.0,
            'reason': 'model response was not valid JSON',
            'raw_text': raw_text,
        }
    complete = bool(obj.get('complete', False))
    try:
        confidence = float(obj.get('confidence', 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    reason = str(obj.get('reason', ''))[:400]
    return {
        'complete': complete,
        'confidence': confidence,
        'reason': reason,
        'raw_text': raw_text,
    }


class TaskCompletionObserver:
    def __init__(self, config: TaskObserverConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._session_id = 0
        self._session_prompt = ''
        self._session_active = False
        self._latest_snapshot: _Snapshot | None = None
        self._request_id = 0
        self._request_in_flight = False
        self._last_request_ts = 0.0
        self._completion_result: TaskObserverResult | None = None
        self._proc: subprocess.Popen[str] | None = None
        self._stderr_fh = None
        self._reader_thread: threading.Thread | None = None
        self._loop_thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def start(self) -> None:
        if not self.enabled:
            return
        self._config.captures_dir.mkdir(parents=True, exist_ok=True)
        self._config.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_fh = open(self._config.log_path, 'a', encoding='utf-8')
        env = dict(os.environ)
        env['PYTHONUNBUFFERED'] = '1'
        env['PYTHONNOUSERSITE'] = '1'
        self._proc = subprocess.Popen(
            [
                str(self._config.python_bin),
                str(self._config.worker_script),
                '--worker',
                '--model',
                str(self._config.model_path),
                '--max-new-tokens',
                str(self._config.max_new_tokens),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_fh,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()
        self._loop_thread = threading.Thread(target=self._schedule_loop, daemon=True)
        self._loop_thread.start()
        self._append_log({'event': 'observer_start', 'pid': int(self._proc.pid)})

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.stdin:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
        if self._reader_thread:
            self._reader_thread.join(timeout=2)
        if self._loop_thread:
            self._loop_thread.join(timeout=2)
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._stderr_fh is not None:
            self._stderr_fh.close()
            self._stderr_fh = None

    def begin_session(self, task_prompt: str) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._session_id += 1
            self._session_prompt = task_prompt
            self._session_active = True
            self._latest_snapshot = None
            self._completion_result = None
            self._request_in_flight = False
            self._last_request_ts = 0.0

    def end_session(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._session_active = False
            self._latest_snapshot = None
            self._request_in_flight = False
            self._completion_result = None

    def update_observation(self, *, step: int, aligned_obs: Any) -> None:
        if not self.enabled:
            return
        snapshot = _Snapshot(
            step=int(step),
            tcp_pose_sim=np.asarray(aligned_obs.aligned_tcp_pose_sim, dtype=np.float64).reshape(6).copy(),
            gripper_open_scalar=float(aligned_obs.gripper_open_scalar),
            yaw_readback_scalar=None if aligned_obs.yaw_readback_scalar is None else float(aligned_obs.yaw_readback_scalar),
            main_bgr=np.asarray(aligned_obs.main_frame.image_bgr).copy(),
            wrist_bgr=np.asarray(aligned_obs.wrist_frame.image_bgr).copy(),
        )
        with self._lock:
            if self._session_active:
                self._latest_snapshot = snapshot

    def pop_completion(self) -> TaskObserverResult | None:
        if not self.enabled:
            return None
        with self._lock:
            result = self._completion_result
            self._completion_result = None
            return result

    def status_text(self) -> str:
        if not self.enabled:
            return 'disabled'
        with self._lock:
            return (
                f'session_active={self._session_active} '
                f'in_flight={self._request_in_flight} '
                f'has_snapshot={self._latest_snapshot is not None}'
            )

    def _append_log(self, event: dict[str, object]) -> None:
        payload = {'ts_ms': int(time.time() * 1000), **event}
        with open(self._config.log_path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + '\n')

    def _reader_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for raw_line in self._proc.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                self._append_log({'event': 'observer_bad_json', 'line': line[:400]})
                continue
            event = str(msg.get('event', ''))
            if event == 'ready':
                self._append_log({'event': 'observer_ready'})
                continue
            if event == 'result':
                with self._lock:
                    current_session_id = self._session_id
                    self._request_in_flight = False
                if int(msg.get('session_id', -1)) != current_session_id:
                    continue
                result = TaskObserverResult(
                    session_id=int(msg.get('session_id', 0)),
                    request_id=int(msg.get('id', 0)),
                    step=int(msg.get('step', 0)),
                    complete=bool(msg.get('complete', False)),
                    confidence=float(msg.get('confidence', 0.0)),
                    reason=str(msg.get('reason', ''))[:400],
                    raw_text=str(msg.get('raw_text', ''))[:1000],
                )
                self._append_log(
                    {
                        'event': 'observer_result',
                        'session_id': result.session_id,
                        'request_id': result.request_id,
                        'complete': result.complete,
                        'confidence': result.confidence,
                        'reason': result.reason,
                    }
                )
                if result.complete:
                    with self._lock:
                        self._completion_result = result
                continue
            if event == 'error':
                with self._lock:
                    self._request_in_flight = False
                self._append_log({'event': 'observer_error', 'error': str(msg.get('error', ''))[:500]})
        self._append_log({'event': 'observer_stdout_closed'})

    def _schedule_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.1)
            if not self.enabled:
                continue
            with self._lock:
                if not self._session_active or self._completion_result is not None:
                    continue
                if self._request_in_flight or self._latest_snapshot is None:
                    continue
                if time.monotonic() - self._last_request_ts < self._config.interval_s:
                    continue
                self._request_id += 1
                request_id = self._request_id
                self._request_in_flight = True
                self._last_request_ts = time.monotonic()
                session_id = self._session_id
                session_prompt = self._session_prompt
                snapshot = self._latest_snapshot
            try:
                assert snapshot is not None
                main_path, wrist_path = self._write_snapshot_images(session_id, request_id, snapshot)
                payload = {
                    'id': request_id,
                    'session_id': session_id,
                    'step': int(snapshot.step),
                    'task_prompt': session_prompt,
                    'task_spec': self._config.task_spec,
                    'state_summary': self._build_state_summary(snapshot),
                    'main_image_path': str(main_path),
                    'wrist_image_path': str(wrist_path),
                }
                self._send_payload(payload)
                self._append_log({'event': 'observer_request', 'session_id': session_id, 'request_id': request_id, 'step': snapshot.step})
            except Exception as exc:
                with self._lock:
                    self._request_in_flight = False
                self._append_log({'event': 'observer_request_error', 'error': f'{type(exc).__name__}: {exc}'})

    def _send_payload(self, payload: dict[str, object]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError('observer worker is not running')
        self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + '\n')
        self._proc.stdin.flush()

    def _write_snapshot_images(self, session_id: int, request_id: int, snapshot: _Snapshot) -> tuple[Path, Path]:
        base = self._config.captures_dir / f'session_{session_id:04d}'
        base.mkdir(parents=True, exist_ok=True)
        main_path = base / f'{request_id:06d}_main.jpg'
        wrist_path = base / f'{request_id:06d}_wrist.jpg'
        import cv2

        if not cv2.imwrite(str(main_path), snapshot.main_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
            raise RuntimeError(f'failed to write {main_path}')
        if not cv2.imwrite(str(wrist_path), snapshot.wrist_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
            raise RuntimeError(f'failed to write {wrist_path}')
        return main_path, wrist_path

    @staticmethod
    def _build_state_summary(snapshot: _Snapshot) -> str:
        pose = [round(float(v), 4) for v in snapshot.tcp_pose_sim.tolist()]
        yaw_str = 'none' if snapshot.yaw_readback_scalar is None else f'{float(snapshot.yaw_readback_scalar):.4f}'
        return (
            f'step={snapshot.step}; '
            f'gripper_open_scalar={snapshot.gripper_open_scalar:.3f}; '
            f'yaw_readback={yaw_str}; '
            f'tcp_pose_sim={pose}'
        )


def worker_main(argv: list[str] | None = None) -> int:
    import argparse
    import traceback

    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
    os.environ.setdefault('TQDM_DISABLE', '1')
    os.environ.setdefault('TRANSFORMERS_VERBOSITY', 'error')

    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model', required=True)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    args = parser.parse_args(argv)

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map='cuda',
    )

    print(json.dumps({'event': 'ready'}), flush=True)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            request_id = int(req.get('id', 0))
            session_id = int(req.get('session_id', 0))
            step = int(req.get('step', 0))
            task_prompt = str(req.get('task_prompt', ''))
            task_spec = str(req.get('task_spec', ''))
            state_summary = str(req.get('state_summary', ''))
            main_image = str(req.get('main_image_path', ''))
            wrist_image = str(req.get('wrist_image_path', ''))

            user_text = (
                'You are a robot task completion observer. '
                'Decide whether the current task is already complete from the images and context.\n\n'
                f'Task prompt:\n{task_prompt}\n\n'
                f'Completion rules:\n{task_spec or "Use the task prompt and visible scene to judge completion. If unsure, return complete=false."}\n\n'
                f'Robot state summary:\n{state_summary}\n\n'
                'Return strict JSON only, with this schema:\n'
                '{"complete": true|false, "confidence": 0.0-1.0, "reason": "short reason"}\n'
                'If you are unsure, return complete=false. Do not use markdown.'
            )
            messages = [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'url': main_image},
                        {'type': 'image', 'url': wrist_image},
                        {'type': 'text', 'text': user_text},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                return_dict=True,
                return_tensors='pt',
                add_generation_prompt=True,
            ).to(model.device)
            input_len = inputs['input_ids'].shape[-1]
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            text = processor.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            result = _normalize_worker_result(_extract_json(text), text)
            payload = {
                'event': 'result',
                'id': request_id,
                'session_id': session_id,
                'step': step,
                **result,
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)
        except Exception as exc:
            payload = {
                'event': 'error',
                'id': int(req.get('id', 0)) if 'req' in locals() and isinstance(req, dict) else 0,
                'session_id': int(req.get('session_id', 0)) if 'req' in locals() and isinstance(req, dict) else 0,
                'error': f'{type(exc).__name__}: {exc}',
                'traceback': traceback.format_exc(limit=2),
            }
            print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(worker_main())
