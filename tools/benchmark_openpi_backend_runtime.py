#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from support.load_policy import PolicyLoadSpec, create_local_policy


def _build_synthetic_observation(*, prompt: str, state_dim: int, image_size: int) -> dict[str, object]:
    return {
        "observation/state": np.zeros((state_dim,), dtype=np.float32),
        "observation/image": np.zeros((image_size, image_size, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((image_size, image_size, 3), dtype=np.uint8),
        "prompt": prompt,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark configured OpenPI runtime backend through support/load_policy.py.")
    parser.add_argument("--prompt", type=str, default="pick up the blue cube")
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    load_t0 = time.perf_counter()
    runner = create_local_policy(PolicyLoadSpec())
    load_s = time.perf_counter() - load_t0
    obs = _build_synthetic_observation(
        prompt=args.prompt,
        state_dim=int(args.state_dim),
        image_size=int(args.image_size),
    )

    policy_timings_ms: list[float] = []
    roundtrip_timings_ms: list[float] = []
    action_preview: list[float] | None = None

    try:
        for index in range(int(args.iterations)):
            infer_t0 = time.perf_counter()
            result = runner.infer(obs)
            roundtrip_ms = (time.perf_counter() - infer_t0) * 1000.0
            roundtrip_timings_ms.append(roundtrip_ms)

            policy_timing = result.get("policy_timing", {}) if isinstance(result, dict) else {}
            policy_infer_ms = policy_timing.get("infer_ms")
            if policy_infer_ms is not None:
                policy_timings_ms.append(float(policy_infer_ms))

            actions = np.asarray(result["actions"], dtype=np.float32)
            if action_preview is None:
                action_preview = actions[0].astype(float).tolist()

            payload = {
                "iteration": index,
                "roundtrip_ms": round(roundtrip_ms, 3),
                "policy_infer_ms": None if policy_infer_ms is None else round(float(policy_infer_ms), 3),
                "action_shape": list(actions.shape),
                "first_action": [round(value, 6) for value in actions[0].astype(float).tolist()],
            }
            print(json.dumps(payload, ensure_ascii=False))
    finally:
        runner.close()

    summary = {
        "load_s": round(load_s, 3),
        "iterations": int(args.iterations),
        "backend": runner.metadata.get("policy_backend"),
        "roundtrip_ms_first": round(roundtrip_timings_ms[0], 3),
        "roundtrip_ms_best": round(min(roundtrip_timings_ms), 3),
        "roundtrip_ms_last": round(roundtrip_timings_ms[-1], 3),
        "roundtrip_ms_mean_excl_first": round(float(np.mean(roundtrip_timings_ms[1:] or roundtrip_timings_ms)), 3),
        "policy_infer_ms_first": None if not policy_timings_ms else round(policy_timings_ms[0], 3),
        "policy_infer_ms_best": None if not policy_timings_ms else round(min(policy_timings_ms), 3),
        "policy_infer_ms_last": None if not policy_timings_ms else round(policy_timings_ms[-1], 3),
        "policy_infer_ms_mean_excl_first": (
            None if not policy_timings_ms else round(float(np.mean(policy_timings_ms[1:] or policy_timings_ms)), 3)
        ),
        "metadata": runner.metadata,
        "action_preview": [round(value, 6) for value in (action_preview or [])],
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
