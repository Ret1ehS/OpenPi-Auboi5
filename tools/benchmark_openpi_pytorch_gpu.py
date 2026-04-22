#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

if __package__ in (None, ""):
    _PARENT = Path(__file__).resolve().parent.parent
    if str(_PARENT) not in sys.path:
        sys.path.insert(0, str(_PARENT))

from support.openpi_pytorch_policy import (
    OpenPIPyTorchPolicy,
    SyntheticObservationSpec,
    build_synthetic_observation,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark OpenPI PyTorch checkpoint on GPU.")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--prompt", type=str, default="pick up the blue cube")
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--compile-mode", type=str, default="none")
    args = parser.parse_args()

    compile_mode = None if args.compile_mode.strip().lower() in {"", "none", "null"} else args.compile_mode.strip()
    load_t0 = time.perf_counter()
    policy = OpenPIPyTorchPolicy(
        repo_root=args.repo_root,
        checkpoint_dir=args.checkpoint_dir,
        pytorch_device=args.device,
        sample_kwargs={"num_steps": int(args.num_steps)},
        compile_mode=compile_mode,
    )
    load_s = time.perf_counter() - load_t0
    obs = build_synthetic_observation(
        SyntheticObservationSpec(
            prompt=args.prompt,
            state_dim=int(args.state_dim),
            image_size=int(args.image_size),
        )
    )

    timings_ms: list[float] = []
    action_preview: list[float] | None = None
    for index in range(args.iterations):
        infer_t0 = time.perf_counter()
        result = policy.infer(obs)
        roundtrip_ms = (time.perf_counter() - infer_t0) * 1000.0
        policy_infer_ms = float(result["policy_timing"]["infer_ms"])
        timings_ms.append(policy_infer_ms)
        actions_np = np.asarray(result["actions"], dtype=np.float32)
        if action_preview is None:
            action_preview = actions_np[0].astype(float).tolist()
        print(
            json.dumps(
                {
                    "iteration": index,
                    "infer_ms": round(policy_infer_ms, 3),
                    "roundtrip_ms": round(roundtrip_ms, 3),
                    "action_shape": list(actions_np.shape),
                    "first_action": [round(value, 6) for value in actions_np[0].astype(float).tolist()],
                },
                ensure_ascii=False,
            )
        )

    summary = {
        "device": str(policy.metadata.get("pytorch_device", args.device)),
        "load_s": round(load_s, 3),
        "iterations": len(timings_ms),
        "num_steps": int(args.num_steps),
        "compile_mode": compile_mode,
        "infer_ms_first": round(timings_ms[0], 3),
        "infer_ms_best": round(min(timings_ms), 3),
        "infer_ms_last": round(timings_ms[-1], 3),
        "infer_ms_mean_excl_first": round(float(np.mean(timings_ms[1:] or timings_ms)), 3),
        "action_preview": [round(value, 6) for value in (action_preview or [])],
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
