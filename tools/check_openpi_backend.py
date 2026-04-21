#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from support.load_policy import PolicyLoadSpec, create_local_policy


def main() -> int:
    runner = create_local_policy(PolicyLoadSpec())
    print("runner", type(runner).__name__)
    print("metadata", runner.metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
