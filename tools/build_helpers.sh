#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

"${OPENPI_PYTHON}" - <<'PY'
from support.gripper_control import build_tool_io_helper
from support.joint_control import build_joint_helper
from support.tcp_control import build_helper

print(f"Using Python: {__import__('sys').executable}")
print(f"tool_io_helper: {build_tool_io_helper()}")
print(f"joint_control_helper: {build_joint_helper()}")
print(f"tcp_control_helper: {build_helper()}")
PY
