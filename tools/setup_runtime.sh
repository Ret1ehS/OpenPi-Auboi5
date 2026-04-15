#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

echo "[runtime] Python: ${OPENPI_PYTHON}"
"${OPENPI_PYTHON}" "${SCRIPTS_ROOT}/tools/doctor.py" --section runtime
