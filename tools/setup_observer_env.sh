#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

echo "[observer] Runtime env loaded from: ${OPENPI_ENV_FILE:-auto}"
"${OPENPI_PYTHON}" "${SCRIPTS_ROOT}/tools/doctor.py" --section observer
