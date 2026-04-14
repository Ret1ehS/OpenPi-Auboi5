#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
exec "${OPENPI_PYTHON}" "${SCRIPTS_ROOT}/main.py" "$@"
