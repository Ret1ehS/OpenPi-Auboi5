#!/usr/bin/env bash
set -euo pipefail

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_ROOT="$(cd "${TOOLS_DIR}/.." && pwd)"
OPENPI_ROOT="$(cd "${SCRIPTS_ROOT}/.." && pwd)"
export SCRIPTS_ROOT
export OPENPI_ROOT
export PYTHONPATH="${SCRIPTS_ROOT}:${PYTHONPATH:-}"

if [[ -n "${OPENPI_ENV_FILE:-}" ]]; then
  _env_file="${OPENPI_ENV_FILE}"
elif [[ -f "${SCRIPTS_ROOT}/config" ]]; then
  _env_file="${SCRIPTS_ROOT}/config"
else
  _env_file=""
fi

if [[ -n "${_env_file}" && -f "${_env_file}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${_env_file}"
  set +a
fi

if [[ -n "${OPENPI_RUNTIME_PYTHON:-}" ]]; then
  OPENPI_PYTHON="${OPENPI_RUNTIME_PYTHON}"
elif [[ -x "${OPENPI_ROOT}/repo/.venv/bin/python" ]]; then
  OPENPI_PYTHON="${OPENPI_ROOT}/repo/.venv/bin/python"
else
  OPENPI_PYTHON="python3"
fi

export OPENPI_PYTHON
