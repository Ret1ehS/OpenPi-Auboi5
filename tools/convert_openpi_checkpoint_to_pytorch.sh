#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <repo_python> <config_name> <jax_checkpoint_dir> [output_dir]" >&2
  echo "Example: $0 /home/niic/openpi/repo/.venv/bin/python pi05_aubo_agv_lora /home/niic/openpi/repo/checkpoints/pi05_aubo_agv_lora/my_eighth_run/29999 /home/niic/openpi/repo/checkpoints/pi05_aubo_agv_lora_pytorch/my_eighth_run/29999" >&2
  exit 2
fi

repo_python="$1"
config_name="$2"
jax_checkpoint_dir="$3"
output_dir="${4:-${jax_checkpoint_dir}_pytorch}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../repo" && pwd)"

echo "repo_python=${repo_python}"
echo "config_name=${config_name}"
echo "jax_checkpoint_dir=${jax_checkpoint_dir}"
echo "output_dir=${output_dir}"

exec "${repo_python}" "${repo_root}/examples/convert_jax_model_to_pytorch.py" \
  --checkpoint_dir "${jax_checkpoint_dir}" \
  --config_name "${config_name}" \
  --output_path "${output_dir}"
