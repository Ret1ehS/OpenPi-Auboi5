#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <repo_python> <config_name> <jax_checkpoint_dir> [output_dir] [precision]" >&2
  echo "Example: $0 /home/niic/openpi/repo/.venv/bin/python pi05_aubo_agv_lora /home/niic/openpi/repo/checkpoints/pi05_aubo_agv_lora/my_eighth_run/29999 /home/niic/openpi/repo/checkpoints/pi05_aubo_agv_lora_pytorch/my_eighth_run/29999 float32" >&2
  exit 2
fi

repo_python="$1"
config_name="$2"
jax_checkpoint_dir="$3"
scripts_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
openpi_root="$(cd "${scripts_root}/.." && pwd)"
repo_root="${openpi_root}/repo"
precision="${5:-${OPENPI_CONVERT_PRECISION:-float32}}"

if [[ $# -ge 4 ]]; then
  output_dir="$4"
else
  rel_checkpoint="${jax_checkpoint_dir#${repo_root}/checkpoints/}"
  output_suffix="${config_name}_pytorch_lora_merged"
  if [[ "${precision}" != "bfloat16" ]]; then
    output_suffix="${output_suffix}_${precision}"
  fi
  if [[ "${rel_checkpoint}" != "${jax_checkpoint_dir}" ]]; then
    output_dir="${repo_root}/checkpoints/${output_suffix}/${rel_checkpoint#*/}"
  else
    output_dir="${jax_checkpoint_dir}_pytorch_lora_merged_${precision}"
  fi
fi

echo "repo_python=${repo_python}"
echo "config_name=${config_name}"
echo "jax_checkpoint_dir=${jax_checkpoint_dir}"
echo "output_dir=${output_dir}"
echo "precision=${precision}"

exec "${repo_python}" "${scripts_root}/tools/convert_openpi_checkpoint_to_pytorch.py" \
  --repo-root "${repo_root}" \
  --checkpoint_dir "${jax_checkpoint_dir}" \
  --config_name "${config_name}" \
  --output_path "${output_dir}" \
  --precision "${precision}"
