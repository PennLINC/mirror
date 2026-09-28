#!/usr/bin/env bash
# Run one or more subject-level Python processing scripts for a Slurm array row.

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 PROCESSING_SCRIPT [PROCESSING_SCRIPT ...]" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"

task_id="${SLURM_ARRAY_TASK_ID:-}"
if [[ -z "${task_id}" ]]; then
  echo 'SLURM_ARRAY_TASK_ID is required for subject-level jobs.' >&2
  exit 2
fi

subject="$(awk -F '\t' -v row="$((task_id + 1))" 'NR == row {sub(/^sub-/, "", $1); print $1}' "${MIRROR_BIDS_DIR}/participants.tsv")"
if [[ -z "${subject}" ]]; then
  echo "No participant found for array task ${task_id}." >&2
  exit 1
fi

mkdir -p "${MIRROR_LOGS_DIR}/jobs"
for processing_script in "$@"; do
  echo "Running $(basename "${processing_script}") for sub-${subject}"
  "${PYTHON_BIN}" -u "${processing_script}" --subject-id "${subject}"
done

job_id="${SLURM_ARRAY_JOB_ID:-local}"
printf 'sub-%s\t%s\t0\n' "${subject}" "${task_id}" \
  >> "${MIRROR_LOGS_DIR}/jobs/${SLURM_JOB_NAME:-processing}.${job_id}.tsv"
