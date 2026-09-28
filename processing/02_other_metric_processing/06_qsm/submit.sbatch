#!/usr/bin/env bash
#SBATCH --job-name=qsm
#SBATCH --output=logs/qsm/%x-%A_%a.out
#SBATCH --array=1-26
#SBATCH --cpus-per-task=4
#SBATCH --mem=48gb
#SBATCH --time=48:00:00

set -euo pipefail
if command -v module >/dev/null 2>&1; then
  module load "${MATLAB_MODULE:-matlab/R2023A}"
fi
if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/processing/02_other_metric_processing/06_qsm"
exec bash "${SCRIPT_DIR}/../../run_subject_job.sh" \
  "${SCRIPT_DIR}/process_qsm.py" \
  "${SCRIPT_DIR}/process_qsm_post.py"
