#!/usr/bin/env bash
#SBATCH --job-name=coreg_qc
#SBATCH --output=logs/coreg_qc/%x-%A_%a.out
#SBATCH --array=1-26
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=12:00:00

set -euo pipefail
if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/processing/04_quality_control/01_coregistration_reports"
exec bash "${SCRIPT_DIR}/../../run_subject_job.sh" \
  "${SCRIPT_DIR}/generate_coregistration_reports.py"
