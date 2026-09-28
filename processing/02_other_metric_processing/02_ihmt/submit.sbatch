#!/usr/bin/env bash
#SBATCH --job-name=ihmt
#SBATCH --output=logs/ihmt/%x-%A_%a.out
#SBATCH --array=1-26
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00

set -euo pipefail
if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/processing/02_other_metric_processing/02_ihmt"
exec bash "${SCRIPT_DIR}/../../run_subject_job.sh" "${SCRIPT_DIR}/process_ihmt.py"
