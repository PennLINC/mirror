#!/usr/bin/env bash
#SBATCH --job-name=g_ratio_aggregate
#SBATCH --output=logs/g_ratio_aggregate/%x-%j.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb
#SBATCH --time=02:00:00

set -euo pipefail
if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/processing/02_other_metric_processing/08_g_ratio"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
"${PYTHON_BIN}" -u "${SCRIPT_DIR}/aggregate_g_ratio_scaling_factors.py" "$@"
