#!/bin/bash
#SBATCH --job-name=prep_mni_ribbon
#SBATCH --output=logs/prep_mni_ribbon/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
set -euo pipefail

if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/analysis/00_prepare_inputs/01_mni_ribbon_masks"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
QSIPREP_CONTAINER="${QSIPREP_CONTAINER:-${MIRROR_APPTAINER_QSIPREP}}"

if [[ ! -f "${QSIPREP_CONTAINER}" ]]; then
  echo "QSIPrep container not found: ${QSIPREP_CONTAINER}" >&2
  exit 1
fi

bash "${SCRIPT_DIR}/prepare_mni_ribbon_masks.sh" \
  --project-root "${MIRROR_PROJECT_ROOT}" \
  --source-derivatives-dir "${MIRROR_SOURCE_DERIVATIVES}" \
  --run-derivatives-dir "${MIRROR_OUTPUT_DERIVATIVES}" \
  --qsiprep-container "${QSIPREP_CONTAINER}" \
  "$@"
