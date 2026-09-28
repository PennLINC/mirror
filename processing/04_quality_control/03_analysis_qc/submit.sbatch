#!/bin/bash
#SBATCH --job-name=qc_report
#SBATCH --output=logs/qc_report/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=24:00:00
set -euo pipefail

if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/processing/04_quality_control/03_analysis_qc"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
PYTHON="${PYTHON:-${PYTHON_BIN}}"
OUTPUT="${OUTPUT:-${MIRROR_OUTPUT_DERIVATIVES}/qc_report/qc_report.pdf}"
WORK_DIR="${WORK_DIR:-${MIRROR_OUTPUT_DERIVATIVES}/qc_report/work}"

mkdir -p "$(dirname "${OUTPUT}")" "${WORK_DIR}"

# Keep package caches writable and limit libraries to the CPUs Slurm assigned.
export MPLCONFIGDIR="${WORK_DIR}/matplotlib"
export TMPDIR="${WORK_DIR}/tmp"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
mkdir -p "${MPLCONFIGDIR}" "${TMPDIR}"

PYTHON_REQUESTED="${PYTHON}"
if ! PYTHON="$(command -v "${PYTHON_REQUESTED}")"; then
  echo "Python executable not found or not executable: ${PYTHON_REQUESTED}" >&2
  exit 1
fi

SCRIPT="${SCRIPT_DIR}/generate_qc_report.py"
if [[ ! -f "${SCRIPT}" ]]; then
  echo "Report script not found: ${SCRIPT}" >&2
  exit 1
fi

echo "Host: $(hostname)"
echo "Started: $(date)"
echo "Python: ${PYTHON}"
echo "Output: ${OUTPUT}"
echo "Work directory: ${WORK_DIR}"
echo "Additional arguments: $*"

"${PYTHON}" -u "${SCRIPT}" \
  --project-root "${MIRROR_PROJECT_ROOT}" \
  --source-derivatives-root "${MIRROR_SOURCE_DERIVATIVES}" \
  --run-derivatives-root "${MIRROR_OUTPUT_DERIVATIVES}" \
  --output "${OUTPUT}" \
  --work-dir "${WORK_DIR}" \
  "$@"

echo "Finished: $(date)"
echo "Report: ${OUTPUT}"
