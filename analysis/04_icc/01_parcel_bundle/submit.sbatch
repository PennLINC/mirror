#!/bin/bash
#SBATCH --job-name=parcel_icc
#SBATCH --output=logs/parcel_icc/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=24:00:00
set -euo pipefail

if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/analysis/04_icc/01_parcel_bundle"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
PYTHON="${PYTHON:-${PYTHON_BIN}}"
ANALYSIS_SET="${ANALYSIS_SET:-primary}"

"${PYTHON}" -u "${SCRIPT_DIR}/compute_parcel_bundle_icc.py" \
  --patterns-file "${MIRROR_CODE_ROOT}/configuration/metrics.yml" \
  --qc-file "${MIRROR_CODE_ROOT}/data/qc/manual_qc_modality.tsv" \
  --outdir "${MIRROR_OUTPUT_DERIVATIVES}/parcel_bundle_icc" \
  --analysis both \
  --analysis-set "${ANALYSIS_SET}" \
  --qc-mode metricqc \
  --stat median \
  --prefer-masked
