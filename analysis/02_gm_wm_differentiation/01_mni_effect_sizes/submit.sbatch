#!/usr/bin/env bash
#SBATCH --job-name=mni_gmwm_effects
#SBATCH --output=logs/mni_gmwm_effects/%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=12gb
#SBATCH --time=24:00:00
set -euo pipefail

if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/analysis/02_gm_wm_differentiation/01_mni_effect_sizes"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
LOGS_DIR="${MIRROR_LOGS_DIR}"
ANALYSIS_SET="${ANALYSIS_SET:-primary}"
OUTPUT_DIR="${OUTPUT_DIR:-${MIRROR_OUTPUT_DERIVATIVES}/mni_gm_wm_effect_sizes}"
RECALCULATE="${RECALCULATE:-1}"

case "${ANALYSIS_SET}" in
  primary)
    analysis_sets=(primary)
    ;;
  full)
    analysis_sets=(primary full)
    ;;
  *)
    echo "ANALYSIS_SET must be primary or full; received: ${ANALYSIS_SET}" >&2
    exit 2
    ;;
esac

mkdir -p "${LOGS_DIR}/jobs"

for current_set in "${analysis_sets[@]}"; do
  subject_tsv="${OUTPUT_DIR}/mni_gm_wm_effect_sizes_${current_set}_subject.tsv"
  if [[ "${RECALCULATE}" == "1" || ! -s "${subject_tsv}" ]] \
    || ! head -n 1 "${subject_tsv}" | grep -q 'gm_tissue'; then
    "${PYTHON_BIN}" "${SCRIPT_DIR}/compute_mni_gm_wm_effect_sizes.py" \
      --project-root "${MIRROR_PROJECT_ROOT}" \
      --derivatives-dir "${MIRROR_SOURCE_DERIVATIVES}" \
      --run-derivatives-dir "${MIRROR_OUTPUT_DERIVATIVES}" \
      --analysis-set "${current_set}" \
      --output-dir "${OUTPUT_DIR}" \
      "$@"
  else
    echo "Using cached subject-level effect sizes: ${subject_tsv}"
    echo "Set RECALCULATE=1 to recompute voxelwise GM/WM effects."
  fi
done

exitcode=0
job_id="${SLURM_JOB_ID:-local}"
echo "all_subjects   ${job_id}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME:-mni_gmwm_effects}.${job_id}.tsv"
echo "Finished MNI GM/WM effect-size job with exit code ${exitcode}"
exit "${exitcode}"
