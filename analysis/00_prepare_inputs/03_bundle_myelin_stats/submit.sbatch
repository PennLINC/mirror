#!/bin/bash
#SBATCH --job-name=bundle_myelin_stats
#SBATCH --output=logs/bundle_myelin_stats/%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=24:00:00
set -euo pipefail

if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/analysis/00_prepare_inputs/03_bundle_myelin_stats"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
BIDS_DIR="${MIRROR_BIDS_DIR}"
LOGS_DIR="${MIRROR_LOGS_DIR}"

mkdir -p "${LOGS_DIR}/jobs"

# Prefer an explicit host override, then tckmap from the activated environment.
# Fall back to running the complete participant task in the QSIRecon container.
if [[ -z "${TCKMAP_BIN:-}" ]]; then
  if path_tckmap=$(command -v tckmap 2>/dev/null); then
    export TCKMAP_BIN="${path_tckmap}"
  fi
fi

if [[ -z "${TCKMAP_BIN:-}" ]]; then
  if [[ -z "${MIRROR_APPTAINER_QSIRECON:-}" || ! -f "${MIRROR_APPTAINER_QSIRECON}" ]]; then
    echo "QSIRecon image not found: ${MIRROR_APPTAINER_QSIRECON:-<not configured>}" >&2
    echo "Set apptainer.qsirecon in the selected MIRROR profile." >&2
    exit 1
  fi
  if [[ -n "${APPTAINER_BIN:-}" ]]; then
    if ! command -v "${APPTAINER_BIN}" >/dev/null 2>&1; then
      echo "APPTAINER_BIN is not executable: ${APPTAINER_BIN}" >&2
      exit 1
    fi
  elif command -v apptainer >/dev/null 2>&1; then
    export APPTAINER_BIN=apptainer
  elif command -v singularity >/dev/null 2>&1; then
    export APPTAINER_BIN=singularity
  else
    echo "Neither apptainer nor singularity is available on this compute node." >&2
    exit 1
  fi
fi

# Parse participants.tsv line corresponding to this array index.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" "${BIDS_DIR}/participants.tsv" )

if [[ -z "${subject}" ]]; then
  echo "No subject found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

python_args=(
  "${SCRIPT_DIR}/compute_bundle_myelin_stats.py"
  --subject-id "${subject}"
  --source-derivatives-dir "${MIRROR_SOURCE_DERIVATIVES}"
  --run-derivatives-dir "${MIRROR_OUTPUT_DERIVATIVES}"
  --patterns-file "${MIRROR_CODE_ROOT}/configuration/metrics.yml"
  --out-root "${MIRROR_OUTPUT_DERIVATIVES}/bundle_myelin_stats"
)

if [[ -n "${TCKMAP_BIN:-}" ]]; then
  # Use the activated host environment and resolved MRtrix binary.
  cmd=("${PYTHON_BIN}" "${python_args[@]}")
else
  # Default: enter QSIRecon once per subject. All tckmap calls then execute
  # directly inside this process namespace instead of starting one container
  # for every bundle.
  container_args=(
    exec
    --cleanenv
    --bind "${MIRROR_PROJECT_ROOT}:${MIRROR_PROJECT_ROOT}"
    --bind "${MIRROR_CODE_ROOT}:${MIRROR_CODE_ROOT}:ro"
    --pwd "${MIRROR_CODE_ROOT}"
    --env TCKMAP_BIN=tckmap
    --env PYTHONDONTWRITEBYTECODE=1
  )
  if [[ -n "${MIRROR_CONFIG:-}" ]]; then
    container_args+=(--env "MIRROR_CONFIG=${MIRROR_CONFIG}")
  fi
  cmd=(
    "${APPTAINER_BIN}"
    "${container_args[@]}"
    "${MIRROR_APPTAINER_QSIRECON}"
    python
    "${python_args[@]}"
  )
fi

echo "Running task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
printf 'Commandline:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
exitcode=$?

echo "sub-${subject}   ${SLURM_ARRAY_TASK_ID}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit ${exitcode}
