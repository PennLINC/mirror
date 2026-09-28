#!/bin/bash
#SBATCH --job-name=dkt_stats
#SBATCH --output=logs/dkt_stats/%x-%A_%a.out
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
SCRIPT_DIR="${MIRROR_CODE_ROOT}/analysis/00_prepare_inputs/02_dkt_parcel_stats"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
BIDS_DIR="${MIRROR_BIDS_DIR}"
LOGS_DIR="${MIRROR_LOGS_DIR}"

mkdir -p "${LOGS_DIR}/jobs"

# Parse participants.tsv line corresponding to this array index.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" "${BIDS_DIR}/participants.tsv" )

if [[ -z "${subject}" ]]; then
  echo "No subject found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

cmd="${PYTHON_BIN} ${SCRIPT_DIR}/compute_dkt_parcel_stats.py \
  --subject-id ${subject} \
  --source-derivatives-dir ${MIRROR_SOURCE_DERIVATIVES} \
  --run-derivatives-dir ${MIRROR_OUTPUT_DERIVATIVES} \
  --patterns-file ${MIRROR_CODE_ROOT}/configuration/metrics.yml"

echo "Running task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
echo "Commandline: ${cmd}"
eval "${cmd}"
exitcode=$?

echo "sub-${subject}   ${SLURM_ARRAY_TASK_ID}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit ${exitcode}
