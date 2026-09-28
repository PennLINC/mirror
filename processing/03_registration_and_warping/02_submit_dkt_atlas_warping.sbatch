#!/bin/bash
#SBATCH --job-name=dkt_acpc
#SBATCH --output=logs/dkt_acpc/%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=06:00:00
set -euo pipefail

if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/processing/03_registration_and_warping"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
BIDS_DIR="${MIRROR_BIDS_DIR}"
SOURCE_DERIVATIVES_ROOT="${MIRROR_SOURCE_DERIVATIVES}"
OUTPUT_DERIVATIVES_ROOT="${MIRROR_OUTPUT_DERIVATIVES}"
SMRIPREP_DIR="${SOURCE_DERIVATIVES_ROOT}/smriprep"
QSIPREP_DIR="${SOURCE_DERIVATIVES_ROOT}/qsiprep"
XFM_DIR="${OUTPUT_DERIVATIVES_ROOT}/t1w_registration"
LOGS_DIR="${MIRROR_LOGS_DIR}"

mkdir -p "${LOGS_DIR}/jobs"

# Parse participants.tsv line corresponding to this array index.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" "${BIDS_DIR}/participants.tsv" )

if [[ -z "${subject}" ]]; then
  echo "No subject found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

FS_MGZ="${SMRIPREP_DIR}/sourcedata/freesurfer/sub-${subject}/mri/aparc.DKTatlas+aseg.mgz"
OUT_DIR="${XFM_DIR}/sub-${subject}/anat"
OUT_T1W_DSEG="${OUT_DIR}/sub-${subject}_space-T1w_desc-DKTatlas_dseg.nii.gz"
OUT_ACPC_DSEG="${OUT_DIR}/sub-${subject}_space-ACPC_desc-DKTatlas_dseg.nii.gz"
REF_ACPC="${QSIPREP_DIR}/sub-${subject}/anat/sub-${subject}_space-ACPC_desc-preproc_T1w.nii.gz"
T1W_TO_ACPC_XFM="${XFM_DIR}/sub-${subject}/anat/sub-${subject}_from-T1w_to-ACPC_mode-image_xfm.h5"

mkdir -p "${OUT_DIR}"

if [[ ! -f "${FS_MGZ}" ]]; then
  echo "Missing FreeSurfer segmentation: ${FS_MGZ}"
  exit 1
fi

if [[ ! -f "${REF_ACPC}" ]]; then
  echo "Missing ACPC reference image: ${REF_ACPC}"
  exit 1
fi

if [[ ! -f "${T1W_TO_ACPC_XFM}" ]]; then
  echo "Missing T1w->ACPC transform: ${T1W_TO_ACPC_XFM}"
  exit 1
fi

cmd_convert="mri_convert \"${FS_MGZ}\" \"${OUT_T1W_DSEG}\""
cmd_apply="antsApplyTransforms \
  --dimensionality 3 \
  --input \"${OUT_T1W_DSEG}\" \
  --reference-image \"${REF_ACPC}\" \
  --output \"${OUT_ACPC_DSEG}\" \
  --transform \"${T1W_TO_ACPC_XFM}\" \
  --interpolation genericLabel"

echo "Running task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
echo "Convert command: ${cmd_convert}"
eval "${cmd_convert}"

echo "Transform command: ${cmd_apply}"
eval "${cmd_apply}"
exitcode=$?

echo "sub-${subject}   ${SLURM_ARRAY_TASK_ID}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit ${exitcode}
