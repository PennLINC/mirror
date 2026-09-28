#!/bin/bash
#SBATCH --job-name=t1w_reg
#SBATCH --output=logs/t1w_reg/%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8gb
#SBATCH --time=12:00:00
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
QSIPREP_DIR="${SOURCE_DERIVATIVES_ROOT}/qsiprep"
SMRIPREP_DIR="${SOURCE_DERIVATIVES_ROOT}/smriprep"
OUT_ROOT="${OUTPUT_DERIVATIVES_ROOT}/t1w_registration"
TMP_ROOT="${MIRROR_WORK_DIR}/t1w_registration"
LOGS_DIR="${MIRROR_LOGS_DIR}"

mkdir -p "${OUT_ROOT}"
mkdir -p "${TMP_ROOT}"
mkdir -p "${LOGS_DIR}/jobs"

# Parse participants.tsv line corresponding to this array index.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" "${BIDS_DIR}/participants.tsv" )

if [[ -z "${subject}" ]]; then
  echo "No subject found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

FIXED="${QSIPREP_DIR}/sub-${subject}/anat/sub-${subject}_space-ACPC_desc-preproc_T1w.nii.gz"
FIXED_MASK="${QSIPREP_DIR}/sub-${subject}/anat/sub-${subject}_space-ACPC_desc-brain_mask.nii.gz"

MOVING="${SMRIPREP_DIR}/sub-${subject}/anat/sub-${subject}_acq-MPRAGE_rec-refaced_run-01_desc-preproc_T1w.nii.gz"
MOVING_MASK="${SMRIPREP_DIR}/sub-${subject}/anat/sub-${subject}_acq-MPRAGE_rec-refaced_run-01_desc-brain_mask.nii.gz"

if [[ ! -f "${MOVING}" ]]; then
  moving_candidates=( "${SMRIPREP_DIR}/sub-${subject}"/ses-*/anat/sub-${subject}_ses-*_acq-MPRAGE_rec-refaced_run-01_desc-preproc_T1w.nii.gz )
  if [[ ${#moving_candidates[@]} -gt 0 && -f "${moving_candidates[0]}" ]]; then
    MOVING="${moving_candidates[0]}"
    echo "Using session-level moving image: ${MOVING}"
  fi
fi

if [[ ! -f "${MOVING_MASK}" ]]; then
  moving_mask_candidates=( "${SMRIPREP_DIR}/sub-${subject}"/ses-*/anat/sub-${subject}_ses-*_acq-MPRAGE_rec-refaced_run-01_desc-brain_mask.nii.gz )
  if [[ ${#moving_mask_candidates[@]} -gt 0 && -f "${moving_mask_candidates[0]}" ]]; then
    MOVING_MASK="${moving_mask_candidates[0]}"
    echo "Using session-level moving mask: ${MOVING_MASK}"
  fi
fi

OUT_DIR="${OUT_ROOT}/sub-${subject}/anat"
mkdir -p "${OUT_DIR}"

TMP_SUBJ_DIR="${TMP_ROOT}/sub-${subject}"
mkdir -p "${TMP_SUBJ_DIR}"

WORK_PREFIX="${TMP_SUBJ_DIR}/sub-${subject}_acpc_to_t1w_"
WARPED_IMAGE="${TMP_SUBJ_DIR}/sub-${subject}_space-ACPC_desc-preproc_from-T1w_warped.nii.gz"
FIXED_MASKED="${TMP_SUBJ_DIR}/sub-${subject}_space-ACPC_desc-brainmasked_T1w.nii.gz"
MOVING_MASKED="${TMP_SUBJ_DIR}/sub-${subject}_space-T1w_desc-brainmasked_T1w.nii.gz"

echo "Resolved registration inputs:"
echo "  Fixed image:  ${FIXED}"
echo "  Fixed mask:   ${FIXED_MASK}"
echo "  Moving image: ${MOVING}"
echo "  Moving mask:  ${MOVING_MASK}"

if [[ ! -f "${FIXED}" ]]; then
  echo "Missing fixed image: ${FIXED}"
  exit 1
fi

if [[ ! -f "${MOVING}" ]]; then
  echo "Missing moving image: ${MOVING}"
  exit 1
fi

if [[ ! -f "${FIXED_MASK}" ]]; then
  echo "Missing fixed mask: ${FIXED_MASK}"
  exit 1
fi

if [[ ! -f "${MOVING_MASK}" ]]; then
  echo "Missing moving mask: ${MOVING_MASK}"
  exit 1
fi

if ! command -v ImageMath >/dev/null 2>&1; then
  echo "ImageMath is not available on PATH"
  exit 1
fi

echo "Creating brain-masked registration images:"
echo "  Masked fixed:  ${FIXED_MASKED}"
echo "  Masked moving: ${MOVING_MASKED}"
ImageMath 3 "${FIXED_MASKED}" m "${FIXED}" "${FIXED_MASK}"
ImageMath 3 "${MOVING_MASKED}" m "${MOVING}" "${MOVING_MASK}"

if [[ ! -f "${FIXED_MASKED}" ]]; then
  echo "Failed to create masked fixed image: ${FIXED_MASKED}"
  exit 1
fi

if [[ ! -f "${MOVING_MASKED}" ]]; then
  echo "Failed to create masked moving image: ${MOVING_MASKED}"
  exit 1
fi

cmd="antsRegistration \
  --collapse-output-transforms 1 \
  --dimensionality 3 \
  --float 0 \
  --initial-moving-transform [ ${FIXED_MASKED}, ${MOVING_MASKED}, 1 ] \
  --initialize-transforms-per-stage 0 \
  --interpolation BSpline \
  --output [ ${WORK_PREFIX}, ${WARPED_IMAGE} ] \
  --transform Rigid[ 0.1 ] \
  --metric Mattes[ ${FIXED_MASKED}, ${MOVING_MASKED}, 1, 32, Random, 0.25 ] \
  --convergence [ 1000x500x250x100, 1e-06, 10 ] \
  --smoothing-sigmas 3.0x2.0x1.0x0.0mm \
  --shrink-factors 8x4x2x1 \
  --use-histogram-matching 0 \
  --winsorize-image-intensities [ 0.002, 0.998 ] \
  --write-composite-transform 1"

echo "Running task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
echo "Commandline: ${cmd}"
eval "${cmd}"
exitcode=$?

if [[ ! -f "${WORK_PREFIX}Composite.h5" ]]; then
  echo "Expected output missing: ${WORK_PREFIX}Composite.h5"
  exit 1
fi

if [[ ! -f "${WORK_PREFIX}InverseComposite.h5" ]]; then
  echo "Expected output missing: ${WORK_PREFIX}InverseComposite.h5"
  exit 1
fi

# Fixed is ACPC and moving is native T1w.
# Composite maps T1w -> ACPC, InverseComposite maps ACPC -> T1w.
mv "${WORK_PREFIX}Composite.h5" \
  "${OUT_DIR}/sub-${subject}_from-T1w_to-ACPC_mode-image_xfm.h5"
mv "${WORK_PREFIX}InverseComposite.h5" \
  "${OUT_DIR}/sub-${subject}_from-ACPC_to-T1w_mode-image_xfm.h5"

echo "sub-${subject}   ${SLURM_ARRAY_TASK_ID}    ${exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${exitcode}"
exit ${exitcode}
