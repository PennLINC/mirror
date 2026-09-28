#!/bin/bash
#SBATCH --job-name=warp_bundles
#SBATCH --output=logs/warp_bundles/%x-%A_%a.out
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
SCRIPT_DIR="${MIRROR_CODE_ROOT}/processing/03_registration_and_warping"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
BIDS_DIR="${MIRROR_BIDS_DIR}"
SOURCE_DERIVATIVES_ROOT="${MIRROR_SOURCE_DERIVATIVES}"
OUTPUT_DERIVATIVES_ROOT="${MIRROR_OUTPUT_DERIVATIVES}"
IN_ROOT="${SOURCE_DERIVATIVES_ROOT}/qsirecon/derivatives/qsirecon-MSMTAutoTrack"
OUT_ROOT="${OUTPUT_DERIVATIVES_ROOT}/warped_bundles"
XFM_ROOT="${OUTPUT_DERIVATIVES_ROOT}/t1w_registration"
TMP_ROOT="${MIRROR_WORK_DIR}/warp_bundles"
LOGS_DIR="${MIRROR_LOGS_DIR}"

mkdir -p "${LOGS_DIR}/jobs"
mkdir -p "${TMP_ROOT}"

# Parse participants.tsv line corresponding to this array index.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" "${BIDS_DIR}/participants.tsv" )

if [[ -z "${subject}" ]]; then
  echo "No subject found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

SUBJ_IN_DIR="${IN_ROOT}/sub-${subject}"
if [[ ! -d "${SUBJ_IN_DIR}" ]]; then
  echo "Input subject directory not found: ${SUBJ_IN_DIR}"
  exit 1
fi

echo "Running task ${SLURM_ARRAY_TASK_ID} for sub-${subject}"
overall_exitcode=0

shopt -s nullglob
session_dirs=( "${SUBJ_IN_DIR}"/ses-* )
shopt -u nullglob

if [[ ${#session_dirs[@]} -eq 0 ]]; then
  echo "No session directories found under ${SUBJ_IN_DIR}"
  exit 1
fi

for session_dir in "${session_dirs[@]}"; do
  session=$(basename "${session_dir}")
  in_dwi="${session_dir}/dwi"
  out_dwi="${OUT_ROOT}/sub-${subject}/${session}/dwi"
  acpc_to_t1w_xfm="${XFM_ROOT}/sub-${subject}/anat/sub-${subject}_from-ACPC_to-T1w_mode-image_xfm.h5"
  mkdir -p "${out_dwi}"

  if [[ ! -d "${in_dwi}" ]]; then
    echo "Skipping ${session}: missing dwi directory ${in_dwi}"
    continue
  fi

  if [[ ! -f "${acpc_to_t1w_xfm}" ]]; then
    echo "Skipping ${session}: missing transform ${acpc_to_t1w_xfm}"
    continue
  fi

  ref_candidates=( "${in_dwi}"/*space-ACPC*param-norm_dwimap.nii.gz )
  if [[ ${#ref_candidates[@]} -eq 0 ]]; then
    echo "Skipping ${session}: no ACPC HA reference found in ${in_dwi}"
    continue
  fi
  ref="${ref_candidates[0]}"

  bundle_files=( "${in_dwi}"/sub-*_bundle-*_streamlines.tck.gz )
  if [[ ${#bundle_files[@]} -eq 0 ]]; then
    echo "Skipping ${session}: no bundle streamlines found in ${in_dwi}"
    continue
  fi

  echo "Session ${session}"
  echo "Reference: ${ref}"

  for bundle_file in "${bundle_files[@]}"; do
    base=$(basename "${bundle_file}")
    bundle_name="${base##*_bundle-}"
    bundle_name="${bundle_name%_streamlines.tck.gz}"
    out_tck="${out_dwi}/sub-${subject}_${session}_acq-HBCD75_run-01_space-T1w_model-msmt_bundle-${bundle_name}_streamlines.tck"

    if [[ -f "${out_tck}" ]]; then
      echo "Output exists, skipping: ${out_tck}"
      continue
    fi

    tmp_dir="${TMP_ROOT}/sub-${subject}/${session}"
    mkdir -p "${tmp_dir}"
    tmp_trx="${tmp_dir}/${base%.tck.gz}.trx"
    tmp_warped_trx="${tmp_dir}/${base%.tck.gz}_space-T1w.trx"

    echo "Processing bundle ${bundle_name}"
    echo "Input: ${bundle_file}"
    echo "Output: ${out_tck}"

    # Convert input bundle tractogram to TRX.
    trxrs convert "${bundle_file}" "${tmp_trx}" || {
      overall_exitcode=$?
      rm -f "${tmp_trx}" "${tmp_warped_trx}"
      continue
    }

    # antsApplyTransformsToTRX uses point-transform direction conventions.
    # Use inverse of ACPC->T1w image transform for streamline vertices.
    antsApplyTransformsToTRX -i "${tmp_trx}" -o "${tmp_warped_trx}" -t "[${acpc_to_t1w_xfm},1]" || {
      overall_exitcode=$?
      rm -f "${tmp_trx}" "${tmp_warped_trx}"
      continue
    }

    # Convert warped TRX back to final TCK.
    trxrs convert "${tmp_warped_trx}" "${out_tck}" || {
      overall_exitcode=$?
      rm -f "${tmp_trx}" "${tmp_warped_trx}"
      continue
    }

    # Clean up intermediates.
    rm -f "${tmp_trx}" "${tmp_warped_trx}"
  done
done

echo "sub-${subject}   ${SLURM_ARRAY_TASK_ID}    ${overall_exitcode}" \
  >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo "Finished task ${SLURM_ARRAY_TASK_ID} with exit code ${overall_exitcode}"
exit ${overall_exitcode}
