#!/bin/bash
#SBATCH --job-name=smriprep
#SBATCH --output=logs/smriprep/%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=48:00:00
set -euo pipefail

#==============Shell script==============#
if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/processing/01_smri_dmri"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
BIDS_DIR="${MIRROR_BIDS_DIR}"
CODE_DIR="${SCRIPT_DIR}"
LOGS_DIR="${MIRROR_LOGS_DIR}"
SCRATCH_DIR="${MIRROR_WORK_DIR}/smriprep"
DERIVS_DIR="${MIRROR_SOURCE_DERIVATIVES}/smriprep"
mkdir -p "${SCRATCH_DIR}" "${DERIVS_DIR}" "${LOGS_DIR}/jobs"

# Make sure FS_LICENSE is defined in the container.
FS_LICENSE="${MIRROR_FREESURFER_LICENSE}"

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

# We'll run fMRIPrep in anat-only mode instead of sMRIPrep because we're more familiar with fMRIPrep.
# Only use the MPRAGE scan. Ignore the SPACE T1w and T2w.
cmd="apptainer run --home $HOME --cleanenv --containall --writable-tmpfs \
      -B ${BIDS_DIR}:/data \
      -B ${CODE_DIR}:/code \
      -B ${DERIVS_DIR}:/out \
      -B ${SCRATCH_DIR}:/work \
      -B ${FS_LICENSE}:/license.txt \
      ${MIRROR_APPTAINER_FMRIPREP} \
      /data \
      /out \
      participant \
      --participant-label ${subject} \
      -w /work \
      --omp-nthreads ${SLURM_CPUS_PER_TASK} \
      --nprocs ${SLURM_CPUS_PER_TASK} \
      --mem_mb 39000 \
      --output-spaces T1w MNI152NLin2009cAsym \
      --output-layout bids \
      --anat-only \
      --project-goodvoxels \
      --cifti-output 170k \
      --fs-license-file /license.txt \
      --skip_bids_validation \
      --level full \
      --longitudinal \
      --bids-filter-file /code/smriprep_filter.json \
      --ignore t2w"

# Setup done, run the command
echo Running task "${SLURM_ARRAY_TASK_ID}"
echo Commandline: "$cmd"
eval "$cmd"
exitcode=$?

# Output results to a table
echo "sub-$subject   ${SLURM_ARRAY_TASK_ID}    $exitcode" \
      >> "${LOGS_DIR}/jobs/${SLURM_JOB_NAME}.${SLURM_ARRAY_JOB_ID}.tsv"
echo Finished tasks "${SLURM_ARRAY_TASK_ID}" with exit code $exitcode
exit $exitcode
