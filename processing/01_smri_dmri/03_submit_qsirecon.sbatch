#!/bin/bash
#SBATCH --job-name=qsirecon
#SBATCH --output=logs/qsirecon/%x-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-26
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=4gb
#SBATCH --time=36:00:00
set -euo pipefail

#==============Shell script==============#
# Location of the *reduced* BIDS dataset
if [[ ! -f "${MIRROR_CODE_ROOT:-}/configuration/load_profile.sh" ]]; then
  export MIRROR_CODE_ROOT="$(git -C "${SLURM_SUBMIT_DIR:-$PWD}" rev-parse --show-toplevel)"
fi
SCRIPT_DIR="${MIRROR_CODE_ROOT}/processing/01_smri_dmri"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
LOGS_DIR="${MIRROR_LOGS_DIR}"
CODE_DIR="${SCRIPT_DIR}"
BIDS_DIR="${MIRROR_BIDS_DIR}"
SCRATCH_DIR="${MIRROR_WORK_DIR}/qsirecon"
DERIV_DIR="${MIRROR_SOURCE_DERIVATIVES}"
mkdir -p "${SCRATCH_DIR}" "${LOGS_DIR}/jobs"

# Make sure FS_LICENSE is defined in the container.
FS_LICENSE="${MIRROR_FREESURFER_LICENSE}"

# Parse the participants.tsv file and extract one subject ID from the line corresponding to this SLURM task.
subject=$( sed -n -E "$((${SLURM_ARRAY_TASK_ID} + 1))s/sub-(\S*)\>.*/\1/gp" ${BIDS_DIR}/participants.tsv )

cmd="apptainer run --cleanenv \
      --env DIPY_HOME=/home/qsirecon/.dipy \
      -B ${CODE_DIR}:/code \
      -B ${DERIV_DIR}:/data \
      -B ${SCRATCH_DIR}:/work \
      -B ${FS_LICENSE}:/license.txt \
      ${MIRROR_APPTAINER_QSIRECON} \
      /data/qsiprep \
      /data/qsirecon \
      participant \
      --participant-label ${subject} \
      -w /work/ \
      --omp-nthreads ${SLURM_CPUS_PER_TASK} \
      --nprocs ${SLURM_CPUS_PER_TASK} \
      --mem 79000 \
      --fs-license-file /license.txt \
      --output-resolution 1.7 \
      --recon-spec /code/qsirecon_spec.yml \
      --report-output-level session"

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
