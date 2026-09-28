#!/usr/bin/env bash
#
# reface_anatomicals.sh - Deface T1w and T2w anatomical images in a BIDS dataset
# ============================================================================
#
# DESCRIPTION:
#   This script processes T1w and T2w anatomical images in a BIDS dataset to
#   remove facial features for privacy protection. It uses AFNI's refacer for
#   T1w images and pydeface for T2w and MP2RAGE images. The script creates a SLURM array
#   job where each task processes one anatomical image.
#
# USAGE:
#   bash reface_anatomicals.sh BIDS_DIR LOG_DIR
#
# ARGUMENTS:
#   BIDS_DIR  - Path to the BIDS dataset directory (required)
#   LOG_DIR   - Path to store log files (required)
#
# EXAMPLES:
#   bash ./reface_anatomicals.sh /data/myproject/bids /data/myproject/logs/reface
#
# OUTPUTS:
#   - Creates defaced versions of anatomical images with "rec-refaced" (T1w) or "rec-defaced" (T2w and MP2RAGE) in the filename
#   - Removes original images after processing
#   - Logs are stored in LOG_DIR
#
# REQUIREMENTS:
#   - SLURM job scheduler
#   - AFNI (for T1w defacing)
#   - pydeface (for T2w and MP2RAGE defacing) installed in a micromamba environment
#
# NOTES:
#   - The script automatically determines the array size based on the number of files
#   - Only processes files that don't already have "rec-defaced" or "rec-refaced" in their name
#   - If performing on a datalad dataset, you will need to run a datalad save command after checking
#     outputs. e.g. `datalad save -d BIDS_DIR -m "Reface T1w images with afni_refacer_run and deface
#     T2w and MP2RAGE images with pydeface"`
#
# ============================================================================

# First argument is the BIDS root directory
bids_root="$1"

# Second argument is the base path for logs
log_base_path="$2"

# Ensure log directory exists
mkdir -p "${log_base_path}"

# Create a temporary file with the list of files to process
temp_file_list="${log_base_path}/anat_files_to_process.txt"

# Find files and save to the temporary file - using -L to follow symlinks
find -L "${bids_root}"/sub-* -type f \
  \( -name "*_T1w.nii.gz" -o -name "*_T2w.nii.gz" -o -name "*_inv-2*part-mag*MP2RAGE.nii.gz" \) \
  | grep -v -e "rec-defaced" -e "rec-refaced" | sort > "${temp_file_list}"

# Count the files
file_count=$(wc -l < "${temp_file_list}")

# Subtract 1 for zero-based array indexing
max_array=$((file_count - 1))

if [ $max_array -lt 0 ]; then
  echo "No files found to process. Exiting."
  exit 1
fi

echo "Found $file_count files to process. Setting array size to 0-$max_array."
echo "File list saved to: ${temp_file_list}"

# Submit the job with the calculated array size
sbatch --array=0-$max_array \
  --output="${log_base_path}/reface_%A_%a.out" \
  --error="${log_base_path}/reface_%A_%a.err" \
  --export=ALL,BIDS_ROOT="${bids_root}",FILE_LIST="${temp_file_list}" <<'SBATCH_SCRIPT'
#!/usr/bin/env bash
#SBATCH --job-name=reface
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G

set -eux

# Load AFNI (for T1w refacing)
module add afni/2022_05_03

# Get the BIDS directory from the environment variable
bids_root="${BIDS_ROOT}"
file_list="${FILE_LIST}"

echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "BIDS root: ${bids_root}"
echo "File list: ${file_list}"

# Check if file list exists
if [ ! -f "${file_list}" ]; then
  echo "ERROR: File list not found: ${file_list}"
  exit 1
fi

# Get the file for this array task
ANAT=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${file_list}")

# Check if we got a valid file
if [ -z "${ANAT}" ]; then
  echo "ERROR: No file found for index ${SLURM_ARRAY_TASK_ID}"
  echo "File list contents:"
  cat "${file_list}"
  exit 1
fi

echo "Processing file: ${ANAT}"

ANAT_DIR="$(dirname "$ANAT")"
ANAT_BASENAME="$(basename "$ANAT")"

# Move to the directory containing the file
cd "$ANAT_DIR" || { echo "ERROR: Could not change to directory: $ANAT_DIR"; exit 1; }

# Build the defaced filename with correct BIDS entity order (rec before run)
if [[ "$ANAT_BASENAME" == *"_T1w.nii.gz" ]]; then
  # Replace rec-norm with rec-refaced
  DEFACED_BASENAME="$(echo "$ANAT_BASENAME" | sed 's/_rec-norm/_rec-refaced/')"
elif [[ "$ANAT_BASENAME" == *"_T2w.nii.gz" ]]; then
  DEFACED_BASENAME="$(echo "$ANAT_BASENAME" | sed 's/_rec-norm/_rec-defaced/')"
elif [[ "$ANAT_BASENAME" == *"_MP2RAGE.nii.gz" ]]; then
  INV1_BASENAME="$(echo "$ANAT_BASENAME" | sed 's/_inv-2/_inv-1/')"
  DEFACED_BASENAME="$(echo "$ANAT_BASENAME" | sed 's/_rec-norm/_rec-defaced/')"
  DEFACED_INV1_BASENAME="$(echo "$INV1_BASENAME" | sed 's/_rec-norm/_rec-defaced/')"
else
  echo "ERROR: Unrecognized file type: $ANAT_BASENAME"
  exit 1
fi

echo "SLURM_ARRAY_TASK_ID:   $SLURM_ARRAY_TASK_ID"
echo "Anatomical directory:  $ANAT_DIR"
echo "Anatomical file:       $ANAT_BASENAME"
echo "Defaced file:          $DEFACED_BASENAME"

# Decide which defacing tool to use
if [[ "$ANAT_BASENAME" == *"_T1w.nii.gz" ]]; then
  echo "Using @afni_refacer_run (AFNI) for T1w"
  @afni_refacer_run \
    -input "$ANAT_BASENAME" \
    -mode_reface_plus \
    -prefix "$DEFACED_BASENAME"

elif [[ "$ANAT_BASENAME" == *"_T2w.nii.gz" ]]; then
  echo "Using pydeface for T2w"
  pydeface --outfile "$DEFACED_BASENAME" "$ANAT_BASENAME"

elif [[ "$ANAT_BASENAME" == *"_MP2RAGE.nii.gz" ]]; then
  echo "Using pydeface for MP2RAGE"
  pydeface "$ANAT_BASENAME" --outfile "$DEFACED_BASENAME" --applyto "$INV1_BASENAME"
  # The defaced INV1 filename will be appended with _defaced, so we need to find it and rename it to DEFACED_INV1_BASENAME
  # Remove .nii.gz from the filename
  INV1_BASENAME_BASE="${INV1_BASENAME%.nii.gz}"
  INV1_DEFACER_OUTPUT="${INV1_BASENAME_BASE}_defaced.nii.gz"
  mv "$INV1_DEFACER_OUTPUT" "$DEFACED_INV1_BASENAME"
  # Remove original INV1 file
  rm -f "$INV1_BASENAME"

fi

# Remove the original NIfTI file
rm -f "${ANAT_BASENAME}"

# Handle the JSON sidecar (if it exists)
JSON_BASENAME="${ANAT_BASENAME%.nii.gz}.json"
if [ -f "$JSON_BASENAME" ]; then
  # Use the same naming convention as the NIfTI file
  if [[ "$ANAT_BASENAME" == *"_T1w.nii.gz" ]]; then
    DEFACED_JSON_BASENAME="$(echo "$JSON_BASENAME" | sed 's/_rec-norm/_rec-refaced/')"
  elif [[ "$ANAT_BASENAME" == *"_T2w.nii.gz" ]]; then
    DEFACED_JSON_BASENAME="$(echo "$JSON_BASENAME" | sed 's/_rec-norm/_rec-defaced/')"
  elif [[ "$ANAT_BASENAME" == *"_MP2RAGE.nii.gz" ]]; then
    DEFACED_JSON_BASENAME="$(echo "$JSON_BASENAME" | sed 's/_rec-norm/_rec-defaced/')"
    INV1_JSON_BASENAME="$(echo "$JSON_BASENAME" | sed 's/_inv-2/_inv-1/')"
    DEFACED_INV1_JSON_BASENAME="$(echo "$INV1_JSON_BASENAME" | sed 's/_rec-norm/_rec-defaced/')"

    cp "$INV1_JSON_BASENAME" "$DEFACED_INV1_JSON_BASENAME"
    rm -f "$INV1_JSON_BASENAME"
  fi

  echo "JSON sidecar found:   $JSON_BASENAME"
  echo "Renaming to:          $DEFACED_JSON_BASENAME"

  # Copy the content to the new filename
  cp "${JSON_BASENAME}" "$DEFACED_JSON_BASENAME"

  # Remove the original JSON
  rm -f "${JSON_BASENAME}"
fi

# Clean up only if T1w (AFNI refacer leaves extra files)
if [[ "$ANAT_BASENAME" == *"_T1w.nii.gz" ]]; then
  rm -f *rec-refaced*face_plus*
  rm -rf *rec-refaced*_QC/
fi

echo "Done processing $ANAT_BASENAME"
SBATCH_SCRIPT

echo "Job submitted with:"
echo "  BIDS directory: $bids_root"
echo "  Log directory: $log_base_path"
echo "  File list: ${temp_file_list}"
echo "Remember to datalad save your changes after reviewing. e.g."
echo "datalad save -d BIDS_DIR -m "Reface T1w images with afni_refacer_run and deface T2w and MP2RAGE images with pydeface""
