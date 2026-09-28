#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=48:00:00

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
SCITRAN_ROOT="${MIRROR_PROJECT_ROOT}/sourcedata/scitran/bbl/NIBS_857664"

# Run heudiconv on the first session
sessions="01 02"

# Loop over sessions
for session in $sessions; do
    # Find subjects with the requested session
    subjects=($(ls -d "${SCITRAN_ROOT}"/*_"$session" 2>/dev/null | sed 's|.*/\([0-9a-zA-Z]*\)_.*|\1|' | sort -u))

    if [ ${#subjects[@]} -eq 0 ]; then
        echo "No subjects found for session $session"
        continue
    fi

    echo "Found ${#subjects[@]} subjects for session $session: ${subjects[*]}"

    # Filter out already-converted subjects
    subjects=($(for s in "${subjects[@]}"; do [ ! -d "${MIRROR_BIDS_DIR}/sub-$s/ses-$session" ] && echo "$s"; done))

    for sub in "${subjects[@]}"
    do
        echo "$sub"
        heudiconv \
            -f "${SCRIPT_DIR}/heuristic.py" \
            -o "${MIRROR_BIDS_DIR}" \
            -d "${SCITRAN_ROOT}/{subject}_{session}/*/*/*/*.dcm" \
            --subjects "$sub" \
            --ses "$session" \
            -g all \
            --bids \
            --queue SLURM
    done
done
