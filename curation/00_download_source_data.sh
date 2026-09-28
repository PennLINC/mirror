#!/usr/bin/env bash
set -euo pipefail
unset LD_LIBRARY_PATH

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
FW_BIN="${FW_BIN:-fw}"

subjects=(
    "153-327"
    "24037_13298"
    "24037_13302"
    "60501_13333"
    "60501_13334"
    "60505_13327"
    "60505_13329"
    "60514_13330"
    "60514_13331"
    "60516_13324"
    "60516_13325"
    "60518_13313"
    "60518_13315"
    "60519_13320"
    "60519_13321"
    "60520_13322"
    "60520_13323"
    "60522_13307"
    "60522_13309"
    "60526_13316"
    "60526_13317"
    "60531_13336"
    "60531_13337"
    "24762_13340"
    "24762_13341"
    "60515_13343"
    "60515_13344"
    "60517_13350"
    "60517_13351"
    "60532_13352"
    "60532_13353"
    "22449_13354"
    "22449_13355"
    "60556_13361"
    "60556_13362"
    "60476_13377"
    "60476_13378"
    "60545_13356"
    "60545_13357"
    "60641_13393"
    "60641_13394"
    "60553_13387"
    "60553_13388"
    "60563_13385"
    "60563_13386"
    "techdev_dt_myelin"
    "techdev_human_myelin_02"
    "techdev_ihmt"
)

token="$(<"${MIRROR_PROJECT_ROOT}/tokens/flywheel.txt")"
#~/bin/glibc-2.34/lib/ld-linux-x86-64.so.2 ~/bin/linux_amd64/fw login "$token"
"${FW_BIN}" login "$token"
cd "${MIRROR_PROJECT_ROOT}/sourcedata" || exit

# Initialize download status file if it doesn't exist
download_status_file="${SCRIPT_DIR}/status/downloaded_subjects.txt"
touch "$download_status_file"

# Read already downloaded subjects into an array
if [[ -f "$download_status_file" ]]; then
    mapfile -t downloaded_subjects < "$download_status_file"
else
    downloaded_subjects=()
fi

for subject in "${subjects[@]}"; do
    # Check if subject is already downloaded
    if [[ " ${downloaded_subjects[*]} " =~ " ${subject} " ]]; then
        echo "Subject ${subject} already downloaded, skipping..."
        continue
    fi

    echo "Downloading subject ${subject}..."
    #~/bin/glibc-2.34/lib/ld-linux-x86-64.so.2 ~/bin/linux_amd64/fw download --yes --zip "fw://bbl/NIBS_857664/${subject}"
    if "${FW_BIN}" download --yes --zip "fw://bbl/NIBS_857664/${subject}"; then
        echo "Successfully downloaded ${subject}, adding to status file..."
        echo "${subject}" >> "$download_status_file"
    else
        echo "Failed to download ${subject}"
    fi
done
