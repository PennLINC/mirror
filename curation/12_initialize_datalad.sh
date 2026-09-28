#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"

# Add .heudiconv/ and sourcedata/ to the .gitignore file
grep -qxF '.heudiconv/' "${MIRROR_BIDS_DIR}/.gitignore" || echo '.heudiconv/' >> "${MIRROR_BIDS_DIR}/.gitignore"
grep -qxF 'sourcedata/' "${MIRROR_BIDS_DIR}/.gitignore" || echo 'sourcedata/' >> "${MIRROR_BIDS_DIR}/.gitignore"

# Create the datalad dataset after anonymizing anatomical images and metadata
datalad create --force -c text2git "${MIRROR_BIDS_DIR}"

# Save the datalad dataset
datalad save -d "${MIRROR_BIDS_DIR}" -m "Initial commit"
