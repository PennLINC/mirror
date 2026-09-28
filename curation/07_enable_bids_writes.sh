#!/usr/bin/env bash
# Ensure the following curation steps can update BIDS JSON sidecars.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
chmod -R +w "${MIRROR_BIDS_DIR}"
