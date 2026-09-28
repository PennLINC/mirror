#!/usr/bin/env bash
# Run the BIDS validator at this stage (pre-CuBIDS)
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIRROR_CODE_ROOT="${MIRROR_CODE_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
source "${MIRROR_CODE_ROOT}/configuration/load_profile.sh"
deno run -ERWN jsr:@bids/validator "${MIRROR_BIDS_DIR}"
