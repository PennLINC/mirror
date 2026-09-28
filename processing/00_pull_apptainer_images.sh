#!/usr/bin/env bash
# Download the containers used by MIRROR preprocessing.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${MIRROR_PROJECT_ROOT:-$(cd -- "${CODE_ROOT}/.." && pwd)}"
APPTAINER_DIR="${MIRROR_APPTAINER_DIR:-${PROJECT_ROOT}/apptainer}"
APPTAINER_BIN="${APPTAINER_BIN:-apptainer}"

mkdir -p "${APPTAINER_DIR}"

"${APPTAINER_BIN}" pull "${APPTAINER_DIR}/qsiprep-26.0.0.sif" docker://pennlinc/qsiprep:26.0.0
"${APPTAINER_BIN}" pull "${APPTAINER_DIR}/qsirecon-26.0.0.sif" docker://pennlinc/qsirecon:26.0.0
"${APPTAINER_BIN}" pull "${APPTAINER_DIR}/fmriprep-25.0.0.sif" docker://nipreps/fmriprep:25.0.0
"${APPTAINER_BIN}" pull "${APPTAINER_DIR}/synthstrip-1.7.sif" docker://freesurfer/synthstrip:1.7
