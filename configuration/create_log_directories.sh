#!/usr/bin/env bash
# Create the code-root log directories required by the SBATCH launchers.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
LOGS_ROOT="${CODE_ROOT}/logs"

mkdir -p "${LOGS_ROOT}/jobs"

while IFS= read -r -d '' launcher; do
  job_name="$(sed -n 's/^#SBATCH --job-name=//p' "${launcher}" | head -n 1)"
  if [[ -z "${job_name}" ]]; then
    echo "Missing #SBATCH --job-name in ${launcher}" >&2
    exit 2
  fi
  mkdir -p "${LOGS_ROOT}/${job_name}"
done < <(
  find "${CODE_ROOT}/analysis" "${CODE_ROOT}/processing" \
    -type f -name '*.sbatch' -print0
)

echo "Created SBATCH log directories below ${LOGS_ROOT}"
