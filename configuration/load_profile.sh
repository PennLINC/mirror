#!/usr/bin/env bash
# Load the selected MIRROR profile into the current shell.

if [[ -z "${MIRROR_CODE_ROOT:-}" ]]; then
  echo 'MIRROR_CODE_ROOT must point to this repository before sourcing load_profile.sh.' >&2
  return 2
fi

_mirror_python="${PYTHON_BIN:-${PYTHON:-python}}"
_mirror_checkout="${MIRROR_CODE_ROOT}"
eval "$("${_mirror_python}" "${MIRROR_CODE_ROOT}/configuration/resolve_paths.py" --shell)"
export MIRROR_CODE_ROOT="${_mirror_checkout}"
unset _mirror_python _mirror_checkout
