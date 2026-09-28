#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${MIRROR_PROJECT_ROOT:-$(cd -- "${CODE_ROOT}/.." && pwd)}"
SOURCE_DERIVATIVES_DIR="${SOURCE_DERIVATIVES_DIR:-${DERIVATIVES_DIR:-}}"
RUN_DERIVATIVES_DIR="${RUN_DERIVATIVES_DIR:-}"
QSIPREP_CONTAINER="${QSIPREP_CONTAINER:-${PROJECT_ROOT}/apptainer/qsiprep-26.0.0.sif}"
APPTAINER="${APPTAINER:-apptainer}"
SPACE="${SPACE:-MNI152NLin2009cAsym}"
FORCE=0
DRY_RUN=0
SUBJECT_IDS=()
SESSION_IDS=()

usage() {
  cat <<'USAGE'
Usage: prepare_mni_ribbon_masks.sh [options]

Transform native-space sMRIPrep ribbon masks to MNI space using:
  apptainer exec ~/apptainer/qsiprep-26.0.0.sif antsApplyTransforms

Options:
  --project-root PATH        Project root. Default: parent of this code checkout
  --source-derivatives-dir PATH
                             Read-only source derivatives. Default: <project-root>/derivatives
  --derivatives-dir PATH     Deprecated alias for --source-derivatives-dir.
  --run-derivatives-dir PATH Run-specific output root. Required unless set in the environment.
  --qsiprep-container PATH   QSIPrep image. Default: <project-root>/apptainer/qsiprep-26.0.0.sif
  --subject-id ID            Subject to process, with or without sub-. May repeat.
  --session-id ID            Session to process, with or without ses-. May repeat.
  --force                    Regenerate existing MNI ribbon masks.
  --dry-run                  Print planned commands without running them.
  -h, --help                 Show this help.
USAGE
}

normalize_subject() {
  local value="$1"
  [[ "${value}" == sub-* ]] && printf '%s\n' "${value}" || printf 'sub-%s\n' "${value}"
}

normalize_session() {
  local value="$1"
  [[ "${value}" == ses-* ]] && printf '%s\n' "${value}" || printf 'ses-%s\n' "${value}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="$2"
      shift 2
      ;;
    --source-derivatives-dir|--derivatives-dir)
      SOURCE_DERIVATIVES_DIR="$2"
      shift 2
      ;;
    --run-derivatives-dir)
      RUN_DERIVATIVES_DIR="$2"
      shift 2
      ;;
    --qsiprep-container)
      QSIPREP_CONTAINER="$2"
      shift 2
      ;;
    --subject-id)
      SUBJECT_IDS+=("$(normalize_subject "$2")")
      shift 2
      ;;
    --session-id)
      SESSION_IDS+=("$(normalize_session "$2")")
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

PROJECT_ROOT="${PROJECT_ROOT/#\~/${HOME}}"
QSIPREP_CONTAINER="${QSIPREP_CONTAINER/#\~/${HOME}}"
if [[ -z "${SOURCE_DERIVATIVES_DIR}" ]]; then
  SOURCE_DERIVATIVES_DIR="${PROJECT_ROOT}/derivatives"
fi
SOURCE_DERIVATIVES_DIR="${SOURCE_DERIVATIVES_DIR/#\~/${HOME}}"
RUN_DERIVATIVES_DIR="${RUN_DERIVATIVES_DIR/#\~/${HOME}}"

if [[ -z "${RUN_DERIVATIVES_DIR}" ]]; then
  echo "Set --run-derivatives-dir to a run-specific output root." >&2
  exit 2
fi
if [[ "${RUN_DERIVATIVES_DIR}" == "${SOURCE_DERIVATIVES_DIR}" ]]; then
  echo "Run derivatives must differ from read-only source derivatives." >&2
  exit 2
fi
if [[ ! -d "${SOURCE_DERIVATIVES_DIR}/smriprep" ]]; then
  echo "sMRIPrep directory not found: ${SOURCE_DERIVATIVES_DIR}/smriprep" >&2
  exit 1
fi
if [[ ! -f "${QSIPREP_CONTAINER}" ]]; then
  echo "QSIPrep container not found: ${QSIPREP_CONTAINER}" >&2
  exit 1
fi
if ! command -v "${APPTAINER}" >/dev/null 2>&1; then
  echo "Apptainer executable not found on PATH: ${APPTAINER}" >&2
  exit 1
fi

shopt -s nullglob

preferred_match() {
  local -a candidates=("$@")
  local -a filtered=()
  local -a sorted=()
  local token path
  [[ ${#candidates[@]} -eq 0 ]] && return 1
  while IFS= read -r path; do
    sorted+=("${path}")
  done < <(printf '%s\n' "${candidates[@]}" | sort -u)
  candidates=("${sorted[@]}")
  for token in '_acq-MPRAGE_' '_rec-refaced_' '_run-01_'; do
    filtered=()
    for path in "${candidates[@]}"; do
      [[ "$(basename "${path}")" == *"${token}"* ]] && filtered+=("${path}")
    done
    if [[ ${#filtered[@]} -gt 0 ]]; then
      candidates=("${filtered[@]}")
    fi
  done
  printf '%s\n' "${candidates[0]}"
}

discover_subjects() {
  local -a dirs=("${SOURCE_DERIVATIVES_DIR}/smriprep"/sub-*)
  local dir
  for dir in "${dirs[@]}"; do
    [[ -d "${dir}" ]] || continue
    [[ "$(basename "${dir}")" == sub-PILOT* ]] && continue
    basename "${dir}"
  done | sort -u
}

discover_sessions() {
  local subject="$1"
  local subject_root="${SOURCE_DERIVATIVES_DIR}/smriprep/${subject}"
  local found=0
  local session_dir
  for session_dir in "${subject_root}"/ses-*; do
    [[ -d "${session_dir}" ]] || continue
    basename "${session_dir}"
    found=1
  done
  if [[ ${found} -eq 0 ]]; then
    echo "ses-01"
  fi
}

anat_dirs_for() {
  local subject="$1"
  local session="$2"
  printf '%s\n' \
    "${SOURCE_DERIVATIVES_DIR}/smriprep/${subject}/anat" \
    "${SOURCE_DERIVATIVES_DIR}/smriprep/${subject}/${session}/anat"
}

find_dseg() {
  local subject="$1"
  local session="$2"
  local anat_dir match path count
  local -a candidates=()
  while IFS= read -r anat_dir; do
    [[ -d "${anat_dir}" ]] || continue
    candidates=()
    count=0
    for path in "${anat_dir}/${subject}"*"_space-${SPACE}_dseg.nii"*; do
      candidates[$count]="${path}"
      count=$((count + 1))
    done
    [[ ${count} -eq 0 ]] && continue
    match="$(preferred_match "${candidates[@]}")" || true
    [[ -n "${match:-}" ]] && printf '%s\n' "${match}" && return 0
  done < <(anat_dirs_for "${subject}" "${session}")
  return 1
}

find_native_ribbon() {
  local subject="$1"
  local session="$2"
  local anat_dir path match count filtered_count
  local -a candidates=()
  local -a filtered=()
  while IFS= read -r anat_dir; do
    [[ -d "${anat_dir}" ]] || continue
    candidates=()
    count=0
    for path in "${anat_dir}/${subject}"*"_desc-ribbon_mask.nii"*; do
      candidates[$count]="${path}"
      count=$((count + 1))
    done
    [[ ${count} -eq 0 ]] && continue
    filtered=()
    filtered_count=0
    for path in "${candidates[@]}"; do
      if [[ "$(basename "${path}")" != *"_space-"* ]]; then
        filtered[$filtered_count]="${path}"
        filtered_count=$((filtered_count + 1))
      fi
    done
    [[ ${filtered_count} -eq 0 ]] && continue
    match="$(preferred_match "${filtered[@]}")" || true
    [[ -n "${match:-}" ]] && printf '%s\n' "${match}" && return 0
  done < <(anat_dirs_for "${subject}" "${session}")
  return 1
}

find_transform() {
  local native_ribbon="$1"
  local subject="$2"
  local anat_dir path count
  local -a candidates=()
  anat_dir="$(dirname "${native_ribbon}")"
  count=0
  for path in "${anat_dir}/${subject}"*"_from-T1w_to-${SPACE}_mode-image_xfm.h5"; do
    candidates[$count]="${path}"
    count=$((count + 1))
  done
  [[ ${count} -eq 0 ]] && return 1
  preferred_match "${candidates[@]}"
}

mni_ribbon_path() {
  local native_ribbon="$1"
  local subject="$2"
  local session="$3"
  local stem
  stem="$(basename "${native_ribbon%.nii.gz}")"
  if [[ "${stem}" == "$(basename "${native_ribbon}")" ]]; then
    stem="$(basename "${native_ribbon%.nii}")"
  fi
  stem="${stem%_desc-ribbon_mask}_space-${SPACE}_desc-ribbon_mask.nii.gz"
  printf '%s\n' \
    "${RUN_DERIVATIVES_DIR}/mni_ribbon_masks/${subject}/${session}/anat/${stem}"
}

find_existing_mni_ribbon() {
  local subject="$1"
  local session="$2"
  local anat_dir match path count
  local -a candidates=()
  anat_dir="${RUN_DERIVATIVES_DIR}/mni_ribbon_masks/${subject}/${session}/anat"
  [[ -d "${anat_dir}" ]] || return 1
  count=0
  for path in "${anat_dir}/${subject}"*"_space-${SPACE}_desc-ribbon_mask.nii"*; do
    candidates[$count]="${path}"
    count=$((count + 1))
  done
  [[ ${count} -eq 0 ]] && return 1
  match="$(preferred_match "${candidates[@]}")" || true
  [[ -n "${match:-}" ]] && printf '%s\n' "${match}" && return 0
  return 1
}

if [[ ${#SUBJECT_IDS[@]} -eq 0 ]]; then
  while IFS= read -r subject; do
    SUBJECT_IDS+=("${subject}")
  done < <(discover_subjects)
fi

created=0
existing_count=0
planned=0
missing=0

for subject in "${SUBJECT_IDS[@]}"; do
  if [[ ${#SESSION_IDS[@]} -gt 0 ]]; then
    sessions=("${SESSION_IDS[@]}")
  else
    sessions=()
    while IFS= read -r session; do
      sessions+=("${session}")
    done < <(discover_sessions "${subject}")
  fi
  for session in "${sessions[@]}"; do
    dseg="$(find_dseg "${subject}" "${session}")" || dseg=""
    native_ribbon="$(find_native_ribbon "${subject}" "${session}")" || native_ribbon=""
    existing="$(find_existing_mni_ribbon "${subject}" "${session}")" || existing=""
    if [[ -z "${dseg}" ]]; then
      echo "MISSING ${subject} ${session}: no sMRIPrep MNI dseg reference"
      missing=$((missing + 1))
      continue
    fi
    if [[ -z "${native_ribbon}" ]]; then
      echo "MISSING ${subject} ${session}: no native-space sMRIPrep ribbon"
      missing=$((missing + 1))
      continue
    fi
    output="$(mni_ribbon_path "${native_ribbon}" "${subject}" "${session}")"
    if [[ -n "${existing}" && ${FORCE} -eq 0 ]]; then
      echo "EXISTS ${subject} ${session}: ${existing}"
      existing_count=$((existing_count + 1))
      continue
    fi
    transform="$(find_transform "${native_ribbon}" "${subject}")" || transform=""
    if [[ -z "${transform}" ]]; then
      echo "MISSING ${subject} ${session}: no T1w-to-${SPACE} transform beside ${native_ribbon}"
      missing=$((missing + 1))
      continue
    fi
    temp_output="$(dirname "${output}")/.${RANDOM}.$(basename "${output}")"
    command=(
      "${APPTAINER}" exec "${QSIPREP_CONTAINER}" antsApplyTransforms
      --dimensionality 3
      --input "${native_ribbon}"
      --reference-image "${dseg}"
      --output "${temp_output}"
      --interpolation GenericLabel
      --transform "${transform}"
    )
    if [[ ${DRY_RUN} -eq 1 ]]; then
      printf 'PLAN %s %s:' "${subject}" "${session}"
      printf ' %q' "${command[@]}"
      printf '\n'
      planned=$((planned + 1))
      continue
    fi
    mkdir -p "$(dirname "${output}")"
    rm -f "${temp_output}"
    "${command[@]}"
    mv -f "${temp_output}" "${output}"
    echo "CREATED ${subject} ${session}: ${output}"
    created=$((created + 1))
  done
done

echo "Summary: created=${created}, existing=${existing_count}, planned=${planned}, missing=${missing}"
