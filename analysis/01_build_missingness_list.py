#!/usr/bin/env python3
"""Build the modality-availability matrix consumed by Figure 2."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover - checked after argparse handles --help
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from configuration import load_config  # noqa: E402
from utils.paths import OUTPUT_DERIVATIVES_ROOT  # noqa: E402


PATTERNS = {
    'MPRAGE T1w': ('anat/*acq-MPRAGE*T1w.nii.gz',),
    'SPACE T1w': ('anat/*acq-SPACE*T1w.nii.gz',),
    'SPACE T2w': ('anat/*acq-SPACE*T2w.nii.gz',),
    'B1+': ('fmap/*TB1TFL.nii.gz',),
    'MP2RAGE': ('anat/*part-mag*MP2RAGE.nii.gz',),
    'ihMTRAGE': ('anat/*ihMTRAGE.nii.gz',),
    'dMRI': ('dwi/*dir-AP*dwi.nii.gz', 'dwi/*dir-PA*dwi.nii.gz'),
    'MESE': ('anat/*dir-AP*MESE.nii.gz',),
    'MEGRE': ('anat/*MEGRE.nii.gz',),
}
SESSIONS = {'Session 01': 'ses-01', 'Session 02': 'ses-02'}


def require_dependencies() -> None:
    """Raise an actionable error when the tabular dependency is unavailable."""

    if pd is None:
        raise RuntimeError(
            'Missing required Python package: pandas. '
            'Activate the MIRROR analysis environment first.'
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bids-dir', type=Path, default=Path(config['bids_dir']))
    parser.add_argument(
        '--output',
        type=Path,
        default=OUTPUT_DERIVATIVES_ROOT / 'missingness' / 'missingness_list.tsv',
    )
    return parser


def modality_status(session_dir: Path, modality: str) -> float:
    """Return 0, 0.5, or 1 for one modality/session."""

    matches_by_pattern = [sorted(session_dir.glob(pattern)) for pattern in PATTERNS[modality]]
    if any(not matches for matches in matches_by_pattern):
        return 0.0
    first_file = matches_by_pattern[0][0]
    if modality == 'MP2RAGE':
        return 1.0 if Path(str(first_file).replace('part-mag', 'part-phase')).is_file() else 0.5
    if modality == 'MESE':
        return 1.0 if Path(str(first_file).replace('dir-AP', 'dir-PA')).is_file() else 0.5
    return 1.0


def main() -> None:
    """Build and save the availability matrix."""

    args = build_parser().parse_args()
    require_dependencies()
    bids_dir = args.bids_dir.expanduser().resolve()
    participants = pd.read_table(bids_dir / 'participants.tsv')
    subject_ids = participants['participant_id'].astype(str).tolist()
    subject_ids = [sid for sid in subject_ids if 'PILOT' in sid] + [
        sid for sid in subject_ids if 'PILOT' not in sid
    ]
    columns = [f'{session}--{modality}' for session in SESSIONS for modality in PATTERNS]
    table = pd.DataFrame(index=subject_ids, columns=columns, dtype=float)
    for subject_id in subject_ids:
        for session_name, session_id in SESSIONS.items():
            session_dir = bids_dir / subject_id / session_id
            for modality in PATTERNS:
                table.loc[subject_id, f'{session_name}--{modality}'] = modality_status(
                    session_dir,
                    modality,
                )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, sep='\t', index=True, index_label='participant_id')
    print(f'Wrote {output}', flush=True)


if __name__ == '__main__':
    main()
