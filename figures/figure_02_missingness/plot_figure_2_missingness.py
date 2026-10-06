#!/usr/bin/env python3
"""Plot acquisition availability and manual-QC exclusions (Figure 2)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

try:
    import pandas as pd
    from matplotlib.colors import to_rgb
except ImportError:  # pragma: no cover - checked after argparse handles --help
    pd = None
    to_rgb = None

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import SOURCE_IMAGE_COLORS  # noqa: E402
from utils.paths import CODE_ROOT, OUTPUT_DERIVATIVES_ROOT, RUN_FIGURES_ROOT  # noqa: E402


LABEL_REPLACEMENTS = {
    'T1w': 'T₁w',
    'T2w': 'T₂w',
    'B1+': 'B₁⁺',
}

MODALITY_COLORS = {
    'MPRAGE T1w': SOURCE_IMAGE_COLORS['T1w/T2w'],
    'SPACE T1w': SOURCE_IMAGE_COLORS['T1w/T2w'],
    'SPACE T2w': SOURCE_IMAGE_COLORS['T1w/T2w'],
    'B1+': '#000000',
    'MP2RAGE': SOURCE_IMAGE_COLORS['R1'],
    'ihMTRAGE': SOURCE_IMAGE_COLORS['ihMT'],
    'dMRI': SOURCE_IMAGE_COLORS['dMRI'],
    'MESE': SOURCE_IMAGE_COLORS['MESE'],
    'MEGRE': SOURCE_IMAGE_COLORS['MEGRE'],
}
SESSION_DIVIDER_WIDTH = 2.4


def default_missingness_input() -> Path:
    """Prefer the run-specific table, with the checked-in legacy table as fallback."""

    run_table = OUTPUT_DERIVATIVES_ROOT / 'missingness' / 'missingness_list.tsv'
    legacy_table = CODE_ROOT / 'data' / 'qc' / 'missingness_list.tsv'
    return run_table if run_table.exists() or not legacy_table.exists() else legacy_table


def require_dependencies() -> None:
    """Raise an actionable error when plotting dependencies are unavailable."""

    missing = []
    if pd is None:
        missing.append('pandas')
    if to_rgb is None:
        missing.append('matplotlib')
    if missing:
        raise RuntimeError(
            'Missing required Python packages: '
            f'{", ".join(missing)}. Activate the MIRROR analysis environment first.'
        )


def relabel(column):
    """Replace acquisition abbreviations with their subscripted forms."""
    return next(
        (column.replace(old, new) for old, new in LABEL_REPLACEMENTS.items() if old in column),
        column,
    )


def modality_from_column(column):
    return column.split('--', 1)[1] if '--' in column else column


def palette_for_columns(columns):
    return [
        to_rgb(MODALITY_COLORS.get(modality_from_column(column), SOURCE_IMAGE_COLORS['Other']))
        for column in columns
    ]


def session_boundary(columns, left_session='Session 01', right_session='Session 02'):
    left_count = sum(column.startswith(left_session) for column in columns)
    if left_count == 0 or left_count == len(columns):
        return None
    if not all(column.startswith(left_session) for column in columns[:left_count]):
        return None
    if not all(column.startswith(right_session) for column in columns[left_count:]):
        return None
    return left_count - 0.5


def build_parser() -> argparse.ArgumentParser:
    """Build the Figure 2 command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--input',
        type=Path,
        default=default_missingness_input(),
        help='Acquisition availability table produced by build_missingness_list.py.',
    )
    parser.add_argument(
        '--qc-file',
        type=Path,
        default=CODE_ROOT / 'data' / 'qc' / 'manual_qc_modality.tsv',
        help='Manual modality-level QC table.',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=RUN_FIGURES_ROOT / 'figure_02_missingness' / 'Figure2',
        help='Output stem; both PNG and PDF are written.',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Render Figure 2 from availability and QC tables."""

    args = build_parser().parse_args(argv)
    require_dependencies()
    from utils.analysis import convert_to_multindex, matrix

    df = pd.read_table(args.input.expanduser().resolve(), index_col='participant_id')
    df = df.fillna(0)
    # Legacy tables store MP2RAGE magnitude and phase as separate columns;
    # current tables encode the same status directly in MP2RAGE as 0, 0.5, or 1.
    for session in ('Session 01', 'Session 02'):
        mp2rage = f'{session}--MP2RAGE'
        phase = f'{session}--MP2RAGE-P'
        if phase in df.columns:
            df[mp2rage] = df[[mp2rage, phase]].mean(axis=1)
    columns = df.columns.tolist()
    columns = [c for c in columns if not c.endswith('MP2RAGE-P')]
    # Preserve the acquisition order used in the manuscript.
    columns = [
        'Session 01--MPRAGE T1w',
        'Session 01--SPACE T1w',
        'Session 01--SPACE T2w',
        'Session 01--B1+',
        'Session 01--MP2RAGE',
        'Session 01--ihMTRAGE',
        'Session 01--dMRI',
        'Session 01--MESE',
        'Session 01--MEGRE',
        'Session 02--MPRAGE T1w',
        'Session 02--SPACE T1w',
        'Session 02--SPACE T2w',
        'Session 02--B1+',
        'Session 02--MP2RAGE',
        'Session 02--ihMTRAGE',
        'Session 02--dMRI',
        'Session 02--MESE',
        'Session 02--MEGRE',
    ]
    pal = palette_for_columns(columns)
    df = df[columns]

    # Ratings of 'n/a' parse as NaN and so compare False. Columns without a QC counterpart
    # (e.g., G-Ratio) and subjects absent from the QC table are never grayed out.
    qc_df = pd.read_table(args.qc_file.expanduser().resolve(), index_col='participant_id')
    excluded = (qc_df == 0).reindex(index=df.index, columns=df.columns, fill_value=False)
    qc_usable_acquisition_count = df.where(~excluded, 0).sum(axis=1)

    df = df.rename(columns=relabel)
    excluded = excluded.rename(columns=relabel)
    subjects = df.index.tolist()
    pilot_subjects = [subj for subj in subjects if subj.startswith('sub-PILOT')]
    other_subjects = [subj for subj in subjects if not subj.startswith('sub-PILOT')]
    subjects = pilot_subjects + other_subjects
    df = df.loc[subjects]
    excluded = excluded.loc[subjects]
    qc_usable_acquisition_count = qc_usable_acquisition_count.loc[subjects]
    df = convert_to_multindex(df)
    excluded = convert_to_multindex(excluded)
    ax = matrix(
        df,
        palette=pal,
        excluded=excluded,
        sparkline_values=qc_usable_acquisition_count,
    )
    boundary = session_boundary(columns)
    if boundary is not None:
        ax.axvline(
            boundary, color='black', linewidth=SESSION_DIVIDER_WIDTH, zorder=5, clip_on=False
        )
    output_stem = args.output.expanduser().resolve()
    if output_stem.suffix.lower() in {'.png', '.pdf'}:
        output_stem = output_stem.with_suffix('')
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_stem.with_suffix('.png'), bbox_inches='tight', dpi=400)
    ax.figure.savefig(output_stem.with_suffix('.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    main()
