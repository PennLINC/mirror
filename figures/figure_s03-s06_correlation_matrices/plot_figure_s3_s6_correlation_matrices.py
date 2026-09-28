#!/usr/bin/env python3
"""Render Supplementary Figures S3-S6 with one command."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from figures._shared.correlation_matrices import main as render_correlations  # noqa: E402
from utils.paths import RUN_FIGURES_ROOT  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    """Write FigureS3 through FigureS6 into their combined artifact folder."""

    extra = list(sys.argv[1:] if argv is None else argv)
    output_dir = RUN_FIGURES_ROOT / 'figure_s03-s06_correlation_matrices'
    jobs = (
        ('FigureS3', 'wm', False),
        ('FigureS4', 'gm', False),
        ('FigureS5', 'wm', True),
        ('FigureS6', 'gm', True),
    )
    for output_name, tissue, regional in jobs:
        domain_args = (
            ['--skip-mni', '--parcel-correlation', 'pearson']
            if regional
            else ['--skip-parcel', '--mni-correlation', 'pearson']
        )
        render_correlations(
            [
                '--analysis-set',
                'full',
                '--tissue',
                tissue,
                '--parcel-stat',
                'median',
                '--strict',
                '--output-dir',
                str(output_dir),
                '--output-name',
                output_name,
                *domain_args,
                *extra,
            ]
        )


if __name__ == '__main__':
    main()
