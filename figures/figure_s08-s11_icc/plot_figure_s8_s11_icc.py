#!/usr/bin/env python3
"""Render Supplementary Figures S8-S11 with one command."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from figures._shared.supplemental_icc import main as render_icc  # noqa: E402
from utils.paths import RUN_FIGURES_ROOT  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    """Write FigureS8 through FigureS11 into their combined artifact folder."""

    extra = list(sys.argv[1:] if argv is None else argv)
    output_dir = RUN_FIGURES_ROOT / 'figure_s08-s11_icc'
    jobs = (
        ('FigureS8', 'wm', False),
        ('FigureS9', 'gm', False),
        ('FigureS10', 'wm', True),
        ('FigureS11', 'gm', True),
    )
    for output_name, tissue, regional in jobs:
        render_icc(
            [
                '--analysis-set',
                'full',
                '--tissue',
                tissue,
                '--strict',
                '--output-dir',
                str(output_dir),
                '--output-name',
                output_name,
                '--skip-voxelwise' if regional else '--skip-regional',
                *extra,
            ]
        )


if __name__ == '__main__':
    main()
