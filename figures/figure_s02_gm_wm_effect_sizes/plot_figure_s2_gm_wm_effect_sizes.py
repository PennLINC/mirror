#!/usr/bin/env python3
"""Render expanded-set GM/WM differentiation (Supplementary Figure S2)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from figures._shared.gm_wm_effect_sizes import main as render_effect_sizes  # noqa: E402
from utils.paths import RUN_FIGURES_ROOT  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    """Render the expanded metric set with Figure S2 defaults."""

    extra = list(sys.argv[1:] if argv is None else argv)
    render_effect_sizes(
        [
            '--analysis-set',
            'full',
            '--gm-tissue',
            'cortical_gm',
            '--effect',
            'robust_median_d',
            '--output',
            str(RUN_FIGURES_ROOT / 'figure_s02_gm_wm_effect_sizes' / 'FigureS2'),
            *extra,
        ]
    )


if __name__ == '__main__':
    main()
