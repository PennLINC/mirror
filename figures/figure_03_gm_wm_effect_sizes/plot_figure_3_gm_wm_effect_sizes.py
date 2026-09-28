#!/usr/bin/env python3
"""Render primary GM/WM differentiation (Figure 3)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from figures._shared.gm_wm_effect_sizes import main as render_effect_sizes  # noqa: E402
from utils.paths import RUN_FIGURES_ROOT  # noqa: E402


def main(argv: Sequence[str] | None = None) -> None:
    """Render the primary metric set with Figure 3 defaults."""

    extra = list(sys.argv[1:] if argv is None else argv)
    render_effect_sizes(
        [
            '--analysis-set',
            'primary',
            '--gm-tissue',
            'cortical_gm',
            '--effect',
            'robust_median_d',
            '--output',
            str(RUN_FIGURES_ROOT / 'figure_03_gm_wm_effect_sizes' / 'Figure3'),
            *extra,
        ]
    )


if __name__ == '__main__':
    main()
