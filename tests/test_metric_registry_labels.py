"""Tests for publication-facing metric labels."""

import pytest

from utils.metrics import metric_plot_label, selected_analysis_sets


@pytest.mark.parametrize(
    ('source', 'expected'),
    [
        ('PAth', 'PAθ'),
        ('NG Parallel', 'NG ∥'),
        ('NG (Parallel)', 'NG ∥'),
        ('NG (Perpendicular)', 'NG⊥'),
        ('FA (DSIStudio)', 'FA (DSI Studio)'),
        ('Q-Ratio-E5-B1c', r'$\it{q}$-Ratio-E5-B₁c'),
        ('q-Ratio-E4', r'$\it{q}$-Ratio-E4'),
        ('G-Ratio', r'$\it{g}$-Ratio'),
        ('G-ihMTsat', r'$\it{g}$-ihMTsat'),
        ('G-ihMTR', r'$\it{g}$-ihMTR'),
        ('QSM-X-R2pnet-E4-Para', 'QSM-χ-R₂pnet-E4-Para'),
        ('QSM-X-R2*-E5-Dia', 'QSM-χ-R₂*-E5-Dia'),
    ],
)
def test_metric_plot_label_symbols(source, expected):
    assert metric_plot_label(source) == expected


@pytest.mark.parametrize(
    ('requested', 'expected'),
    [
        ('primary', ('primary',)),
        ('full', ('primary', 'full')),
    ],
)
def test_selected_analysis_sets(requested, expected):
    assert selected_analysis_sets(requested) == expected


def test_selected_analysis_sets_rejects_unknown_value():
    with pytest.raises(ValueError, match='Unsupported metric set'):
        selected_analysis_sets('unknown')
