"""Tests for the angular functions and model builders in run_b0_analysis."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import b0_tractometry as bt  # noqa: E402
import run_b0_analysis as ra  # noqa: E402


def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _node_moments(rng, n_nodes=6, n_points=40, spread=0.2):
    """Moments (n_nodes, 21) of dispersed directions, and the directions and weights."""
    mean_dirs = _unit(rng.normal(size=(n_nodes, 3)))
    dirs = _unit(mean_dirs[None] + spread * rng.normal(size=(n_points, n_nodes, 3)))
    w = rng.random((n_points, n_nodes))
    w /= w.sum(0)
    mom = np.column_stack([bt.weighted_scatter(dirs, w), bt.weighted_moment4(dirs, w)])
    return mom, dirs, w


def test_angular_matches_direct_means():
    rng = np.random.default_rng(0)
    mom, dirs, w = _node_moments(rng)
    b = np.broadcast_to(_unit(rng.normal(size=3)), (mom.shape[0], 3))
    sin2 = 1 - np.einsum('sni,ni->sn', dirs, b) ** 2
    np.testing.assert_allclose(ra.angular(mom, b, 'sin2'), (w * sin2).sum(0))
    np.testing.assert_allclose(ra.angular(mom, b, 'sin4'), (w * sin2**2).sum(0))
    with pytest.raises(ValueError):
        ra.angular(mom, b, 'sin6')


@pytest.mark.parametrize('func', ra.FUNCS)
def test_pose_decomposition_isolates_each_source(func):
    rng = np.random.default_rng(1)
    m1, _, _ = _node_moments(rng)
    m2, _, _ = _node_moments(rng)
    b1 = np.broadcast_to(_unit([0.0, 0.2, 1.0]), (m1.shape[0], 3))
    b2 = np.broadcast_to(_unit([0.05, 0.3, 1.0]), (m1.shape[0], 3))
    # Same pose in both sessions: no pose component.
    x_pose, x_est = ra.pose_decomposition(m1, m2, b1, b1, func)
    np.testing.assert_allclose(x_pose, 0, atol=1e-12)
    np.testing.assert_allclose(x_est, ra.angular(m2, b1, func) - ra.angular(m1, b1, func))
    # Same orientation estimate in both sessions: no estimate component.
    x_pose, x_est = ra.pose_decomposition(m1, m1, b1, b2, func)
    np.testing.assert_allclose(x_est, 0, atol=1e-12)
    np.testing.assert_allclose(x_pose, ra.angular(m1, b2, func) - ra.angular(m1, b1, func))


def test_donor_table_matches_direct_permutation():
    rng = np.random.default_rng(5)
    n_subj, n_per = 4, 3
    m1, _, _ = _node_moments(rng, n_nodes=n_subj * n_per)
    m2, _, _ = _node_moments(rng, n_nodes=n_subj * n_per)
    b1, b2 = _unit(rng.normal(size=(n_subj, 3))), _unit(rng.normal(size=(n_subj, 3)))
    subj_idx = np.repeat(np.arange(n_subj), n_per)

    def regressors(donor):
        return np.column_stack(ra.pose_decomposition(m1, m2, b1[donor], b2[donor], 'sin4'))

    table = ra.donor_table(regressors, len(subj_idx), n_subj)
    assert table.shape == (n_subj * n_per, n_subj, 2)
    perm = np.array([2, 0, 3, 1])
    np.testing.assert_allclose(table[np.arange(len(subj_idx)), perm[subj_idx]],
                               regressors(perm[subj_idx]))
    # One regressor: a trailing axis is added.
    single = ra.donor_table(lambda donor: ra.angular(m1, b1[donor], 'sin2'), len(subj_idx), n_subj)
    assert single.shape == (n_subj * n_per, n_subj, 1)


def test_node_dispersion():
    along_x = np.zeros((1, 21))
    along_x[0, 0] = 1.0  # M_xx = 1: perfectly coherent
    assert ra.node_dispersion_deg(along_x)[0] == pytest.approx(0, abs=1e-6)
    crossing = np.zeros((1, 21))
    crossing[0, [0, 3]] = 0.5  # half along x, half along y
    assert ra.node_dispersion_deg(crossing)[0] == pytest.approx(45)
    assert np.isnan(ra.node_dispersion_deg(np.full((1, 21), np.nan))[0])


def test_reference_comparison():
    out = ra.reference_comparison('inner_fa', ra.REFERENCE_FUNC, 0.0, -0.02, 0.04, 0.015)
    ref, lo, hi = ra.REFERENCE_EFFECTS['inner_fa']
    assert out['ref_beta'] == ref
    assert not out['ci_contains_ref']
    assert out['ci_overlaps_ref'] == (0.04 >= lo)
    assert 0.5 < out['power_ref'] < 1
    # Power is 80% when the reference effect equals the minimal detectable effect.
    se = abs(ref) / ra.bs.Z_80_POWER
    assert ra.reference_comparison('inner_fa', ra.REFERENCE_FUNC, 0, -1, 1, se)[
        'power_ref'] == pytest.approx(0.80, abs=1e-3)
    # No reference for other outcomes or angular functions.
    assert np.isnan(ra.reference_comparison('r2star', ra.REFERENCE_FUNC, 0, -1, 1, 1)['ref_beta'])
    assert np.isnan(ra.reference_comparison('inner_fa', 'sin2', 0, -1, 1, 1)['ref_beta'])


def test_delta_table_node_shift_filter():
    rng = np.random.default_rng(2)
    mom, _, _ = _node_moments(rng, n_nodes=4)
    rows = []
    for participant, shift in (('a', 0.5), ('b', 9.0)):
        for session in ('01', '02'):
            frame = pd.DataFrame(mom, columns=ra.moment_columns('fixel'))
            frame['participant_id'], frame['session_id'] = participant, session
            frame['bundle'], frame['node'] = 'X', np.arange(4)
            frame['inner_fa_mean'] = rng.random(4)
            frame['incl_primary'] = True
            frame['pos_x'] = np.arange(4) + (shift if session == '02' else 0.0)
            frame['pos_y'] = frame['pos_z'] = 0.0
            frame['f_sin4_fixel_dwi'] = rng.random(4)
            rows.append(frame)
    nodes = pd.concat(rows, ignore_index=True)
    vecs = {'dwi': {(p, s): np.array([0, 0, 1.0]) for p in 'ab' for s in ('01', '02')}}
    spec = ra.within_spec('x', 'x')
    assert set(ra.delta_table(nodes, spec, vecs)['participant_id']) == {'a', 'b'}
    spec = ra.within_spec('x', 'x', max_shift=5.0)
    kept = ra.delta_table(nodes, spec, vecs)
    assert set(kept['participant_id']) == {'a'}
    np.testing.assert_allclose(kept['shift_mm'], 0.5)
