"""Tests for b0_tractometry geometry and node summaries."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import b0_tractometry as bt  # noqa: E402


def test_resample_equal_arclength():
    sl = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], float)
    out = bt.resample_streamline(sl, 4)
    np.testing.assert_allclose(out[:, 0], [0, 1, 2, 3])
    assert bt.resample_streamline(np.zeros((3, 3))) is None


def test_orient_streamlines_flips_reversed():
    base = np.column_stack([np.linspace(0, 10, 20), np.zeros(20), np.zeros(20)])
    pts = np.stack([base, base[::-1] + [0, 0.1, 0], base + [0, -0.1, 0]])
    out = bt.orient_streamlines(pts)
    assert np.all(out[:, 0, 0] < out[:, -1, 0])


def test_scatter_reproduces_mean_sin2_for_any_b0():
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(30, 4, 3))
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    w = rng.random((30, 4))
    w /= w.sum(0)
    m = bt.weighted_scatter(dirs, w)
    for _ in range(5):
        b = rng.normal(size=3)
        b /= np.linalg.norm(b)
        direct = (w * (1 - np.einsum('sni,i->sn', dirs, b) ** 2)).sum(0)
        np.testing.assert_allclose(bt.sin2_from_scatter(m, np.broadcast_to(b, (4, 3))), direct)


def test_scatter_ignores_nan_directions():
    dirs = np.array([[[1.0, 0, 0]], [[np.nan, np.nan, np.nan]]])
    w = np.array([[0.5], [0.5]])
    m = bt.weighted_scatter(dirs, w)
    assert bt.sin2_from_scatter(m, np.array([[1.0, 0, 0]]))[0] == pytest.approx(0)
    assert bt.sin2_from_scatter(m, np.array([[0, 0, 1.0]]))[0] == pytest.approx(1)


def test_moments_reproduce_mean_sin4_for_any_b0():
    rng = np.random.default_rng(3)
    dirs = rng.normal(size=(30, 4, 3))
    dirs /= np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs[::7] *= -1  # axial: sign must not matter
    w = rng.random((30, 4))
    w /= w.sum(0)
    m, q = bt.weighted_scatter(dirs, w), bt.weighted_moment4(dirs, w)
    assert q.shape == (4, 15)
    assert len(set(bt.MOMENT4_KEYS)) == 15
    assert sum(bt.MOMENT4_COEFS) == 3**4
    for _ in range(5):
        b = rng.normal(size=3)
        b /= np.linalg.norm(b)
        sin2 = 1 - np.einsum('sni,i->sn', dirs, b) ** 2
        b_rows = np.broadcast_to(b, (4, 3))
        np.testing.assert_allclose(bt.sin4_from_moments(m, q, b_rows), (w * sin2**2).sum(0))
        np.testing.assert_allclose(bt.cos4_from_moment4(q, b_rows), (w * (1 - sin2) ** 2).sum(0))


def test_sin4_exceeds_squared_mean_sin2_with_dispersion():
    # Jensen: <sin^4> >= <sin^2>^2, with equality only without dispersion.
    dirs = np.array([[[1.0, 0, 0]], [[0, 0, 1.0]]])
    w = np.array([[0.5], [0.5]])
    m, q = bt.weighted_scatter(dirs, w), bt.weighted_moment4(dirs, w)
    b = np.array([[0, 0, 1.0]])
    assert bt.sin2_from_scatter(m, b)[0] == pytest.approx(0.5)
    assert bt.sin4_from_moments(m, q, b)[0] == pytest.approx(0.5)


def test_moments_nan_without_any_direction():
    dirs = np.full((3, 2, 3), np.nan)
    dirs[:, 0] = [0, 1.0, 0]
    w = np.full((3, 2), 1 / 3)
    m, q = bt.weighted_scatter(dirs, w), bt.weighted_moment4(dirs, w)
    assert np.all(np.isfinite(m[0])) and np.all(np.isfinite(q[0]))
    assert np.all(np.isnan(m[1])) and np.all(np.isnan(q[1]))


def test_weighted_median_and_mean():
    v = np.array([[1.0], [2.0], [10.0], [np.nan]])
    w = np.array([[0.2], [0.5], [0.3], [0.0]])
    assert bt.weighted_median(v, w)[0] == 2.0
    assert bt.weighted_mean(v, w)[0] == pytest.approx(0.2 + 1.0 + 3.0)


def test_voxel_to_world_rotation_lps_grid():
    affine = np.diag([-1.7, -1.7, 1.7, 1.0])
    rot = bt.voxel_to_world_rotation(affine)
    np.testing.assert_allclose(rot, np.diag([-1.0, -1.0, 1.0]))
