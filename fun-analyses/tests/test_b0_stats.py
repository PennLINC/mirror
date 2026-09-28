"""Tests for b0_stats estimators and resampling."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import b0_stats as bs  # noqa: E402


def _delta_data(beta=0.5, n_subj=20, n_groups=5, n_per=30, seed=0):
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_subj), n_groups * n_per)
    groups = np.repeat(np.arange(n_subj * n_groups), n_per)
    x = rng.normal(size=subj.size) + rng.normal(size=n_subj * n_groups)[groups]
    y = beta * x + 3 * rng.normal(size=n_subj * n_groups)[groups] + rng.normal(size=subj.size)
    return y, x, subj, groups


def test_fit_delta_recovers_beta_and_loso():
    y, x, subj, groups = _delta_data()
    fit = bs.fit_delta(y, x, subj, groups, n_boot=500)
    assert fit.beta == pytest.approx(0.5, abs=0.05)
    lo, hi = fit.ci
    assert lo < 0.5 < hi
    # LOSO equals refitting without that participant.
    keep = subj != 3
    refit = bs.fit_delta(y[keep], x[keep], subj[keep], groups[keep], n_boot=10)
    assert fit.loso[3] == pytest.approx(refit.beta)


def test_fit_delta_with_covariate_matches_ols():
    y, x, subj, groups = _delta_data(beta=0.3, seed=2)
    rng = np.random.default_rng(9)
    c = rng.normal(size=x.size)
    y = y - 0.8 * c
    fit = bs.fit_delta(y, np.column_stack([x, c]), subj, groups, n_boot=200)
    Xd = bs.demean(np.column_stack([x, c]), groups)
    ref = np.linalg.lstsq(Xd, bs.demean(y, groups), rcond=None)[0]
    assert fit.beta == pytest.approx(ref[0])
    assert fit.gamma[0] == pytest.approx(ref[1])
    keep = subj != 5
    refit = bs.fit_delta(y[keep], np.column_stack([x, c])[keep], subj[keep], groups[keep], n_boot=5)
    assert fit.loso[5] == pytest.approx(refit.beta)


def test_fit_delta_absorbs_group_offsets():
    y, x, subj, groups = _delta_data(beta=0.0)
    y = y + 100 * groups  # huge group offsets must not matter
    fit = bs.fit_delta(y, x, subj, groups, n_boot=10)
    assert abs(fit.beta) < 0.05


def test_permutation_null_centered():
    y, x, subj, groups = _delta_data(beta=0.0, seed=4)
    fit = bs.fit_delta(y, x, subj, groups, n_boot=10)
    subj_x = {s: x[subj == s] for s in np.unique(subj)}

    def make_dx(perm):
        # Swap predictor blocks between participants of equal size.
        return np.concatenate([subj_x[perm[s]] for s in range(len(perm))])

    null = bs.permute_delta(fit, make_dx, groups, n_perm=300)
    assert abs(np.mean(null)) < 3 * np.std(null) / np.sqrt(len(null)) + 0.01
    assert 0 < fit.p_perm <= 1


def test_multiway_demean_matches_dummy_regression():
    rng = np.random.default_rng(1)
    n = 400
    a = rng.integers(0, 8, n)
    b = rng.integers(0, 12, n)
    x = rng.normal(size=n)
    y = 0.7 * x + rng.normal(size=8)[a] + rng.normal(size=12)[b] + 0.1 * rng.normal(size=n)
    fit = bs.fit_cross(y, x[:, None], ['x'], subjects=a, fe_codes=[a, b])
    dummies = np.column_stack([x, np.eye(8)[a], np.eye(12)[b][:, 1:]])
    ref = np.linalg.lstsq(dummies, y, rcond=None)[0][0]
    assert fit.beta[0] == pytest.approx(ref, abs=1e-6)


def test_group_codes_combinations():
    codes = bs.group_codes(['a', 'a', 'b', 'a'], [1, 2, 1, 1])
    assert codes[0] == codes[3]
    assert len(np.unique(codes)) == 3
