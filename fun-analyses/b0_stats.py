"""Fixed-effect estimators, cluster bootstrap, and permutation tests for the B0 analysis.

All models are linear with high-dimensional fixed effects absorbed by within-transformation
(demeaning). Inference is clustered on participants, because head pose is a participant-level
rotation and nodes within a participant are strongly dependent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Z_80_POWER = 1.959964 + 0.841621  # two-sided alpha 0.05, power 0.80


def group_codes(*columns) -> np.ndarray:
    """Integer codes for the unique combinations of one or more aligned columns."""
    stacked = np.rec.fromarrays([np.asarray(c) for c in columns])
    _, codes = np.unique(stacked, return_inverse=True)
    return codes.ravel()


def demean(x: np.ndarray, codes: np.ndarray) -> np.ndarray:
    """Subtract group means (codes are 0..G-1) from x of shape (n,) or (n, k)."""
    x = np.asarray(x, float)
    counts = np.maximum(np.bincount(codes), 1)  # empty groups are never indexed
    if x.ndim == 1:
        return x - (np.bincount(codes, x) / counts)[codes]
    means = np.stack([np.bincount(codes, x[:, j]) for j in range(x.shape[1])], 1)
    return x - (means / counts[:, None])[codes]


def multiway_demean(x: np.ndarray, code_list, tol: float = 1e-10, max_iter: int = 500):
    """Absorb several (possibly non-nested) fixed effects by alternating projections."""
    x = np.asarray(x, float).copy()
    for _ in range(max_iter):
        prev = x.copy()
        for codes in code_list:
            x = demean(x, codes)
        if np.max(np.abs(x - prev)) < tol * (1 + np.max(np.abs(x))):
            break
    return x


def cluster_se(X: np.ndarray, resid: np.ndarray, clusters: np.ndarray, n_absorbed: int = 0):
    """CR1 cluster-robust covariance for OLS on already-demeaned X (n, k)."""
    X = np.atleast_2d(X.T).T
    n, k = X.shape
    bread = np.linalg.inv(X.T @ X)
    scores = X * resid[:, None]
    g_scores = np.stack([np.bincount(clusters, scores[:, j]) for j in range(k)], 1)
    n_clusters = len(np.unique(clusters))
    meat = g_scores.T @ g_scores
    adj = n_clusters / (n_clusters - 1) * (n - 1) / max(n - k - n_absorbed, 1)
    return bread @ meat @ bread * adj


@dataclass
class DeltaFit:
    """Within-subject between-session slope with participant-level resampling summaries.

    ``beta`` is the coefficient of the first regressor; ``gamma`` holds any covariates'
    coefficients. ``x_dm``/``y_dm`` are the first regressor and outcome after absorbing the
    fixed effects and partialling out the covariates (Frisch-Waugh-Lovell), for display and
    standardized effects.
    """

    beta: float
    gamma: np.ndarray
    n_rows: int
    n_subjects: int
    r2_within: float
    se_cluster: float
    boot: np.ndarray = field(repr=False)
    loso: np.ndarray = field(repr=False)
    subjects: np.ndarray = field(repr=False)
    perm: np.ndarray | None = field(default=None, repr=False)
    x_dm: np.ndarray | None = field(default=None, repr=False)
    y_dm: np.ndarray | None = field(default=None, repr=False)
    y_fe: np.ndarray | None = field(default=None, repr=False)

    @property
    def se_boot(self) -> float:
        return float(np.nanstd(self.boot, ddof=1))

    @property
    def ci(self) -> tuple[float, float]:
        return tuple(np.nanpercentile(self.boot, [2.5, 97.5]))

    @property
    def p_perm(self) -> float:
        if self.perm is None:
            return float('nan')
        return float((1 + np.sum(np.abs(self.perm) >= abs(self.beta))) / (1 + len(self.perm)))

    @property
    def mde(self) -> float:
        """Minimal detectable |beta| (80% power, two-sided 0.05) from the bootstrap SE."""
        return Z_80_POWER * self.se_boot

    @property
    def std_beta(self) -> float:
        """Slope in SD units of the demeaned outcome per SD of the demeaned predictor."""
        return float(self.beta * np.std(self.x_dm) / np.std(self.y_dm))


def _partial_out(v: np.ndarray, others: np.ndarray) -> np.ndarray:
    if others.shape[1] == 0:
        return v
    return v - others @ np.linalg.lstsq(others, v, rcond=None)[0]


def fit_delta(dy, dx, subjects, groups, n_boot=2000, rng=None) -> DeltaFit:
    """Fit ``dy ~ beta * dx[:, 0] + gamma . dx[:, 1:]`` with ``groups`` fixed effects.

    ``groups`` must be nested in ``subjects``. Then each participant's demeaned data are
    unchanged by resampling participants, so the cluster bootstrap and leave-one-subject-out
    estimates reduce to re-solving the normal equations from per-participant sums.
    """
    rng = rng or np.random.default_rng(0)
    dy = np.asarray(dy, float)
    X = np.asarray(dx, float)
    X = X[:, None] if X.ndim == 1 else X
    k = X.shape[1]
    y_dm, X_dm = demean(dy, groups), demean(X, groups)
    subj_codes = group_codes(subjects)
    n_subj = subj_codes.max() + 1
    sxx = np.stack(
        [np.stack([np.bincount(subj_codes, X_dm[:, i] * X_dm[:, j], n_subj) for j in range(k)], -1)
         for i in range(k)], -2,
    )  # (S, k, k)
    sxy = np.stack([np.bincount(subj_codes, X_dm[:, i] * y_dm, n_subj) for i in range(k)], -1)
    coef = np.linalg.solve(sxx.sum(0), sxy.sum(0))
    resid = y_dm - X_dm @ coef
    n_groups = len(np.unique(groups))
    se = float(np.sqrt(cluster_se(X_dm, resid, subj_codes, n_absorbed=n_groups)[0, 0]))

    counts = rng.multinomial(n_subj, np.full(n_subj, 1 / n_subj), size=n_boot).astype(float)
    boot = np.linalg.solve(np.einsum('bs,sij->bij', counts, sxx), (counts @ sxy)[..., None])
    boot = boot[:, 0, 0]
    loso = np.linalg.solve(sxx.sum(0)[None] - sxx, (sxy.sum(0)[None] - sxy)[..., None])[:, 0, 0]
    r2 = 1 - np.sum(resid**2) / np.sum(y_dm**2)
    others = X_dm[:, 1:]
    return DeltaFit(
        beta=float(coef[0]), gamma=coef[1:], n_rows=len(dy), n_subjects=int(n_subj),
        r2_within=float(r2), se_cluster=se, boot=boot, loso=loso,
        subjects=np.unique(np.asarray(subjects)),
        x_dm=_partial_out(X_dm[:, 0], others), y_dm=_partial_out(y_dm, others), y_fe=y_dm,
    )


def permute_delta(fit: DeltaFit, make_dx, groups, n_perm=2000, rng=None) -> np.ndarray:
    """Null distribution of ``beta`` from recomputing regressors under participant permutations.

    ``make_dx(perm)`` returns the regressor(s), (n,) or (n, k), for a permutation ``perm`` of
    participant indices (0..S-1 in ``fit.subjects`` order). The demeaned outcome is fixed.
    """
    rng = rng or np.random.default_rng(1)
    null = np.empty(n_perm)
    for i in range(n_perm):
        X = np.asarray(make_dx(rng.permutation(fit.n_subjects)), float)
        X = demean(X[:, None] if X.ndim == 1 else X, groups)
        null[i] = np.linalg.lstsq(X, fit.y_fe, rcond=None)[0][0]
    fit.perm = null
    return null


@dataclass
class CrossFit:
    """Cross-sectional fixed-effects fit with cluster-robust SEs."""

    names: list
    beta: np.ndarray
    cov: np.ndarray
    n_rows: int
    n_subjects: int
    r2_within: float
    r2_increment: float
    perm: np.ndarray | None = field(default=None, repr=False)
    x_dm: np.ndarray | None = field(default=None, repr=False)
    y_dm: np.ndarray | None = field(default=None, repr=False)

    def se(self, j=0) -> float:
        return float(np.sqrt(self.cov[j, j]))

    def ci(self, j=0) -> tuple[float, float]:
        return (self.beta[j] - 1.959964 * self.se(j), self.beta[j] + 1.959964 * self.se(j))

    @property
    def p_perm(self) -> float:
        if self.perm is None:
            return float('nan')
        return float((1 + np.sum(np.abs(self.perm) >= abs(self.beta[0]))) / (1 + len(self.perm)))

    @property
    def std_beta(self) -> float:
        return float(self.beta[0] * np.std(self.x_dm[:, 0]) / np.std(self.y_dm))


def fit_cross(y, X, names, subjects, fe_codes) -> CrossFit:
    """OLS of y on X (first column is the effect of interest) with absorbed fixed effects."""
    y_dm = multiway_demean(np.asarray(y, float), fe_codes)
    X_dm = multiway_demean(np.atleast_2d(np.asarray(X, float).T).T, fe_codes)
    beta, *_ = np.linalg.lstsq(X_dm, y_dm, rcond=None)
    resid = y_dm - X_dm @ beta
    subj_codes = group_codes(subjects)
    n_absorbed = sum(len(np.unique(c)) for c in fe_codes)
    cov = cluster_se(X_dm, resid, subj_codes, n_absorbed=n_absorbed)
    r2 = 1 - np.sum(resid**2) / np.sum(y_dm**2)
    if X_dm.shape[1] > 1:
        b_rest, *_ = np.linalg.lstsq(X_dm[:, 1:], y_dm, rcond=None)
        r2_rest = 1 - np.sum((y_dm - X_dm[:, 1:] @ b_rest) ** 2) / np.sum(y_dm**2)
    else:
        r2_rest = 0.0
    return CrossFit(
        names=list(names), beta=beta, cov=cov, n_rows=len(y_dm),
        n_subjects=len(np.unique(subj_codes)), r2_within=float(r2),
        r2_increment=float(r2 - r2_rest), x_dm=X_dm, y_dm=y_dm,
    )


def permute_cross(fit: CrossFit, make_x0, fe_codes, n_perm=500, rng=None) -> np.ndarray:
    """Null distribution of the first coefficient, recomputing only the first regressor."""
    rng = rng or np.random.default_rng(2)
    null = np.empty(n_perm)
    rest = fit.x_dm[:, 1:]
    for i in range(n_perm):
        x0 = multiway_demean(make_x0(rng.permutation(fit.n_subjects)), fe_codes)
        Xp = np.column_stack([x0, rest])
        null[i] = np.linalg.lstsq(Xp, fit.y_dm, rcond=None)[0][0]
    fit.perm = null
    return null
