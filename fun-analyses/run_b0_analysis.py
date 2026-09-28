#!/usr/bin/env python3
"""Run the B0-orientation analysis (plan Sections 6-20) and write an HTML report.

Inputs (from ``--in-dir``, default ``derivatives/b0_orientation``):

- ``b0_pose.tsv`` from ``extract_b0_pose.py``;
- ``nodes/sub-*_ses-*_nodes.parquet`` and ``*_qc.json`` from ``b0_tractometry.py``.

Outputs: ``results/*.tsv`` and ``b0_orientation_report.html``.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys
from datetime import date
from glob import glob

# Many small solves: multithreaded BLAS oversubscribes cores and is far slower here.
for _var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_var, '1')

import matplotlib  # noqa: E402

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b0_stats as bs  # noqa: E402
from b0_tractometry import (  # noqa: E402
    MOMENT4_KEYS,
    N_NODES,
    SCATTER_KEYS,
    sin2_from_scatter,
    sin4_from_moments,
)
from extract_b0_pose import default_out_dir, mean_axial  # noqa: E402

# ---------------------------------------------------------------------------
# Pre-specified settings (plan Sections 5, 7, 8, 15)
# ---------------------------------------------------------------------------

END_NODES = 5  # first/last 10% of 50 nodes
MIN_SUPPORT = 20  # streamlines per bundle (after subsampling) in each session
MIN_GROUP_FA = 0.25  # group-average inner-shell FA at the node
PEAK_RATIO_PRIMARY = 0.5  # second/first fixel amplitude
PEAK_RATIO_STRICT = 0.3
PEAK_RATIO_REFERENCE = 0.1  # single-fibre criterion of the reference study
MAX_NODE_SHIFT_MM = 5.0  # between-session node centroid distance (sensitivity analysis)
# Angular functions f(theta). The reference study models re-orientation effects with sin^4.
PRIMARY_FUNC = 'sin4'
FUNCS = ('sin4', 'sin2')
FUNC_LABEL = {'sin4': 'sin⁴θ', 'sin2': 'sin²θ'}
FL = FUNC_LABEL[PRIMARY_FUNC]
N_BOOT = 5000
N_PERM_PRIMARY = 5000
N_PERM_SECONDARY = 2000
N_PERM_CROSS = 1000
SEED = 20260928
# Bundles excluded from the ICC analyses for inconsistent recognition.
EXCLUDED_BUNDLE_PATTERNS = ('AnteriorCommissure', 'DentatorubrothalamicTract')
ESTIMATORS = ('fixel', 'tangent', 'v1')
# 'megre' (cross-contrast registration) is a diagnostic pose, used in sensitivity analyses only.
POSES = ('dwi', 'anat-chain', 'megre')

OUTCOMES = {
    'inner_fa': ('FA', 1),
    'inner_ad': ('AD', 1),
    'inner_rd': ('RD', -1),
    'inner_md': ('MD', 0),
    'r2star': ('R2*', 1),
}

# Reference effect sizes: Kleban, Jones & Tax (2023), Imaging Neuroscience 1,
# doi:10.1162/imag_a_00012, Fig. 4B (tract-segment-wise differences between 0 and 18 degree
# head tilt, slope B per unit sin^4(theta); 5 participants, 3 T, b <= 1500 s/mm^2).
# Values are (B, lower, upper), read from the published bar plot (bars and error bars as
# drawn; accuracy about 0.005) at TE = 75 and 100 ms and linearly interpolated to the NIBS
# TE of 88 ms. Diffusivities are converted from um^2/ms to the um^2/s of the NIBS maps.
REFERENCE_TE_MS = 88
REFERENCE_EFFECTS = {
    'inner_fa': (0.061, 0.034, 0.088),
    'inner_ad': (118.0, 67.0, 170.0),
    'inner_rd': (-38.0, -62.0, -14.0),
    'inner_md': (15.0, -2.0, 32.0),
}
REFERENCE_FUNC = 'sin4'

# Palette (dataviz reference instance, light surface).
SURFACE, INK, INK2, MUTED, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#8a8983', '#e7e6e1'
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
BLUES = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------


def load_inputs(in_dir):
    """Load node tables, QC, and pose."""
    frames = [pd.read_parquet(p) for p in sorted(glob(f'{in_dir}/nodes/*_nodes.parquet'))]
    if not frames:
        raise FileNotFoundError(f'No node tables in {in_dir}/nodes')
    nodes = pd.concat(frames, ignore_index=True)
    nodes['participant_id'] = nodes['participant_id'].astype(str)
    nodes['session_id'] = nodes['session_id'].astype(str)
    qc = pd.DataFrame(
        [json.load(open(p)) for p in sorted(glob(f'{in_dir}/nodes/*_qc.json'))]
    ).astype({'participant_id': str, 'session_id': str})
    pose = pd.read_table(f'{in_dir}/b0_pose.tsv', dtype={'participant_id': str, 'session_id': str})
    return nodes, qc, pose


def align_node_order(nodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reverse node order where needed so node n is the same end of a bundle everywhere.

    Each session's bundle centroid profile is compared, as is and reversed, with a group
    reference (initially the first profile, then the mean aligned profile).
    """
    nodes = nodes.sort_values(['bundle', 'participant_id', 'session_id', 'node']).copy()
    flips = []
    for bundle, grp in nodes.groupby('bundle', sort=False):
        keys = grp[['participant_id', 'session_id']].drop_duplicates().to_numpy()
        prof = grp[['pos_x', 'pos_y', 'pos_z']].to_numpy().reshape(len(keys), N_NODES, 3)
        ref = prof[0]
        for _ in range(3):
            d_same = np.linalg.norm(prof - ref, axis=2).mean(1)
            d_flip = np.linalg.norm(prof[:, ::-1] - ref, axis=2).mean(1)
            flip = d_flip < d_same
            ref = np.where(flip[:, None, None], prof[:, ::-1], prof).mean(0)
        for (p, s), f in zip(keys, flip):
            flips.append((bundle, p, s, bool(f)))
    flips = pd.DataFrame(flips, columns=['bundle', 'participant_id', 'session_id', 'flipped'])
    nodes = nodes.merge(flips, on=['bundle', 'participant_id', 'session_id'])
    nodes['node'] = np.where(nodes['flipped'], N_NODES - 1 - nodes['node'], nodes['node'])
    return nodes.drop(columns='flipped'), flips


def pose_vectors(pose: pd.DataFrame) -> dict:
    """{source: {(participant, session): unit B0 vector in ACPC}}."""
    out = {}
    for source in POSES:
        sub = pose.loc[(pose['source'] == source) & pose['direction_ok'].astype(bool)]
        out[source] = {
            (r.participant_id, r.session_id): np.array([r.b0_x, r.b0_y, r.b0_z])
            for r in sub.itertuples()
        }
    return out


def moment_columns(est: str, suffix: str = '') -> list:
    """Column names of a node's orientation moments: scatter tensor, then fourth order."""
    return ([f'M_{est}_{k}{suffix}' for k in SCATTER_KEYS]
            + [f'Q_{est}_{k}{suffix}' for k in MOMENT4_KEYS])


def moments(frame: pd.DataFrame, est: str, suffix: str = '') -> np.ndarray:
    """Orientation moments (n, 21): 6 scatter-tensor and 15 fourth-order components."""
    return frame[moment_columns(est, suffix)].to_numpy(float)


def angular(mom: np.ndarray, b0: np.ndarray, func: str) -> np.ndarray:
    """Node-mean f(theta) for B0 vector(s) ``b0``; linear in the moments ``mom``."""
    n_scatter = len(SCATTER_KEYS)
    if func == 'sin2':
        return sin2_from_scatter(mom[..., :n_scatter], b0)
    if func == 'sin4':
        return sin4_from_moments(mom[..., :n_scatter], mom[..., n_scatter:], b0)
    raise ValueError(f'Unknown angular function {func!r}')


def node_dispersion_deg(mom: np.ndarray) -> np.ndarray:
    """Within-node angular dispersion about the node's principal direction, in degrees.

    ``arcsin(sqrt(1 - lambda_max))`` of the scatter tensor: the root-mean-square sine of the
    angle between the per-point fiber directions and their principal axis.
    """
    xx, xy, xz, yy, yz, zz = np.moveaxis(mom[..., :len(SCATTER_KEYS)], -1, 0)
    full = np.stack([np.stack([xx, xy, xz], -1), np.stack([xy, yy, yz], -1),
                     np.stack([xz, yz, zz], -1)], -2)
    ok = np.all(np.isfinite(full), axis=(-1, -2))
    lam = np.linalg.eigvalsh(np.where(ok[..., None, None], full, np.eye(3)))[..., -1]
    return np.where(ok, np.degrees(np.arcsin(np.sqrt(np.clip(1 - lam, 0, 1)))), np.nan)


def add_angles(nodes, vecs, b0_ref):
    """Add f(theta) columns for every function x estimator x pose, and reference-B0 versions."""
    keys = list(zip(nodes['participant_id'], nodes['session_id']))
    new = {}
    for est in ESTIMATORS:
        mom = moments(nodes, est)
        for func in FUNCS:
            for pose in POSES:
                b0 = np.array([vecs[pose].get(k, np.full(3, np.nan)) for k in keys])
                new[f'f_{func}_{est}_{pose}'] = angular(mom, b0, func)
            new[f'fref_{func}_{est}'] = angular(mom, b0_ref, func)
        new[f'dispersion_{est}'] = node_dispersion_deg(mom)
    return pd.concat([nodes, pd.DataFrame(new, index=nodes.index)], axis=1)


def add_inclusion(nodes):
    """Boolean inclusion columns for the pre-specified variants (plan Section 8.2)."""
    excluded = nodes['bundle'].str.contains('|'.join(EXCLUDED_BUNDLE_PATTERNS))
    nodes['group_fa'] = nodes.groupby(['bundle', 'node'])['inner_fa_mean'].transform('mean')
    nodes['group_peak_ratio'] = nodes.groupby(['bundle', 'node'])['peak_ratio'].transform('mean')
    base = ~excluded & (nodes['group_fa'] >= MIN_GROUP_FA) & (nodes['n_streamlines'] >= MIN_SUPPORT)
    interior = nodes['node'].between(END_NODES, N_NODES - 1 - END_NODES)
    nodes['incl_primary'] = base & interior & (nodes['peak_ratio'] < PEAK_RATIO_PRIMARY)
    nodes['incl_strict'] = base & interior & (nodes['peak_ratio'] < PEAK_RATIO_STRICT)
    nodes['incl_reference'] = base & interior & (nodes['peak_ratio'] < PEAK_RATIO_REFERENCE)
    # Anatomical (group-average) coherence criterion: no session-specific selection.
    nodes['incl_group_ratio'] = base & interior & (nodes['group_peak_ratio'] < PEAK_RATIO_PRIMARY)
    nodes['incl_no_coherence'] = base & interior
    nodes['incl_with_endpoints'] = base & (nodes['peak_ratio'] < PEAK_RATIO_PRIMARY)
    nodes['excluded_bundle'] = excluded
    return nodes


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def within_spec(name, label, outcome='inner_fa', est='fixel', pose='dwi', incl='primary',
                summary='mean', group='sensitivity', n_perm=None, decompose=True,
                func=PRIMARY_FUNC, fe='subject:bundle', max_shift=None):
    n_perm = N_PERM_SECONDARY if n_perm is None else n_perm
    return dict(name=name, label=label, outcome=outcome, est=est, pose=pose, incl=incl,
                summary=summary, group=group, n_perm=n_perm, decompose=decompose, func=func,
                fe=fe, max_shift=max_shift)


def pose_decomposition(m1, m2, b1, b2, func=PRIMARY_FUNC):
    """Split the between-session change in f(theta) into pose and estimate components.

    ``m1``/``m2`` are the sessions' orientation moments (see ``moments``), in which f is linear.
    The naive change f(M2, b2) - f(M1, b1) also varies with session-to-session differences in
    the *estimated* fiber orientation (M1 -> M2), which are driven by noise and tractography
    and can co-vary with the outcome. Returns
    ``x_pose = f(Mbar, b2) - f(Mbar, b1)`` (varies only through head pose; Mbar is the
    two-session mean of the moments) and ``x_est = f(M2, bbar) - f(M1, bbar)``
    (orientation-estimate change at the mean pose), used as a covariate.
    """
    mbar = 0.5 * (m1 + m2)
    bbar = b1 + b2
    bbar = bbar / np.linalg.norm(bbar, axis=-1, keepdims=True)
    x_pose = angular(mbar, b2, func) - angular(mbar, b1, func)
    x_est = angular(m2, bbar, func) - angular(m1, bbar, func)
    return x_pose, x_est


def delta_table(nodes, spec, vecs):
    """Paired session table for one spec; only participants with both poses available."""
    ycol = f'{spec["outcome"]}_{spec["summary"]}'
    fcol = f'f_{spec["func"]}_{spec["est"]}_{spec["pose"]}'
    pos = ['pos_x', 'pos_y', 'pos_z']
    cols = (['participant_id', 'bundle', 'node', ycol, fcol, f'incl_{spec["incl"]}'] + pos
            + moment_columns(spec['est']))
    s1 = nodes.loc[nodes['session_id'] == '01', cols]
    s2 = nodes.loc[nodes['session_id'] == '02', cols]
    m = s1.merge(s2, on=['participant_id', 'bundle', 'node'], suffixes=('_1', '_2'))
    ok = (
        m[f'incl_{spec["incl"]}_1'] & m[f'incl_{spec["incl"]}_2']
        & np.isfinite(m[f'{ycol}_1']) & np.isfinite(m[f'{ycol}_2'])
        & np.isfinite(m[f'{fcol}_1']) & np.isfinite(m[f'{fcol}_2'])
    )
    has_pose = m['participant_id'].map(
        lambda p: (p, '01') in vecs[spec['pose']] and (p, '02') in vecs[spec['pose']]
    )
    m = m.loc[ok & has_pose].reset_index(drop=True)
    m['shift_mm'] = np.linalg.norm(
        m[[f'{p}_2' for p in pos]].to_numpy() - m[[f'{p}_1' for p in pos]].to_numpy(), axis=1
    )
    if spec.get('max_shift') is not None:
        # Drop participant x bundle pairs whose nodes moved too far between sessions.
        pair_shift = m.groupby(['participant_id', 'bundle'])['shift_mm'].transform('median')
        m = m.loc[pair_shift <= spec['max_shift']].reset_index(drop=True)
    m['dy'] = m[f'{ycol}_2'] - m[f'{ycol}_1']
    m['dx'] = m[f'{fcol}_2'] - m[f'{fcol}_1']
    return m


def fixed_effect_codes(m, fe):
    """Group codes of the absorbed fixed effects (all nested in participants)."""
    if fe == 'subject:bundle':
        return bs.group_codes(m['participant_id'], m['bundle'])
    if fe == 'subject':
        return bs.group_codes(m['participant_id'])
    raise ValueError(f'Unknown fixed effects {fe!r}')


def donor_table(make_x, n_rows, n_subjects):
    """Regressors for every row under every donor participant's pose, (n_rows, S, k).

    ``make_x(donor)`` returns the regressor(s) for an array of per-row donor indices. A
    permutation then only selects one donor per row, so f(theta) is not recomputed.
    """
    cols = [np.asarray(make_x(np.full(n_rows, d)), float) for d in range(n_subjects)]
    table = np.stack(cols, axis=1)
    return table[..., None] if table.ndim == 2 else table


def run_within(nodes, spec, vecs, n_boot=None, rng_seed=SEED):
    """Fit the within-subject between-session model with bootstrap and permutation."""
    n_boot = N_BOOT if n_boot is None else n_boot
    m = delta_table(nodes, spec, vecs)
    if m['participant_id'].nunique() < 5:
        return None, m
    groups = fixed_effect_codes(m, spec['fe'])
    subjects = sorted(m['participant_id'].unique())
    subj_idx = m['participant_id'].map({s: i for i, s in enumerate(subjects)}).to_numpy()
    b1 = np.array([vecs[spec['pose']][(s, '01')] for s in subjects])
    b2 = np.array([vecs[spec['pose']][(s, '02')] for s in subjects])
    est, func = spec['est'], spec['func']
    m1, m2 = moments(m, est, '_1'), moments(m, est, '_2')

    def regressors(donor):
        if spec['decompose']:
            return np.column_stack(pose_decomposition(m1, m2, b1[donor], b2[donor], func))
        return angular(m2, b2[donor], func) - angular(m1, b1[donor], func)

    by_donor = donor_table(regressors, len(m), len(subjects))
    rows_idx = np.arange(len(m))

    def make_dx(perm):
        return by_donor[rows_idx, perm[subj_idx]]

    X = make_dx(np.arange(len(subjects)))
    if spec['decompose']:
        m['dx'], m['dx_est'] = X[:, 0], X[:, 1]
    fit = bs.fit_delta(m['dy'], X, m['participant_id'], groups, n_boot=n_boot,
                       rng=np.random.default_rng(rng_seed))
    assert list(fit.subjects) == subjects

    if spec['n_perm']:
        bs.permute_delta(fit, make_dx, groups, n_perm=spec['n_perm'],
                         rng=np.random.default_rng(rng_seed + 1))
    return fit, m


def run_joint(nodes, outcome, vecs, pose='dwi', est='fixel', n_boot=N_BOOT):
    """Within-subject model with sin^2 and sin^4 terms together (plan Section 7, secondary).

    Returns one row per angular term, each with its own participant-level bootstrap CI.
    """
    spec = within_spec('joint', 'joint', outcome=outcome, est=est, pose=pose)
    m = delta_table(nodes, spec, vecs)
    groups = fixed_effect_codes(m, spec['fe'])
    subjects = sorted(m['participant_id'].unique())
    subj_idx = m['participant_id'].map({s: i for i, s in enumerate(subjects)}).to_numpy()
    b1 = np.array([vecs[pose][(s, '01')] for s in subjects])[subj_idx]
    b2 = np.array([vecs[pose][(s, '02')] for s in subjects])[subj_idx]
    m1, m2 = moments(m, est, '_1'), moments(m, est, '_2')
    parts = {func: pose_decomposition(m1, m2, b1, b2, func) for func in FUNCS}
    x_pose = {func: parts[func][0] for func in FUNCS}
    x_est = np.column_stack([parts[func][1] for func in FUNCS])
    pose_dm = bs.demean(np.column_stack([x_pose[func] for func in FUNCS]), groups)
    rows = []
    for func in FUNCS:
        other = [f for f in FUNCS if f != func][0]
        X = np.column_stack([x_pose[func], x_pose[other], x_est])
        fit = bs.fit_delta(m['dy'], X, m['participant_id'], groups, n_boot=n_boot,
                           rng=np.random.default_rng(SEED + 11))
        lo, hi = fit.ci
        rows.append(dict(outcome=OUTCOMES[outcome][0], term=f'Δ{FUNC_LABEL[func]} (pose)',
                         beta=fit.beta, ci_lo=lo, ci_hi=hi, se_boot=fit.se_boot,
                         corr_terms=float(np.corrcoef(pose_dm.T)[0, 1]),
                         r2_within=fit.r2_within, n_subjects=fit.n_subjects,
                         n_rows=fit.n_rows))
    return rows


def run_cross(nodes, outcome, vecs, session='01', naive=False, n_perm=N_PERM_CROSS,
              est='fixel', pose='dwi', func=PRIMARY_FUNC):
    """Cross-sectional pose-isolated model within one session (plan Section 10)."""
    ycol = f'{outcome}_mean'
    fcol, rcol = f'f_{func}_{est}_{pose}', f'fref_{func}_{est}'
    d = nodes.loc[
        (nodes['session_id'] == session) & nodes['incl_primary']
        & np.isfinite(nodes[ycol]) & np.isfinite(nodes[fcol])
    ].reset_index(drop=True)
    fe = [bs.group_codes(d['bundle'], d['node']), bs.group_codes(d['participant_id'], d['bundle'])]
    X = d[[fcol]].to_numpy() if naive else d[[fcol, rcol]].to_numpy()
    names = ['f'] if naive else ['f', 'f_ref']
    fit = bs.fit_cross(d[ycol].to_numpy(), X, names, d['participant_id'], fe)
    if n_perm and not naive:
        subjects = sorted(d['participant_id'].unique())
        subj_idx = d['participant_id'].map({s: i for i, s in enumerate(subjects)}).to_numpy()
        b = np.array([vecs[pose][(s, session)] for s in subjects])
        mm = moments(d, est)
        by_donor = donor_table(lambda donor: angular(mm, b[donor], func), len(d),
                               len(subjects))[..., 0]
        rows_idx = np.arange(len(d))
        bs.permute_cross(fit, lambda perm: by_donor[rows_idx, perm[subj_idx]], fe,
                         n_perm=n_perm, rng=np.random.default_rng(SEED + 7))
    return fit, d


def reference_comparison(outcome, func, beta, ci_lo, ci_hi, se):
    """Compare an estimate with the reference study's effect for the same outcome and f.

    ``power_ref`` is the design-based power (two-sided 0.05) to detect the reference point
    estimate given the bootstrap SE; it does not depend on the observed estimate.
    """
    keys = ('ref_beta', 'ref_lo', 'ref_hi', 'power_ref', 'ci_contains_ref', 'ci_overlaps_ref')
    if outcome not in REFERENCE_EFFECTS or func != REFERENCE_FUNC or not np.isfinite(se):
        return dict.fromkeys(keys, np.nan)
    ref, lo, hi = REFERENCE_EFFECTS[outcome]
    z = abs(ref) / se
    power = norm.cdf(z - norm.ppf(0.975)) + norm.cdf(-z - norm.ppf(0.975))
    return dict(ref_beta=ref, ref_lo=lo, ref_hi=hi, power_ref=float(power),
                ci_contains_ref=bool(ci_lo <= ref <= ci_hi),
                ci_overlaps_ref=bool(ci_lo <= hi and ci_hi >= lo))


def run_per_tract(m, n_boot=2000, min_subjects=10, min_rows=100):
    rows = []
    for bundle, g in m.groupby('bundle'):
        if g['participant_id'].nunique() < min_subjects or len(g) < min_rows:
            continue
        groups = bs.group_codes(g['participant_id'])
        fit = bs.fit_delta(g['dy'], g[['dx', 'dx_est']].to_numpy(), g['participant_id'], groups,
                           n_boot=n_boot, rng=np.random.default_rng(SEED + 3))
        lo, hi = fit.ci
        rows.append(dict(bundle=bundle, beta=fit.beta, ci_lo=lo, ci_hi=hi,
                         n_subjects=fit.n_subjects, n_rows=fit.n_rows,
                         sd_dx=float(np.std(fit.x_dm))))
    return pd.DataFrame(rows).sort_values('beta') if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def set_style():
    plt.rcParams.update({
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': MUTED, 'axes.labelcolor': INK2, 'xtick.color': INK2,
        'ytick.color': INK2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.6, 'axes.spines.top': False, 'axes.spines.right': False,
        'font.size': 9, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
        'axes.titlelocation': 'left', 'lines.linewidth': 1.5, 'legend.frameon': False,
        'axes.axisbelow': True,
    })


def fig_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def binned(ax, x, y, n_bins=20, color=BLUE, beta=None, ci=None):
    """Quantile-binned means (+/- SE) of demeaned y vs demeaned x, with the fitted slope."""
    edges = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, len(edges) - 2)
    bx = np.array([x[idx == i].mean() for i in range(len(edges) - 1)])
    by = np.array([y[idx == i].mean() for i in range(len(edges) - 1)])
    be = np.array([y[idx == i].std(ddof=1) / np.sqrt((idx == i).sum()) for i in range(len(edges) - 1)])
    xs = np.linspace(bx.min(), bx.max(), 50)
    if ci is not None:
        ax.fill_between(xs, ci[0] * xs, ci[1] * xs, color=color, alpha=0.15, linewidth=0)
    if beta is not None:
        ax.plot(xs, beta * xs, color=color, linewidth=1.5)
    ax.errorbar(bx, by, yerr=be, fmt='o', ms=4.5, color=color, ecolor=color, elinewidth=1,
                mec=SURFACE, mew=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=0.8)


def forest(ax, labels, est, lo, hi, colors=None, ref=0.0):
    y = np.arange(len(labels))[::-1]
    colors = colors or [BLUE] * len(labels)
    for yi, e, a, b, c in zip(y, est, lo, hi, colors):
        ax.plot([a, b], [yi, yi], color=c, linewidth=1.5, solid_capstyle='round')
        ax.plot(e, yi, 'o', color=c, ms=5.5, mec=SURFACE, mew=0.8)
    ax.axvline(ref, color=MUTED, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(axis='y', visible=False)


def figure_pose(pose, between, agreement):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    dwi = pose.loc[pose['source'] == 'dwi'].pivot(index='participant_id', columns='session_id',
                                                   values='tilt_deg').dropna()
    dwi = dwi.sort_values('01')
    ax = axes[0]
    y = np.arange(len(dwi))
    for yi, (a, b) in zip(y, dwi[['01', '02']].to_numpy()):
        ax.plot([a, b], [yi, yi], color=GRID, linewidth=2, zorder=1)
    ax.plot(dwi['01'], y, 'o', color=BLUE, ms=5, label='Session 1', mec=SURFACE, mew=0.8)
    ax.plot(dwi['02'], y, 'o', color=ORANGE, ms=5, label='Session 2', mec=SURFACE, mew=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(dwi.index, fontsize=7)
    ax.set_xlabel('Head tilt: angle between B0 and ACPC z (deg)')
    ax.set_title('Head tilt per session (DWI pose)')
    ax.legend(loc='lower right')
    ax.grid(axis='y', visible=False)

    ax = axes[1]
    d = between.loc[between['source'] == 'dwi', 'delta_b0_deg'].dropna()
    ax.hist(d, bins=np.arange(0, np.ceil(d.max()) + 0.5, 0.5), color=BLUE, edgecolor=SURFACE,
            linewidth=1.5)
    ax.axvline(d.median(), color=INK2, linewidth=1, linestyle='--')
    ax.text(d.median(), ax.get_ylim()[1] * 0.95, f'  median {d.median():.1f} deg', color=INK2,
            va='top', fontsize=8)
    ax.set_xlabel('Between-session change in B0 direction (deg)')
    ax.set_ylabel('Participants')
    ax.set_title('Natural pose change between sessions')

    ax = axes[2]
    col = [c for c in agreement.columns if c.endswith('_deg')][0]
    ax.hist(agreement[col], bins=np.arange(0, agreement[col].max() + 0.25, 0.25), color=AQUA,
            edgecolor=SURFACE, linewidth=1.5)
    ax.set_xlabel('DWI registration vs. anatomical chain (deg)')
    ax.set_ylabel('Sessions')
    ax.set_title('Agreement of pose estimates')
    from matplotlib.ticker import MaxNLocator
    for a in axes[1:]:
        a.yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    return fig_b64(fig)


def figure_leverage(m_primary):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 1.6]})
    ax = axes[0]
    ax.hist(m_primary['dx'], bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.set_xlabel(f'Between-session change in {FL} (Δf)')
    ax.set_ylabel('Retained nodes')
    ax.set_title(f'Leverage: distribution of Δ {FL}')
    ax = axes[1]
    lev = m_primary.assign(adx=m_primary['dx'].abs()).groupby(['bundle', 'node'])['adx'].median()
    lev = lev.unstack('node').reindex(columns=range(END_NODES, N_NODES - END_NODES))
    n_all = len(lev)
    lev = lev.loc[lev.mean(1).sort_values(ascending=False).index[:30]]
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blues', BLUES)
    cmap.set_bad(SURFACE)
    im = ax.imshow(lev.to_numpy(), aspect='auto', cmap=cmap, interpolation='nearest',
                   extent=(END_NODES - 0.5, N_NODES - END_NODES - 0.5, len(lev) - 0.5, -0.5))
    ax.set_yticks(range(len(lev)))
    ax.set_yticklabels([short_bundle(b) for b in lev.index], fontsize=7)
    ax.set_xlabel('Node')
    ax.set_title(f'Median |Δ {FL}| by node: 30 highest-leverage of {n_all} bundles')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig_b64(fig)


def short_bundle(name):
    for prefix in ('ProjectionBasalGanglia', 'ProjectionBrainstem', 'Association', 'Commissure',
                   'Cerebellum'):
        if name.startswith(prefix) and name != prefix:
            return name[len(prefix):]
    return name


def figure_profiles(nodes):
    preferred = ['ProjectionBrainstemCorticospinalTractL', 'CommissureCorpusCallosumBody',
                 'AssociationArcuateFasciculusL']
    bundles = [b for b in preferred if b in set(nodes['bundle'])][:3]
    if len(bundles) < 3:
        extra = nodes['bundle'].value_counts().index
        bundles += [b for b in extra if b not in bundles][: 3 - len(bundles)]
    fig, axes = plt.subplots(2, len(bundles), figsize=(12, 5.2), sharex=True)
    d = nodes.loc[nodes['session_id'] == '01']
    for j, bundle in enumerate(bundles):
        g = d.loc[d['bundle'] == bundle]
        theta = np.degrees(np.arcsin(np.sqrt(np.clip(g['fref_sin2_fixel'], 0, 1))))
        g = g.assign(theta_ref=theta)
        stats = g.groupby('node').agg(fa=('inner_fa_mean', 'mean'), fa_sd=('inner_fa_mean', 'std'),
                                      th=('theta_ref', 'mean'), th_sd=('theta_ref', 'std'))
        for i, (col, sd, color, lab) in enumerate(
            [('fa', 'fa_sd', BLUE, 'FA (inner-shell)'), ('th', 'th_sd', ORANGE,
                                                        'θ to reference B0 (deg)')]
        ):
            ax = axes[i, j]
            ax.fill_between(stats.index, stats[col] - stats[sd], stats[col] + stats[sd],
                            color=color, alpha=0.18, linewidth=0)
            ax.plot(stats.index, stats[col], color=color)
            ax.axvspan(0, END_NODES - 0.5, color=GRID, alpha=0.6, linewidth=0)
            ax.axvspan(N_NODES - END_NODES - 0.5, N_NODES - 1, color=GRID, alpha=0.6, linewidth=0)
            ax.set_ylabel(lab)
            if i == 0:
                ax.set_title(short_bundle(bundle))
            else:
                ax.set_xlabel('Node')
    fig.tight_layout()
    return fig_b64(fig)


def figure_primary(fit, label, unit):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    binned(axes[0], fit.x_dm, fit.y_dm, beta=fit.beta, ci=fit.ci)
    axes[0].set_xlabel(f'Pose-driven Δ {FL} (fixed effects and Δf_est partialled out)')
    axes[0].set_ylabel(f'Δ {label} (demeaned){unit}')
    axes[0].set_title(f'Within-subject: Δ{label} vs Δ {FL} (binned means ± SE)')
    ax = axes[1]
    ax.hist(fit.perm, bins=60, color=MUTED, edgecolor=SURFACE, linewidth=0.3, alpha=0.8)
    ax.axvline(fit.beta, color=BLUE, linewidth=2)
    ax.text(fit.beta, ax.get_ylim()[1] * 0.95, f'  observed β = {fit.beta:.3g}', color=INK,
            va='top', fontsize=8)
    ax.set_xlabel('β under pose permutation')
    ax.set_ylabel('Permutations')
    ax.set_title(f'Pose-permutation null (p = {fit.p_perm:.3g})')
    fig.tight_layout()
    return fig_b64(fig)


def figure_naive_vs_pose(naive, pose_fit, label):
    """Permutation nulls: naive vs pose-isolated Δf(θ)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, fit, title in [(axes[0], naive, f'Naive Δ {FL} — {label}'),
                           (axes[1], pose_fit, f'Pose-isolated Δ {FL} — {label}')]:
        ax.hist(fit.perm, bins=50, color=MUTED, edgecolor=SURFACE, linewidth=0.3, alpha=0.8)
        ax.axvline(fit.beta, color=BLUE, linewidth=2)
        ax.axvline(0, color=INK2, linewidth=0.8, linestyle=':')
        ax.set_title(f'{title}\nobserved β = {fit.beta:.3g}; null mean = {np.mean(fit.perm):.3g}; '
                     f'p = {fit.p_perm:.3g}', fontsize=9)
        ax.set_xlabel('β under pose permutation')
    axes[0].set_ylabel('Permutations')
    fig.tight_layout()
    return fig_b64(fig)


def figure_outcomes(fits):
    fig, axes = plt.subplots(1, len(fits), figsize=(3.1 * len(fits), 3.4))
    for ax, (label, fit) in zip(np.atleast_1d(axes), fits.items()):
        color = ORANGE if label == 'R2*' else BLUE
        binned(ax, fit.x_dm, fit.y_dm, beta=fit.beta, ci=fit.ci, color=color, n_bins=15)
        ax.set_title(f'{label}  (p_perm = {fit.p_perm:.2g})')
        ax.set_xlabel(f'Δ {FL} (demeaned)')
    np.atleast_1d(axes)[0].set_ylabel('Δ outcome (demeaned)')
    fig.tight_layout()
    return fig_b64(fig)


def figure_std_forest(rows):
    fig, ax = plt.subplots(figsize=(7, 0.34 * len(rows) + 1.2))
    colors = [ORANGE if 'R2*' in r['label'] else (AQUA if 'cross' in r['label'] else BLUE)
              for r in rows]
    forest(ax, [r['label'] for r in rows], [r['std'] for r in rows],
           [r['std_lo'] for r in rows], [r['std_hi'] for r in rows], colors)
    ax.set_xlabel(f'Standardized slope (SD outcome per SD {FL}), 95% CI')
    ax.set_title('Effects across outcomes and designs')
    fig.tight_layout()
    return fig_b64(fig)


def figure_sensitivity(table):
    fig, ax = plt.subplots(figsize=(7.5, 0.36 * len(table) + 1.2))
    colors = [BLUE if g == 'primary' else INK2 for g in table['group']]
    forest(ax, list(table['label']), table['beta'], table['ci_lo'], table['ci_hi'], colors)
    ax.set_xlabel(f'β: Δ FA per unit Δ {FL} (95% bootstrap CI)')
    ax.set_title('Sensitivity of the FA estimate')
    fig.tight_layout()
    return fig_b64(fig)


def figure_loso(fit, label):
    fig, ax = plt.subplots(figsize=(7, 3))
    order = np.argsort(fit.loso)
    ax.plot(np.arange(len(order)), fit.loso[order], 'o', color=BLUE, ms=5, mec=SURFACE, mew=0.8)
    ax.axhline(fit.beta, color=INK2, linewidth=1, linestyle='--')
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels(fit.subjects[order], rotation=90, fontsize=7)
    ax.set_ylabel(f'β for {label} without participant')
    ax.set_title('Leave-one-subject-out (dashed: all participants)')
    ax.grid(axis='x', visible=False)
    fig.tight_layout()
    return fig_b64(fig)


def figure_tracts(tr):
    fig, ax = plt.subplots(figsize=(7.5, 0.2 * len(tr) + 1.2))
    forest(ax, [short_bundle(b) for b in tr['bundle']], tr['beta'], tr['ci_lo'], tr['ci_hi'])
    ax.tick_params(axis='y', labelsize=6)
    ax.set_xlabel(f'β: Δ FA per unit Δ {FL} (95% bootstrap CI)')
    ax.set_title('Per-bundle within-subject estimates (exploratory)')
    fig.tight_layout()
    return fig_b64(fig)


def figure_pooled(nodes):
    d = nodes.loc[(nodes['session_id'] == '01') & nodes['incl_primary']]
    theta = np.degrees(np.arcsin(np.sqrt(np.clip(d['f_sin2_fixel_dwi'], 0, 1))))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('blues', BLUES[1:])
    for ax, col, lab in [(axes[0], 'inner_fa_mean', 'FA (inner-shell)'),
                         (axes[1], 'r2star_mean', 'R2* (s⁻¹)')]:
        ok = np.isfinite(d[col]) & np.isfinite(theta)
        hb = ax.hexbin(theta[ok], d[col][ok], gridsize=45, cmap=cmap, mincnt=1, bins='log',
                       linewidths=0.2, edgecolors=SURFACE)
        ax.set_xlabel('Fiber-to-B0 angle θ (deg)')
        ax.set_ylabel(lab)
        ax.set_title(f'Pooled {lab.split(" ")[0]} vs θ — descriptive only (session 1)')
        cb = fig.colorbar(hb, ax=ax, fraction=0.04)
        cb.set_label('Nodes (log)')
        cb.outline.set_visible(False)
    fig.tight_layout()
    return fig_b64(fig)


def figure_cross(fit, label, color=BLUE):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    # Partial out f_ref for display: residualize both on f_ref (Frisch-Waugh).
    xr = fit.x_dm[:, 1]
    x = fit.x_dm[:, 0] - xr * np.dot(xr, fit.x_dm[:, 0]) / np.dot(xr, xr)
    y = fit.y_dm - xr * np.dot(xr, fit.y_dm) / np.dot(xr, xr)
    binned(ax, x, y, beta=fit.beta[0], ci=fit.ci(0), color=color)
    ax.set_xlabel(f'{FL}, residualized on {FL}_ref and fixed effects')
    ax.set_ylabel(f'{label}, residualized')
    ax.set_title(f'Cross-sectional pose-isolated: {label}')
    fig.tight_layout()
    return fig_b64(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def fmt(x, digits=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return 'n/a'
    if isinstance(x, (int, np.integer)):
        return f'{x:,}'
    ax_ = abs(x)
    if ax_ != 0 and (ax_ < 1e-3 or ax_ >= 1e4):
        return f'{x:.{digits - 1}e}'
    return f'{x:.{digits}g}'


def fmt_p(p):
    if not np.isfinite(p):
        return 'n/a'
    return f'{p:.3f}' if p >= 0.001 else f'{p:.1e}'


def table_html(df, cols=None, formats=None):
    cols = cols or list(df.columns)
    formats = formats or {}
    head = ''.join(f'<th>{html.escape(str(c))}</th>' for c in cols)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = formats.get(c)
            s = f(v) if f else (fmt(v) if isinstance(v, (float, np.floating)) else str(v))
            num = isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
            cells.append(f'<td class="{"num" if num else ""}">{html.escape(s)}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    return f'<div class="tbl"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def img(b64, alt):
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{b64}"></figure>'


CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8983;--line:#e7e6e1;
--accent:#2a78d6;--good:#0ca30c;--warn:#fab219;--crit:#d03b3b;--figbg:#fcfcfb}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232322;
--ink:#ffffff;--ink2:#c3c2b7;--muted:#8f8e86;--line:#383835;--accent:#3987e5}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232322;--ink:#ffffff;--ink2:#c3c2b7;--muted:#8f8e86;
--line:#383835;--accent:#3987e5}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:28px;margin:0 0 4px}h2{font-size:20px;margin:40px 0 8px;padding-top:8px;
border-top:1px solid var(--line)}h3{font-size:16px;margin:24px 0 6px}
.sub{color:var(--ink2);margin:0 0 24px}.note{color:var(--ink2);font-size:13px}
.summary{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 20px}
.summary li{margin:4px 0}
figure{margin:12px 0;background:var(--figbg);border:1px solid var(--line);border-radius:8px;
padding:8px;overflow-x:auto}figure img{max-width:100%;height:auto;display:block;margin:auto}
.tbl{overflow-x:auto;margin:10px 0}table{border-collapse:collapse;font-size:13px;width:100%}
th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--ink2);font-weight:600}td.num{text-align:right;font-variant-numeric:tabular-nums}
.crit{display:grid;grid-template-columns:auto 1fr;gap:6px 12px;align-items:start}
.badge{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600;
border:1px solid var(--line);white-space:nowrap}
.b-yes{color:var(--good)}.b-no{color:var(--crit)}.b-na{color:var(--muted)}
code{font-size:13px}
"""


def badge(state):
    label, cls = {True: ('✓ met', 'b-yes'), False: ('✗ not met', 'b-no'),
                  None: ('– n/a', 'b-na')}[state]
    return f'<span class="badge {cls}">{label}</span>'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='pc', choices=['pc', 'cubic'])
    parser.add_argument('--in-dir')
    parser.add_argument('--quick', action='store_true', help='Few resamples (for testing).')
    args = parser.parse_args(argv)

    global N_BOOT, N_PERM_PRIMARY, N_PERM_SECONDARY, N_PERM_CROSS
    if args.quick:
        N_BOOT, N_PERM_PRIMARY, N_PERM_SECONDARY, N_PERM_CROSS = 200, 100, 50, 20

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from configuration.config import load_config

    cfg = load_config(args.config)
    in_dir = args.in_dir or default_out_dir(cfg)
    res_dir = os.path.join(in_dir, 'results')
    os.makedirs(res_dir, exist_ok=True)
    set_style()
    log = print

    # --- data ---------------------------------------------------------------------------
    nodes, qc, pose = load_inputs(in_dir)
    n_raw_rows = len(nodes)
    nodes, flips = align_node_order(nodes)
    vecs = pose_vectors(pose)
    b0_ref = mean_axial(list(vecs['dwi'].values()))
    nodes = add_angles(nodes, vecs, b0_ref)
    nodes = add_inclusion(nodes)
    between = pd.read_table(f'{in_dir}/b0_pose_between_sessions.tsv', dtype={'participant_id': str})
    agreement = pd.read_table(f'{in_dir}/b0_pose_source_agreement.tsv',
                              dtype={'participant_id': str, 'session_id': str})
    log(f'[INFO] {len(nodes)} node rows, {nodes["participant_id"].nunique()} participants')

    # --- within-subject models ----------------------------------------------------------
    specs = [within_spec('primary', 'FA — primary', group='primary', n_perm=N_PERM_PRIMARY)]
    for oc in ('inner_ad', 'inner_rd', 'inner_md'):
        specs.append(within_spec(oc, OUTCOMES[oc][0], outcome=oc, group='secondary'))
    specs.append(within_spec('r2star', 'R2* (positive control)', outcome='r2star',
                             pose='anat-chain', group='control', n_perm=N_PERM_PRIMARY))
    specs += [
        within_spec('func_sin2', f'FA — angular function {FUNC_LABEL["sin2"]}', func='sin2'),
        within_spec('est_tangent', 'FA — orientation: bundle tangent', est='tangent'),
        within_spec('est_v1', 'FA — orientation: DTI V1', est='v1'),
        within_spec('incl_strict', f'FA — stricter single-fiber (ratio < {PEAK_RATIO_STRICT})',
                    incl='strict'),
        within_spec('incl_reference',
                    f'FA — reference-like single-fiber (ratio < {PEAK_RATIO_REFERENCE})',
                    incl='reference'),
        within_spec('incl_group_ratio',
                    f'FA — group-average coherence (ratio < {PEAK_RATIO_PRIMARY})',
                    incl='group_ratio'),
        within_spec('incl_none', 'FA — no coherence filter', incl='no_coherence'),
        within_spec('incl_endpoints', 'FA — endpoints kept', incl='with_endpoints'),
        within_spec('node_shift', f'FA — node shift ≤ {MAX_NODE_SHIFT_MM:g} mm',
                    max_shift=MAX_NODE_SHIFT_MM),
        within_spec('summary_median', 'FA — weighted median node summary', summary='median'),
        within_spec('pose_chain', 'FA — pose from anatomical chain', pose='anat-chain'),
        within_spec('fe_subject', 'FA — participant fixed effects only', fe='subject'),
        within_spec('tensor_full', 'FA — full-shell tensor', outcome='full_fa'),
        within_spec('tensor_dki', 'FA — DKI', outcome='dki_fa'),
        within_spec('full_ad', 'AD — full-shell tensor', outcome='full_ad', group='secondary-fit'),
        within_spec('full_rd', 'RD — full-shell tensor', outcome='full_rd', group='secondary-fit'),
        within_spec('dki_ad', 'AD — DKI', outcome='dki_ad', group='secondary-fit'),
        within_spec('dki_rd', 'RD — DKI', outcome='dki_rd', group='secondary-fit'),
        within_spec('r2_sin2', f'R2* — angular function {FUNC_LABEL["sin2"]}', outcome='r2star',
                    pose='anat-chain', func='sin2', group='control-sens'),
        within_spec('r2_tangent', 'R2* — orientation: bundle tangent', outcome='r2star',
                    est='tangent', pose='anat-chain', group='control-sens'),
        within_spec('r2_median', 'R2* — weighted median', outcome='r2star', summary='median',
                    pose='anat-chain', group='control-sens'),
        within_spec('r2_pose_dwi', 'R2* — pose from DWI registration', outcome='r2star',
                    pose='dwi', group='control-sens'),
        within_spec('r2_pose_megre', 'R2* — pose from MEGRE registration (diagnostic)',
                    outcome='r2star', pose='megre', group='control-sens'),
        within_spec('r2_fe_subject', 'R2* — participant fixed effects only', outcome='r2star',
                    pose='anat-chain', fe='subject', group='control-sens'),
        # Naive Δf (pose + orientation-estimate change, no covariate): shows the artifact.
        within_spec('naive_fa', f'FA — naive Δ {FL} (not pose-isolated)', decompose=False,
                    group='naive'),
        within_spec('naive_r2', f'R2* — naive Δ {FL} (not pose-isolated)', outcome='r2star',
                    pose='anat-chain', decompose=False, group='naive'),
    ]
    fits, tables = {}, {}
    rows = []
    for spec in specs:
        log(f'[INFO] within-subject: {spec["label"]}')
        fit, m = run_within(nodes, spec, vecs)
        if fit is None:
            continue
        fits[spec['name']], tables[spec['name']] = fit, m
        lo, hi = fit.ci
        rows.append(dict(
            name=spec['name'], label=spec['label'], group=spec['group'], outcome=spec['outcome'],
            estimator=spec['est'], pose=spec['pose'], inclusion=spec['incl'],
            summary=spec['summary'], angular_function=spec['func'], fixed_effects=spec['fe'],
            decomposed=spec['decompose'], beta=fit.beta, ci_lo=lo,
            ci_hi=hi, beta_est=(float(fit.gamma[0]) if len(fit.gamma) else np.nan),
            perm_mean=float(np.mean(fit.perm)), se_boot=fit.se_boot,
            se_cluster=fit.se_cluster, p_perm=fit.p_perm, n_perm=len(fit.perm),
            std_beta=fit.std_beta, r2_within=fit.r2_within, mde=fit.mde,
            loso_min=float(fit.loso.min()), loso_max=float(fit.loso.max()),
            n_subjects=fit.n_subjects, n_rows=fit.n_rows,
            sd_dx=float(np.std(fit.x_dm)), sd_dy=float(np.std(fit.y_dm)),
            # Reference effects apply to pose-isolated estimates from the primary tensor fit.
            **reference_comparison(spec['outcome'] if spec['decompose'] else None, spec['func'],
                                   fit.beta, lo, hi, fit.se_boot),
        ))
    within = pd.DataFrame(rows)
    within.to_csv(f'{res_dir}/within_subject_models.tsv', sep='\t', index=False)
    prim = fits['primary']
    m_prim = tables['primary']

    # --- joint sin^2 + sin^4 model (secondary) ------------------------------------------
    joint_rows = []
    for oc, pose_src in (('inner_fa', 'dwi'), ('r2star', 'anat-chain')):
        log(f'[INFO] joint angular terms: {OUTCOMES[oc][0]}')
        joint_rows += run_joint(nodes, oc, vecs, pose=pose_src, n_boot=N_BOOT)
    joint = pd.DataFrame(joint_rows)
    joint.to_csv(f'{res_dir}/joint_angular_models.tsv', sep='\t', index=False)

    # --- cross-sectional models ---------------------------------------------------------
    cross_rows, cross_fits = [], {}
    for oc, (lab, _) in OUTCOMES.items():
        pose_src = 'anat-chain' if oc == 'r2star' else 'dwi'
        for session in ('01', '02'):
            if session == '02' and oc not in ('inner_fa', 'r2star'):
                continue
            for naive in (False, True):
                log(f'[INFO] cross-sectional: {lab} ses-{session} naive={naive}')
                n_perm = 0 if naive else (N_PERM_CROSS if oc in ('inner_fa', 'r2star')
                                          else N_PERM_CROSS // 2)
                fit, _ = run_cross(nodes, oc, vecs, session=session, naive=naive,
                                   n_perm=n_perm, pose=pose_src)
                cross_fits[(oc, session, naive)] = fit
                lo, hi = fit.ci(0)
                cross_rows.append(dict(
                    outcome=lab, session=session,
                    model='naive (no θ_ref)' if naive else 'pose-isolated (+ θ_ref)',
                    beta=fit.beta[0], ci_lo=lo, ci_hi=hi, se_cluster=fit.se(0),
                    p_perm=fit.p_perm,
                    beta_ref=(fit.beta[1] if not naive else np.nan),
                    std_beta=fit.std_beta, r2_increment=fit.r2_increment,
                    corr_f_fref=(float(np.corrcoef(fit.x_dm[:, 0], fit.x_dm[:, 1])[0, 1])
                                 if not naive else np.nan),
                    n_subjects=fit.n_subjects, n_rows=fit.n_rows,
                ))
    cross = pd.DataFrame(cross_rows)
    cross.to_csv(f'{res_dir}/cross_sectional_models.tsv', sep='\t', index=False)

    # --- per-tract, noise, dataset summaries -------------------------------------------
    per_tract = run_per_tract(m_prim)
    per_tract.to_csv(f'{res_dir}/per_bundle_primary.tsv', sep='\t', index=False)
    loso = pd.DataFrame({'participant_id': prim.subjects, 'beta_without': prim.loso})
    loso.to_csv(f'{res_dir}/loso_primary.tsv', sep='\t', index=False)
    retest_sd = float(np.std(prim.y_dm) / np.sqrt(2))

    s1 = nodes.loc[nodes['session_id'] == '01']
    retention = pd.DataFrame([
        ('All nodes (session 1)', len(s1)),
        ('Bundle not excluded', int((~s1['excluded_bundle']).sum())),
        (f'+ group FA ≥ {MIN_GROUP_FA}', int((~s1['excluded_bundle'] & (s1['group_fa'] >= MIN_GROUP_FA)).sum())),
        (f'+ support ≥ {MIN_SUPPORT} streamlines', int((~s1['excluded_bundle'] & (s1['group_fa'] >= MIN_GROUP_FA) & (s1['n_streamlines'] >= MIN_SUPPORT)).sum())),
        ('+ endpoints excluded', int(s1['incl_no_coherence'].sum())),
        (f'+ peak ratio < {PEAK_RATIO_PRIMARY} (primary)', int(s1['incl_primary'].sum())),
        ('Paired nodes in primary within-subject model', prim.n_rows),
    ], columns=['Step', 'Rows'])
    retention['%'] = 100 * retention['Rows'] / len(s1)

    dataset = pd.DataFrame([
        ('Participants with node data', nodes['participant_id'].nunique()),
        ('Sessions with node data', len(qc)),
        ('Participants in within-subject model', prim.n_subjects),
        ('Bundles (AutoTrack)', nodes['bundle'].nunique()),
        ('Nodes per bundle', N_NODES),
        ('Node rows (all sessions)', n_raw_rows),
        ('Bundle profiles reversed during node alignment', int(flips['flipped'].sum())),
        ('Median fixel–tangent |cos| (per-session QC)', float(qc['qc_fixel_tangent_cos'].median())),
        ('Sessions with R2* sampled', int(qc['r2star_available'].sum())),
        ('Median within-node angular dispersion, primary nodes (deg)',
         float(nodes.loc[nodes['incl_primary'], 'dispersion_fixel'].median())),
        ('Median between-session node centroid shift, primary nodes (mm)',
         float(m_prim['shift_mm'].median())),
    ], columns=['Quantity', 'Value'])

    # --- figures ------------------------------------------------------------------------
    log('[INFO] figures')
    figs = {
        'pose': figure_pose(pose, between, agreement),
        'leverage': figure_leverage(m_prim),
        'profiles': figure_profiles(nodes),
        'primary': figure_primary(prim, 'FA', ''),
        'outcomes': figure_outcomes({lab: fits[k] for k, lab in
                                     [('primary', 'FA'), ('inner_ad', 'AD'), ('inner_rd', 'RD'),
                                      ('inner_md', 'MD'), ('r2star', 'R2*')] if k in fits}),
        'loso': figure_loso(prim, 'FA'),
        'naive': figure_naive_vs_pose(fits['naive_fa'], prim, 'FA'),
        'pooled': figure_pooled(nodes),
        'cross_fa': figure_cross(cross_fits[('inner_fa', '01', False)], 'FA'),
        'cross_r2': figure_cross(cross_fits[('r2star', '01', False)], 'R2*', ORANGE),
    }
    sens = within.loc[within['group'].isin(['primary', 'sensitivity'])]
    figs['sensitivity'] = figure_sensitivity(sens)
    if len(per_tract):
        figs['tracts'] = figure_tracts(per_tract)
    std_rows = []
    for k in ('primary', 'inner_ad', 'inner_rd', 'inner_md', 'r2star'):
        if k in fits:
            f = fits[k]
            scale = np.std(f.x_dm) / np.std(f.y_dm)
            lo, hi = f.ci
            std_rows.append(dict(label=f'{OUTCOMES.get(k, ("FA",))[0] if k != "primary" else "FA"} — within',
                                 std=f.std_beta, std_lo=lo * scale, std_hi=hi * scale))
    for oc in OUTCOMES:
        f = cross_fits[(oc, '01', False)]
        scale = np.std(f.x_dm[:, 0]) / np.std(f.y_dm)
        lo, hi = f.ci(0)
        std_rows.append(dict(label=f'{OUTCOMES[oc][0]} — cross-sectional', std=f.std_beta,
                             std_lo=lo * scale, std_hi=hi * scale))
    figs['std_forest'] = figure_std_forest(std_rows)

    # --- interpretation -----------------------------------------------------------------
    def row(name):
        r = within.loc[within['name'] == name]
        return r.iloc[0] if len(r) else None

    med_delta = float(between.loc[between['source'] == 'dwi', 'delta_b0_deg'].median())
    r_fa, r_ad, r_rd, r_r2 = row('primary'), row('inner_ad'), row('inner_rd'), row('r2star')
    c_r2 = cross.loc[(cross['outcome'] == 'R2*') & (cross['session'] == '01')
                     & cross['model'].str.startswith('pose')].iloc[0]
    sig = lambda r: r is not None and r['p_perm'] < 0.05 and (r['ci_lo'] > 0 or r['ci_hi'] < 0)  # noqa: E731
    r2_recovered = (sig(r_r2) and r_r2['beta'] > 0) or (c_r2['p_perm'] < 0.05 and c_r2['beta'] > 0)
    fa_supported = sig(r_fa) and r_fa['beta'] > 0
    strict = row('incl_strict')
    excludes_zero = lambda r: r['ci_lo'] > 0 or r['ci_hi'] < 0  # noqa: E731
    ad_rd_signs = bool(r_ad['beta'] > 0 and r_rd['beta'] < 0)
    ad_rd_supported = ad_rd_signs and excludes_zero(r_ad) and excludes_zero(r_rd)
    ref_label = (f'Kleban et al. (2023) tract-segment estimate at TE ≈ {REFERENCE_TE_MS} ms: '
                 f'{fmt(r_fa["ref_beta"])} [{fmt(r_fa["ref_lo"])}, {fmt(r_fa["ref_hi"])}] FA '
                 f'per unit {FUNC_LABEL[REFERENCE_FUNC]}')
    criteria = [
        ('Meaningful between-session pose variation (median ΔB0 ≥ 2°)', med_delta >= 2,
         f'median ΔB0 = {med_delta:.2f}°'),
        ('R2* positive control recovered (β > 0, p_perm < 0.05, in within-subject or '
         'cross-sectional model)', bool(r2_recovered),
         f'within β = {fmt(r_r2["beta"])} (p = {fmt_p(r_r2["p_perm"])}); cross-sectional β = '
         f'{fmt(c_r2["beta"])} (p = {fmt_p(c_r2["p_perm"])})'),
        ('FA β > 0, permutation p < 0.05, bootstrap CI excludes 0', bool(fa_supported),
         f'β = {fmt(r_fa["beta"])} [{fmt(r_fa["ci_lo"])}, {fmt(r_fa["ci_hi"])}], '
         f'p = {fmt_p(r_fa["p_perm"])}'),
        ('AD and RD move in the expected complementary directions (AD β > 0, RD β < 0, both '
         'bootstrap CIs exclude 0)', bool(ad_rd_supported),
         f'AD β = {fmt(r_ad["beta"])} [{fmt(r_ad["ci_lo"])}, {fmt(r_ad["ci_hi"])}], RD β = '
         f'{fmt(r_rd["beta"])} [{fmt(r_rd["ci_lo"])}, {fmt(r_rd["ci_hi"])}]; signs '
         f'{"as" if ad_rd_signs else "not as"} expected'),
        ('Robust: same sign under stricter single-fiber criteria and in every LOSO fit',
         bool(np.sign(strict['beta']) == np.sign(r_fa['beta'])
              and np.sign(r_fa['loso_min']) == np.sign(r_fa['loso_max'])),
         f'strict β = {fmt(strict["beta"])}; LOSO range [{fmt(r_fa["loso_min"])}, '
         f'{fmt(r_fa["loso_max"])}]'),
        ('Magnitude compatible with the reference multi-head-position study (bootstrap CI '
         'contains the reference estimate)', bool(r_fa['ci_contains_ref']),
         f'{ref_label}. NIBS CI [{fmt(r_fa["ci_lo"])}, {fmt(r_fa["ci_hi"])}] '
         f'{"overlaps" if r_fa["ci_overlaps_ref"] else "does not overlap"} the reference '
         'interval.'),
    ]

    # --- reference comparison and design-based power (plan Section 6) ---------------------
    ref_names = [('primary', 'FA'), ('inner_ad', 'AD (µm²/s)'), ('inner_rd', 'RD (µm²/s)'),
                 ('inner_md', 'MD (µm²/s)'), ('incl_strict', 'FA — stricter single-fiber'),
                 ('incl_reference', 'FA — reference-like single-fiber'),
                 ('fe_subject', 'FA — participant fixed effects only')]
    ref_tbl = pd.DataFrame([
        dict(estimate=lab, reference=row(k)['ref_beta'], ref_lo=row(k)['ref_lo'],
             ref_hi=row(k)['ref_hi'], beta=row(k)['beta'], ci_lo=row(k)['ci_lo'],
             ci_hi=row(k)['ci_hi'], se_boot=row(k)['se_boot'], mde=row(k)['mde'],
             power_ref=row(k)['power_ref'],
             ci_contains_ref='yes' if row(k)['ci_contains_ref'] else 'no',
             ci_overlaps_ref='yes' if row(k)['ci_overlaps_ref'] else 'no')
        for k, lab in ref_names if row(k) is not None
    ])
    ref_tbl.to_csv(f'{res_dir}/reference_comparison.tsv', sep='\t', index=False)

    # --- narrative ----------------------------------------------------------------------
    fa_unit_note = (f'A β of {fmt(r_fa["beta"])} means FA would differ by {fmt(r_fa["beta"])} '
                    'between a fiber parallel and perpendicular to B0 (extrapolated from small '
                    'angle changes).')
    exp_dfa = abs(r_fa['beta']) * r_fa['sd_dx']
    ref_dfa = abs(r_fa['ref_beta']) * r_fa['sd_dx']
    summary_items = [
        f'<b>Design.</b> Within-subject comparison of the two NIBS sessions: {prim.n_subjects} '
        f'participants, {fmt(prim.n_rows)} paired bundle nodes. Natural head-pose changes '
        f'between sessions had a median of {med_delta:.1f}° (the "manipulation").',
        f'<b>Positive control (R2*).</b> Within-subject β = {fmt(r_r2["beta"])} s⁻¹ per unit '
        f'{FL} (95% CI {fmt(r_r2["ci_lo"])} to {fmt(r_r2["ci_hi"])}, permutation p = '
        f'{fmt_p(r_r2["p_perm"])}); cross-sectional pose-isolated β = {fmt(c_r2["beta"])} '
        f'(p = {fmt_p(c_r2["p_perm"])}). '
        + ('The pipeline <b>does</b> recover the known R2* orientation effect.' if r2_recovered
           else 'The pipeline did <b>not</b> recover the known R2* orientation effect, so FA '
                'nulls are uninformative.'),
        f'<b>Primary result (FA).</b> β = {fmt(r_fa["beta"])} per unit {FL} (95% bootstrap CI '
        f'{fmt(r_fa["ci_lo"])} to {fmt(r_fa["ci_hi"])}; pose-permutation p = '
        f'{fmt_p(r_fa["p_perm"])}). ' + fa_unit_note,
        f'<b>Comparison with the reference study.</b> {ref_label}. The NIBS 95% CI '
        f'{"contains" if r_fa["ci_contains_ref"] else "does not contain"} that estimate and '
        f'{"overlaps" if r_fa["ci_overlaps_ref"] else "does not overlap"} its interval. '
        f'Design-based power to detect the reference estimate was '
        f'{100 * r_fa["power_ref"]:.0f}%. The reference used much stricter single-fiber voxels, '
        'so its effect is an upper bound on what these nodes should show (see Section 4).',
        f'<b>Sensitivity/power.</b> Minimal detectable |β| (80% power) is {fmt(r_fa["mde"])} FA '
        f'per unit {FL}. The typical between-session Δ {FL} at a node has SD '
        f'{fmt(r_fa["sd_dx"])}, so the within-subject signal would be about {fmt(ref_dfa)} FA '
        f'at the reference β and is about {fmt(exp_dfa)} FA at the observed β, versus '
        f'node-level test–retest noise SD ≈ {fmt(retest_sd)} FA.',
        f'<b>Methodological caveat.</b> A naive between-session Δ {FL} gives FA β = '
        f'{fmt(row("naive_fa")["beta"])}, but its pose-permutation null is centered at '
        f'{fmt(row("naive_fa")["perm_mean"])} (p = {fmt_p(row("naive_fa")["p_perm"])}): that slope '
        'comes from session-to-session changes in the estimated fiber orientation, not head pose. '
        f'All primary estimates use the pose-isolated Δ {FL}.',
        f'<b>Secondary tensor metrics.</b> AD β = {fmt(r_ad["beta"])} (95% CI '
        f'{fmt(r_ad["ci_lo"])} to {fmt(r_ad["ci_hi"])}, p = {fmt_p(r_ad["p_perm"])}), RD β = '
        f'{fmt(r_rd["beta"])} (95% CI {fmt(r_rd["ci_lo"])} to {fmt(r_rd["ci_hi"])}, p = '
        f'{fmt_p(r_rd["p_perm"])}). Neither is distinguishable from zero.'
        if not ad_rd_supported else
        f'<b>Secondary tensor metrics.</b> AD β = {fmt(r_ad["beta"])} (p = '
        f'{fmt_p(r_ad["p_perm"])}), RD β = {fmt(r_rd["beta"])} (p = {fmt_p(r_rd["p_perm"])}); '
        'both in the expected direction with CIs excluding zero.',
    ]

    # --- HTML ---------------------------------------------------------------------------
    wfmt = {'beta': fmt, 'ci_lo': fmt, 'ci_hi': fmt, 'p_perm': fmt_p, 'mde': fmt,
            'std_beta': fmt, 'r2_within': fmt, 'n_rows': fmt, 'n_subjects': fmt,
            'loso_min': fmt, 'loso_max': fmt}
    wfmt['beta_est'] = fmt
    wfmt['perm_mean'] = fmt
    wcols = ['label', 'beta', 'ci_lo', 'ci_hi', 'p_perm', 'perm_mean', 'beta_est', 'std_beta',
             'mde', 'loso_min', 'loso_max', 'n_subjects', 'n_rows']
    main_tbl = within.loc[within['group'].isin(['primary', 'secondary', 'control'])]
    fit_tbl = within.loc[within['group'].isin(['secondary-fit', 'control-sens'])]
    ccols = ['outcome', 'session', 'model', 'beta', 'ci_lo', 'ci_hi', 'p_perm', 'beta_ref',
             'std_beta', 'r2_increment', 'corr_f_fref', 'n_subjects', 'n_rows']
    cfmt = {k: fmt for k in ('beta', 'ci_lo', 'ci_hi', 'beta_ref', 'std_beta', 'r2_increment',
                             'corr_f_fref', 'n_rows', 'n_subjects')}
    cfmt['p_perm'] = fmt_p
    pose_tbl = (between.loc[between['source'].isin(['dwi', 'anat-chain', 'megre'])]
                .groupby('source')['delta_b0_deg'].describe()[['count', 'mean', '50%', 'min', 'max']]
                .reset_index().rename(columns={'50%': 'median'}))
    crit_html = ''.join(f'<div>{badge(s)}</div><div><b>{html.escape(t)}</b><br>'
                        f'<span class="note">{html.escape(d)}</span></div>'
                        for t, s, d in criteria)

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>B0 Orientation Report</title><style>{CSS}</style></head><body><main>
<h1>Fiber orientation relative to B0 and DTI metrics</h1>
<p class="sub">NIBS observational sanity check · generated {date.today().isoformat()} ·
pipeline: <code>extract_b0_pose.py</code> → <code>b0_tractometry.py</code> →
<code>run_b0_analysis.py</code></p>

<div class="summary"><h3 style="margin-top:0">Key findings</h3><ul>
{''.join(f'<li>{s}</li>' for s in summary_items)}</ul>
<p class="note">Observational: head pose was not assigned. Estimates bound feasibility and
effect size; they do not establish that B0 orientation causes FA differences.</p></div>

<h2>Interpretation criteria (plan §18.1)</h2>
<div class="crit">{crit_html}</div>

<h2>1. Data and quality control</h2>
{table_html(dataset, formats={'Value': lambda v: fmt(v) if not isinstance(v, str) else v})}
<h3>Node retention (session 1)</h3>
{table_html(retention, formats={'Rows': fmt, '%': lambda v: f'{v:.1f}'})}
<p class="note">Pre-specified inclusion: bundle not in the ICC exclusion list; group-average
inner-shell FA ≥ {MIN_GROUP_FA}; ≥ {MIN_SUPPORT} streamlines; nodes {END_NODES}–{N_NODES - 1 - END_NODES}
of 0–{N_NODES - 1}; MSMT second/first fixel amplitude ratio &lt; {PEAK_RATIO_PRIMARY}. Criteria
do not use subject-level FA. Within-subject rows require inclusion in both sessions.</p>

<h2>2. Head pose (feasibility gate)</h2>
<p>B0 is the scanner z-axis (all runs head-first supine), expressed in each participant's
ACPC space via rigid registration of the raw DWI b0 to the preprocessed reference (AP/PA
averaged). The anatomical-chain estimate (QSIPrep MPRAGE transforms) is a cross-check and is
used for MEGRE.</p>
{img(figs['pose'], 'Head pose summary')}
{table_html(pose_tbl, formats={c: fmt for c in ('count', 'mean', 'median', 'min', 'max')})}
<p class="note">Between-session change in B0 direction (degrees) by pose source.</p>

<h2>3. Leverage and anatomy</h2>
<p>Only the <i>change</i> in {FL} between sessions informs the within-subject estimate. Its
distribution, and where in the bundles it is largest, are shown below. Along-tract profiles show
how strongly FA and the anatomical fiber angle co-vary along bundles — the confound that a pooled
correlation would mistake for a B0 effect.</p>
{img(figs['leverage'], 'Leverage of the within-subject design')}
{img(figs['profiles'], 'Along-tract FA and angle profiles')}

<h2>4. Positive control and primary result</h2>
<p>Model (plan §9, with pose isolation): Δy ~ β·Δf<sub>pose</sub> + γ·Δf<sub>est</sub> with
participant × bundle fixed effects and f(θ) = {FL}, where Δf<sub>pose</sub> = f(M̄, b₂) − f(M̄, b₁)
changes only through head pose (M̄: two-session mean of the node's fiber-orientation moments;
b: session B0) and
Δf<sub>est</sub> = f(M₂, b̄) − f(M₁, b̄) captures session-to-session change in the <i>estimated</i>
fiber orientation. Inference: cluster bootstrap over participants ({N_BOOT} resamples) and a
pose-permutation null that reassigns participants' pairs of B0 vectors ({N_PERM_PRIMARY}
permutations), keeping each participant's anatomy and measurements.</p>
<h3>Why pose isolation is needed</h3>
<p>The naive predictor f(M₂, b₂) − f(M₁, b₁) also varies with orientation-estimate changes
between sessions, which track tractography/noise differences that also change FA. Its
permutation null is therefore centered on the observed slope, not on zero: the naive "effect" is
not attributable to head pose. The pose-isolated predictor removes this (null centered near 0).</p>
{img(figs['naive'], 'Naive vs pose-isolated permutation nulls')}
{table_html(within.loc[within['group'] == 'naive'], ['label', 'beta', 'ci_lo', 'ci_hi', 'perm_mean', 'p_perm', 'n_subjects', 'n_rows'], {**wfmt, 'perm_mean': fmt})}
<p class="note">For the naive model the bootstrap CI is not a valid test of a pose effect,
because the predictor is not exogenous; the permutation p is.</p>
<h3>Pose-isolated results</h3>
{img(figs['primary'], 'Primary FA result')}
{img(figs['outcomes'], 'Within-subject results across outcomes')}
{table_html(main_tbl, wcols, wfmt)}
<p class="note">β units: outcome units per unit {FL} (FA unitless; AD/RD/MD in map units;
R2* in s⁻¹). perm_mean: mean of the pose-permutation null. std_beta: SD of the demeaned outcome
per SD of the demeaned predictor. mde: minimal detectable |β| at 80% power. LOSO: range of β
when leaving out each participant.</p>
{img(figs['loso'], 'Leave-one-subject-out')}
<h3>Comparison with the reference study and design-based power (plan §6)</h3>
<p>Reference: Kleban, Jones &amp; Tax (2023), <i>Imaging Neuroscience</i>
(doi:10.1162/imag_a_00012), Fig. 4B: slopes of tract-segment differences between 0° and 18°
head tilt against Δ{FUNC_LABEL[REFERENCE_FUNC]} (5 participants, 3 T), read from the published
bar plot at TE = 75 and 100 ms and interpolated to the NIBS TE of {REFERENCE_TE_MS} ms. Its
pooled-voxel analysis (FA varying by up to about 20% across angles) mixes anatomy with
orientation and is not the comparison used here.</p>
{table_html(ref_tbl, formats={**{c: fmt for c in ('reference', 'ref_lo', 'ref_hi', 'beta', 'ci_lo', 'ci_hi', 'se_boot', 'mde')}, 'power_ref': lambda v: f'{100 * v:.0f}%'})}
<p class="note">power_ref: power (two-sided 0.05) to detect the reference estimate given the
bootstrap SE; it does not depend on the observed β. The reference selected single-fiber voxels
with a second/first peak ratio below {PEAK_RATIO_REFERENCE} and low dispersion; NIBS nodes
average over streamlines (median within-node dispersion
{fmt(float(nodes.loc[nodes['incl_primary'], 'dispersion_fixel'].median()))}°), and dispersion
reduces the expected effect. Pose error also attenuates β. The reference estimate is therefore
an upper bound on the slope expected here.</p>
<h3>Joint {FUNC_LABEL['sin2']} and {FUNC_LABEL['sin4']} terms (secondary)</h3>
{table_html(joint, formats={**{c: fmt for c in ('beta', 'ci_lo', 'ci_hi', 'se_boot', 'corr_terms', 'r2_within', 'n_rows', 'n_subjects')}})}
<p class="note">Both pose-isolated terms and both Δf<sub>est</sub> covariates in one model.
corr_terms: correlation of the two pose terms after absorbing fixed effects; when it is near 1
the two coefficients are not separately identified.</p>

<h2>5. Cross-sectional, pose-isolated analysis</h2>
<p>Model (plan §10), per session: y ~ {FL} + {FL}_ref with bundle × node and participant ×
bundle fixed effects. {FL}_ref uses the group-mean B0 and absorbs anatomical orientation, so the
{FL} coefficient is identified by head pose only. CIs: cluster-robust (CR1) by participant;
p: permutation of B0 across participants. The naive model omits {FL}_ref.</p>
{img(figs['cross_fa'], 'Cross-sectional FA')}
{img(figs['cross_r2'], 'Cross-sectional R2*')}
{table_html(cross, ccols, cfmt)}
{img(figs['std_forest'], 'Standardized effects')}

<h2>6. Sensitivity analyses</h2>
{img(figs['sensitivity'], 'Sensitivity forest')}
{table_html(within.loc[within['group'] == 'sensitivity'], wcols, wfmt)}
<h3>Other tensor fits and R2* variants</h3>
{table_html(fit_tbl, wcols, wfmt)}
{img(figs['tracts'], 'Per-bundle estimates') if 'tracts' in figs else ''}
<p class="note">Per-bundle estimates are exploratory: look at direction consistency, not
p-values. {int((per_tract['beta'] > 0).sum()) if len(per_tract) else 0} of {len(per_tract)}
bundles have β &gt; 0.</p>

<h2>7. Descriptive pooled relationship</h2>
{img(figs['pooled'], 'Pooled descriptive')}
<p class="note">Pooled plots mix anatomy with orientation and are <b>not</b> evidence for a B0
effect.</p>

<h2>8. Methods summary and limitations</h2>
<ul>
<li>Tractometry: DSI Studio AutoTrack (MSMT) bundles in ACPC space; up to 1,000 streamlines per
bundle resampled to {N_NODES} nodes; Gaussian Mahalanobis node weights; node order aligned across
sessions and participants by centroid profiles.</li>
<li>Fiber direction (primary): at each streamline point, the MSMT-CSD fixel (QSIRecon fib) best
aligned with the local tangent; {FL} averaged over points per node (exactly, from second- and
fourth-order orientation moments; not computed from a mean direction).</li>
<li>Participant-level covariates (change in bore position, change in mean motion) are constant
within participant and are absorbed by the fixed effects of the differenced model.</li>
<li>Outcomes: TORTOISE inner-shell tensor (b ≤ 1200; primary); full-shell tensor and DKI as
sensitivity; R2* (MEGRE, 5 echoes) sampled via rigid sMRIPrep-T1w→ACPC registration.</li>
<li>R2* pose uses the same-session anatomical-chain transform (MEGRE registration to T1w was not
precise enough); within-session head motion is about 1°.</li>
<li>Angle noise attenuates β toward zero; within-subject differences in {FL} are small, which
magnifies this attenuation.</li>
<li>The within-subject design removes stable anatomy but not session-specific confounds that
happen to co-vary with pose change.</li>
</ul>
<p class="note">Result tables: <code>{html.escape(res_dir)}</code></p>
</main></body></html>"""
    out_html = os.path.join(in_dir, 'b0_orientation_report.html')
    with open(out_html, 'w', encoding='utf-8') as fobj:
        fobj.write(doc)
    log(f'[INFO] Wrote {out_html}')


if __name__ == '__main__':
    main()
