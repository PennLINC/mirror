#!/usr/bin/env python3
"""Compute metric correlation matrices from WM bundle and GM parcel profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import (  # noqa: E402
    ANALYSIS_SET_CHOICES,
    build_metric_specs,
    metric_display_labels,
    metric_order,
    selected_analysis_sets,
)
from utils.regional_io import (  # noqa: E402
    DEFAULT_DKT_GLOBS,
    DEFAULT_QC_FILE,
    DEFAULT_WM_GLOBS,
    QC_MODES,
    apply_qc_mode,
    load_dkt_long_df,
    load_qc_table,
    load_wm_long_df,
)
from utils.paths import OUTPUT_DERIVATIVES_ROOT  # noqa: E402


PROFILE_TYPES = ('wm_bundles', 'gm_parcels')
CORRELATION_METHODS = ('spearman', 'pearson')


def write_metric_inclusion(
    out_file: Path,
    profile_type: str,
    analysis_set: str,
    tissue: str,
    expected_labels: list[str],
    observed_before_qc_labels: set[str],
    observed_labels: set[str],
    included_labels: list[str],
    display: dict[str, str],
) -> None:
    included = set(included_labels)
    rows = [
        {
            'profile_type': profile_type,
            'analysis_set': analysis_set,
            'tissue': tissue,
            'metric_key': label,
            'metric': display.get(label, label),
            'expected': True,
            'observed_in_input_before_qc': label in observed_before_qc_labels,
            'observed_in_input_after_qc': label in observed_labels,
            'included': label in included,
            'reason_if_not_included': (
                ''
                if label in included
                else (
                    'not_observed_in_input_before_qc'
                    if label not in observed_before_qc_labels
                    else 'not_observed_in_input_after_qc'
                    if label not in observed_labels
                    else 'fewer_than_two_included_metrics_or_no_valid_correlations'
                )
            ),
        }
        for label in expected_labels
    ]
    pd.DataFrame(rows).to_csv(out_file, sep='\t', index=False)


def selected_profile_types(analysis: str) -> list[str]:
    if analysis == 'wm':
        return ['wm_bundles']
    if analysis == 'gm':
        return ['gm_parcels']
    return list(PROFILE_TYPES)


def pairwise_profile_correlation(
    profile: pd.DataFrame,
    min_features: int,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = list(profile.columns)
    corr = np.full((len(metrics), len(metrics)), np.nan, dtype=float)
    counts = np.zeros((len(metrics), len(metrics)), dtype=np.int64)
    values = profile.to_numpy(dtype=float)
    for i in range(len(metrics)):
        corr[i, i] = 1.0
        counts[i, i] = int(np.count_nonzero(np.isfinite(values[:, i])))
        for j in range(i + 1, len(metrics)):
            valid = np.isfinite(values[:, i]) & np.isfinite(values[:, j])
            n_valid = int(np.count_nonzero(valid))
            counts[i, j] = n_valid
            counts[j, i] = n_valid
            if n_valid < min_features:
                continue
            if method == 'spearman':
                x = rankdata(values[valid, i])
                y = rankdata(values[valid, j])
            elif method == 'pearson':
                x = values[valid, i]
                y = values[valid, j]
            else:
                raise ValueError(f'Unsupported correlation method: {method}')
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            value = float(np.corrcoef(x, y)[0, 1])
            corr[i, j] = value
            corr[j, i] = value
    return (
        pd.DataFrame(corr, index=metrics, columns=metrics),
        pd.DataFrame(counts, index=metrics, columns=metrics),
    )


def mean_correlation_matrix(
    long_df: pd.DataFrame,
    labels: list[str],
    min_features: int,
    min_group_subjects: int,
    method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    subject_z_mats = []
    subject_count_mats = []
    for subject, subject_df in long_df[long_df['metric'].isin(labels)].groupby(
        'subject', sort=True
    ):
        session_z_mats = []
        session_count_mats = []
        for _, dfg in subject_df.groupby('session', sort=True):
            profile = dfg.pivot_table(
                index='feature', columns='metric', values='value', aggfunc='mean'
            )
            profile = profile.reindex(columns=labels)
            profile = profile.dropna(axis=1, how='all')
            if profile.shape[1] < 2:
                continue
            corr, count = pairwise_profile_correlation(profile, min_features, method)
            z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
            np.fill_diagonal(z.values, 0.0)
            session_z_mats.append(z.reindex(index=labels, columns=labels))
            session_count_mats.append(count.reindex(index=labels, columns=labels))
        if not session_z_mats:
            continue
        z_stack = np.stack([mat.to_numpy(float) for mat in session_z_mats])
        count_stack = np.stack([mat.to_numpy(float) for mat in session_count_mats])
        subject_z_mats.append(
            pd.DataFrame(np.nanmean(z_stack, axis=0), index=labels, columns=labels)
        )
        subject_count_mats.append(
            pd.DataFrame(np.nanmean(count_stack, axis=0), index=labels, columns=labels)
        )
    if not subject_z_mats:
        empty = pd.DataFrame(np.nan, index=labels, columns=labels)
        return empty, empty.copy(), pd.DataFrame(0, index=labels, columns=labels)
    group_stack = np.stack([mat.to_numpy(float) for mat in subject_z_mats])
    nsubjects = pd.DataFrame(np.sum(np.isfinite(group_stack), axis=0), index=labels, columns=labels)
    mean_z = pd.DataFrame(np.nanmean(group_stack, axis=0), index=labels, columns=labels)
    mean_r = pd.DataFrame(np.tanh(mean_z), index=labels, columns=labels)
    low_n = nsubjects < int(min_group_subjects)
    mean_r = mean_r.mask(low_n)
    np.fill_diagonal(mean_r.values, 1.0)
    count_stack = np.stack([mat.to_numpy(float) for mat in subject_count_mats])
    mean_counts = pd.DataFrame(np.nanmean(count_stack, axis=0), index=labels, columns=labels)
    mean_counts = mean_counts.mask(low_n)
    return mean_r, mean_counts, nsubjects


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--patterns-file', type=Path, default=REPO_ROOT / 'configuration' / 'metrics.yml'
    )
    parser.add_argument('--qc-file', type=Path, default=DEFAULT_QC_FILE)
    parser.add_argument(
        '--outdir',
        type=Path,
        default=OUTPUT_DERIVATIVES_ROOT / 'parcel_bundle_correlations',
    )
    parser.add_argument('--analysis', choices=('wm', 'gm', 'both'), default='both')
    parser.add_argument(
        '--analysis-set',
        choices=ANALYSIS_SET_CHOICES,
        default='primary',
        help='Metric set to process. Default: primary.',
    )
    parser.add_argument('--qc-mode', choices=QC_MODES, default='metricqc')
    parser.add_argument(
        '--correlation',
        nargs='+',
        choices=(*CORRELATION_METHODS, 'both'),
        default=['both'],
        help='Correlation method(s) to compute.',
    )
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument('--prefer-masked', action='store_true')
    parser.add_argument('--min-features', type=int, default=2)
    parser.add_argument('--min-group-subjects', type=int, default=1)
    parser.add_argument('--wm-input-globs', nargs='+', default=DEFAULT_WM_GLOBS)
    parser.add_argument('--dkt-input-glob', nargs='+', default=DEFAULT_DKT_GLOBS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    specs = build_metric_specs(args.patterns_file)

    qc_df = load_qc_table(args.qc_file)
    correlation_methods = (
        list(CORRELATION_METHODS)
        if 'both' in args.correlation
        else list(dict.fromkeys(args.correlation))
    )

    inputs = []
    if args.analysis in {'wm', 'both'}:
        wm_df = load_wm_long_df(
            args.wm_input_globs,
            stat=args.stat,
            prefer_masked=args.prefer_masked,
            patterns_file=args.patterns_file,
        )
        raw_wm_df = wm_df.copy()
        qc_wm_df = apply_qc_mode(
            wm_df,
            qc_df,
            args.qc_mode,
            profile_type='wm',
            patterns_file=args.patterns_file,
        )
        inputs.append(('wm_bundles', raw_wm_df, qc_wm_df))
    if args.analysis in {'gm', 'both'}:
        gm_df = load_dkt_long_df(
            args.dkt_input_glob,
            stat=args.stat,
            patterns_file=args.patterns_file,
        )
        raw_gm_df = gm_df.copy()
        qc_gm_df = apply_qc_mode(
            gm_df,
            qc_df,
            args.qc_mode,
            profile_type='dkt',
            patterns_file=args.patterns_file,
        )
        inputs.append(('gm_parcels', raw_gm_df, qc_gm_df))

    for profile_type, raw_df, long_df in inputs:
        for analysis_set in selected_analysis_sets(args.analysis_set):
            tissue = 'wm' if profile_type == 'wm_bundles' else 'gm'
            expected_labels = metric_order(
                specs,
                analysis_set,
                tissue=tissue,
            )
            observed_before_qc_labels = set(raw_df['metric'])
            observed_labels = set(long_df['metric'])
            labels = [label for label in expected_labels if label in observed_labels]
            display = metric_display_labels(
                specs,
                analysis_set,
                tissue=tissue,
            )
            for correlation_method in correlation_methods:
                stem = (
                    args.outdir / f'{profile_type}_{analysis_set}_{correlation_method}_{args.stat}'
                )
                write_metric_inclusion(
                    stem.with_name(stem.name + '_metric_inclusion.tsv'),
                    profile_type,
                    analysis_set,
                    tissue,
                    expected_labels,
                    observed_before_qc_labels,
                    observed_labels,
                    labels,
                    display,
                )
                if len(labels) < 2:
                    continue
                corr, counts, nsubjects = mean_correlation_matrix(
                    long_df,
                    labels,
                    args.min_features,
                    args.min_group_subjects,
                    correlation_method,
                )
                corr = corr.rename(index=display, columns=display)
                counts = counts.rename(index=display, columns=display)
                nsubjects = nsubjects.rename(index=display, columns=display)
                corr.to_csv(stem.with_name(stem.name + '_r.tsv'), sep='\t')
                counts.to_csv(stem.with_name(stem.name + '_mean_pairwise_nfeatures.tsv'), sep='\t')
                nsubjects.to_csv(stem.with_name(stem.name + '_nsubjects.tsv'), sep='\t')
                print(f'Wrote: {stem}_r.tsv', flush=True)


if __name__ == '__main__':
    main()
