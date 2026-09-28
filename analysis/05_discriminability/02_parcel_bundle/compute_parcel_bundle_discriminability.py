#!/usr/bin/env python3
"""Compute test-retest discriminability for WM bundle and DKT parcel profiles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    require_regional_analysis_inputs,
)
from utils.paths import OUTPUT_DERIVATIVES_ROOT  # noqa: E402


RESULT_COLUMNS = [
    'profile_type',
    'profile_group',
    'stat',
    'distance_metric',
    'discriminability',
    'nearest_neighbor_accuracy',
    'mean_genuine_distance',
    'mean_impostor_distance',
    'mean_rank_percentile',
    'n_subjects',
    'n_sessions',
    'n_profiles',
    'n_features',
]


def _zscore_columns(matrix: pd.DataFrame) -> pd.DataFrame:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    keep = np.isfinite(stds.to_numpy(dtype=float)) & (stds.to_numpy(dtype=float) > 0)
    matrix = matrix.loc[:, keep]
    stds = stds.loc[matrix.columns]
    means = means.loc[matrix.columns]
    return (matrix - means) / stds


def _pairwise_distances(values: np.ndarray, metric: str) -> np.ndarray:
    if metric == 'euclidean':
        diffs = values[:, None, :] - values[None, :, :]
        return np.sqrt(np.sum(diffs * diffs, axis=2))

    if metric != 'correlation':
        raise ValueError(f'Unsupported distance metric: {metric}')

    centered = values - values.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    scaled = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    corr = np.clip(scaled @ scaled.T, -1.0, 1.0)
    return 1.0 - corr


def _score_profile_matrix(
    matrix: pd.DataFrame,
    group_name: str,
    profile_type: str,
    stat: str,
    distance_metric: str,
) -> dict[str, object] | None:
    if matrix.empty:
        return None

    matrix = keep_paired_profile_rows(matrix).sort_index()
    if matrix.empty:
        return None
    subjects = matrix.index.get_level_values('subject').astype(str).to_numpy()
    sessions = matrix.index.get_level_values('session').astype(str).to_numpy()
    values = matrix.to_numpy(dtype=float)

    if len(np.unique(subjects)) < 2 or len(np.unique(sessions)) < 2 or values.shape[1] < 2:
        return None

    distances = _pairwise_distances(values, distance_metric)
    scores: list[float] = []
    nearest_correct: list[float] = []
    genuine_distances: list[float] = []
    impostor_distances: list[float] = []
    rank_percentiles: list[float] = []

    for i, subject in enumerate(subjects):
        valid = np.arange(len(subjects)) != i
        same = (subjects == subject) & valid
        other = (subjects != subject) & valid
        if not same.any() or not other.any():
            continue

        genuine = distances[i, same]
        impostor = distances[i, other]
        genuine_min = float(np.min(genuine))

        scores.append(float(np.mean(impostor > genuine_min)))
        nearest_idx = np.argmin(np.where(valid, distances[i, :], np.inf))
        nearest_correct.append(float(subjects[nearest_idx] == subject))
        genuine_distances.append(genuine_min)
        impostor_distances.append(float(np.mean(impostor)))

        all_valid_distances = distances[i, valid]
        rank = 1 + int(np.sum(all_valid_distances < genuine_min))
        denom = max(len(all_valid_distances) - 1, 1)
        rank_percentiles.append(float(1.0 - (rank - 1) / denom))

    if not scores:
        return None

    return {
        'profile_type': profile_type,
        'profile_group': group_name,
        'stat': stat,
        'distance_metric': distance_metric,
        'discriminability': float(np.mean(scores)),
        'nearest_neighbor_accuracy': float(np.mean(nearest_correct)),
        'mean_genuine_distance': float(np.mean(genuine_distances)),
        'mean_impostor_distance': float(np.mean(impostor_distances)),
        'mean_rank_percentile': float(np.mean(rank_percentiles)),
        'n_subjects': int(len(np.unique(subjects))),
        'n_sessions': int(len(np.unique(sessions))),
        'n_profiles': int(len(subjects)),
        'n_features': int(values.shape[1]),
    }


def keep_paired_profile_rows(
    matrix: pd.DataFrame,
    min_sessions: int = 2,
) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    subjects = matrix.index.get_level_values('subject').astype(str).to_numpy()
    sessions = matrix.index.get_level_values('session').astype(str).to_numpy()
    session_counts = pd.Series(sessions, index=subjects).groupby(level=0).nunique()
    paired_subjects = set(session_counts[session_counts >= min_sessions].index)
    keep_rows = [subject in paired_subjects for subject in subjects]
    return matrix.iloc[keep_rows, :]


def _build_profile_matrix(
    df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    min_feature_coverage: float,
    min_profile_coverage: float,
    zscore_features: bool,
) -> pd.DataFrame:
    grouped = (
        df[['subject', 'session', feature_col, value_col]]
        .dropna(subset=[value_col])
        .groupby(['subject', 'session', feature_col], as_index=False)[value_col]
        .mean()
    )
    matrix = grouped.pivot_table(
        index=['subject', 'session'],
        columns=feature_col,
        values=value_col,
        aggfunc='mean',
    )
    if matrix.empty:
        return matrix

    matrix = keep_paired_profile_rows(matrix)
    if matrix.empty:
        return matrix

    feature_coverage = matrix.notna().mean(axis=0)
    matrix = matrix.loc[:, feature_coverage >= min_feature_coverage]
    profile_coverage = matrix.notna().mean(axis=1)
    matrix = matrix.loc[profile_coverage >= min_profile_coverage, :]
    if matrix.empty:
        return matrix

    matrix = keep_paired_profile_rows(matrix)
    if matrix.empty:
        return matrix

    matrix = matrix.dropna(axis=1, how='any')

    if matrix.empty:
        return matrix

    matrix = keep_paired_profile_rows(matrix)
    if matrix.empty:
        return matrix

    if zscore_features:
        matrix = _zscore_columns(matrix)
    return matrix


def _compute_discriminability(
    long_df: pd.DataFrame,
    profile_type: str,
    stat: str,
    min_feature_coverage: float,
    min_profile_coverage: float,
    distance_metric: str,
    zscore_features: bool,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    value_col = 'value'

    all_df = long_df.copy()
    all_df['metric_feature'] = all_df['metric'].astype(str) + '|' + all_df['feature'].astype(str)
    all_matrix = _build_profile_matrix(
        all_df,
        feature_col='metric_feature',
        value_col=value_col,
        min_feature_coverage=min_feature_coverage,
        min_profile_coverage=min_profile_coverage,
        zscore_features=zscore_features,
    )
    all_score = _score_profile_matrix(
        all_matrix,
        group_name='ALL_METRICS',
        profile_type=profile_type,
        stat=stat,
        distance_metric=distance_metric,
    )
    if all_score is not None:
        rows.append(all_score)

    for metric_name, metric_df in long_df.groupby('metric', sort=True):
        matrix = _build_profile_matrix(
            metric_df,
            feature_col='feature',
            value_col=value_col,
            min_feature_coverage=min_feature_coverage,
            min_profile_coverage=min_profile_coverage,
            zscore_features=zscore_features,
        )
        score = _score_profile_matrix(
            matrix,
            group_name=str(metric_name),
            profile_type=profile_type,
            stat=stat,
            distance_metric=distance_metric,
        )
        if score is not None:
            rows.append(score)

    if not rows:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    return (
        pd.DataFrame(rows, columns=RESULT_COLUMNS)
        .sort_values(['profile_type', 'profile_group'])
        .reset_index(drop=True)
    )


def filter_analysis_set(
    df: pd.DataFrame,
    specs,
    analysis_set: str,
    tissue: str,
) -> pd.DataFrame:
    allowed = set(metric_order(specs, analysis_set, tissue=tissue))
    return df.loc[df['metric'].isin(allowed)].copy()


def write_metric_inclusion(
    out_file: Path,
    profile_type: str,
    analysis_set: str,
    qc_mode: str,
    tissue: str,
    expected_labels: list[str],
    observed_labels: set[str],
    scored_labels: set[str],
    display: dict[str, str],
) -> None:
    rows = []
    for label in expected_labels:
        observed = label in observed_labels
        scored = label in scored_labels
        rows.append(
            {
                'profile_type': profile_type,
                'analysis_set': analysis_set,
                'qc_mode': qc_mode,
                'tissue': tissue,
                'metric_key': label,
                'metric': display.get(label, label),
                'expected': True,
                'observed_after_qc': observed,
                'scored': scored,
                'reason_if_not_scored': (
                    ''
                    if scored
                    else (
                        'not_observed_after_qc'
                        if not observed
                        else 'insufficient_paired_profiles_or_feature_coverage'
                    )
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out_file, sep='\t', index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--analysis',
        choices=('wm', 'dkt', 'both'),
        default='both',
        help='Which profile discriminability analysis to run.',
    )
    parser.add_argument(
        '--analysis-set',
        choices=ANALYSIS_SET_CHOICES,
        default='primary',
        help='Metric set to process. Default: primary.',
    )
    parser.add_argument(
        '--stat',
        choices=('mean', 'median'),
        default='median',
        help='Scalar statistic to use.',
    )
    parser.add_argument(
        '--prefer-masked',
        action='store_true',
        help='For WM bundle TSVs, prefer masked_mean/masked_median when available.',
    )
    parser.add_argument(
        '--distance-metric',
        choices=('correlation', 'euclidean'),
        default='correlation',
        help='Distance metric between subject-session profiles.',
    )
    parser.add_argument(
        '--no-zscore',
        action='store_true',
        help='Do not z-score features before computing distances.',
    )
    parser.add_argument(
        '--min-feature-coverage',
        type=float,
        default=0.0,
        help='Minimum fraction of profiles with finite data required for a feature.',
    )
    parser.add_argument(
        '--min-profile-coverage',
        type=float,
        default=0.0,
        help='Minimum fraction of retained features required for a subject-session profile.',
    )
    parser.add_argument(
        '--wm-input-globs',
        nargs='+',
        default=DEFAULT_WM_GLOBS,
        help='Input globs for WM bundle scalarstats TSVs.',
    )
    parser.add_argument(
        '--dkt-input-glob',
        nargs='+',
        default=DEFAULT_DKT_GLOBS,
        help=(
            'Input glob(s) or expanded file path(s) for DKT parcel stats CSVs. '
            'Quote shell globs to avoid expansion, or pass multiple files.'
        ),
    )
    parser.add_argument(
        '--qc-file',
        type=Path,
        default=DEFAULT_QC_FILE,
        help='Manual modality QC TSV.',
    )
    parser.add_argument(
        '--qc-mode',
        nargs='+',
        choices=QC_MODES,
        default=list(QC_MODES),
        help='QC-filtered discriminability versions to write.',
    )
    parser.add_argument(
        '--outdir',
        default=str(OUTPUT_DERIVATIVES_ROOT / 'parcel_bundle_discriminability'),
        help='Output directory.',
    )
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=REPO_ROOT / 'configuration' / 'metrics.yml',
        help='Metric pattern registry.',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_regional_analysis_inputs(
        args.wm_input_globs,
        args.dkt_input_glob,
        require_wm=args.analysis in {'wm', 'both'},
        require_dkt=args.analysis in {'dkt', 'both'},
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    zscore_features = not args.no_zscore
    qc_df = load_qc_table(args.qc_file)
    specs = build_metric_specs(args.patterns_file)
    source_by_metric = {spec.label: spec.source_image for spec in specs}

    if args.analysis in {'wm', 'both'}:
        wm_df = load_wm_long_df(
            input_globs=args.wm_input_globs,
            stat=args.stat,
            prefer_masked=args.prefer_masked,
            patterns_file=args.patterns_file,
        )
        suffix = f'{args.stat}_{args.distance_metric}'
        if args.prefer_masked:
            suffix = f'masked_preferred_{suffix}'
        for qc_mode in args.qc_mode:
            filtered_wm = apply_qc_mode(
                wm_df,
                qc_df,
                qc_mode,
                profile_type='wm',
                patterns_file=args.patterns_file,
            )
            for analysis_set in selected_analysis_sets(args.analysis_set):
                expected_labels = metric_order(specs, analysis_set, tissue='wm')
                observed_labels = set(filtered_wm['metric'])
                display = metric_display_labels(specs, analysis_set, tissue='wm')
                set_df = filter_analysis_set(
                    filtered_wm,
                    specs,
                    analysis_set,
                    tissue='wm',
                )
                wm_out = _compute_discriminability(
                    set_df,
                    profile_type='wm_bundles',
                    stat=args.stat,
                    min_feature_coverage=args.min_feature_coverage,
                    min_profile_coverage=args.min_profile_coverage,
                    distance_metric=args.distance_metric,
                    zscore_features=zscore_features,
                )
                if wm_out.empty:
                    write_metric_inclusion(
                        outdir
                        / f'discriminability_wm_bundles_{analysis_set}_{suffix}_metric_inclusion.tsv',
                        'wm_bundles',
                        analysis_set,
                        qc_mode,
                        'wm',
                        expected_labels,
                        observed_labels,
                        set(),
                        display,
                    )
                    continue
                wm_out.insert(0, 'analysis_set', analysis_set)
                wm_out.insert(1, 'qc_mode', qc_mode)
                wm_out.insert(2, 'profile_group_key', wm_out['profile_group'])
                wm_out['profile_group'] = (
                    wm_out['profile_group_key'].map(display).fillna(wm_out['profile_group_key'])
                )
                wm_out['source_image'] = (
                    wm_out['profile_group_key'].map(source_by_metric).fillna('Other')
                )
                out_csv = outdir / f'discriminability_wm_bundles_{analysis_set}_{suffix}.csv'
                wm_out.to_csv(out_csv, index=False)
                write_metric_inclusion(
                    outdir
                    / f'discriminability_wm_bundles_{analysis_set}_{suffix}_metric_inclusion.tsv',
                    'wm_bundles',
                    analysis_set,
                    qc_mode,
                    'wm',
                    expected_labels,
                    observed_labels,
                    set(wm_out['profile_group_key']),
                    display,
                )
                print(f'Wrote: {out_csv}', flush=True)

    if args.analysis in {'dkt', 'both'}:
        dkt_df = load_dkt_long_df(
            args.dkt_input_glob,
            stat=args.stat,
            patterns_file=args.patterns_file,
        )
        for qc_mode in args.qc_mode:
            filtered_dkt = apply_qc_mode(
                dkt_df,
                qc_df,
                qc_mode,
                profile_type='dkt',
                patterns_file=args.patterns_file,
            )
            for analysis_set in selected_analysis_sets(args.analysis_set):
                expected_labels = metric_order(specs, analysis_set, tissue='gm')
                observed_labels = set(filtered_dkt['metric'])
                display = metric_display_labels(specs, analysis_set, tissue='gm')
                set_df = filter_analysis_set(
                    filtered_dkt,
                    specs,
                    analysis_set,
                    tissue='gm',
                )
                dkt_out = _compute_discriminability(
                    set_df,
                    profile_type='DKTatlas_parcels',
                    stat=args.stat,
                    min_feature_coverage=args.min_feature_coverage,
                    min_profile_coverage=args.min_profile_coverage,
                    distance_metric=args.distance_metric,
                    zscore_features=zscore_features,
                )
                if dkt_out.empty:
                    write_metric_inclusion(
                        outdir
                        / (
                            f'discriminability_DKTatlas_{analysis_set}_{args.stat}_'
                            f'{args.distance_metric}_metric_inclusion.tsv'
                        ),
                        'DKTatlas_parcels',
                        analysis_set,
                        qc_mode,
                        'gm',
                        expected_labels,
                        observed_labels,
                        set(),
                        display,
                    )
                    continue
                dkt_out.insert(0, 'analysis_set', analysis_set)
                dkt_out.insert(1, 'qc_mode', qc_mode)
                dkt_out.insert(2, 'profile_group_key', dkt_out['profile_group'])
                dkt_out['profile_group'] = (
                    dkt_out['profile_group_key'].map(display).fillna(dkt_out['profile_group_key'])
                )
                dkt_out['source_image'] = (
                    dkt_out['profile_group_key'].map(source_by_metric).fillna('Other')
                )
                out_csv = outdir / (
                    f'discriminability_DKTatlas_{analysis_set}_{args.stat}_'
                    f'{args.distance_metric}.csv'
                )
                dkt_out.to_csv(out_csv, index=False)
                write_metric_inclusion(
                    outdir
                    / (
                        f'discriminability_DKTatlas_{analysis_set}_{args.stat}_'
                        f'{args.distance_metric}_metric_inclusion.tsv'
                    ),
                    'DKTatlas_parcels',
                    analysis_set,
                    qc_mode,
                    'gm',
                    expected_labels,
                    observed_labels,
                    set(dkt_out['profile_group_key']),
                    display,
                )
                print(f'Wrote: {out_csv}', flush=True)


if __name__ == '__main__':
    main()
