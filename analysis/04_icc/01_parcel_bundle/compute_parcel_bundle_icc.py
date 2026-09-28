#!/usr/bin/env python3
"""Compute ICC for GM parcels and WM bundles with registry metric labels."""

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


def complete_case_matrix(
    values: np.ndarray,
    subjects: np.ndarray,
    sessions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    sub_unique, sub_idx = np.unique(subjects, return_inverse=True)
    ses_unique, ses_idx = np.unique(sessions, return_inverse=True)
    matrix = np.full((len(sub_unique), len(ses_unique)), np.nan, dtype=float)
    for value, i_sub, i_ses in zip(values, sub_idx, ses_idx):
        matrix[i_sub, i_ses] = value
    complete = ~np.any(np.isnan(matrix), axis=1)
    return matrix[complete], ses_unique


def anova_mean_squares(matrix: np.ndarray) -> tuple[float, float, float]:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan, np.nan, np.nan
    n_sub, n_ses = matrix.shape
    grand = matrix.mean()
    row_means = matrix.mean(axis=1)
    col_means = matrix.mean(axis=0)
    ssr = n_ses * np.sum((row_means - grand) ** 2)
    ssc = n_sub * np.sum((col_means - grand) ** 2)
    sse = np.sum((matrix - grand) ** 2) - ssr - ssc
    msr = ssr / (n_sub - 1)
    msc = ssc / (n_ses - 1)
    mse = sse / ((n_sub - 1) * (n_ses - 1))
    return float(msr), float(msc), float(mse)


def compute_icc2(values: np.ndarray, subjects: np.ndarray, sessions: np.ndarray) -> float:
    matrix, _ = complete_case_matrix(values, subjects, sessions)
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan
    n_sub, n_ses = matrix.shape
    msr, msc, mse = anova_mean_squares(matrix)
    denom = msr + (n_ses - 1) * mse + n_ses * (msc - mse) / n_sub
    if denom == 0:
        return np.nan
    return float((msr - mse) / denom)


def compute_icc3_from_matrix(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return np.nan
    _, n_ses = matrix.shape
    msr, _, mse = anova_mean_squares(matrix)
    denom = msr + (n_ses - 1) * mse
    if denom == 0:
        return np.nan
    return float((msr - mse) / denom)


def compute_pairwise_pearson_r(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 3 or matrix.shape[1] < 2:
        return np.nan
    values = []
    for i in range(matrix.shape[1]):
        for j in range(i + 1, matrix.shape[1]):
            x = matrix[:, i]
            y = matrix[:, j]
            if np.std(x) == 0 or np.std(y) == 0:
                continue
            values.append(float(np.corrcoef(x, y)[0, 1]))
    if not values:
        return np.nan
    z_values = np.arctanh(np.clip(values, -0.999999, 0.999999))
    return float(np.tanh(np.mean(z_values)))


def compute_icc_diagnostics(matrix: np.ndarray) -> dict[str, float | int]:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return {
            'ICC3_1': np.nan,
            'pearson_r': np.nan,
            'between_subject_sd': np.nan,
            'within_subject_sd': np.nan,
            'mean_abs_session_diff': np.nan,
            'n_complete_subjects': int(matrix.shape[0]),
            'n_complete_sessions': int(matrix.shape[1]),
        }
    _, _, mse = anova_mean_squares(matrix)
    pairwise_abs_diffs = [
        np.abs(matrix[:, i] - matrix[:, j])
        for i in range(matrix.shape[1])
        for j in range(i + 1, matrix.shape[1])
    ]
    return {
        'ICC3_1': compute_icc3_from_matrix(matrix),
        'pearson_r': compute_pairwise_pearson_r(matrix),
        'between_subject_sd': float(np.std(matrix.mean(axis=1), ddof=1)),
        'within_subject_sd': float(np.sqrt(max(mse, 0.0))) if np.isfinite(mse) else np.nan,
        'mean_abs_session_diff': float(np.mean(np.concatenate(pairwise_abs_diffs))),
        'n_complete_subjects': int(matrix.shape[0]),
        'n_complete_sessions': int(matrix.shape[1]),
    }


def compute_icc_table(long_df: pd.DataFrame, profile_type: str, stat: str) -> pd.DataFrame:
    collapsed = (
        long_df[['subject', 'session', 'metric', 'feature', 'value']]
        .dropna(subset=['value'])
        .groupby(['subject', 'session', 'metric', 'feature'], as_index=False)['value']
        .mean()
    )
    rows = []
    for (metric, feature), dfg in collapsed.groupby(['metric', 'feature'], sort=True):
        finite = dfg[np.isfinite(dfg['value'].to_numpy(float))].copy()
        paired_counts = finite.groupby('subject')['session'].nunique()
        paired_subjects = paired_counts[paired_counts >= 2].index
        paired = finite.loc[finite['subject'].isin(paired_subjects)].copy()
        if paired['subject'].nunique() < 2 or paired['session'].nunique() < 2:
            continue
        values = paired['value'].to_numpy(float)
        subjects = paired['subject'].astype(str).to_numpy()
        sessions = paired['session'].astype(str).to_numpy()
        matrix, _ = complete_case_matrix(values, subjects, sessions)
        diagnostics = compute_icc_diagnostics(matrix)
        rows.append(
            {
                'profile_type': profile_type,
                'metric': metric,
                'feature': feature,
                'stat': stat,
                'ICC2_1': compute_icc2(values, subjects, sessions),
                **diagnostics,
                'n_subjects': int(paired['subject'].nunique()),
                'n_sessions': int(paired['session'].nunique()),
                'n_observations': int(len(paired)),
            }
        )
    return (
        pd.DataFrame(rows).sort_values(['profile_type', 'metric', 'feature']).reset_index(drop=True)
    )


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
                        else 'insufficient_paired_subjects_or_features'
                    )
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out_file, sep='\t', index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', choices=('wm', 'gm', 'both'), default='both')
    parser.add_argument(
        '--analysis-set',
        choices=ANALYSIS_SET_CHOICES,
        default='primary',
        help='Metric set to process. Default: primary.',
    )
    parser.add_argument('--stat', choices=('mean', 'median'), default='median')
    parser.add_argument('--prefer-masked', action='store_true')
    parser.add_argument('--qc-mode', nargs='+', choices=QC_MODES, default=list(QC_MODES))
    parser.add_argument('--wm-input-globs', nargs='+', default=DEFAULT_WM_GLOBS)
    parser.add_argument('--dkt-input-glob', nargs='+', default=DEFAULT_DKT_GLOBS)
    parser.add_argument('--qc-file', type=Path, default=DEFAULT_QC_FILE)
    parser.add_argument(
        '--patterns-file', type=Path, default=REPO_ROOT / 'configuration' / 'metrics.yml'
    )
    parser.add_argument(
        '--outdir',
        type=Path,
        default=OUTPUT_DERIVATIVES_ROOT / 'parcel_bundle_icc',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    require_regional_analysis_inputs(
        args.wm_input_globs,
        args.dkt_input_glob,
        require_wm=args.analysis in {'wm', 'both'},
        require_dkt=args.analysis in {'gm', 'both'},
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    specs = build_metric_specs(args.patterns_file)
    source_by_metric = {spec.label: spec.source_image for spec in specs}
    qc_df = load_qc_table(args.qc_file)

    inputs = []
    if args.analysis in {'wm', 'both'}:
        wm_df = load_wm_long_df(
            args.wm_input_globs,
            stat=args.stat,
            prefer_masked=args.prefer_masked,
            patterns_file=args.patterns_file,
        )
        inputs.append(('wm_bundles', 'wm', wm_df))
    if args.analysis in {'gm', 'both'}:
        gm_df = load_dkt_long_df(
            args.dkt_input_glob,
            stat=args.stat,
            patterns_file=args.patterns_file,
        )
        inputs.append(('gm_parcels', 'dkt', gm_df))

    for profile_name, qc_profile, input_df in inputs:
        for qc_mode in args.qc_mode:
            filtered = apply_qc_mode(
                input_df,
                qc_df,
                qc_mode,
                profile_type=qc_profile,
                patterns_file=args.patterns_file,
            )
            for analysis_set in selected_analysis_sets(args.analysis_set):
                tissue = 'wm' if profile_name == 'wm_bundles' else 'gm'
                expected_labels = metric_order(
                    specs,
                    analysis_set,
                    tissue=tissue,
                )
                observed_labels = set(filtered['metric'])
                labels = set(expected_labels)
                set_df = filtered.loc[filtered['metric'].isin(labels)].copy()
                display = metric_display_labels(specs, analysis_set, tissue=tissue)
                stem = args.outdir / f'icc_{profile_name}_{analysis_set}_{args.stat}'
                if set_df.empty:
                    write_metric_inclusion(
                        stem.with_name(stem.name + '_metric_inclusion.tsv'),
                        profile_name,
                        analysis_set,
                        qc_mode,
                        tissue,
                        expected_labels,
                        observed_labels,
                        set(),
                        display,
                    )
                    continue
                icc_df = compute_icc_table(set_df, profile_name, args.stat)
                if icc_df.empty:
                    write_metric_inclusion(
                        stem.with_name(stem.name + '_metric_inclusion.tsv'),
                        profile_name,
                        analysis_set,
                        qc_mode,
                        tissue,
                        expected_labels,
                        observed_labels,
                        set(),
                        display,
                    )
                    continue
                icc_df.insert(0, 'analysis_set', analysis_set)
                icc_df.insert(1, 'qc_mode', qc_mode)
                icc_df.insert(2, 'metric_key', icc_df['metric'])
                icc_df['metric'] = icc_df['metric_key'].map(display).fillna(icc_df['metric_key'])
                icc_df['source_image'] = icc_df['metric_key'].map(source_by_metric).fillna('Other')
                csv_path = stem.with_name(stem.name + '.csv')
                icc_df.to_csv(csv_path, index=False)
                write_metric_inclusion(
                    stem.with_name(stem.name + '_metric_inclusion.tsv'),
                    profile_name,
                    analysis_set,
                    qc_mode,
                    tissue,
                    expected_labels,
                    observed_labels,
                    set(icc_df['metric_key']),
                    display,
                )
                print(f'Wrote: {csv_path}', flush=True)


if __name__ == '__main__':
    main()
