#!/usr/bin/env python3
"""Compute within-scan voxelwise GM-vs-WM effect sizes for MNI scalar maps.

For each subject/session/metric, this script compares subject-specific cortical
GM with subject-specific WM by default. Cortical GM comes from each subject's
precomputed MNI-space sMRIPrep ribbon; the remaining masks combine the subject
MNI dseg with deterministic template labels. Deep and all-GM comparisons can
be requested explicitly. Signed GM - WM effect sizes are averaged across
repeated sessions and summarized across subjects.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata
except ImportError:  # pragma: no cover - checked after argparse handles --help
    nib = None
    np = None
    pd = None
    rankdata = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.mni_analysis import (  # noqa: E402
    discover_sessions,
    discover_subjects,
    load_patterns,
    load_qc_table,
    metric_paths_for_session,
    normalize_session,
    normalize_subject,
    robust_outlier_mask,
)
from utils.metrics import (  # noqa: E402
    MetricSpec,
    build_metric_specs,
    gm_noddi_hybrid_pairs,
    metric_display_labels,
    metric_specs_for_analysis,
)
from utils.paths import (  # noqa: E402
    CODE_ROOT,
    OUTPUT_DERIVATIVES_ROOT,
    PROJECT_ROOT,
    SOURCE_DERIVATIVES_ROOT,
)
from utils.tissue_masks import (  # noqa: E402
    GM_TISSUES,
    SPACE,
    build_subject_tissue_masks,
    find_existing_mni_ribbon,
    find_smriprep_dseg,
    load_like,
    subject_mask_source,
)


EFFECT_COLUMNS = (
    'mean_difference',
    'median_difference',
    'percent_median_difference',
    'cohen_d',
    'hedges_g',
    'robust_median_d',
    'signed_auc',
)


def require_dependencies() -> None:
    missing = [
        name
        for name, module in (
            ('nibabel', nib),
            ('numpy', np),
            ('pandas', pd),
            ('scipy.stats', rankdata),
        )
        if module is None
    ]
    if missing:
        raise RuntimeError(
            'Missing required Python packages: '
            f'{", ".join(missing)}. Activate the MIRROR analysis environment first.'
        )


def resolve_default_qc(no_qc: bool, requested: Path | None) -> Path | None:
    if no_qc:
        return None
    if requested is not None:
        return requested.expanduser().resolve()
    candidates = (CODE_ROOT / 'data' / 'qc' / 'manual_qc_modality.tsv',)
    return next((path for path in candidates if path.exists()), None)


def finite_nonzero_tissue_values(
    data: np.ndarray,
    mask: np.ndarray,
    outlier_z: float,
) -> np.ndarray:
    values = np.asarray(data[np.asarray(mask, dtype=bool)], dtype=np.float32)
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return values.astype(float)
    keep = robust_outlier_mask(values, outlier_z)
    return values[keep].astype(float)


def downsample_values(
    values: np.ndarray,
    max_voxels: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_voxels is None or values.size <= max_voxels:
        return values
    indices = rng.choice(values.size, size=max_voxels, replace=False)
    return values[indices]


def safe_percent_difference(gm_median: float, wm_median: float) -> float:
    denom = abs(wm_median)
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return 100.0 * (gm_median - wm_median) / denom


def cohen_d(gm: np.ndarray, wm: np.ndarray) -> float:
    if gm.size < 2 or wm.size < 2:
        return np.nan
    gm_sd = float(np.std(gm, ddof=1))
    wm_sd = float(np.std(wm, ddof=1))
    pooled = np.sqrt((gm_sd**2 + wm_sd**2) / 2.0)
    if not np.isfinite(pooled) or pooled <= 0:
        return np.nan
    return (float(np.mean(gm)) - float(np.mean(wm))) / pooled


def hedges_g_from_d(d_value: float, n_gm: int, n_wm: int) -> float:
    if not np.isfinite(d_value):
        return np.nan
    df = n_gm + n_wm - 2
    if df <= 1:
        return np.nan
    correction = 1.0 - 3.0 / (4.0 * df - 1.0)
    return d_value * correction


def robust_median_d(gm: np.ndarray, wm: np.ndarray) -> float:
    gm_median = float(np.median(gm))
    wm_median = float(np.median(wm))
    gm_mad = float(np.median(np.abs(gm - gm_median)))
    wm_mad = float(np.median(np.abs(wm - wm_median)))
    pooled_mad_sd = 1.4826 * np.sqrt((gm_mad**2 + wm_mad**2) / 2.0)
    if not np.isfinite(pooled_mad_sd) or pooled_mad_sd <= 0:
        return np.nan
    return (gm_median - wm_median) / pooled_mad_sd


def signed_auc_effect(gm: np.ndarray, wm: np.ndarray) -> float:
    if gm.size == 0 or wm.size == 0:
        return np.nan
    combined = np.concatenate([gm, wm])
    ranks = rankdata(combined, method='average')
    rank_sum_gm = float(np.sum(ranks[: gm.size]))
    auc = (rank_sum_gm - gm.size * (gm.size + 1.0) / 2.0) / (gm.size * wm.size)
    return 2.0 * auc - 1.0


def compute_effect_row(
    subject: str,
    session: str,
    spec: MetricSpec,
    display_metric: str,
    gm_tissue: str,
    gm_path: Path,
    wm_path: Path,
    tissue_mask_file: str,
    gm_data: np.ndarray,
    wm_data: np.ndarray,
    tissue_masks: dict[str, np.ndarray],
    outlier_z: float,
    min_voxels: int,
    max_voxels_per_tissue: int | None,
    rng: np.random.Generator,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    gm = finite_nonzero_tissue_values(gm_data, tissue_masks[gm_tissue], outlier_z)
    wm = finite_nonzero_tissue_values(wm_data, tissue_masks['wm'], outlier_z)
    metric_file = str(wm_path) if gm_path == wm_path else f'gm={gm_path};wm={wm_path}'
    n_gm_available = int(gm.size)
    n_wm_available = int(wm.size)
    if n_gm_available < min_voxels or n_wm_available < min_voxels:
        return None, {
            'subject': subject,
            'session': session,
            'metric_key': spec.label,
            'display_metric': display_metric,
            'source_image': spec.source_image,
            'gm_tissue': gm_tissue,
            'reason': 'too_few_valid_voxels',
            'n_gm_voxels': n_gm_available,
            'n_wm_voxels': n_wm_available,
            'metric_file': metric_file,
            'gm_metric_file': str(gm_path),
            'wm_metric_file': str(wm_path),
            'tissue_mask_file': str(tissue_mask_file),
        }

    gm = downsample_values(gm, max_voxels_per_tissue, rng)
    wm = downsample_values(wm, max_voxels_per_tissue, rng)
    gm_mean = float(np.mean(gm))
    wm_mean = float(np.mean(wm))
    gm_median = float(np.median(gm))
    wm_median = float(np.median(wm))
    d_value = cohen_d(gm, wm)
    return {
        'subject': subject,
        'session': session,
        'metric_key': spec.label,
        'primary_label': spec.primary_label,
        'display_metric': display_metric,
        'source_image': spec.source_image,
        'gm_tissue': gm_tissue,
        'metric_file': metric_file,
        'gm_metric_file': str(gm_path),
        'wm_metric_file': str(wm_path),
        'tissue_mask_file': str(tissue_mask_file),
        'n_gm_voxels_available': n_gm_available,
        'n_wm_voxels_available': n_wm_available,
        'n_gm_voxels_used': int(gm.size),
        'n_wm_voxels_used': int(wm.size),
        'gm_mean': gm_mean,
        'wm_mean': wm_mean,
        'gm_median': gm_median,
        'wm_median': wm_median,
        'mean_difference': gm_mean - wm_mean,
        'median_difference': gm_median - wm_median,
        'percent_median_difference': safe_percent_difference(gm_median, wm_median),
        'cohen_d': d_value,
        'hedges_g': hedges_g_from_d(d_value, gm.size, wm.size),
        'robust_median_d': robust_median_d(gm, wm),
        'signed_auc': signed_auc_effect(gm, wm),
    }, None


def average_sessions(session_rows: pd.DataFrame) -> pd.DataFrame:
    if session_rows.empty:
        return session_rows.copy()
    group_cols = [
        'subject',
        'gm_tissue',
        'metric_key',
        'primary_label',
        'display_metric',
        'source_image',
    ]
    numeric_cols = [
        column
        for column in session_rows.columns
        if column
        not in {
            *group_cols,
            'session',
            'metric_file',
            'gm_metric_file',
            'wm_metric_file',
            'tissue_mask_file',
        }
        and pd.api.types.is_numeric_dtype(session_rows[column])
    ]
    out = session_rows.groupby(group_cols, sort=False)[numeric_cols].mean().reset_index()
    session_info = (
        session_rows.groupby(group_cols, sort=False)
        .agg(
            n_sessions=('session', 'nunique'),
            sessions=('session', lambda values: ','.join(sorted(map(str, set(values))))),
            metric_files=('metric_file', lambda values: ';'.join(map(str, values))),
            gm_metric_files=('gm_metric_file', lambda values: ';'.join(map(str, values))),
            wm_metric_files=('wm_metric_file', lambda values: ';'.join(map(str, values))),
            tissue_mask_files=(
                'tissue_mask_file',
                lambda values: ';'.join(sorted(map(str, set(values)))),
            ),
        )
        .reset_index()
    )
    return out.merge(session_info, on=group_cols, how='left')


def summarize_subjects(subject_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if subject_rows.empty:
        return pd.DataFrame(rows)
    for (
        gm_tissue,
        metric_key,
        primary_label,
        display_metric,
        source_image,
    ), group in subject_rows.groupby(
        ['gm_tissue', 'metric_key', 'primary_label', 'display_metric', 'source_image'],
        sort=False,
    ):
        row: dict[str, object] = {
            'gm_tissue': gm_tissue,
            'metric_key': metric_key,
            'primary_label': primary_label,
            'display_metric': display_metric,
            'source_image': source_image,
            'n_subjects': int(group['subject'].nunique()),
            'n_subject_sessions': int(group['n_sessions'].sum()),
        }
        for column in EFFECT_COLUMNS:
            values = pd.to_numeric(group[column], errors='coerce').to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            row[f'{column}_mean'] = float(np.mean(finite)) if finite.size else np.nan
            row[f'{column}_median'] = float(np.median(finite)) if finite.size else np.nan
            row[f'{column}_q25'] = float(np.percentile(finite, 25)) if finite.size else np.nan
            row[f'{column}_q75'] = float(np.percentile(finite, 75)) if finite.size else np.nan
            row[f'{column}_sd'] = float(np.std(finite, ddof=1)) if finite.size > 1 else np.nan
            row[f'{column}_sem'] = (
                float(np.std(finite, ddof=1) / np.sqrt(finite.size)) if finite.size > 1 else np.nan
            )
        rows.append(row)
    return (
        pd.DataFrame(rows).sort_values(['gm_tissue', 'robust_median_d_mean']).reset_index(drop=True)
    )


def write_metric_inclusion(
    out_file: Path,
    specs: list[MetricSpec],
    analysis_set: str,
    gm_tissues: list[str],
    subject_rows: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    included = (
        set(zip(subject_rows['gm_tissue'], subject_rows['metric_key']))
        if not subject_rows.empty
        else set()
    )
    observed_failed = (
        set(zip(diagnostics['gm_tissue'], diagnostics['metric_key']))
        if not diagnostics.empty and 'gm_tissue' in diagnostics
        else set()
    )
    globally_failed = (
        set(diagnostics.loc[diagnostics['gm_tissue'] == 'all', 'metric_key'])
        if not diagnostics.empty and 'gm_tissue' in diagnostics
        else set()
    )
    rows = []
    for gm_tissue in gm_tissues:
        for spec in specs:
            key = (gm_tissue, spec.label)
            rows.append(
                {
                    'analysis_set': analysis_set,
                    'gm_tissue': gm_tissue,
                    'metric_key': spec.label,
                    'primary_label': spec.primary_label,
                    'source_image': spec.source_image,
                    'expected': True,
                    'included': key in included,
                    'observed_but_failed': (
                        key in observed_failed or spec.label in globally_failed
                    ),
                    'reason_if_not_included': ''
                    if key in included
                    else (
                        'observed_but_failed'
                        if key in observed_failed or spec.label in globally_failed
                        else 'not_observed'
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(out_file, sep='\t', index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--derivatives-dir', type=Path, default=None)
    parser.add_argument('--run-derivatives-dir', type=Path, default=OUTPUT_DERIVATIVES_ROOT)
    parser.add_argument(
        '--patterns-file', type=Path, default=CODE_ROOT / 'configuration' / 'metrics.yml'
    )
    parser.add_argument('--qc-file', type=Path, default=None)
    parser.add_argument('--subject-id', action='append', help='Subject(s), with or without sub-.')
    parser.add_argument('--session-id', action='append', help='Session(s), with or without ses-.')
    parser.add_argument(
        '--analysis-set',
        choices=('primary', 'full'),
        default='primary',
        help='Metric registry set to analyze.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=OUTPUT_DERIVATIVES_ROOT / 'mni_gm_wm_effect_sizes',
    )
    parser.add_argument('--outlier-z', type=float, default=6.0)
    parser.add_argument(
        '--gm-tissue',
        action='append',
        choices=GM_TISSUES,
        default=None,
        help='GM compartment(s) to compare with WM; defaults to cortical GM.',
    )
    parser.add_argument(
        '--template-dseg',
        type=Path,
        default=None,
        help='Deterministic FreeSurfer aseg dseg used to define deep GM.',
    )
    parser.add_argument('--gm-erosion-mm', type=float, default=0.0)
    parser.add_argument('--wm-erosion-mm', type=float, default=0.0)
    parser.add_argument('--min-voxels', type=int, default=100)
    parser.add_argument(
        '--max-voxels-per-tissue',
        type=int,
        default=0,
        help='Randomly sample at most this many valid voxels per tissue per scan; default 0 uses all voxels.',
    )
    parser.add_argument('--seed', type=int, default=20260818)
    parser.add_argument('--no-qc', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require_dependencies()
    args.project_root = args.project_root.expanduser().resolve()
    args.derivatives_dir = (
        args.derivatives_dir.expanduser().resolve()
        if args.derivatives_dir
        else SOURCE_DERIVATIVES_ROOT
    )
    args.run_derivatives_dir = args.run_derivatives_dir.expanduser().resolve()
    args.patterns_file = args.patterns_file.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The manuscript reports cortical GM versus WM. Deep/all-GM outputs remain
    # available when explicitly requested with one or more --gm-tissue flags.
    args.gm_tissues = args.gm_tissue if args.gm_tissue else ['cortical_gm']
    data_candidates = (CODE_ROOT / 'data' / 'atlases',)
    if args.template_dseg is None:
        args.template_dseg = next(
            (
                directory / f'tpl-{SPACE}_res-01_seg-aseg_dseg.nii.gz'
                for directory in data_candidates
                if (directory / f'tpl-{SPACE}_res-01_seg-aseg_dseg.nii.gz').exists()
            ),
            data_candidates[0] / f'tpl-{SPACE}_res-01_seg-aseg_dseg.nii.gz',
        )
    else:
        args.template_dseg = args.template_dseg.expanduser().resolve()
    if not args.template_dseg.exists():
        raise FileNotFoundError(f'Template aseg dseg not found: {args.template_dseg}')
    if args.max_voxels_per_tissue is not None and args.max_voxels_per_tissue <= 0:
        args.max_voxels_per_tissue = None

    patterns = load_patterns(args.patterns_file)
    all_specs = build_metric_specs(args.patterns_file)
    # A GM-vs-WM effect requires a WM-compatible metric. GM-only NODDI fits
    # are supplied below as the GM half of a hybrid pair and must not also be
    # analyzed as standalone metrics. Likewise, g-ratio is intentionally
    # WM-only and therefore has no valid GM comparison.
    specs = [
        spec
        for spec in metric_specs_for_analysis(all_specs, args.analysis_set)
        if 'wm' in spec.tissues and spec.group != 'G-Ratio'
    ]
    hybrid_pairs = gm_noddi_hybrid_pairs(all_specs)
    hybrid_gm_labels = set(hybrid_pairs.values())
    path_specs_by_label = {spec.label: spec for spec in specs}
    path_specs_by_label.update(
        {spec.label: spec for spec in all_specs if spec.label in hybrid_gm_labels}
    )
    path_specs = list(path_specs_by_label.values())
    display_labels = metric_display_labels(all_specs, args.analysis_set)
    qc = load_qc_table(resolve_default_qc(args.no_qc, args.qc_file))
    subjects = (
        [normalize_subject(subject) for subject in args.subject_id]
        if args.subject_id
        else discover_subjects(args.derivatives_dir)
    )

    session_rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    rng = np.random.default_rng(args.seed)
    for subject in subjects:
        sessions = (
            [normalize_session(session) for session in args.session_id]
            if args.session_id
            else discover_sessions(args.derivatives_dir, subject)
        )
        for session in sessions:
            dseg_path = find_smriprep_dseg(args.derivatives_dir, subject, session, SPACE)
            if dseg_path is None:
                diagnostics.append(
                    {
                        'subject': subject,
                        'session': session,
                        'metric_key': '',
                        'display_metric': '',
                        'source_image': '',
                        'gm_tissue': 'all',
                        'reason': 'missing_subject_dseg',
                        'n_gm_voxels': 0,
                        'n_wm_voxels': 0,
                        'metric_file': '',
                        'tissue_mask_file': '',
                    }
                )
                print(f'Skipping {subject} {session}: missing smriprep MNI dseg')
                continue
            try:
                ribbon_path = find_existing_mni_ribbon(
                    args.run_derivatives_dir, subject, session, SPACE
                )
                if ribbon_path is None:
                    raise FileNotFoundError(
                        'missing precomputed MNI cortical ribbon; run '
                        'analysis/00_prepare_inputs/01_mni_ribbon_masks/'
                        'prepare_mni_ribbon_masks.sh first'
                    )
                reference, tissue_masks = build_subject_tissue_masks(
                    dseg_path,
                    ribbon_path,
                    args.template_dseg,
                    args.gm_erosion_mm,
                    args.wm_erosion_mm,
                )
            except (FileNotFoundError, RuntimeError) as exc:
                diagnostics.append(
                    {
                        'subject': subject,
                        'session': session,
                        'metric_key': '',
                        'display_metric': '',
                        'source_image': '',
                        'gm_tissue': 'all',
                        'reason': str(exc),
                        'n_gm_voxels': 0,
                        'n_wm_voxels': 0,
                        'metric_file': '',
                        'tissue_mask_file': str(dseg_path),
                    }
                )
                print(f'Skipping {subject} {session}: {exc}')
                continue
            tissue_mask_file = subject_mask_source(
                dseg_path,
                ribbon_path,
                args.template_dseg,
            )

            metric_paths = metric_paths_for_session(
                args.derivatives_dir,
                patterns,
                path_specs,
                qc,
                subject,
                session,
                SPACE,
            )
            if not metric_paths:
                print(f'Skipping {subject} {session}: no metric files found after QC')
            spec_by_label = {spec.label: spec for spec in specs}
            for metric_label in [spec.label for spec in specs]:
                path = metric_paths.get(metric_label)
                spec = spec_by_label.get(metric_label)
                if spec is None:
                    continue
                display_metric = display_labels.get(metric_label, metric_label)
                gm_metric_label = hybrid_pairs.get(metric_label, metric_label)
                gm_path = metric_paths.get(gm_metric_label)
                wm_path = path
                if wm_path is None or gm_path is None:
                    missing_parts = []
                    if gm_path is None:
                        missing_parts.append('gm')
                    if wm_path is None:
                        missing_parts.append('wm')
                    diagnostics.append(
                        {
                            'subject': subject,
                            'session': session,
                            'metric_key': metric_label,
                            'display_metric': display_metric,
                            'source_image': spec.source_image,
                            'gm_tissue': 'all',
                            'reason': f'missing_{"_and_".join(missing_parts)}_metric_file_or_qc_failed',
                            'n_gm_voxels': 0,
                            'n_wm_voxels': 0,
                            'metric_file': '',
                            'gm_metric_file': str(gm_path) if gm_path is not None else '',
                            'wm_metric_file': str(wm_path) if wm_path is not None else '',
                            'tissue_mask_file': tissue_mask_file,
                        }
                    )
                    continue
                wm_data = load_like(wm_path, reference, order=1).reshape(-1)
                gm_data = (
                    wm_data
                    if gm_path == wm_path
                    else load_like(gm_path, reference, order=1).reshape(-1)
                )
                for gm_tissue in args.gm_tissues:
                    row, diagnostic = compute_effect_row(
                        subject,
                        session,
                        spec,
                        display_metric,
                        gm_tissue,
                        gm_path,
                        wm_path,
                        tissue_mask_file,
                        gm_data,
                        wm_data,
                        tissue_masks,
                        args.outlier_z,
                        args.min_voxels,
                        args.max_voxels_per_tissue,
                        rng,
                    )
                    if row is not None:
                        session_rows.append(row)
                    if diagnostic is not None:
                        diagnostics.append(diagnostic)

    session_df = pd.DataFrame(session_rows)
    diagnostics_df = pd.DataFrame(diagnostics)
    subject_df = average_sessions(session_df)
    summary_df = summarize_subjects(subject_df)

    session_out = args.output_dir / f'mni_gm_wm_effect_sizes_{args.analysis_set}_session.tsv'
    subject_out = args.output_dir / f'mni_gm_wm_effect_sizes_{args.analysis_set}_subject.tsv'
    summary_out = args.output_dir / f'mni_gm_wm_effect_sizes_{args.analysis_set}_summary.tsv'
    diagnostics_out = (
        args.output_dir / f'mni_gm_wm_effect_sizes_{args.analysis_set}_diagnostics.tsv'
    )
    inclusion_out = (
        args.output_dir / f'mni_gm_wm_effect_sizes_{args.analysis_set}_metric_inclusion.tsv'
    )

    session_df.to_csv(session_out, sep='\t', index=False)
    subject_df.to_csv(subject_out, sep='\t', index=False)
    summary_df.to_csv(summary_out, sep='\t', index=False)
    diagnostics_df.to_csv(diagnostics_out, sep='\t', index=False)
    write_metric_inclusion(
        inclusion_out,
        specs,
        args.analysis_set,
        args.gm_tissues,
        subject_df,
        diagnostics_df,
    )
    for out_file in (session_out, subject_out, summary_out, diagnostics_out, inclusion_out):
        print(f'Wrote: {out_file}')


if __name__ == '__main__':
    main()
