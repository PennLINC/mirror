#!/usr/bin/env python3
"""Compute voxelwise correlation matrices in MNI space.

For each subject/session and tissue mask, this script loads configured
space-MNI152NLin2009cAsym scalar maps, computes pairwise-valid voxelwise
correlations, Fisher-z transforms them, and averages first within subject and
then across subjects. By default, only the primary metric set and
cortical-GM/WM compartments are processed. Cortical GM comes from each
subject's precomputed MNI-space sMRIPrep ribbon, while WM comes from the
subject MNI dseg. Deep GM and all-GM outputs can be requested explicitly.
Selecting the full set also writes the primary result view.
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

from utils.metrics import (  # noqa: E402
    ANALYSIS_SET_CHOICES,
    build_metric_specs,
    metric_display_labels,
    metric_order,
    metric_specs_for_analysis,
    primary_metric_specs,
    selected_analysis_sets,
)
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
from utils.tissue_masks import (  # noqa: E402
    SPACE,
    TISSUES,
    build_subject_tissue_masks,
    build_template_tissue_masks,
    find_existing_mni_ribbon,
    find_smriprep_dseg,
    load_like,
    metric_registry_tissue,
    subject_mask_source,
)
from utils.paths import OUTPUT_DERIVATIVES_ROOT, PROJECT_ROOT, SOURCE_DERIVATIVES_ROOT  # noqa: E402


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
            f'{", ".join(missing)}. Activate the MIRROR processing environment first.'
        )


def compute_profile_correlations(
    data: pd.DataFrame,
    tissue_mask: np.ndarray,
    tissue: str,
    method: str,
    outlier_z: float,
    min_voxels: int,
    outlier_masks: tuple[np.ndarray, ...] | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, dict[str, object]]:
    labels = list(data.columns)
    matrix = data.to_numpy(dtype=np.float32)
    n_metrics = len(labels)
    n_tissue_voxels = int(np.count_nonzero(tissue_mask))
    valid_masks: list[np.ndarray] = []
    compartment_masks = (
        tuple(np.asarray(mask, dtype=bool) for mask in outlier_masks)
        if outlier_masks is not None
        else (np.asarray(tissue_mask, dtype=bool),)
    )

    for col_idx in range(n_metrics):
        valid_mask = np.zeros(tissue_mask.shape, dtype=bool)
        for compartment_mask in compartment_masks:
            finite_nonzero = (
                compartment_mask
                & tissue_mask
                & np.isfinite(matrix[:, col_idx])
                & (matrix[:, col_idx] != 0)
            )
            if np.any(finite_nonzero):
                outlier_mask = robust_outlier_mask(matrix[finite_nonzero, col_idx], outlier_z)
                valid_mask[np.flatnonzero(finite_nonzero)] = outlier_mask
        valid_masks.append(valid_mask)

    corr = np.full((n_metrics, n_metrics), np.nan, dtype=float)
    voxel_counts = np.zeros((n_metrics, n_metrics), dtype=np.int64)
    voxel_proportions = np.full((n_metrics, n_metrics), np.nan, dtype=float)

    for idx in range(n_metrics):
        corr[idx, idx] = 1.0
        n_valid = int(np.count_nonzero(valid_masks[idx]))
        voxel_counts[idx, idx] = n_valid
        voxel_proportions[idx, idx] = n_valid / n_tissue_voxels if n_tissue_voxels else np.nan

    for idx_a in range(n_metrics):
        for idx_b in range(idx_a + 1, n_metrics):
            pair_mask = valid_masks[idx_a] & valid_masks[idx_b]
            n_voxels = int(np.count_nonzero(pair_mask))
            voxel_counts[idx_a, idx_b] = n_voxels
            voxel_counts[idx_b, idx_a] = n_voxels
            pair_proportion = n_voxels / n_tissue_voxels if n_tissue_voxels else np.nan
            voxel_proportions[idx_a, idx_b] = pair_proportion
            voxel_proportions[idx_b, idx_a] = pair_proportion
            if n_voxels < min_voxels:
                continue
            values_a = matrix[pair_mask, idx_a]
            values_b = matrix[pair_mask, idx_b]
            if method == 'spearman':
                values_a = rankdata(values_a).astype(np.float32)
                values_b = rankdata(values_b).astype(np.float32)
            elif method != 'pearson':
                raise ValueError(f'Unsupported correlation method: {method}')
            if np.std(values_a) == 0 or np.std(values_b) == 0:
                continue
            corr_value = float(np.corrcoef(values_a, values_b)[0, 1])
            corr[idx_a, idx_b] = corr_value
            corr[idx_b, idx_a] = corr_value

    max_pair_voxels = int(np.max(voxel_counts)) if voxel_counts.size else 0
    upper_triangle = np.triu(np.ones((n_metrics, n_metrics), dtype=bool), k=1)
    valid_pairs = upper_triangle & np.isfinite(corr)
    pair_counts = voxel_counts[upper_triangle]
    pair_proportions = voxel_proportions[upper_triangle]
    finite_pair_proportions = pair_proportions[np.isfinite(pair_proportions)]
    diagnostics = {
        'tissue': tissue,
        'n_metrics': len(labels),
        'n_tissue_voxels': n_tissue_voxels,
        'n_valid_metric_pairs': int(np.count_nonzero(valid_pairs)),
        'n_total_metric_pairs': int(n_metrics * (n_metrics - 1) / 2),
        'min_pairwise_voxels': int(np.min(pair_counts)) if pair_counts.size else 0,
        'median_pairwise_voxels': float(np.median(pair_counts)) if pair_counts.size else np.nan,
        'max_pairwise_voxels': max_pair_voxels,
        'min_pairwise_proportion': (
            float(np.min(finite_pair_proportions)) if finite_pair_proportions.size else np.nan
        ),
        'median_pairwise_proportion': (
            float(np.median(finite_pair_proportions)) if finite_pair_proportions.size else np.nan
        ),
        'max_pairwise_proportion': (
            float(np.max(finite_pair_proportions)) if finite_pair_proportions.size else np.nan
        ),
    }
    if not np.any(valid_pairs):
        count_df = pd.DataFrame(voxel_counts, index=labels, columns=labels)
        proportion_df = pd.DataFrame(voxel_proportions, index=labels, columns=labels)
        return None, count_df, proportion_df, diagnostics
    corr = np.clip(corr, -0.999999, 0.999999)
    corr_df = pd.DataFrame(corr, index=labels, columns=labels)
    count_df = pd.DataFrame(voxel_counts, index=labels, columns=labels)
    proportion_df = pd.DataFrame(voxel_proportions, index=labels, columns=labels)
    return corr_df, count_df, proportion_df, diagnostics


def pairwise_coverage_rows(
    count_df: pd.DataFrame,
    proportion_df: pd.DataFrame,
    subject: str,
    session: str,
    tissue: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    labels = list(count_df.index)
    for idx_a, metric_a in enumerate(labels):
        for idx_b in range(idx_a + 1, len(labels)):
            metric_b = labels[idx_b]
            rows.append(
                {
                    'subject': subject,
                    'session': session,
                    'tissue': tissue,
                    'metric_a': metric_a,
                    'metric_b': metric_b,
                    'n_voxels': int(count_df.loc[metric_a, metric_b]),
                    'proportion_tissue_voxels': float(proportion_df.loc[metric_a, metric_b]),
                }
            )
    return rows


def write_metric_inclusion(
    out_file: Path,
    tissue: str,
    analysis_set: str,
    expected_labels: list[str],
    observed_labels: set[str],
    included_labels: list[str],
    display: dict[str, str],
) -> None:
    included = set(included_labels)
    rows = []
    for label in expected_labels:
        observed = label in observed_labels
        in_matrix = label in included
        rows.append(
            {
                'analysis_set': analysis_set,
                'tissue': tissue,
                'metric_key': label,
                'metric': display.get(label, label),
                'expected': True,
                'observed_after_qc': observed,
                'included': in_matrix,
                'reason_if_not_included': (
                    ''
                    if in_matrix
                    else (
                        'not_observed_after_qc'
                        if not observed
                        else 'fewer_than_min_metrics_or_no_valid_correlations'
                    )
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out_file, sep='\t', index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--project-root',
        type=Path,
        default=PROJECT_ROOT,
        help='Project root containing derivatives, code, and data.',
    )
    parser.add_argument('--derivatives-dir', type=Path, default=None)
    parser.add_argument('--run-derivatives-dir', type=Path, default=OUTPUT_DERIVATIVES_ROOT)
    parser.add_argument('--patterns-file', type=Path, default=None)
    parser.add_argument('--qc-file', type=Path, default=None)
    parser.add_argument('--subject-id', action='append', help='Subject(s), with or without sub-.')
    parser.add_argument('--session-id', action='append', help='Session(s), with or without ses-.')
    parser.add_argument(
        '--analysis-set',
        choices=ANALYSIS_SET_CHOICES,
        default='primary',
        help='Metric set to process. Default: primary.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Defaults to <project-root>/derivatives/mni_voxelwise_correlations.',
    )
    parser.add_argument('--outlier-z', type=float, default=6.0)
    parser.add_argument('--min-voxels', type=int, default=2)
    parser.add_argument(
        '--correlation',
        choices=('pearson', 'spearman'),
        default='pearson',
        help='Voxelwise correlation method. Spearman ranks voxels within each metric first.',
    )
    parser.add_argument(
        '--min-metrics',
        type=int,
        default=2,
        help='Minimum number of selected metrics required for a subject/session profile.',
    )
    parser.add_argument(
        '--tissue-mask-source',
        choices=('subject', 'template'),
        default='subject',
        help='Use subject-specific ribbon/dseg masks, or fixed deterministic template masks.',
    )
    parser.add_argument(
        '--template-dseg',
        type=Path,
        default=None,
        help='Deterministic FreeSurfer aseg dseg used to define template tissue compartments.',
    )
    parser.add_argument(
        '--tissue',
        action='append',
        choices=TISSUES,
        default=None,
        help='Tissue compartment(s) to process; defaults to cortical GM and WM.',
    )
    parser.add_argument('--gm-erosion-mm', type=float, default=0.0)
    parser.add_argument('--wm-erosion-mm', type=float, default=0.0)
    parser.add_argument(
        '--no-qc',
        action='store_true',
        help='Do not apply manual modality QC even if the default QC file exists.',
    )
    args = parser.parse_args()
    args.project_root = args.project_root.expanduser().resolve()
    args.derivatives_dir = (
        args.derivatives_dir.expanduser().resolve()
        if args.derivatives_dir
        else SOURCE_DERIVATIVES_ROOT
    )
    args.run_derivatives_dir = args.run_derivatives_dir.expanduser().resolve()
    args.patterns_file = (
        args.patterns_file.expanduser().resolve()
        if args.patterns_file
        else REPO_ROOT / 'configuration' / 'metrics.yml'
    )
    if not args.patterns_file.exists():
        fallback = REPO_ROOT / 'configuration' / 'metrics.yml'
        args.patterns_file = fallback
    if args.qc_file is None and not args.no_qc:
        candidates = (REPO_ROOT / 'data' / 'qc' / 'manual_qc_modality.tsv',)
        args.qc_file = next((path for path in candidates if path.exists()), None)
    elif args.qc_file is not None:
        args.qc_file = args.qc_file.expanduser().resolve()
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else OUTPUT_DERIVATIVES_ROOT / 'mni_voxelwise_correlations'
    )
    data_candidates = (REPO_ROOT / 'data' / 'atlases',)
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
    if args.gm_erosion_mm < 0 or args.wm_erosion_mm < 0:
        parser.error('Tissue mask erosion distances must be nonnegative.')
    args.tissues = args.tissue if args.tissue else ['cortical_gm', 'wm']
    return args


def main() -> None:
    args = parse_args()
    require_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patterns = load_patterns(args.patterns_file)
    all_specs = build_metric_specs(args.patterns_file)
    primary_specs = primary_metric_specs(all_specs)
    if len(primary_specs) != len(metric_order(all_specs, 'primary')):
        raise RuntimeError('Primary metric registry contains duplicate labels.')
    analysis_sets = selected_analysis_sets(args.analysis_set)
    processing_set = 'primary' if args.analysis_set == 'primary' else 'full'
    if processing_set == 'primary':
        # Load the union of tissue-specific primary inputs. A global primary
        # selection contains the WM-compatible ICVF record, while cortical GM
        # requires the separate GM-NODDI ICVF map with the same display label.
        selected_pattern_keys = {
            spec.pattern_key
            for tissue in ('gm', 'wm')
            for spec in metric_specs_for_analysis(
                all_specs,
                processing_set,
                tissue=tissue,
            )
        }
        specs = [spec for spec in all_specs if spec.pattern_key in selected_pattern_keys]
    else:
        specs = metric_specs_for_analysis(all_specs, processing_set)
    qc = load_qc_table(args.qc_file)

    subjects = (
        [normalize_subject(subject) for subject in args.subject_id]
        if args.subject_id
        else discover_subjects(args.derivatives_dir)
    )
    diagnostics: list[dict[str, object]] = []
    pairwise_coverage: list[dict[str, object]] = []
    template_reference = None
    template_tissue_masks = None
    if args.tissue_mask_source == 'template':
        template_reference, template_tissue_masks = build_template_tissue_masks(
            args.template_dseg,
            args.gm_erosion_mm,
            args.wm_erosion_mm,
        )
    z_mats: dict[str, list[pd.DataFrame]] = {tissue: [] for tissue in args.tissues}
    count_mats: dict[str, list[pd.DataFrame]] = {tissue: [] for tissue in args.tissues}
    proportion_mats: dict[str, list[pd.DataFrame]] = {
        tissue: [] for tissue in args.tissues
    }

    for subject in subjects:
        sessions = (
            [normalize_session(session) for session in args.session_id]
            if args.session_id
            else discover_sessions(args.derivatives_dir, subject)
        )
        subject_z_mats: dict[str, list[pd.DataFrame]] = {
            tissue: [] for tissue in args.tissues
        }
        subject_count_mats: dict[str, list[pd.DataFrame]] = {
            tissue: [] for tissue in args.tissues
        }
        subject_proportion_mats: dict[str, list[pd.DataFrame]] = {
            tissue: [] for tissue in args.tissues
        }
        for session in sessions:
            metric_paths = metric_paths_for_session(
                args.derivatives_dir,
                patterns,
                specs,
                qc,
                subject,
                session,
                SPACE,
            )
            if len(metric_paths) < args.min_metrics:
                print(f'Skipping {subject} {session}: only {len(metric_paths)} metrics')
                continue

            if args.tissue_mask_source == 'subject':
                dseg_path = find_smriprep_dseg(args.derivatives_dir, subject, session, SPACE)
                if dseg_path is None:
                    diagnostics.append(
                        {
                            'subject': subject,
                            'session': session,
                            'tissue': 'all',
                            'n_metrics': len(metric_paths),
                            'n_tissue_voxels': 0,
                            'n_valid_metric_pairs': 0,
                            'n_total_metric_pairs': 0,
                            'reason': 'missing_subject_dseg',
                            'tissue_mask_source': args.tissue_mask_source,
                            'tissue_mask_file': '',
                            'metrics': ','.join(metric_paths),
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
                            'tissue': 'all',
                            'n_metrics': len(metric_paths),
                            'n_tissue_voxels': 0,
                            'n_valid_metric_pairs': 0,
                            'n_total_metric_pairs': 0,
                            'reason': str(exc),
                            'tissue_mask_source': args.tissue_mask_source,
                            'tissue_mask_file': str(dseg_path),
                            'metrics': ','.join(metric_paths),
                        }
                    )
                    print(f'Skipping {subject} {session}: {exc}')
                    continue
                tissue_mask_file = subject_mask_source(
                    dseg_path,
                    ribbon_path,
                    args.template_dseg,
                )
            else:
                reference = template_reference
                tissue_masks = template_tissue_masks
                if reference is None or tissue_masks is None:
                    raise RuntimeError('Template tissue masks were not initialized.')
                tissue_mask_file = str(args.template_dseg)

            metric_data = {
                label: load_like(path, reference, order=1).reshape(-1)
                for label, path in metric_paths.items()
            }
            data = pd.DataFrame(metric_data)
            for tissue in args.tissues:
                analysis_data = data
                tissue_labels = [
                    label
                    for label in metric_order(
                        all_specs,
                        processing_set,
                        tissue=metric_registry_tissue(tissue),
                    )
                    if label in analysis_data.columns
                ]
                tissue_data = analysis_data.loc[:, tissue_labels]
                if len(tissue_labels) < args.min_metrics:
                    diagnostics.append(
                        {
                            'subject': subject,
                            'session': session,
                            'tissue': tissue,
                            'n_metrics': len(tissue_labels),
                            'n_tissue_voxels': int(np.count_nonzero(tissue_masks[tissue])),
                            'n_valid_metric_pairs': 0,
                            'n_total_metric_pairs': 0,
                            'reason': 'too_few_tissue_eligible_metrics',
                            'tissue_mask_source': args.tissue_mask_source,
                            'tissue_mask_file': tissue_mask_file,
                            'metrics': ','.join(tissue_labels),
                        }
                    )
                    continue
                corr, counts, proportions, diag = compute_profile_correlations(
                    tissue_data,
                    tissue_masks[tissue],
                    tissue,
                    method=args.correlation,
                    outlier_z=args.outlier_z,
                    min_voxels=args.min_voxels,
                    outlier_masks=None,
                )
                diag.update(
                    {
                        'subject': subject,
                        'session': session,
                        'tissue_mask_source': args.tissue_mask_source,
                        'tissue_mask_file': tissue_mask_file,
                        'metrics': ','.join(tissue_data.columns),
                    }
                )
                diagnostics.append(diag)
                if counts is None or proportions is None:
                    continue
                pairwise_coverage.extend(
                    pairwise_coverage_rows(counts, proportions, subject, session, tissue)
                )
                if corr is None:
                    continue
                z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
                np.fill_diagonal(z.values, 0.0)
                subject_z_mats[tissue].append(z)
                subject_count_mats[tissue].append(counts)
                subject_proportion_mats[tissue].append(proportions)

        for tissue, mats in subject_z_mats.items():
            if not mats:
                continue
            all_labels = sorted({label for mat in mats for label in mat.index})
            stack = np.stack(
                [
                    mat.reindex(index=all_labels, columns=all_labels).to_numpy(dtype=float)
                    for mat in mats
                ]
            )
            mean_z = pd.DataFrame(
                np.nanmean(stack, axis=0),
                index=all_labels,
                columns=all_labels,
            )
            z_mats[tissue].append(mean_z)
            count_stack = np.stack(
                [
                    mat.reindex(index=all_labels, columns=all_labels).to_numpy(dtype=float)
                    for mat in subject_count_mats[tissue]
                ]
            )
            proportion_stack = np.stack(
                [
                    mat.reindex(index=all_labels, columns=all_labels).to_numpy(dtype=float)
                    for mat in subject_proportion_mats[tissue]
                ]
            )
            count_mats[tissue].append(
                pd.DataFrame(np.nanmean(count_stack, axis=0), index=all_labels, columns=all_labels)
            )
            proportion_mats[tissue].append(
                pd.DataFrame(
                    np.nanmean(proportion_stack, axis=0),
                    index=all_labels,
                    columns=all_labels,
                )
            )

    pd.DataFrame(diagnostics).to_csv(
        args.output_dir / 'mni_voxelwise_correlation_diagnostics.tsv',
        sep='\t',
        index=False,
    )
    pd.DataFrame(pairwise_coverage).to_csv(
        args.output_dir / 'mni_voxelwise_pairwise_correlation_coverage.tsv',
        sep='\t',
        index=False,
    )

    for tissue, mats in z_mats.items():
        if not mats:
            continue
        orders = {
            analysis_set: metric_order(
                all_specs,
                analysis_set,
                tissue=metric_registry_tissue(tissue),
            )
            for analysis_set in analysis_sets
        }
        display_labels = {
            analysis_set: metric_display_labels(
                all_specs,
                analysis_set,
                tissue=metric_registry_tissue(tissue),
            )
            for analysis_set in orders
        }
        processing_labels = metric_order(
            all_specs,
            processing_set,
            tissue=metric_registry_tissue(tissue),
        )
        full_labels = [
            label for label in processing_labels if any(label in mat.index for mat in mats)
        ]
        observed_labels = set(full_labels)
        stack = np.stack(
            [
                mat.reindex(index=full_labels, columns=full_labels).to_numpy(dtype=float)
                for mat in mats
            ]
        )
        full_mean_z = pd.DataFrame(
            np.nanmean(stack, axis=0), index=full_labels, columns=full_labels
        )
        full_n_subjects = pd.DataFrame(
            np.sum(np.isfinite(stack), axis=0),
            index=full_labels,
            columns=full_labels,
        )
        count_stack = np.stack(
            [
                mat.reindex(index=full_labels, columns=full_labels).to_numpy(dtype=float)
                for mat in count_mats[tissue]
            ]
        )
        proportion_stack = np.stack(
            [
                mat.reindex(index=full_labels, columns=full_labels).to_numpy(dtype=float)
                for mat in proportion_mats[tissue]
            ]
        )
        full_mean_counts = pd.DataFrame(
            np.nanmean(count_stack, axis=0),
            index=full_labels,
            columns=full_labels,
        )
        full_mean_proportions = pd.DataFrame(
            np.nanmean(proportion_stack, axis=0),
            index=full_labels,
            columns=full_labels,
        )

        for analysis_set, ordered_labels in orders.items():
            labels = [label for label in ordered_labels if label in full_mean_z.index]
            label_map = display_labels[analysis_set]
            stem = f'mni_voxelwise_{analysis_set}_{tissue}_{args.correlation}'
            write_metric_inclusion(
                args.output_dir / f'{stem}_metric_inclusion.tsv',
                tissue,
                analysis_set,
                ordered_labels,
                observed_labels,
                labels,
                label_map,
            )
            if len(labels) < 2:
                continue
            mean_z = full_mean_z.reindex(index=labels, columns=labels)
            mean_r = pd.DataFrame(np.tanh(mean_z), index=labels, columns=labels)
            np.fill_diagonal(mean_r.values, 1.0)
            mean_counts = full_mean_counts.reindex(index=labels, columns=labels)
            mean_proportions = full_mean_proportions.reindex(index=labels, columns=labels)
            n_subjects = full_n_subjects.reindex(index=labels, columns=labels)
            mean_z = mean_z.rename(index=label_map, columns=label_map)
            mean_r = mean_r.rename(index=label_map, columns=label_map)
            mean_counts = mean_counts.rename(index=label_map, columns=label_map)
            mean_proportions = mean_proportions.rename(index=label_map, columns=label_map)
            n_subjects = n_subjects.rename(index=label_map, columns=label_map)
            mean_r.to_csv(
                args.output_dir / f'{stem}_r.tsv',
                sep='\t',
                index_label='metric',
            )
            mean_z.to_csv(
                args.output_dir / f'{stem}_fisherz.tsv',
                sep='\t',
                index_label='metric',
            )
            mean_counts.to_csv(
                args.output_dir / f'{stem}_mean_pairwise_nvoxels.tsv',
                sep='\t',
                index_label='metric',
            )
            mean_proportions.to_csv(
                args.output_dir / f'{stem}_mean_pairwise_proportion.tsv',
                sep='\t',
                index_label='metric',
            )
            n_subjects.to_csv(
                args.output_dir / f'{stem}_nsubjects.tsv',
                sep='\t',
                index_label='metric',
            )


if __name__ == '__main__':
    main()
