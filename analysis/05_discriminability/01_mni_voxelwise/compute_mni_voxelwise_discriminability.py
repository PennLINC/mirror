#!/usr/bin/env python3
"""Compute voxelwise test-retest discriminability in MNI space.

Each subject/session image is treated as a voxel-feature profile. For each
metric, tissue mask, and analysis set, the script builds a metric-specific
common voxel mask across included profiles, computes profile distances in
chunks, and reports discriminability plus nearest-neighbor accuracy.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from collections import OrderedDict
from pathlib import Path

try:
    import nibabel as nib
    import numpy as np
    import pandas as pd
except ImportError:  # pragma: no cover - checked after --help
    nib = None
    np = None
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import (  # noqa: E402
    build_metric_specs,
    flatten_metric_patterns,
    gm_noddi_hybrid_pairs,
    metric_display_labels,
    metric_order,
)
from utils.tissue_masks import (  # noqa: E402
    SPACE,
    TISSUES as DETERMINISTIC_TISSUES,
    TISSUE_TITLES,
    build_template_tissue_masks,
    load_like,
    metric_registry_tissue,
)
from utils.paths import OUTPUT_DERIVATIVES_ROOT, PROJECT_ROOT, SOURCE_DERIVATIVES_ROOT  # noqa: E402


TISSUES = (*DETERMINISTIC_TISSUES, 'gmwm')
ANALYSIS_SETS = ('primary', 'full')


def require_dependencies() -> None:
    missing = [
        name
        for name, module in (
            ('nibabel', nib),
            ('numpy', np),
            ('pandas', pd),
        )
        if module is None
    ]
    if missing:
        raise RuntimeError(
            'Missing required Python packages: '
            f'{", ".join(missing)}. Activate the MIRROR processing environment first.'
        )


def normalize_subject(value: str) -> str:
    token = str(value).strip()
    return token if token.startswith('sub-') else f'sub-{token}'


def normalize_session(value: str) -> str:
    token = str(value).strip()
    return token if token.startswith('ses-') else f'ses-{token}'


def subject_for_qc(subject: object) -> str:
    return re.sub(r'^sub-', '', str(subject).strip())


def is_pilot_subject(subject: object) -> bool:
    return subject_for_qc(subject).upper().startswith('PILOT')


def session_label(session: object) -> str:
    match = re.search(r'(\d+)', str(session))
    if match is None:
        raise ValueError(f'Could not parse session number from {session}')
    return f'Session {int(match.group(1)):02d}'


def safe_label(value: object) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '-', str(value)).strip('-')


def number_token(value: float) -> str:
    return f'{float(value):g}'.replace('.', 'p')


def load_patterns(path: Path) -> dict[str, str]:
    return flatten_metric_patterns(path)


def first_glob(patterns) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(pattern.parent.glob(pattern.name)))
    unique = sorted(set(matches))
    return unique[0] if unique else None


def pattern_path(
    derivatives: Path,
    rel_pattern: str,
    subject: str,
    session: str,
    space: str,
) -> Path:
    rel_pattern = rel_pattern.replace('_space-MNI152NLin2009cAsym_', '_space-{space}_')
    return derivatives / rel_pattern.format(subject=subject, session=session, space=space)


def discover_subjects(derivatives: Path) -> list[str]:
    roots = (
        derivatives / 'smriprep',
        derivatives / 'qsirecon' / 'derivatives' / 'qsirecon-DIPYDKI',
        derivatives / 'pymp2rage',
        derivatives / 'ihmt',
    )
    return sorted(
        {
            path.name
            for root in roots
            if root.is_dir()
            for path in root.glob('sub-*')
            if path.is_dir() and not is_pilot_subject(path.name)
        }
    )


def load_qc_table(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    qc = pd.read_csv(path, sep='\t')
    qc['participant_id'] = qc['participant_id'].map(subject_for_qc)
    qc = qc.loc[~qc['participant_id'].map(is_pilot_subject)].copy()
    return qc.set_index('participant_id', drop=False)


def qc_passes(qc: pd.DataFrame | None, subject: str, session: str, spec) -> bool:
    if qc is None:
        return True
    if not spec.qc_modalities:
        warnings.warn(f'No QC modality mapping for {spec.label}; applying no modality QC')
        return True
    subject_id = subject_for_qc(subject)
    if subject_id not in qc.index:
        return False
    row = qc.loc[subject_id]
    prefix = session_label(session)
    for modality in spec.qc_modalities:
        column = f'{prefix}--{modality}'
        if column not in qc.columns:
            raise RuntimeError(f'QC file is missing required column: {column}')
        value = row[column]
        if pd.isna(value) or int(value) != 1:
            return False
    return True


def robust_outlier_mask(values: np.ndarray, z_threshold: float) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return finite
    clean = values[finite]
    median = float(np.median(clean))
    mad = float(np.median(np.abs(clean - median)))
    if np.isfinite(mad) and mad > 0:
        robust_z = 0.67448975 * (values - median) / mad
        return finite & (np.abs(robust_z) <= z_threshold)
    q_low, q_high = np.percentile(clean, [0.1, 99.9])
    return finite & (values >= q_low) & (values <= q_high)


def selected_analysis_sets(analysis_set: str) -> list[str]:
    return ['primary', 'full'] if analysis_set == 'full' else ['primary']


def specs_for_analysis_set(
    specs,
    analysis_set: str,
    tissues: list[str] | tuple[str, ...] | None = None,
) -> list:
    by_label = {spec.label: spec for spec in specs}
    ordered_specs = OrderedDict()
    tissue_values = tuple(tissues) if tissues is not None else (None,)
    for tissue in tissue_values:
        for label in metric_order(specs, analysis_set, tissue=tissue):
            if label in by_label:
                ordered_specs.setdefault(label, by_label[label])
    return list(ordered_specs.values())


def select_metric_specs(
    specs,
    analysis_set: str,
    requested: list[str] | None,
    tissues: list[str] | tuple[str, ...] | None = None,
) -> list:
    selected = OrderedDict()
    for current_set in selected_analysis_sets(analysis_set):
        for spec in specs_for_analysis_set(specs, current_set, tissues=tissues):
            selected.setdefault(spec.label, spec)
    if not requested:
        return list(selected.values())

    by_lower = {}
    for spec in selected.values():
        for label in (spec.label, spec.primary_label, spec.pattern_key):
            by_lower.setdefault(str(label).lower(), []).append(spec)

    out = []
    unknown = []
    for label in requested:
        matches = by_lower.get(str(label).strip().lower())
        if not matches:
            unknown.append(str(label))
            continue
        for spec in matches:
            if spec not in out:
                out.append(spec)
    if unknown:
        raise ValueError(
            'Unknown metric label(s): '
            f'{", ".join(unknown)}. Available labels: {", ".join(selected)}'
        )
    return out


def collect_metric_profiles(
    derivatives: Path,
    patterns: dict[str, str],
    qc: pd.DataFrame | None,
    subjects: list[str],
    sessions: list[str],
    spec,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rel_pattern = patterns.get(spec.pattern_key)
    if rel_pattern is None:
        raise RuntimeError(f'metrics.yml has no entry for {spec.label}: {spec.pattern_key}')

    profiles = []
    diagnostics = []
    for subject in subjects:
        for session in sessions:
            record = {
                'metric': spec.label,
                'subject': subject,
                'session': session,
                'included': False,
                'reason': '',
                'metric_file': '',
            }
            if not qc_passes(qc, subject, session, spec):
                record['reason'] = 'failed_or_missing_qc'
                diagnostics.append(record)
                continue
            metric_path = first_glob(
                (pattern_path(derivatives, rel_pattern, subject, session, SPACE),)
            )
            record['metric_file'] = str(metric_path) if metric_path is not None else ''
            if metric_path is None:
                record['reason'] = 'missing_metric'
                diagnostics.append(record)
                continue
            record['included'] = True
            record['reason'] = 'included'
            diagnostics.append(record)
            profiles.append({'subject': subject, 'session': session, 'metric_file': metric_path})
    return profiles, diagnostics


def pair_gmwm_hybrid_profiles(
    wm_profiles: list[dict[str, object]],
    gm_profiles: list[dict[str, object]],
) -> list[dict[str, object]]:
    gm_by_key = {(profile['subject'], profile['session']): profile for profile in gm_profiles}
    paired = []
    for profile in wm_profiles:
        gm_profile = gm_by_key.get((profile['subject'], profile['session']))
        if gm_profile is None:
            continue
        out = dict(profile)
        out['gm_metric_file'] = gm_profile['metric_file']
        paired.append(out)
    return paired


def keep_subjects_with_multiple_sessions(
    profiles: list[dict[str, object]],
    min_sessions: int = 2,
) -> list[dict[str, object]]:
    sessions_by_subject: dict[str, set[str]] = {}
    for profile in profiles:
        sessions_by_subject.setdefault(str(profile['subject']), set()).add(str(profile['session']))
    paired_subjects = {
        subject
        for subject, sessions in sessions_by_subject.items()
        if len(sessions) >= min_sessions
    }
    return [profile for profile in profiles if str(profile['subject']) in paired_subjects]


def common_metric_mask(
    profiles: list[dict[str, object]],
    reference,
    base_mask: np.ndarray,
    outlier_z: float,
    remove_zeros: bool,
    gm_mask: np.ndarray | None = None,
    outlier_masks: tuple[np.ndarray, ...] | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    common = np.asarray(base_mask, dtype=bool).copy()
    compartment_masks = (
        tuple(np.asarray(mask, dtype=bool) for mask in outlier_masks)
        if outlier_masks is not None
        else (np.asarray(base_mask, dtype=bool),)
    )
    values_by_profile = []
    for profile in profiles:
        values = load_like(Path(profile['metric_file']), reference, order=1).reshape(-1)
        if gm_mask is not None and profile.get('gm_metric_file'):
            gm_mask_bool = np.asarray(gm_mask, dtype=bool)
            gm_values = load_like(Path(profile['gm_metric_file']), reference, order=1).reshape(-1)
            values = values.copy()
            values[gm_mask_bool] = gm_values[gm_mask_bool]
        metric_valid = np.zeros(np.asarray(base_mask, dtype=bool).shape, dtype=bool)
        for compartment_mask in compartment_masks:
            valid = np.asarray(base_mask, dtype=bool) & compartment_mask & np.isfinite(values)
            if remove_zeros:
                valid &= values != 0
            if np.any(valid):
                local = robust_outlier_mask(values[valid], outlier_z)
                metric_valid[np.flatnonzero(valid)] = local
        common &= metric_valid
        values_by_profile.append(values)
    return common, values_by_profile


def zscore_feature_chunk(chunk: np.ndarray) -> np.ndarray:
    means = chunk.mean(axis=0, keepdims=True)
    stds = chunk.std(axis=0, ddof=0, keepdims=True)
    return np.divide(chunk - means, stds, out=np.zeros_like(chunk), where=stds > 0)


def pairwise_distances_chunked(
    values_by_profile: list[np.ndarray],
    mask: np.ndarray,
    distance_metric: str,
    zscore_features: bool,
    chunk_size: int,
) -> np.ndarray:
    indices = np.flatnonzero(mask)
    n_profiles = len(values_by_profile)
    if distance_metric == 'euclidean':
        sums = np.zeros((n_profiles, n_profiles), dtype=np.float64)
        for start in range(0, len(indices), chunk_size):
            chunk_idx = indices[start : start + chunk_size]
            chunk = np.vstack([values[chunk_idx] for values in values_by_profile]).astype(
                np.float64
            )
            if zscore_features:
                chunk = zscore_feature_chunk(chunk)
            for i in range(n_profiles):
                diff = chunk[i][None, :] - chunk
                sums[i, :] += np.sum(diff * diff, axis=1)
        return np.sqrt(sums)

    if distance_metric != 'correlation':
        raise ValueError(f'Unsupported distance metric: {distance_metric}')

    sums = np.zeros(n_profiles, dtype=np.float64)
    sums_sq = np.zeros(n_profiles, dtype=np.float64)
    cross = np.zeros((n_profiles, n_profiles), dtype=np.float64)
    n_features = len(indices)

    for start in range(0, n_features, chunk_size):
        chunk_idx = indices[start : start + chunk_size]
        chunk = np.vstack([values[chunk_idx] for values in values_by_profile]).astype(np.float64)
        if zscore_features:
            chunk = zscore_feature_chunk(chunk)
        sums += chunk.sum(axis=1)
        sums_sq += np.sum(chunk * chunk, axis=1)
        cross += chunk @ chunk.T

    cov = cross - np.outer(sums, sums) / float(n_features)
    ss = sums_sq - (sums * sums) / float(n_features)
    denom = np.sqrt(np.outer(ss, ss))
    corr = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0)
    corr = np.clip(corr, -1.0, 1.0)
    return 1.0 - corr


def score_distances(
    distances: np.ndarray,
    profiles: list[dict[str, object]],
    n_features: int,
    analysis_set: str,
    metric_label: str,
    metric_key: str,
    tissue: str,
    source_image: str,
    distance_metric: str,
) -> dict[str, object] | None:
    subjects = np.asarray([str(profile['subject']) for profile in profiles])
    sessions = np.asarray([str(profile['session']) for profile in profiles])
    paired_subjects = pd.Series(sessions, index=subjects).groupby(level=0).nunique()
    paired_subjects = set(paired_subjects[paired_subjects >= 2].index)
    keep = np.asarray([subject in paired_subjects for subject in subjects], dtype=bool)
    if np.count_nonzero(keep) < 4:
        return None

    distances = distances[np.ix_(keep, keep)]
    subjects = subjects[keep]
    sessions = sessions[keep]
    scores = []
    nearest_correct = []
    genuine_distances = []
    impostor_distances = []
    rank_percentiles = []

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
        'analysis_set': analysis_set,
        'profile_type': 'mni_voxelwise',
        'metric': metric_label,
        'metric_key': metric_key,
        'source_image': source_image,
        'tissue': tissue,
        'distance_metric': distance_metric,
        'discriminability': float(np.mean(scores)),
        'nearest_neighbor_accuracy': float(np.mean(nearest_correct)),
        'mean_genuine_distance': float(np.mean(genuine_distances)),
        'mean_impostor_distance': float(np.mean(impostor_distances)),
        'mean_rank_percentile': float(np.mean(rank_percentiles)),
        'n_subjects': int(len(np.unique(subjects))),
        'n_sessions': int(len(np.unique(sessions))),
        'n_profiles': int(len(subjects)),
        'n_features': int(n_features),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--derivatives-dir', type=Path, default=None)
    parser.add_argument('--patterns-file', type=Path, default=None)
    parser.add_argument('--qc-file', type=Path, default=None)
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--subject-id', action='append')
    parser.add_argument('--session-id', action='append', default=None)
    parser.add_argument('--metric', action='append')
    parser.add_argument('--analysis-set', choices=ANALYSIS_SETS, default='primary')
    parser.add_argument('--tissue', action='append', choices=TISSUES, default=None)
    parser.add_argument(
        '--distance-metric', choices=('correlation', 'euclidean'), default='correlation'
    )
    parser.add_argument('--zscore-features', action='store_true')
    parser.add_argument(
        '--template-dseg',
        type=Path,
        default=None,
        help='Deterministic FreeSurfer aseg dseg used for all common tissue masks.',
    )
    parser.add_argument('--gm-erosion-mm', type=float, default=0.0)
    parser.add_argument('--wm-erosion-mm', type=float, default=0.0)
    parser.add_argument('--outlier-z', type=float, default=6.0)
    parser.add_argument('--chunk-size', type=int, default=50000)
    parser.add_argument('--min-features', type=int, default=2)
    parser.add_argument('--allow-zero', action='store_true')
    parser.add_argument('--no-qc', action='store_true')
    args = parser.parse_args()

    args.project_root = args.project_root.expanduser().resolve()
    args.derivatives_dir = (
        args.derivatives_dir.expanduser().resolve()
        if args.derivatives_dir
        else SOURCE_DERIVATIVES_ROOT
    )
    args.patterns_file = (
        args.patterns_file.expanduser().resolve()
        if args.patterns_file
        else REPO_ROOT / 'configuration' / 'metrics.yml'
    )
    if not args.patterns_file.exists():
        args.patterns_file = REPO_ROOT / 'configuration' / 'metrics.yml'
    if args.qc_file is None and not args.no_qc:
        candidates = (REPO_ROOT / 'data' / 'qc' / 'manual_qc_modality.tsv',)
        args.qc_file = next((path for path in candidates if path.exists()), None)
    elif args.qc_file is not None:
        args.qc_file = args.qc_file.expanduser().resolve()
    args.output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else OUTPUT_DERIVATIVES_ROOT / 'mni_voxelwise_discriminability'
    )
    args.sessions = (
        [normalize_session(session) for session in args.session_id]
        if args.session_id
        else ['ses-01', 'ses-02']
    )
    args.tissues = args.tissue if args.tissue else list(TISSUES)
    if len(args.sessions) < 2:
        parser.error('At least two --session-id values are required.')
    if args.chunk_size < 1:
        parser.error('--chunk-size must be positive.')
    if args.min_features < 1:
        parser.error('--min-features must be positive.')
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
        parser.error('Template mask erosion distances must be nonnegative.')
    return args


def main() -> None:
    args = parse_args()
    require_dependencies()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    patterns = load_patterns(args.patterns_file)
    specs = build_metric_specs(args.patterns_file)
    spec_by_label = {spec.label: spec for spec in specs}
    hybrid_pairs = gm_noddi_hybrid_pairs(specs)
    metric_specs = select_metric_specs(
        specs,
        args.analysis_set,
        args.metric,
        tissues=list(dict.fromkeys(metric_registry_tissue(tissue) for tissue in args.tissues)),
    )
    analysis_sets = selected_analysis_sets(args.analysis_set)
    display_labels = {
        tissue: {
            analysis_set: metric_display_labels(
                specs,
                analysis_set,
                tissue=metric_registry_tissue(tissue),
            )
            for analysis_set in analysis_sets
        }
        for tissue in args.tissues
    }
    labels_by_tissue = {
        tissue: {
            analysis_set: set(
                metric_order(
                    specs,
                    analysis_set,
                    tissue=metric_registry_tissue(tissue),
                )
            )
            for analysis_set in analysis_sets
        }
        for tissue in args.tissues
    }
    qc = load_qc_table(args.qc_file)

    subjects = (
        [normalize_subject(subject) for subject in args.subject_id]
        if args.subject_id
        else discover_subjects(args.derivatives_dir)
    )
    reference, masks = build_template_tissue_masks(
        args.template_dseg,
        args.gm_erosion_mm,
        args.wm_erosion_mm,
    )

    rows = []
    diagnostics = []
    coverage_rows = []
    inclusion_rows = []
    for metric_index, spec in enumerate(metric_specs):
        print(f'Metric {metric_index + 1}/{len(metric_specs)}: {spec.label}', flush=True)
        profiles, metric_diagnostics = collect_metric_profiles(
            args.derivatives_dir,
            patterns,
            qc,
            subjects,
            args.sessions,
            spec,
        )
        diagnostics.extend(metric_diagnostics)
        profiles = keep_subjects_with_multiple_sessions(profiles)
        if len(profiles) < 4:
            continue
        gm_hybrid_profiles = None
        gm_counterpart = spec_by_label.get(hybrid_pairs.get(spec.label))
        if gm_counterpart is not None and 'gmwm' in args.tissues:
            gm_hybrid_profiles, gm_metric_diagnostics = collect_metric_profiles(
                args.derivatives_dir,
                patterns,
                qc,
                subjects,
                args.sessions,
                gm_counterpart,
            )
            diagnostics.extend(gm_metric_diagnostics)
            gm_hybrid_profiles = keep_subjects_with_multiple_sessions(gm_hybrid_profiles)
        for tissue in args.tissues:
            if not any(
                spec.label in labels_by_tissue[tissue][analysis_set]
                for analysis_set in analysis_sets
            ):
                continue
            tissue_profiles = profiles
            tissue_gm_mask = None
            if tissue == 'gmwm' and gm_hybrid_profiles is not None:
                tissue_profiles = pair_gmwm_hybrid_profiles(profiles, gm_hybrid_profiles)
                tissue_profiles = keep_subjects_with_multiple_sessions(tissue_profiles)
                tissue_gm_mask = masks['all_gm']
                if len(tissue_profiles) < 4:
                    continue
            print(f'  {TISSUE_TITLES[tissue]}', flush=True)
            common_mask, values_by_profile = common_metric_mask(
                tissue_profiles,
                reference,
                masks[tissue],
                outlier_z=args.outlier_z,
                remove_zeros=not args.allow_zero,
                gm_mask=tissue_gm_mask,
                outlier_masks=((masks['all_gm'], masks['wm']) if tissue == 'gmwm' else None),
            )
            n_features = int(np.count_nonzero(common_mask))
            coverage_rows.append(
                {
                    'metric': spec.label,
                    'tissue': tissue,
                    'n_profiles': len(tissue_profiles),
                    'n_tissue_voxels': int(np.count_nonzero(masks[tissue])),
                    'n_common_voxels': n_features,
                    'proportion_tissue_voxels': (
                        n_features / float(np.count_nonzero(masks[tissue]))
                        if np.count_nonzero(masks[tissue])
                        else np.nan
                    ),
                }
            )
            if n_features < args.min_features:
                continue
            distances = pairwise_distances_chunked(
                values_by_profile,
                common_mask,
                distance_metric=args.distance_metric,
                zscore_features=args.zscore_features,
                chunk_size=args.chunk_size,
            )
            for analysis_set in analysis_sets:
                if spec.label not in labels_by_tissue[tissue][analysis_set]:
                    continue
                metric_label = display_labels[tissue][analysis_set].get(spec.label, spec.label)
                score = score_distances(
                    distances,
                    tissue_profiles,
                    n_features=n_features,
                    analysis_set=analysis_set,
                    metric_label=metric_label,
                    metric_key=spec.label,
                    tissue=tissue,
                    source_image=spec.source_image,
                    distance_metric=args.distance_metric,
                )
                if score is not None:
                    rows.append(score)

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(['analysis_set', 'tissue', 'metric']).reset_index(drop=True)
    coverage_df = pd.DataFrame(coverage_rows)
    scored_keys = (
        set(zip(summary['analysis_set'], summary['tissue'], summary['metric_key']))
        if not summary.empty
        else set()
    )
    for tissue in args.tissues:
        for analysis_set in analysis_sets:
            display = display_labels[tissue][analysis_set]
            for label in metric_order(
                specs,
                analysis_set,
                tissue=metric_registry_tissue(tissue),
            ):
                observed = not coverage_df.empty and bool(
                    ((coverage_df['metric'] == label) & (coverage_df['tissue'] == tissue)).any()
                )
                scored = (analysis_set, tissue, label) in scored_keys
                inclusion_rows.append(
                    {
                        'analysis_set': analysis_set,
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
                                else 'insufficient_profiles_or_common_voxels'
                            )
                        ),
                    }
                )
    suffix = args.distance_metric
    if args.zscore_features:
        suffix = f'zscore_{suffix}'
    summary.to_csv(
        args.output_dir / f'mni_voxelwise_discriminability_{suffix}.tsv',
        sep='\t',
        index=False,
    )
    coverage_df.to_csv(
        args.output_dir / f'mni_voxelwise_discriminability_mask_coverage_{suffix}.tsv',
        sep='\t',
        index=False,
    )
    pd.DataFrame(inclusion_rows).to_csv(
        args.output_dir / f'mni_voxelwise_discriminability_metric_inclusion_{suffix}.tsv',
        sep='\t',
        index=False,
    )
    pd.DataFrame(diagnostics).to_csv(
        args.output_dir / 'mni_voxelwise_discriminability_subject_diagnostics.tsv',
        sep='\t',
        index=False,
    )


if __name__ == '__main__':
    main()
