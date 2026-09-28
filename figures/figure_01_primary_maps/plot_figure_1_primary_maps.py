#!/usr/bin/env python3
"""Plot compact group-average axial slices of the primary MIRROR brain maps.

The figure uses one axial slice in MNI152NLin2009cAsym space for every map.
Scalar overlays are displayed with one shared colormap, while each map's color
limits are its 5th and 95th percentiles within GM+WM tissue voxels after
voxelwise averaging across QC-passing images. The primary ICVF panel is a
GM/WM hybrid: WM voxels come from the regular NODDI fit and GM voxels come
from the gmNODDI fit.

This manuscript renderer uses tight slice cropping and compact panel spacing.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

if 'MPLCONFIGDIR' not in os.environ:
    mpl_config_dir = Path(os.environ.get('TMPDIR', '/tmp')) / 'mirror_matplotlib'
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(mpl_config_dir)
if 'XDG_CACHE_HOME' not in os.environ:
    xdg_cache_dir = Path(os.environ.get('TMPDIR', '/tmp')) / 'mirror_cache'
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['XDG_CACHE_HOME'] = str(xdg_cache_dir)

import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['svg.fonttype'] = 'none'
matplotlib.rcParams['font.family'] = 'Arial'
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.images import (  # noqa: E402
    first_glob,
    load_canonical,
    resample_image,
)
from utils.metrics import (  # noqa: E402
    SOURCE_IMAGE_COLORS,
    MetricSpec,
    build_metric_specs,
    load_patterns,
    metric_plot_label,
    primary_metric_specs,
)
from utils.tissue_masks import build_template_tissue_masks  # noqa: E402
from utils.paths import (  # noqa: E402
    PROJECT_ROOT,
    RUN_FIGURES_ROOT,
    SOURCE_DERIVATIVES_ROOT,
)

LOGGER = logging.getLogger('primary_maps')

MNI_SPACE = 'MNI152NLin2009cAsym'
DISPLAY_PERCENTILES = (5.0, 95.0)
CACHE_VERSION = 10
SLICE_CROP_PADDING = 4
B1_PATTERN = (
    'pymp2rage/{subject}/{session}/fmap/'
    '{subject}_{session}_run-01_space-MNI152NLin2009cAsym_TB1map.nii.gz'
)
FIGURE_SOURCE_COLORS = {
    **SOURCE_IMAGE_COLORS,
    'B1': '#000000',
}
INVERTED_DISPLAY_METRICS = {
    'MD',
    'RD',
    'G-ihMTR',
    'G-ihMTsat',
    'QSM-SEPIA-E5-X',
    'QSM-X-R2p-E5-X',
    'QSM-X-R2p-E5-Para',
}
INVERTED_DISPLAY_SYMBOL = r'$^\dagger$'

TABLE_ORDER = {
    'dMRI': (
        'ICVF',
        'MKT',
        'RK',
        'AWF',
        'GFA',
        'FA',
        'MD',
        'RD',
        'NG',
        'NG (Perpendicular)',
        'RTAP',
        'RTOP',
    ),
    'B1': ('B1',),
    'R1': (
        'R1',
        'R1-B1c',
    ),
    'MESE': ('R2',),
    'ihMT': (
        'ihMTR',
        'ihMTsat-B1c',
    ),
    'MEGRE': (
        'R2*',
        'Q-Ratio-E5-B1c',
    ),
    'QSM': (
        'QSM-SEPIA-E5-X',
        'QSM-X-R2p-E5-X',
        'QSM-X-R2p-E5-Dia',
        'QSM-X-R2p-E5-Para',
    ),
    'T1w/T2w': (
        'MPRAGE-MyelinW',
        'SPACE-MyelinW',
    ),
    'g-ratio': (
        'G-ihMTR',
        'G-ihMTsat',
    ),
}

LAYOUT_ROWS = (
    ('dMRI',),
    ('B1', 'R1', 'MESE', 'ihMT'),
    ('MEGRE', 'QSM'),
    ('T1w/T2w', 'g-ratio'),
)

PANEL_LABELS = {
    'MPRAGE-MyelinW': 'MPRAGE\nT1w/T2w Ratio',
    'SPACE-MyelinW': 'SPACE\nT1w/T2w Ratio',
    'B1': 'B₁⁺ map',
}

STACKED_PANEL_GROUPS = {'T1w/T2w'}

GROUP_PANEL_STYLE = {
    'dMRI': {
        'xpad': 0.070,
        'top_pad': 0.115,
        'bottom_pad': 0.018,
        'col_gap': 0.045,
        'row_gap': 0.145,
        'label_offset': 0.010,
        'panel_width_cap': 0.105,
        'label_fontsize': 8.1,
        'group_fontsize': 10.5,
    },
    'QSM': {
        'xpad': 0.040,
        'top_pad': 0.190,
        'bottom_pad': 0.025,
        'col_gap': 0.030,
        'row_gap': 0.080,
        'label_offset': 0.010,
        'panel_width_cap': 0.190,
        'label_fontsize': 7.8,
        'group_fontsize': 10.3,
    },
    'T1w/T2w': {
        'xpad': 0.160,
        'top_pad': 0.105,
        'bottom_pad': 0.035,
        'col_gap': 0.055,
        'row_gap': 0.150,
        'label_offset': 0.010,
        'panel_width_cap': 0.460,
        'label_fontsize': 7.2,
        'group_fontsize': 10.0,
    },
}

DEFAULT_PANEL_STYLE = {
    'xpad': 0.075,
    'top_pad': 0.175,
    'bottom_pad': 0.030,
    'col_gap': 0.060,
    'row_gap': 0.075,
    'label_offset': 0.012,
    'panel_width_cap': 0.270,
    'label_fontsize': 8.3,
    'group_fontsize': 10.3,
}

PANEL_WIDTH_IN = 0.62
PANEL_HEIGHT_IN = 0.86
PANEL_GAP_IN = 0.18
PANEL_ROW_GAP_IN = 0.24
GROUP_XPAD_IN = 0.13
GROUP_TOP_PAD_IN = 0.28
GROUP_BOTTOM_PAD_IN = 0.08
GROUP_GAP_IN = 0.035
ROW_GAP_IN = 0.070
FIGURE_MARGIN_IN = 0.12
GROUP_INCH_STYLE = {
    'dMRI': {
        'panel_gap': 0.075,
        'row_gap': 0.18,
        'xpad': 0.12,
        'top_pad': 0.42,
        'bottom_pad': 0.08,
        'label_fontsize': 7.9,
        'group_fontsize': 10.0,
    },
    'QSM': {
        'panel_gap': 0.42,
        'row_gap': 0.24,
        'xpad': 0.20,
        'top_pad': 0.44,
        'bottom_pad': 0.08,
        'label_fontsize': 6.9,
        'group_fontsize': 9.8,
    },
    'T1w/T2w': {
        'panel_gap': 0.18,
        'row_gap': 0.34,
        'xpad': 0.28,
        'top_pad': 0.42,
        'bottom_pad': 0.10,
        'label_fontsize': 7.1,
        'group_fontsize': 9.5,
        'min_width': 2.25,
    },
}
LONG_LABEL_WIDTH_IN = {
    'T1w/T2w': 1.92,
}


@dataclass(frozen=True)
class ResolvedMetric:
    spec: MetricSpec
    pattern: str
    source_group: str
    space: str
    paths: tuple[Path, ...]
    contributors: tuple[tuple[str, str], ...]
    hybrid_gm_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class TissueSpace:
    reference: nib.spatialimages.SpatialImage
    mask: np.ndarray
    scaling_mask: np.ndarray
    gm_mask: np.ndarray
    brain_mask: np.ndarray
    background: np.ndarray
    slice_index: int
    slice_crop: tuple[slice, slice]
    source_file: str


@dataclass(frozen=True)
class PreparedPanel:
    metric: ResolvedMetric
    data: np.ndarray
    reference: nib.spatialimages.SpatialImage
    tissue_mask: np.ndarray
    contributor_count: np.ndarray
    limits: tuple[float, float]


def normalize_subject(value: str) -> str:
    token = value.strip()
    return token if token.startswith('sub-') else f'sub-{token}'


def normalize_session(value: str) -> str:
    token = value.strip()
    return token if token.startswith('ses-') else f'ses-{token}'


def session_number(session: str) -> str:
    return normalize_session(session).removeprefix('ses-')


def is_pilot_subject(subject: str) -> bool:
    return normalize_subject(subject).upper().startswith('SUB-PILOT')


def load_qc_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as fobj:
        return list(csv.DictReader(fobj, delimiter='\t'))


def sessions_from_qc(rows: Sequence[dict[str, str]]) -> list[str]:
    sessions: set[str] = set()
    for row in rows:
        for column in row:
            if column.startswith('Session ') and '--' in column:
                sessions.add(normalize_session(column.split('--', 1)[0].replace('Session ', '')))
    return sorted(sessions)


def qc_passes(row: dict[str, str], specs: Sequence[MetricSpec], session: str) -> bool:
    ses_num = session_number(session)
    for spec in specs:
        for modality in spec.qc_modalities:
            column = f'Session {ses_num}--{modality}'
            if row.get(column) != '1':
                return False
    return True


def table_metric_specs(patterns_file: Path) -> list[tuple[str, MetricSpec, str]]:
    patterns = load_patterns(patterns_file)
    specs = primary_metric_specs(build_metric_specs(patterns_file))
    by_primary = {spec.primary_label: spec for spec in specs}
    ordered: list[tuple[str, MetricSpec, str]] = []
    for source_group, labels in TABLE_ORDER.items():
        for label in labels:
            if label == 'B1':
                ordered.append(
                    (
                        source_group,
                        MetricSpec(
                            label='B1',
                            primary_label='B1',
                            pattern_key='B1',
                            group='B1',
                            family='B1',
                            source_image='B1',
                            qc_modalities=('B1+',),
                            tissues=('gm', 'wm'),
                            primary=False,
                        ),
                        B1_PATTERN,
                    )
                )
                continue
            spec = by_primary.get(label)
            if spec is None:
                raise KeyError(f'Primary metric {label!r} was not found in {patterns_file}')
            ordered.append((source_group, spec, patterns[spec.group][spec.pattern_key]))
    if len(ordered) != 28:
        raise RuntimeError(f'Expected 27 primary metrics plus B1, found {len(ordered)}')
    return ordered


def metric_space(spec: MetricSpec) -> str:
    return MNI_SPACE


def find_metric_path(
    derivatives: Path,
    subject: str,
    session: str,
    pattern: str,
    space: str,
) -> Path | None:
    formatted = pattern.format(subject=subject, session=session, space=space)
    return first_glob([derivatives / formatted])


def gm_noddi_pattern_for(spec: MetricSpec, patterns_file: Path) -> str | None:
    if spec.group != 'dMRI' or spec.pattern_key != 'ICVF':
        return None
    patterns = load_patterns(patterns_file)
    return patterns.get('dMRI', {}).get('ICVF (GM)')


def candidate_sessions(
    qc_rows: Sequence[dict[str, str]],
    requested_subjects: Sequence[str] | None,
    requested_sessions: Sequence[str] | None,
) -> list[tuple[str, str, dict[str, str] | None]]:
    if requested_subjects:
        subjects = [normalize_subject(subject) for subject in requested_subjects]
    else:
        subjects = [row['participant_id'] for row in qc_rows]
    subjects = [subject for subject in subjects if not is_pilot_subject(subject)]

    if requested_sessions:
        sessions = [normalize_session(session) for session in requested_sessions]
    else:
        sessions = sessions_from_qc(qc_rows)

    rows_by_subject = {row['participant_id']: row for row in qc_rows}
    return [
        (subject, session, rows_by_subject.get(subject))
        for subject in subjects
        for session in sessions
    ]


def resolve_average_metrics(
    derivatives: Path,
    metric_specs: Sequence[tuple[str, MetricSpec, str]],
    qc_rows: Sequence[dict[str, str]],
    requested_subjects: Sequence[str] | None,
    requested_sessions: Sequence[str] | None,
    patterns_file: Path,
) -> list[ResolvedMetric]:
    candidates = candidate_sessions(qc_rows, requested_subjects, requested_sessions)
    resolved: list[ResolvedMetric] = []
    missing: list[str] = []
    for source_group, spec, pattern in metric_specs:
        gm_pattern = gm_noddi_pattern_for(spec, patterns_file)
        paths: list[Path] = []
        hybrid_gm_paths: list[Path] = []
        contributors: list[tuple[str, str]] = []
        for subject, session, qc_row in candidates:
            if qc_row is None or not qc_passes(qc_row, [spec], session):
                continue
            space = metric_space(spec)
            path = find_metric_path(derivatives, subject, session, pattern, space)
            if path is None:
                continue
            gm_path = None
            if gm_pattern is not None:
                gm_path = find_metric_path(derivatives, subject, session, gm_pattern, space)
                if gm_path is None:
                    continue
            paths.append(path)
            if gm_path is not None:
                hybrid_gm_paths.append(gm_path)
            contributors.append((subject, session))
        if not paths:
            missing.append(spec.primary_label)
            continue
        resolved.append(
            ResolvedMetric(
                spec=spec,
                pattern=pattern,
                source_group=source_group,
                space=metric_space(spec),
                paths=tuple(paths),
                contributors=tuple(contributors),
                hybrid_gm_paths=tuple(hybrid_gm_paths),
            )
        )
        if hybrid_gm_paths:
            LOGGER.info(
                '%s: averaging %d matched regular/gmNODDI image pair(s)',
                spec.primary_label,
                len(paths),
            )
        else:
            LOGGER.info('%s: averaging %d QC-passing image(s)', spec.primary_label, len(paths))
    if missing:
        raise RuntimeError(
            'No QC-passing MNI image was found for these primary metrics: ' + ', '.join(missing)
        )
    return resolved


def middle_axial_slice(mask: np.ndarray) -> int:
    coordinates = np.argwhere(mask)
    if not coordinates.size:
        return int(mask.shape[2] // 2)
    z_min = int(coordinates[:, 2].min())
    z_max = int(coordinates[:, 2].max())
    return int(round((z_min + z_max) / 2))


def load_mni_tissue_space(template_dseg: Path) -> TissueSpace:
    if not template_dseg.exists():
        raise FileNotFoundError(f'Template aseg dseg not found: {template_dseg}')
    reference, masks = build_template_tissue_masks(template_dseg)
    shape = reference.shape[:3]
    gm_mask = np.asarray(masks['all_gm'], dtype=bool).reshape(shape)
    scaling_mask = gm_mask | np.asarray(masks['wm'], dtype=bool).reshape(shape)
    brain_mask = np.asanyarray(reference.dataobj).astype(np.float32) > 0
    slice_index = middle_axial_slice(brain_mask)
    return TissueSpace(
        reference=reference,
        mask=brain_mask,
        scaling_mask=scaling_mask,
        gm_mask=gm_mask,
        brain_mask=brain_mask,
        background=brain_mask.astype(np.float32),
        slice_index=slice_index,
        slice_crop=axial_crop(brain_mask, slice_index, SLICE_CROP_PADDING),
        source_file=str(template_dseg),
    )


def axial_crop(mask: np.ndarray, slice_index: int, padding: int) -> tuple[slice, slice]:
    slice_mask = axial_slice(mask, slice_index)
    coordinates = np.argwhere(slice_mask)
    if not coordinates.size:
        return slice(None), slice(None)
    row_min, col_min = coordinates.min(axis=0)
    row_max, col_max = coordinates.max(axis=0)
    row_min = max(int(row_min) - padding, 0)
    col_min = max(int(col_min) - padding, 0)
    row_max = min(int(row_max) + padding + 1, slice_mask.shape[0])
    col_max = min(int(col_max) + padding + 1, slice_mask.shape[1])
    return slice(row_min, row_max), slice(col_min, col_max)


def robust_background_limits(data: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = data[np.isfinite(data) & mask]
    if not values.size:
        values = data[np.isfinite(data)]
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, [1, 99])
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def scalar_limits(data: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = data[np.isfinite(data) & mask & (np.abs(data) > 0)]
    if not values.size:
        raise RuntimeError('No finite nonzero GM/WM values were found')
    low, high = np.percentile(values, DISPLAY_PERCENTILES)
    if low == high:
        padding = max(abs(float(low)) * 0.05, 1.0)
        low -= padding
        high += padding
    return float(low), float(high)


def is_display_inverted(metric: ResolvedMetric | MetricSpec) -> bool:
    spec = metric.spec if isinstance(metric, ResolvedMetric) else metric
    return spec.primary_label in INVERTED_DISPLAY_METRICS


def voxelwise_average(
    paths: Sequence[Path],
    reference: nib.spatialimages.SpatialImage,
) -> tuple[np.ndarray, np.ndarray]:
    total: np.ndarray | None = None
    count: np.ndarray | None = None
    for path in paths:
        scalar_img = resample_image(load_canonical(path), reference, order=1)
        data = np.asarray(scalar_img.get_fdata(), dtype=np.float32)
        finite = np.isfinite(data)
        if total is None:
            total = np.zeros(data.shape, dtype=np.float64)
            count = np.zeros(data.shape, dtype=np.uint16)
        total[finite] += data[finite]
        count[finite] += 1
    if total is None or count is None:
        raise RuntimeError('Cannot average an empty image list')
    average = np.full(total.shape, np.nan, dtype=np.float32)
    valid = count > 0
    average[valid] = (total[valid] / count[valid]).astype(np.float32)
    return average, count


def hybrid_gm_wm_average(
    wm_paths: Sequence[Path],
    gm_paths: Sequence[Path],
    reference: nib.spatialimages.SpatialImage,
    gm_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if len(wm_paths) != len(gm_paths):
        raise RuntimeError('Hybrid NODDI averaging requires matched WM and GM image lists')
    wm_average, wm_count = voxelwise_average(wm_paths, reference)
    gm_average, gm_count = voxelwise_average(gm_paths, reference)
    data = np.full(wm_average.shape, np.nan, dtype=np.float32)
    contributor_count = np.zeros(wm_count.shape, dtype=np.uint16)
    wm_valid = np.isfinite(wm_average) & (wm_count > 0)
    gm_valid = gm_mask & np.isfinite(gm_average) & (gm_count > 0)
    data[wm_valid] = wm_average[wm_valid]
    data[gm_valid] = gm_average[gm_valid]
    contributor_count[wm_valid] = wm_count[wm_valid]
    contributor_count[gm_valid] = gm_count[gm_valid]
    return data, contributor_count


def safe_cache_slug(value: str) -> str:
    return ''.join(char if char.isalnum() else '-' for char in value).strip('-')


def panel_cache_path(cache_dir: Path, metric: ResolvedMetric) -> Path:
    group_slug = safe_cache_slug(metric.source_group)
    slug = safe_cache_slug(metric.spec.primary_label)
    return cache_dir / f'{group_slug}_{slug}_{metric.space}_average.npz'


def panel_cache_metadata(
    metric: ResolvedMetric,
    tissue_space: TissueSpace,
) -> dict[str, object]:
    reference = tissue_space.reference
    return {
        'version': CACHE_VERSION,
        'metric': metric.spec.primary_label,
        'source_group': metric.source_group,
        'space': metric.space,
        'pattern': metric.pattern,
        'paths': [str(path) for path in metric.paths],
        'hybrid_gm_paths': [str(path) for path in metric.hybrid_gm_paths],
        'contributors': [list(item) for item in metric.contributors],
        'reference_shape': list(reference.shape),
        'reference_affine': np.asarray(reference.affine).round(6).tolist(),
        'tissue_source_file': tissue_space.source_file,
        'display_percentiles': list(DISPLAY_PERCENTILES),
        'inverted_display_metrics': sorted(INVERTED_DISPLAY_METRICS),
    }


def load_cached_panel(
    cache_dir: Path,
    metric: ResolvedMetric,
    tissue_space: TissueSpace,
) -> PreparedPanel | None:
    cache_path = panel_cache_path(cache_dir, metric)
    if not cache_path.exists():
        return None
    expected_metadata = panel_cache_metadata(metric, tissue_space)
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            metadata = json.loads(str(cache['metadata'].item()))
            if metadata != expected_metadata:
                LOGGER.info('%s: cached average is stale', metric.spec.primary_label)
                return None
            data = np.asarray(cache['data'], dtype=np.float32)
            contributor_count = np.asarray(cache['contributor_count'], dtype=np.uint16)
            limits = tuple(float(value) for value in cache['limits'])
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        LOGGER.warning(
            '%s: ignoring unreadable average cache %s (%s)',
            metric.spec.primary_label,
            cache_path,
            exc,
        )
        return None
    tissue_mask = tissue_space.mask
    LOGGER.info('%s: loaded cached average map', metric.spec.primary_label)
    return PreparedPanel(
        metric=metric,
        data=data,
        reference=tissue_space.reference,
        tissue_mask=tissue_mask,
        contributor_count=contributor_count,
        limits=(limits[0], limits[1]),
    )


def save_cached_panel(
    cache_dir: Path,
    panel: PreparedPanel,
    tissue_space: TissueSpace,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = panel_cache_path(cache_dir, panel.metric)
    metadata = json.dumps(
        panel_cache_metadata(panel.metric, tissue_space),
        sort_keys=True,
        separators=(',', ':'),
    )
    np.savez_compressed(
        cache_path,
        data=panel.data,
        contributor_count=panel.contributor_count,
        limits=np.asarray(panel.limits, dtype=np.float32),
        metadata=np.asarray(metadata),
    )


def prepare_panels(
    resolved_metrics: Sequence[ResolvedMetric],
    tissue_spaces: dict[str, TissueSpace],
    average_cache_dir: Path | None = None,
    recalculate_average_maps: bool = False,
) -> list[PreparedPanel]:
    panels: list[PreparedPanel] = []
    for metric in resolved_metrics:
        tissue_space = tissue_spaces[metric.space]
        if average_cache_dir is not None and not recalculate_average_maps:
            cached_panel = load_cached_panel(average_cache_dir, metric, tissue_space)
            if cached_panel is not None:
                panels.append(cached_panel)
                continue
        if metric.hybrid_gm_paths:
            data, contributor_count = hybrid_gm_wm_average(
                metric.paths,
                metric.hybrid_gm_paths,
                tissue_space.reference,
                tissue_space.gm_mask,
            )
        else:
            data, contributor_count = voxelwise_average(metric.paths, tissue_space.reference)
        if is_display_inverted(metric):
            data = -data
        tissue_mask = tissue_space.mask
        limits = scalar_limits(data, tissue_space.scaling_mask)
        panel = PreparedPanel(
            metric=metric,
            data=data,
            reference=tissue_space.reference,
            tissue_mask=tissue_mask,
            contributor_count=contributor_count,
            limits=limits,
        )
        if average_cache_dir is not None:
            save_cached_panel(average_cache_dir, panel, tissue_space)
            LOGGER.info('%s: wrote cached average map', metric.spec.primary_label)
        panels.append(panel)
    return panels


def axial_slice(data: np.ndarray, slice_index: int) -> np.ndarray:
    return np.rot90(data[:, :, slice_index])


def cropped_axial_slice(
    data: np.ndarray | np.ma.MaskedArray,
    slice_index: int,
    crop: tuple[slice, slice],
) -> np.ndarray | np.ma.MaskedArray:
    return axial_slice(data, slice_index)[crop]


def format_label(label: str) -> str:
    return metric_plot_label(label)


def math_label(label: str, bold: bool = False) -> str:
    text = format_label(label)
    replacements = {
        '₁': r'$_1$',
        '₂': r'$_2$',
        '⁺': r'$^{+}$',
        'χ': r'$\chi$',
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    if bold:
        text = text.replace(r'$_1$', r'$_{\mathbf{1}}$')
        text = text.replace(r'$_2$', r'$_{\mathbf{2}}$')
        text = text.replace(r'$^{+}$', r'$^{\mathbf{+}}$')
    return text


def display_label(spec: MetricSpec) -> str:
    label = math_label(PANEL_LABELS.get(spec.primary_label, spec.primary_label))
    if is_display_inverted(spec):
        label = f'{label}{INVERTED_DISPLAY_SYMBOL}'
    return label


def group_label(group: str) -> str:
    if group == 'B1':
        return math_label('B₁⁺', bold=True)
    if group == 'R1':
        return 'MP2RAGE'
    if group == 'MESE':
        return math_label('R₂', bold=True)
    if group == 'g-ratio':
        return r'$\it{g}$-Ratio'
    return math_label(group, bold=True)


def add_group_box(fig: plt.Figure, spec, color: str):
    axis = fig.add_subplot(spec)
    axis.set_zorder(0)
    axis.set_facecolor('white')
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.25)
        spine.set_edgecolor(mcolors.to_rgba(color, 0.72))
    return axis


def add_group_label(
    axis: plt.Axes,
    group: str,
    color: str,
) -> None:
    axis.annotate(
        group_label(group),
        xy=(0, 1),
        xycoords='axes fraction',
        xytext=(5, -4),
        textcoords='offset points',
        ha='left',
        va='top',
        fontsize=10.2,
        fontweight='bold',
        color=color,
    )


def plot_panel(
    ax: plt.Axes,
    panel: PreparedPanel,
    tissue_space: TissueSpace,
    cmap: str,
    bg_limits: tuple[float, float],
) -> None:
    background = np.asarray(tissue_space.background, dtype=np.float32)
    background = np.ma.masked_where(~tissue_space.brain_mask, background)
    z_index = tissue_space.slice_index
    gray_cmap = plt.get_cmap('gray').copy()
    gray_cmap.set_bad((1, 1, 1, 0))
    scalar_cmap = plt.get_cmap(cmap).copy()
    scalar_cmap.set_bad((1, 1, 1, 0))
    crop = tissue_space.slice_crop
    ax.imshow(
        cropped_axial_slice(background, z_index, crop),
        cmap=gray_cmap,
        vmin=bg_limits[0],
        vmax=bg_limits[1],
        interpolation='nearest',
    )
    overlay_data = np.where(np.isfinite(panel.data), panel.data, panel.limits[0])
    overlay = np.ma.masked_where(~panel.tissue_mask, overlay_data)
    ax.imshow(
        cropped_axial_slice(overlay, z_index, crop),
        cmap=scalar_cmap,
        vmin=panel.limits[0],
        vmax=panel.limits[1],
        alpha=0.82,
        interpolation='nearest',
    )
    slice_shape = cropped_axial_slice(background, z_index, crop).shape
    bottom_margin = max(1.5, 0.025 * slice_shape[0])
    ax.set_xlim(-0.5, slice_shape[1] - 0.5)
    ax.set_ylim(slice_shape[0] - 0.5 + bottom_margin, -0.5)
    ax.set_aspect('equal')
    label = display_label(panel.metric.spec)
    title_fontsize = 7.2 if '\n' in label else 8.2
    title_pad = 4.4 if '\n' in label else 1.2
    ax.set_title(label, fontsize=title_fontsize, pad=title_pad, linespacing=0.92)
    ax.set_axis_off()


def panel_style(group: str) -> dict[str, float]:
    style = DEFAULT_PANEL_STYLE.copy()
    style.update(GROUP_PANEL_STYLE.get(group, {}))
    return style


def plot_group_panels(
    host_axis: plt.Axes,
    group: str,
    group_panels: Sequence[PreparedPanel],
    rows: int,
    width: int,
    tissue_spaces: dict[str, TissueSpace],
    bg_limits: dict[str, tuple[float, float]],
    cmap: str,
    color: str,
) -> None:
    style = panel_style(group)
    host_axis.text(
        0.018,
        0.965,
        group_label(group),
        ha='left',
        va='top',
        fontsize=style['group_fontsize'],
        fontweight='bold',
        color=color,
        transform=host_axis.transAxes,
    )

    panel_area_width = 1.0 - 2.0 * style['xpad']
    panel_area_height = 1.0 - style['top_pad'] - style['bottom_pad']
    panel_height = (panel_area_height - style['row_gap'] * max(rows - 1, 0)) / rows
    cell_width = (panel_area_width - style['col_gap'] * max(width - 1, 0)) / width
    panel_width = min(cell_width, style['panel_width_cap'])
    total_width = panel_width * width + style['col_gap'] * max(width - 1, 0)
    x_start = 0.5 - total_width / 2.0

    for panel_index in range(rows * width):
        if panel_index >= len(group_panels):
            continue
        panel_row = panel_index // width
        panel_column = panel_index % width
        x0 = x_start + panel_column * (panel_width + style['col_gap'])
        y0 = style['bottom_pad'] + (rows - panel_row - 1) * (panel_height + style['row_gap'])
        label = display_label(group_panels[panel_index].metric.spec)
        host_axis.text(
            x0 + panel_width / 2.0,
            y0 + panel_height + style['label_offset'],
            label,
            ha='center',
            va='bottom',
            fontsize=style['label_fontsize'],
            color='black',
            transform=host_axis.transAxes,
        )
        axis = host_axis.inset_axes([x0, y0, panel_width, panel_height])
        axis.set_zorder(2)
        axis.set_facecolor((1, 1, 1, 0))
        panel = group_panels[panel_index]
        plot_panel(
            axis,
            panel,
            tissue_spaces[panel.metric.space],
            cmap,
            bg_limits[panel.metric.space],
        )


def group_grid_shape(group: str, n_panels: int) -> tuple[int, int]:
    if group in STACKED_PANEL_GROUPS:
        return n_panels, 1
    if group == 'dMRI':
        return 2, int(np.ceil(n_panels / 2))
    return 1, n_panels


def group_inch_style(group: str) -> dict[str, float]:
    style = {
        'panel_gap': PANEL_GAP_IN,
        'row_gap': PANEL_ROW_GAP_IN,
        'xpad': GROUP_XPAD_IN,
        'top_pad': GROUP_TOP_PAD_IN,
        'bottom_pad': GROUP_BOTTOM_PAD_IN,
        'label_fontsize': 7.9,
        'group_fontsize': 9.8,
        'min_width': LONG_LABEL_WIDTH_IN.get(group, 0.0),
    }
    style.update(GROUP_INCH_STYLE.get(group, {}))
    return style


def group_size_inches(group: str, n_panels: int) -> tuple[float, float, int, int]:
    style = group_inch_style(group)
    rows, columns = group_grid_shape(group, n_panels)
    panel_area_width = columns * PANEL_WIDTH_IN + max(columns - 1, 0) * style['panel_gap']
    width = max(panel_area_width + 2 * style['xpad'], style['min_width'])
    height = (
        style['top_pad']
        + rows * PANEL_HEIGHT_IN
        + max(rows - 1, 0) * style['row_gap']
        + style['bottom_pad']
    )
    return width, height, rows, columns


def add_group_box_at(
    fig: plt.Figure,
    left: float,
    bottom: float,
    width: float,
    height: float,
    figure_width: float,
    figure_height: float,
    color: str,
) -> plt.Axes:
    axis = fig.add_axes(
        [
            left / figure_width,
            bottom / figure_height,
            width / figure_width,
            height / figure_height,
        ]
    )
    axis.set_zorder(0)
    axis.set_facecolor('white')
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.25)
        spine.set_edgecolor(mcolors.to_rgba(color, 0.72))
    return axis


def plot_group_panels_inches(
    fig: plt.Figure,
    host_axis: plt.Axes,
    group: str,
    group_panels: Sequence[PreparedPanel],
    left: float,
    bottom: float,
    group_width: float,
    group_height: float,
    rows: int,
    columns: int,
    figure_width: float,
    figure_height: float,
    tissue_spaces: dict[str, TissueSpace],
    bg_limits: dict[str, tuple[float, float]],
    cmap: str,
    color: str,
) -> None:
    style = group_inch_style(group)
    host_axis.text(
        0.015,
        0.965,
        group_label(group),
        ha='left',
        va='top',
        fontsize=style['group_fontsize'],
        fontweight='bold',
        color=color,
        transform=host_axis.transAxes,
    )
    total_panel_width = columns * PANEL_WIDTH_IN + max(columns - 1, 0) * style['panel_gap']
    x_start = left + (group_width - total_panel_width) / 2.0

    for panel_index, panel in enumerate(group_panels):
        panel_row = panel_index // columns
        panel_column = panel_index % columns
        x0 = x_start + panel_column * (PANEL_WIDTH_IN + style['panel_gap'])
        y0 = (
            bottom
            + style['bottom_pad']
            + (rows - panel_row - 1) * (PANEL_HEIGHT_IN + style['row_gap'])
        )
        fig.text(
            (x0 + PANEL_WIDTH_IN / 2.0) / figure_width,
            (y0 + PANEL_HEIGHT_IN + 0.026) / figure_height,
            display_label(panel.metric.spec),
            ha='center',
            va='bottom',
            fontsize=style['label_fontsize'],
            color='black',
        )
        axis = fig.add_axes(
            [
                x0 / figure_width,
                y0 / figure_height,
                PANEL_WIDTH_IN / figure_width,
                PANEL_HEIGHT_IN / figure_height,
            ]
        )
        axis.set_zorder(2)
        axis.set_facecolor((1, 1, 1, 0))
        plot_panel(
            axis,
            panel,
            tissue_spaces[panel.metric.space],
            cmap,
            bg_limits[panel.metric.space],
        )


def plot_figure(
    panels: Sequence[PreparedPanel],
    tissue_spaces: dict[str, TissueSpace],
    output_base: Path,
    cmap: str,
    max_columns: int,
) -> None:
    panels_by_group = {
        group: [panel for panel in panels if panel.metric.source_group == group]
        for group in TABLE_ORDER
    }

    packed_rows: list[list[tuple[str, list[PreparedPanel], int, int, int]]] = []
    for layout_row in LAYOUT_ROWS:
        packed_row: list[tuple[str, list[PreparedPanel], int, int, int]] = []
        current_width = 0
        for group in layout_row:
            group_panels = panels_by_group.get(group, [])
            if not group_panels:
                continue
            width = min(max_columns - current_width, len(group_panels))
            rows = int(np.ceil(len(group_panels) / width))
            packed_row.append((group, group_panels, current_width, width, rows))
            current_width += width
        if packed_row:
            packed_rows.append(packed_row)

    row_panel_counts = [max(group[-1] for group in row) for row in packed_rows]
    row_header_ratios = []
    for packed_row in packed_rows:
        groups = {group for group, *_ in packed_row}
        if 'dMRI' in groups:
            row_header_ratios.append(0.26)
        elif {'T1w/T2w', 'g-ratio'} & groups:
            row_header_ratios.append(0.50)
        else:
            row_header_ratios.append(0.34)
    height_ratios = [header + rows for header, rows in zip(row_header_ratios, row_panel_counts)]
    figure_width = 7.30
    figure_height = 0.22 + 0.99 * sum(height_ratios) + 0.13 * len(packed_rows)
    fig = plt.figure(figsize=(figure_width, figure_height), facecolor='white')
    outer = fig.add_gridspec(
        len(packed_rows),
        max_columns,
        left=0.022,
        right=0.997,
        top=0.990,
        bottom=0.035,
        hspace=0.045,
        # Leave a narrow white gutter between adjacent group boxes so their
        # colored spines remain visually distinct instead of being overdrawn.
        wspace=0.025,
        height_ratios=height_ratios,
    )
    bg_limits = {
        space: robust_background_limits(
            np.asarray(tissue.background, dtype=np.float32),
            tissue.brain_mask,
        )
        for space, tissue in tissue_spaces.items()
    }

    for row_index, packed_row in enumerate(packed_rows):
        for group, group_panels, start_column, width, rows in packed_row:
            color = FIGURE_SOURCE_COLORS[group]
            group_spec = outer[row_index, start_column : start_column + width]
            group_axis = add_group_box(fig, group_spec, color)
            add_group_label(group_axis, group, color)
            header_ratio = row_header_ratios[row_index]
            nested = group_spec.subgridspec(
                rows + 1,
                width,
                height_ratios=[header_ratio] + [1] * rows,
                hspace=0.20 if rows > 1 else 0.075,
                wspace=0.0,
            )
            title_axis = fig.add_subplot(nested[0, :])
            title_axis.set_axis_off()

            for panel_index in range(rows * width):
                panel_row = panel_index // width
                panel_column = panel_index % width
                axis = fig.add_subplot(nested[panel_row + 1, panel_column])
                axis.set_zorder(2)
                axis.set_facecolor((1, 1, 1, 0))
                if panel_index >= len(group_panels):
                    axis.set_axis_off()
                    continue
                panel = group_panels[panel_index]
                plot_panel(
                    axis,
                    panel,
                    tissue_spaces[panel.metric.space],
                    cmap,
                    bg_limits[panel.metric.space],
                )

    bottom_used_width = max(
        start_column + width for _, _, start_column, width, _ in packed_rows[-1]
    )
    if bottom_used_width < max_columns:
        colorbar_spec = outer[-1, bottom_used_width:max_columns]
        colorbar_host = fig.add_subplot(colorbar_spec)
        colorbar_host.set_axis_off()
        colorbar_axis = colorbar_host.inset_axes([0.20, 0.52, 0.60, 0.075])
    else:
        colorbar_axis = fig.add_axes([0.40, 0.024, 0.20, 0.012])
    colorbar = fig.colorbar(
        ScalarMappable(norm=mcolors.Normalize(vmin=5, vmax=95), cmap=cmap),
        cax=colorbar_axis,
        orientation='horizontal',
    )
    colorbar.set_ticks([5, 50, 95])
    colorbar.set_ticklabels(['5th', '50th', '95th'])
    colorbar.ax.tick_params(labelsize=8.2, length=0)
    colorbar.set_label('Percentile', fontsize=9.0, fontweight='bold', labelpad=2)

    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix('.png'), dpi=300, bbox_inches='tight')
    fig.savefig(output_base.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def write_status_tsv(
    output: Path,
    panels: Sequence[PreparedPanel],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as fobj:
        writer = csv.writer(fobj, delimiter='\t')
        writer.writerow(
            [
                'participant_id',
                'session',
                'source_group',
                'metric',
                'space',
                'n_images_for_metric',
                'vmin_5th_gmwm',
                'vmax_95th_gmwm',
                'inverted_for_display',
                'path',
                'hybrid_gm_path',
            ]
        )
        for panel in panels:
            gm_paths = panel.metric.hybrid_gm_paths or ('',) * len(panel.metric.paths)
            for (subject, session), path, gm_path in zip(
                panel.metric.contributors,
                panel.metric.paths,
                gm_paths,
                strict=True,
            ):
                writer.writerow(
                    [
                        subject,
                        session,
                        panel.metric.source_group,
                        panel.metric.spec.primary_label,
                        panel.metric.space,
                        len(panel.metric.paths),
                        f'{panel.limits[0]:.10g}',
                        f'{panel.limits[1]:.10g}',
                        int(is_display_inverted(panel.metric)),
                        path,
                        gm_path,
                    ]
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--project-root',
        default=PROJECT_ROOT,
        help='Project root containing derivatives/.',
    )
    parser.add_argument(
        '--derivatives-root',
        default=SOURCE_DERIVATIVES_ROOT,
        help='Reusable source-scalar derivatives directory.',
    )
    parser.add_argument(
        '--patterns-file',
        default=REPO_ROOT / 'configuration' / 'metrics.yml',
        type=Path,
        help='Scalar path pattern registry.',
    )
    parser.add_argument(
        '--qc-file',
        default=REPO_ROOT / 'data' / 'qc' / 'manual_qc_modality.tsv',
        type=Path,
        help='Manual modality QC table.',
    )
    parser.add_argument(
        '--template-dseg',
        default=REPO_ROOT / 'data' / 'atlases' / f'tpl-{MNI_SPACE}_res-01_seg-aseg_dseg.nii.gz',
        type=Path,
        help='Deterministic template aseg dseg used as the reference grid and GM selector for NODDI hybrids.',
    )
    parser.add_argument(
        '--subject-id',
        action='append',
        help='Limit averaging to this subject, with or without sub-. Repeat as needed.',
    )
    parser.add_argument(
        '--session-id',
        action='append',
        help='Limit averaging to this session, with or without ses-. Repeat as needed.',
    )
    parser.add_argument(
        '--output',
        default=RUN_FIGURES_ROOT / 'figure_01_primary_maps' / 'Figure1',
        type=Path,
        help='Output path stem, or a .png/.pdf path whose suffix will be replaced.',
    )
    parser.add_argument(
        '--average-cache-dir',
        type=Path,
        help='Directory for cached group-average maps. Defaults to <output>_average_cache.',
    )
    parser.add_argument(
        '--recalculate-average-maps',
        action='store_true',
        help='Recompute group-average maps even if matching cached maps exist.',
    )
    parser.add_argument(
        '--cmap',
        default='viridis',
        help='Matplotlib colormap used for every scalar overlay.',
    )
    parser.add_argument(
        '--max-columns',
        type=int,
        default=6,
        help='Maximum number of map panels per source-group row.',
    )
    parser.add_argument(
        '--log-level',
        choices=('DEBUG', 'INFO', 'WARNING', 'ERROR'),
        default='INFO',
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_columns < 1:
        parser.error('--max-columns must be at least 1')
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s | %(levelname)s | %(message)s',
    )

    project_root = Path(args.project_root).expanduser().resolve()
    derivatives = (
        Path(args.derivatives_root).expanduser().resolve()
        if args.derivatives_root
        else project_root / 'derivatives'
    )
    output_base = Path(args.output).expanduser().resolve()
    if output_base.suffix.lower() in {'.png', '.pdf'}:
        output_base = output_base.with_suffix('')
    average_cache_dir = (
        Path(args.average_cache_dir).expanduser().resolve()
        if args.average_cache_dir
        else output_base.with_name(f'{output_base.name}_average_cache')
    )

    patterns_file = Path(args.patterns_file).expanduser().resolve()
    metric_specs = table_metric_specs(patterns_file)
    qc_rows = load_qc_rows(Path(args.qc_file).expanduser().resolve())
    resolved_metrics = resolve_average_metrics(
        derivatives,
        metric_specs,
        qc_rows,
        args.subject_id,
        args.session_id,
        patterns_file,
    )
    tissue_spaces = {
        MNI_SPACE: load_mni_tissue_space(
            Path(args.template_dseg).expanduser().resolve(),
        )
    }
    panels = prepare_panels(
        resolved_metrics,
        tissue_spaces,
        average_cache_dir=average_cache_dir,
        recalculate_average_maps=args.recalculate_average_maps,
    )
    plot_figure(
        panels,
        tissue_spaces,
        output_base=output_base,
        cmap=args.cmap,
        max_columns=args.max_columns,
    )
    write_status_tsv(
        output_base.with_name(f'{output_base.name}_inputs.tsv'),
        panels,
    )
    LOGGER.info(
        'Wrote %s and %s',
        output_base.with_suffix('.png'),
        output_base.with_suffix('.pdf'),
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
