#!/usr/bin/env python3
"""Generate a multipage PDF for ACPC/T1w quality control.

The report checks spatial alignment and scalar coverage for every discovered
subject/session:

1. Selected ACPC-space tract bundles and their T1w-warped counterparts.
2. The native T1w DKT parcellation and its ACPC-space warp.
3. T1w-space primary non-dMRI scalar maps.
4. DKT parcel coverage for T1w-space and ACPC-space primary scalar maps.
5. Native-ACPC primary dMRI maps with GM/WM tissue distributions.

The script never modifies source derivatives. Temporary tractogram files and
generated ACPC tissue-label warps are kept under the report work directory so
subsequent report runs can reuse them.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from nibabel.affines import apply_affine
from scipy.stats import gaussian_kde

from utils.images import first_glob, load_canonical, resample_image  # noqa: E402
from utils.metrics import build_metric_specs, load_patterns, primary_metric_specs  # noqa: E402
from utils.paths import OUTPUT_DERIVATIVES_ROOT, PROJECT_ROOT, SOURCE_DERIVATIVES_ROOT  # noqa: E402

LOGGER = logging.getLogger('qc')

PAGE_SIZE = (16.0, 10.0)
DEFAULT_PATTERNS_FILE = REPO_ROOT / 'configuration' / 'metrics.yml'
SPACE_COLORS = {'ACPC': '#2A9D8F', 'T1w': '#E76F51'}
BUNDLE_COLORS = ('#00A6D6', '#F28E2B', '#59A14F', '#AF7AA1')
MISSING_COLOR = '#9B2226'
TISSUE_COLORS = {'GM': '#D55E00', 'WM': '#0072B2'}
SESSION_LINESTYLES = ('-', '--', ':', '-.')
SMRIPREP_DSEG_LABELS = {'GM': 1, 'WM': 2}
ATLAS_DESC = 'DKTatlas'
ATLAS_DISPLAY = 'DKT'
RUN_DERIVATIVE_DATASETS = {
    'bundle_myelin_stats',
    'DKTatlas_myelin_stats',
    'mni_ribbon_masks',
    't1w_registration',
    'warped_bundles',
}


@dataclass(frozen=True)
class DerivativeOverlay:
    """Route reusable inputs and run-generated derivatives to separate roots."""

    source_root: Path
    run_root: Path

    def __truediv__(self, value: str | Path) -> Path:
        relative = Path(value)
        first = relative.parts[0] if relative.parts else ''
        root = self.run_root if first in RUN_DERIVATIVE_DATASETS else self.source_root
        return root / relative

    def __str__(self) -> str:
        return f'source={self.source_root}; run={self.run_root}'


DKT_LABEL_IDS = {
    1002,
    1003,
    1005,
    1006,
    1007,
    1008,
    1009,
    1010,
    1011,
    1012,
    1013,
    1014,
    1015,
    1016,
    1017,
    1018,
    1019,
    1020,
    1021,
    1022,
    1023,
    1024,
    1025,
    1026,
    1027,
    1028,
    1029,
    1030,
    1031,
    1034,
    1035,
    2002,
    2003,
    2005,
    2006,
    2007,
    2008,
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
    2015,
    2016,
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
    2027,
    2028,
    2029,
    2030,
    2031,
    2034,
    2035,
}


@dataclass(frozen=True)
class MetricPattern:
    """A scalar map label and path pattern."""

    label: str
    pattern: str


def primary_metric_patterns(
    patterns_file: Path,
    source_image: str | None,
    space: str,
    include_gm_noddi_icvf: bool = False,
) -> tuple[MetricPattern, ...]:
    patterns = load_patterns(patterns_file)
    metrics: list[MetricPattern] = []
    for spec in primary_metric_specs(build_metric_specs(patterns_file)):
        if source_image is None:
            if spec.source_image == 'dMRI':
                continue
        elif spec.source_image != source_image:
            continue
        pattern = patterns[spec.group][spec.pattern_key].replace('{space}', space)
        metrics.append(MetricPattern(spec.primary_label, pattern))
        if (
            include_gm_noddi_icvf
            and spec.group == 'dMRI'
            and spec.pattern_key == 'ICVF'
            and 'ICVF (GM)' in patterns.get('dMRI', {})
        ):
            gm_pattern = patterns['dMRI']['ICVF (GM)'].replace('{space}', space)
            metrics.append(MetricPattern('ICVF (GM)', gm_pattern))
    return tuple(metrics)


MYELIN_METRICS = primary_metric_patterns(DEFAULT_PATTERNS_FILE, None, 'T1w')
METRIC_BY_LABEL = {metric.label: metric for metric in MYELIN_METRICS}

DWI_METRICS = primary_metric_patterns(
    DEFAULT_PATTERNS_FILE,
    'dMRI',
    'ACPC',
    include_gm_noddi_icvf=True,
)
DWI_METRIC_BY_LABEL = {metric.label: metric for metric in DWI_METRICS}


@dataclass(frozen=True)
class SessionKey:
    subject: str
    session: str


@dataclass
class SessionInputs:
    key: SessionKey
    acpc_t1w: Path | None
    t1w: Path | None
    dseg_acpc: Path | None
    dseg_t1w: Path | None
    t1w_to_acpc_xfm: Path | None
    acpc_bundle_dir: Path | None
    t1w_bundle_dir: Path | None
    tissue_dseg: Path | None
    t1w_brain_mask: Path | None
    acpc_brain_mask: Path | None


@dataclass
class DensityResult:
    data: np.ndarray
    total_streamlines: int
    used_streamlines: int


@dataclass
class StatusEntry:
    subject: str
    session: str
    section: str
    item: str
    status: str
    detail: str


@dataclass
class ScalarPanelData:
    """Prepared native T1w scalar data for one subject/session."""

    source_path: Path
    t1w_reference: nib.spatialimages.SpatialImage
    source_data: np.ndarray
    source_brain_mask: np.ndarray


@dataclass
class DwiPanelData:
    """Prepared ACPC dMRI scalar data for one subject/session."""

    source_path: Path
    acpc_reference: nib.spatialimages.SpatialImage
    data: np.ndarray
    brain_mask: np.ndarray


def normalize_subject(value: str) -> str:
    token = value.strip()
    return token if token.startswith('sub-') else f'sub-{token}'


def normalize_session(value: str) -> str:
    token = value.strip()
    return token if token.startswith('ses-') else f'ses-{token}'


def normalize_name(value: str) -> str:
    return re.sub(r'[^a-z0-9]+', '', value.lower())


def safe_filename(value: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '-', value).strip('-')


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def glob_string(pattern: str) -> list[str]:
    """Glob an absolute pattern without relying on Path.glob's root handling."""
    from glob import glob

    return glob(pattern)


def discover_subjects(derivatives: Path) -> list[str]:
    roots = (
        derivatives / 'warped_bundles',
        derivatives / 't1w_registration',
        derivatives / 'qsiprep',
        derivatives / 'smriprep',
    )
    subjects = {
        path.name for root in roots if root.is_dir() for path in root.glob('sub-*') if path.is_dir()
    }
    return sorted(subjects)


def discover_sessions(derivatives: Path, subject: str) -> list[str]:
    roots = [
        derivatives / 'warped_bundles' / subject,
        derivatives / 'ihmt' / subject,
        derivatives / 'pymp2rage' / subject,
        derivatives / 'qsm' / subject,
        derivatives / 't1wt2w_ratio' / subject,
        derivatives / 'g_ratio' / subject,
    ]
    recon_root = derivatives / 'qsirecon' / 'derivatives'
    if recon_root.is_dir():
        roots.extend(path / subject for path in recon_root.glob('qsirecon-*MSMTAutoTrack*'))
    sessions = {
        path.name for root in roots if root.is_dir() for path in root.glob('ses-*') if path.is_dir()
    }
    return sorted(sessions)


def find_acpc_bundle_dir(derivatives: Path, subject: str, session: str) -> Path | None:
    recon_root = derivatives / 'qsirecon' / 'derivatives'
    candidates = sorted(
        path / subject / session / 'dwi'
        for path in recon_root.glob('qsirecon-*MSMTAutoTrack*')
        if (path / subject / session / 'dwi').is_dir()
    )
    return candidates[0] if candidates else None


def collect_session_inputs(derivatives: Path, key: SessionKey) -> SessionInputs:
    subject = key.subject
    session = key.session
    registration_dir = derivatives / 't1w_registration' / subject / 'anat'

    acpc_t1w = first_glob(
        (
            derivatives
            / 'qsiprep'
            / subject
            / 'anat'
            / f'{subject}_space-ACPC_desc-preproc_T1w.nii*',
            derivatives
            / 'qsiprep'
            / subject
            / session
            / 'anat'
            / f'{subject}_{session}_space-ACPC_desc-preproc_T1w.nii*',
        )
    )
    t1w = first_glob(
        (
            derivatives
            / 'smriprep'
            / subject
            / session
            / 'anat'
            / f'{subject}_{session}_acq-MPRAGE*run-01_desc-preproc_T1w.nii*',
            derivatives
            / 'smriprep'
            / subject
            / 'anat'
            / f'{subject}_acq-MPRAGE*run-01_desc-preproc_T1w.nii*',
            derivatives
            / 'smriprep'
            / subject
            / session
            / 'anat'
            / f'{subject}_{session}_*desc-preproc_T1w.nii*',
            derivatives / 'smriprep' / subject / 'anat' / f'{subject}_*desc-preproc_T1w.nii*',
        )
    )
    return SessionInputs(
        key=key,
        acpc_t1w=acpc_t1w,
        t1w=t1w,
        dseg_acpc=first_existing(
            [registration_dir / f'{subject}_space-ACPC_desc-{ATLAS_DESC}_dseg.nii.gz']
        ),
        dseg_t1w=first_existing(
            [registration_dir / f'{subject}_space-T1w_desc-{ATLAS_DESC}_dseg.nii.gz']
        ),
        t1w_to_acpc_xfm=first_existing(
            [registration_dir / f'{subject}_from-T1w_to-ACPC_mode-image_xfm.h5']
        ),
        acpc_bundle_dir=find_acpc_bundle_dir(derivatives, subject, session),
        t1w_bundle_dir=first_existing_directory(
            [derivatives / 'warped_bundles' / subject / session / 'dwi']
        ),
        tissue_dseg=first_glob(
            (
                derivatives
                / 'smriprep'
                / subject
                / session
                / 'anat'
                / f'{subject}_{session}_acq-MPRAGE*run-01_dseg.nii*',
                derivatives
                / 'smriprep'
                / subject
                / session
                / 'anat'
                / f'{subject}_{session}_*dseg.nii*',
                derivatives
                / 'smriprep'
                / subject
                / 'anat'
                / f'{subject}_acq-MPRAGE*run-01_dseg.nii*',
                derivatives / 'smriprep' / subject / 'anat' / f'{subject}_*dseg.nii*',
            )
        ),
        t1w_brain_mask=first_glob(
            (
                derivatives
                / 'smriprep'
                / subject
                / session
                / 'anat'
                / f'{subject}_{session}_acq-MPRAGE*run-01_desc-brain_mask.nii*',
                derivatives
                / 'smriprep'
                / subject
                / 'anat'
                / f'{subject}_acq-MPRAGE*run-01_desc-brain_mask.nii*',
                derivatives
                / 'smriprep'
                / subject
                / session
                / 'anat'
                / f'{subject}_{session}_*desc-brain_mask.nii*',
                derivatives / 'smriprep' / subject / 'anat' / f'{subject}_*desc-brain_mask.nii*',
            )
        ),
        acpc_brain_mask=first_glob(
            (
                derivatives
                / 'qsiprep'
                / subject
                / 'anat'
                / f'{subject}_space-ACPC_desc-brain_mask.nii*',
                derivatives
                / 'qsiprep'
                / subject
                / session
                / 'anat'
                / f'{subject}_{session}_space-ACPC_desc-brain_mask.nii*',
            )
        ),
    )


def first_existing_directory(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.is_dir():
            return path
    return None


def extract_bundle_name(path: Path) -> str:
    match = re.search(r'_bundle-(.+?)_streamlines\.(?:tck|trx)(?:\.gz)?$', path.name)
    return match.group(1) if match else path.stem


def find_bundle_file(directory: Path | None, requested: str) -> Path | None:
    if directory is None:
        return None
    candidates = sorted(
        path
        for path in directory.glob('*bundle-*_streamlines.*')
        if path.suffix in {'.tck', '.gz', '.trx'} or path.name.endswith(('.tck.gz', '.trx.gz'))
    )
    target = normalize_name(requested)
    exact = [path for path in candidates if normalize_name(extract_bundle_name(path)) == target]
    if exact:
        return exact[0]
    relaxed_target = target.removeprefix('association')
    relaxed = [
        path
        for path in candidates
        if normalize_name(extract_bundle_name(path)).removeprefix('association') == relaxed_target
    ]
    return relaxed[0] if relaxed else None


def robust_limits(data: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    values = np.asarray(data, dtype=float)
    valid = np.isfinite(values)
    if mask is not None:
        valid &= mask
    elif np.any(valid & (np.abs(values) > 0)):
        valid &= np.abs(values) > 0
    selected = values[valid]
    if not selected.size:
        return 0.0, 1.0
    low, high = np.percentile(selected, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low = float(np.nanmin(selected))
        high = float(np.nanmax(selected))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def decompress_tractogram(path: Path, temp_dir: Path) -> Path:
    if not path.name.endswith('.gz'):
        return path
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = '.tck' if path.name.endswith('.tck.gz') else '.trx'
    out_path = temp_dir / f'{safe_filename(path.name)}{suffix}'
    with gzip.open(path, 'rb') as source, out_path.open('wb') as target:
        shutil.copyfileobj(source, target)
    return out_path


def reservoir_streamlines(
    tractogram_path: Path,
    max_streamlines: int,
    temp_dir: Path,
) -> tuple[list[np.ndarray], int]:
    load_path = decompress_tractogram(tractogram_path, temp_dir)
    loaded = nib.streamlines.load(str(load_path), lazy_load=True)
    tractogram = loaded.tractogram
    tractogram.to_world()
    rng = np.random.default_rng(20260727)
    sampled: list[np.ndarray] = []
    total = 0
    for total, streamline in enumerate(tractogram.streamlines, start=1):
        points = np.asarray(streamline, dtype=np.float32)
        if len(points) < 2:
            continue
        if len(sampled) < max_streamlines:
            sampled.append(points)
        else:
            replacement = int(rng.integers(0, total))
            if replacement < max_streamlines:
                sampled[replacement] = points
    return sampled, total


def rasterize_streamlines(
    tractogram_path: Path,
    reference: nib.spatialimages.SpatialImage,
    max_streamlines: int,
    temp_dir: Path,
) -> DensityResult:
    streamlines, total = reservoir_streamlines(
        tractogram_path, max_streamlines=max_streamlines, temp_dir=temp_dir
    )
    density = np.zeros(reference.shape[:3], dtype=np.float32)
    inverse_affine = np.linalg.inv(reference.affine)
    shape = np.asarray(reference.shape[:3], dtype=int)

    for streamline in streamlines:
        voxels = apply_affine(inverse_affine, streamline)
        deltas = np.diff(voxels, axis=0)
        segment_steps = np.maximum(1, np.ceil(np.max(np.abs(deltas), axis=1) * 1.5).astype(int))
        sampled_segments = [
            np.linspace(voxels[index], voxels[index + 1], steps + 1, endpoint=True)
            for index, steps in enumerate(segment_steps)
        ]
        if not sampled_segments:
            continue
        indices = np.rint(np.concatenate(sampled_segments, axis=0)).astype(int)
        inside = np.all((indices >= 0) & (indices < shape), axis=1)
        indices = np.unique(indices[inside], axis=0)
        if indices.size:
            density[tuple(indices.T)] += 1
    return DensityResult(
        data=density,
        total_streamlines=total,
        used_streamlines=len(streamlines),
    )


def foreground_center(
    overlay: np.ndarray | None,
    background: np.ndarray,
) -> tuple[int, int, int]:
    if overlay is not None:
        finite = np.isfinite(overlay)
        if np.issubdtype(overlay.dtype, np.bool_) or np.issubdtype(overlay.dtype, np.integer):
            foreground = finite & (overlay != 0)
        else:
            nonzero = np.abs(overlay[finite])
            threshold = np.percentile(nonzero, 60) if nonzero.size else 0
            foreground = finite & (np.abs(overlay) > threshold)
        coordinates = np.argwhere(foreground)
        if coordinates.size:
            return tuple(np.rint(np.median(coordinates, axis=0)).astype(int))
    finite_background = np.isfinite(background)
    if finite_background.any():
        low, _ = robust_limits(background)
        coordinates = np.argwhere(finite_background & (background > low))
        if coordinates.size:
            return tuple(np.rint(np.median(coordinates, axis=0)).astype(int))
    return tuple(np.asarray(background.shape[:3]) // 2)


def segmentation_boundaries_2d(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    boundary = np.zeros(labels.shape, dtype=bool)
    horizontal = labels[:, :-1] != labels[:, 1:]
    horizontal &= (labels[:, :-1] != 0) | (labels[:, 1:] != 0)
    vertical = labels[:-1, :] != labels[1:, :]
    vertical &= (labels[:-1, :] != 0) | (labels[1:, :] != 0)
    boundary[:, :-1] |= horizontal
    boundary[:, 1:] |= horizontal
    boundary[:-1, :] |= vertical
    boundary[1:, :] |= vertical
    return boundary


def slice_data(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.rot90(np.take(volume, index, axis=axis))


def transparent_color_map(color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        f'transparent_{safe_filename(color)}',
        [(0.0, (1, 1, 1, 0)), (1.0, (*matplotlib.colors.to_rgb(color), 1))],
    )


def plot_orthogonal_montage(
    figure: plt.Figure,
    spec,
    background: nib.spatialimages.SpatialImage | None,
    overlay: np.ndarray | None = None,
    *,
    overlay_kind: str = 'scalar',
    color: str = '#00A6D6',
    title: str = '',
    limits: tuple[float, float] | None = None,
    note: str = '',
    center_world: np.ndarray | None = None,
) -> None:
    nested = spec.subgridspec(1, 3, wspace=0.02)
    axes = [figure.add_subplot(nested[0, index]) for index in range(3)]
    if background is None:
        for axis in axes:
            axis.set_axis_off()
        axes[1].text(
            0.5,
            0.5,
            'Missing anatomical reference',
            ha='center',
            va='center',
            color=MISSING_COLOR,
            transform=axes[1].transAxes,
        )
        axes[1].set_title(title, fontsize=9)
        return

    background_data = np.asarray(background.get_fdata(), dtype=np.float32)
    if center_world is None:
        center = foreground_center(overlay, background_data)
    else:
        center_voxel = apply_affine(np.linalg.inv(background.affine), center_world)
        shape = np.asarray(background_data.shape[:3], dtype=int)
        center = tuple(np.clip(np.rint(center_voxel).astype(int), 0, shape - 1))
    bg_limits = robust_limits(background_data)
    orientation_names = ('Sagittal', 'Coronal', 'Axial')
    for view_axis, axis in enumerate(axes):
        background_slice = slice_data(background_data, view_axis, center[view_axis])
        axis.imshow(background_slice, cmap='gray', vmin=bg_limits[0], vmax=bg_limits[1])
        if overlay is not None:
            overlay_slice = slice_data(overlay, view_axis, center[view_axis])
            if overlay_kind in {'bundle', 'boundary'}:
                mask = (
                    segmentation_boundaries_2d(overlay_slice)
                    if overlay_kind == 'boundary'
                    else overlay_slice > 0
                )
                axis.imshow(
                    mask.astype(float),
                    cmap=transparent_color_map(color),
                    vmin=0,
                    vmax=1,
                    alpha=0.88 if overlay_kind == 'boundary' else 0.72,
                    interpolation='nearest',
                )
            else:
                scalar_limits = limits or robust_limits(overlay)
                masked = np.ma.masked_invalid(overlay_slice)
                masked = np.ma.masked_where(np.abs(masked) == 0, masked)
                axis.imshow(
                    masked,
                    cmap='magma',
                    vmin=scalar_limits[0],
                    vmax=scalar_limits[1],
                    alpha=0.68,
                    interpolation='nearest',
                )
        axis.set_axis_off()
        axis.set_title(orientation_names[view_axis], fontsize=7, color='#555555', pad=2)
    axes[1].text(
        0.5,
        1.16,
        title,
        ha='center',
        va='bottom',
        fontsize=9,
        fontweight='semibold',
        transform=axes[1].transAxes,
    )
    if note:
        axes[1].text(
            0.5,
            -0.08,
            note,
            ha='center',
            va='top',
            fontsize=6.5,
            color='#555555',
            transform=axes[1].transAxes,
        )


def missing_panel(figure: plt.Figure, spec, title: str, detail: str) -> None:
    axis = figure.add_subplot(spec)
    axis.set_axis_off()
    axis.text(
        0.5,
        0.57,
        title,
        ha='center',
        va='center',
        fontsize=9,
        fontweight='semibold',
        transform=axis.transAxes,
    )
    axis.text(
        0.5,
        0.42,
        textwrap.fill(detail, 55),
        ha='center',
        va='center',
        fontsize=7,
        color=MISSING_COLOR,
        transform=axis.transAxes,
    )


def add_page_header(figure: plt.Figure, title: str, subtitle: str) -> None:
    figure.text(0.04, 0.965, title, fontsize=18, fontweight='bold', va='top')
    figure.text(0.04, 0.925, subtitle, fontsize=9, color='#555555', va='top')


def add_page_footer(figure: plt.Figure, page_number: int) -> None:
    figure.text(
        0.04,
        0.018,
        'MIRROR spatial QC | visual inspection required',
        fontsize=7,
        color='#666666',
    )
    figure.text(
        0.96,
        0.018,
        str(page_number),
        fontsize=7,
        color='#666666',
        ha='right',
    )


def save_page(pdf: PdfPages, figure: plt.Figure, page_number: int) -> int:
    add_page_footer(figure, page_number)
    pdf.savefig(figure, bbox_inches='tight', dpi=180)
    plt.close(figure)
    return page_number + 1


def status(
    entries: list[StatusEntry],
    key: SessionKey,
    section: str,
    item: str,
    ok: bool,
    detail: str,
) -> None:
    entries.append(
        StatusEntry(
            subject=key.subject,
            session=key.session,
            section=section,
            item=item,
            status='OK' if ok else 'MISSING/FAILED',
            detail=detail,
        )
    )


def bundle_page(
    pdf: PdfPages,
    inputs: SessionInputs,
    bundle_names: Sequence[str],
    max_streamlines: int,
    work_dir: Path,
    statuses: list[StatusEntry],
    page_number: int,
) -> int:
    key = inputs.key
    figure = plt.figure(figsize=PAGE_SIZE, facecolor='white')
    add_page_header(
        figure,
        f'{key.subject} {key.session} | Bundle transformation',
        'Top: original ACPC tractograms on ACPC T1w. Bottom: warped tractograms on native T1w.',
    )
    grid = figure.add_gridspec(
        2,
        len(bundle_names),
        left=0.04,
        right=0.98,
        top=0.87,
        bottom=0.07,
        hspace=0.22,
        wspace=0.10,
    )

    references: dict[str, nib.spatialimages.SpatialImage | None] = {'ACPC': None, 'T1w': None}
    for space, path in (('ACPC', inputs.acpc_t1w), ('T1w', inputs.t1w)):
        try:
            references[space] = load_canonical(path) if path else None
            status(
                statuses,
                key,
                'bundles',
                f'{space} T1w',
                references[space] is not None,
                str(path) if path else 'Not found',
            )
        except Exception as error:
            status(statuses, key, 'bundles', f'{space} T1w', False, str(error))

    temp_dir = work_dir / key.subject / key.session / 'tractograms'
    temp_dir.mkdir(parents=True, exist_ok=True)
    for column, bundle_name in enumerate(bundle_names):
        for row, (space, directory) in enumerate(
            (('ACPC', inputs.acpc_bundle_dir), ('T1w', inputs.t1w_bundle_dir))
        ):
            tractogram_path = find_bundle_file(directory, bundle_name)
            reference = references[space]
            title = f'{bundle_name} | {space}'
            if tractogram_path is None or reference is None:
                detail = (
                    'Bundle not found'
                    if tractogram_path is None
                    else 'Anatomical reference not found'
                )
                missing_panel(figure, grid[row, column], title, detail)
                status(statuses, key, 'bundles', f'{space} {bundle_name}', False, detail)
                continue
            try:
                density = rasterize_streamlines(
                    tractogram_path,
                    reference,
                    max_streamlines=max_streamlines,
                    temp_dir=temp_dir,
                )
                note = (
                    f'{density.used_streamlines:,}/{density.total_streamlines:,} '
                    f'streamlines displayed'
                )
                plot_orthogonal_montage(
                    figure,
                    grid[row, column],
                    reference,
                    density.data,
                    overlay_kind='bundle',
                    color=BUNDLE_COLORS[column % len(BUNDLE_COLORS)],
                    title=title,
                    note=note,
                )
                status(
                    statuses,
                    key,
                    'bundles',
                    f'{space} {bundle_name}',
                    bool(np.count_nonzero(density.data)),
                    f'{tractogram_path}; {note}',
                )
            except Exception as error:
                LOGGER.exception('Failed to render %s', tractogram_path)
                missing_panel(figure, grid[row, column], title, str(error))
                status(statuses, key, 'bundles', f'{space} {bundle_name}', False, str(error))
    return save_page(pdf, figure, page_number)


def parcellation_page(
    pdf: PdfPages,
    inputs: SessionInputs,
    statuses: list[StatusEntry],
    page_number: int,
) -> int:
    key = inputs.key
    figure = plt.figure(figsize=PAGE_SIZE, facecolor='white')
    add_page_header(
        figure,
        f'{key.subject} {key.session} | FreeSurfer {ATLAS_DISPLAY} parcellation',
        'Parcel boundaries should follow cortical anatomy in native T1w and remain aligned after T1w-to-ACPC warping.',
    )
    grid = figure.add_gridspec(1, 2, left=0.04, right=0.98, top=0.84, bottom=0.10, wspace=0.10)
    for column, (space, t1w_path, dseg_path) in enumerate(
        (
            ('T1w', inputs.t1w, inputs.dseg_t1w),
            ('ACPC', inputs.acpc_t1w, inputs.dseg_acpc),
        )
    ):
        if t1w_path is None or dseg_path is None:
            detail = (
                f'Missing {"anatomical" if t1w_path is None else "dseg"}: '
                f'T1w={t1w_path}, dseg={dseg_path}'
            )
            missing_panel(figure, grid[0, column], f'{ATLAS_DISPLAY} | {space}', detail)
            status(statuses, key, 'parcellation', space, False, detail)
            continue
        try:
            reference = load_canonical(t1w_path)
            dseg = resample_image(load_canonical(dseg_path), reference, order=0)
            labels = np.rint(dseg.get_fdata()).astype(np.int32)
            label_count = len(np.unique(labels[labels > 0]))
            plot_orthogonal_montage(
                figure,
                grid[0, column],
                reference,
                labels,
                overlay_kind='boundary',
                color=SPACE_COLORS[space],
                title=f'{ATLAS_DISPLAY} | {space}',
                note=f'{label_count} nonzero labels',
            )
            status(
                statuses,
                key,
                'parcellation',
                space,
                bool(label_count),
                f'{dseg_path}; {label_count} labels',
            )
        except Exception as error:
            LOGGER.exception('Failed to render parcellation %s', dseg_path)
            missing_panel(figure, grid[0, column], f'{ATLAS_DISPLAY} | {space}', str(error))
            status(statuses, key, 'parcellation', space, False, str(error))
    return save_page(pdf, figure, page_number)


def find_metric(
    derivatives: Path,
    key: SessionKey,
    metric: MetricPattern,
) -> Path | None:
    pattern = metric.pattern.format(subject=key.subject, session=key.session)
    return first_glob([derivatives / pattern])


def resolve_ants_command(requested: str | None) -> str | None:
    if requested:
        expanded = str(Path(requested).expanduser())
        if Path(expanded).is_file():
            return expanded
        return shutil.which(requested)
    env_value = os.environ.get('ANTS_APPLY_TRANSFORMS')
    if env_value:
        return resolve_ants_command(env_value)
    return shutil.which('antsApplyTransforms')


def warp_scalar_to_acpc(
    source: Path,
    reference: Path,
    transform: Path,
    output: Path,
    ants_command: str | None,
    overwrite: bool,
    interpolation: str = 'Linear',
) -> Path:
    if output.is_file() and not overwrite:
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    if ants_command:
        command = [
            ants_command,
            '--dimensionality',
            '3',
            '--input',
            str(source),
            '--reference-image',
            str(reference),
            '--output',
            str(output),
            '--transform',
            str(transform),
            '--interpolation',
            interpolation,
            '--float',
            '1',
        ]
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f'antsApplyTransforms failed ({completed.returncode}):\n{completed.stdout[-2000:]}'
            )
    else:
        try:
            import ants
        except ImportError as error:
            raise RuntimeError(
                'Neither antsApplyTransforms nor the antspyx Python package is available.'
            ) from error
        antspy_interpolation = {
            'Linear': 'linear',
            'GenericLabel': 'genericLabel',
            'NearestNeighbor': 'nearestNeighbor',
        }.get(interpolation, interpolation)
        warped = ants.apply_transforms(
            fixed=ants.image_read(str(reference)),
            moving=ants.image_read(str(source)),
            transformlist=[str(transform)],
            interpolator=antspy_interpolation,
        )
        ants.image_write(warped, str(output))
    if not output.is_file():
        raise RuntimeError(f'Expected warped scalar was not created: {output}')
    return output


def scalar_group_limits(
    arrays: Sequence[np.ndarray],
    brain_masks: Sequence[np.ndarray],
    percentiles: tuple[float, float],
) -> tuple[float, float]:
    """Return robust limits from brain voxels, shared across sessions."""
    if len(arrays) != len(brain_masks):
        raise ValueError('Each scalar array must have a corresponding brain mask')
    value_arrays = [
        array[np.isfinite(array) & (np.abs(array) > 0) & np.asarray(brain_mask, dtype=bool)]
        for array, brain_mask in zip(arrays, brain_masks)
        if np.any(np.isfinite(array) & (np.abs(array) > 0) & np.asarray(brain_mask, dtype=bool))
    ]
    if not value_arrays:
        return 0.0, 1.0
    values = np.concatenate(value_arrays)
    low, high = np.percentile(values, percentiles)
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def shared_metric_centers(
    prepared: Iterable[ScalarPanelData],
) -> dict[str, np.ndarray]:
    """Choose one T1w world-coordinate slice center across sessions."""
    world_centers: dict[str, list[np.ndarray]] = {'T1w': []}
    for panel_data in prepared:
        data = np.where(
            panel_data.source_brain_mask,
            panel_data.source_data,
            np.nan,
        )
        background = np.asarray(panel_data.t1w_reference.get_fdata(), dtype=np.float32)
        voxel_center = foreground_center(data, background)
        world_centers['T1w'].append(
            np.asarray(apply_affine(panel_data.t1w_reference.affine, voxel_center), dtype=float)
        )
    return {
        space: np.median(np.vstack(centers), axis=0)
        for space, centers in world_centers.items()
        if centers
    }


def tissue_metric_values(
    panel_data: ScalarPanelData,
    dseg_path: Path,
) -> dict[str, np.ndarray]:
    """Extract finite native-space scalar values from sMRIPrep GM and WM."""
    dseg = resample_image(load_canonical(dseg_path), panel_data.t1w_reference, order=0)
    labels = np.rint(dseg.get_fdata()).astype(np.int16)
    return tissue_values_from_labels(panel_data.source_data, labels)


def tissue_values_from_labels(
    data: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    finite = np.isfinite(data)
    return {
        tissue: data[finite & (labels == label)].astype(np.float64, copy=False)
        for tissue, label in SMRIPREP_DSEG_LABELS.items()
    }


def parcel_coverage_values(
    metric_path: Path,
    dseg_path: Path,
) -> dict[int, float]:
    """Return DKT valid-voxel coverage for a scalar map and same-space dseg."""
    metric_image = load_canonical(metric_path)
    metric_data = np.asarray(metric_image.get_fdata(), dtype=np.float32)
    dseg = resample_image(load_canonical(dseg_path), metric_image, order=0)
    labels = np.rint(dseg.get_fdata()).astype(np.int32)
    valid = np.isfinite(metric_data) & (metric_data != 0)
    coverage: dict[int, float] = {}
    for label in sorted(int(value) for value in np.unique(labels) if value > 0):
        if label not in DKT_LABEL_IDS:
            continue
        parcel = labels == label
        n_total = int(np.count_nonzero(parcel))
        if n_total:
            coverage[label] = float(np.count_nonzero(valid & parcel) / n_total)
    return coverage


def parcel_coverage_pages(
    pdf: PdfPages,
    derivatives: Path,
    subject_inputs: Sequence[SessionInputs],
    metrics: Sequence[MetricPattern],
    statuses: list[StatusEntry],
    page_number: int,
    space_label: str = 'T1w',
) -> int:
    """Plot DKT parcel coverage for each metric and session in one space."""
    if not subject_inputs:
        return page_number
    subject_inputs = sorted(subject_inputs, key=lambda item: item.key.session)
    subject = subject_inputs[0].key.subject
    records: list[tuple[SessionKey, str, dict[int, float]]] = []
    if space_label not in {'T1w', 'ACPC'}:
        raise ValueError(f'Unsupported DKT coverage space: {space_label}')
    section = f'{space_label} parcel coverage'

    for metric in metrics:
        for inputs in subject_inputs:
            key = inputs.key
            metric_path = find_metric(derivatives, key, metric)
            dseg_path = inputs.dseg_acpc if space_label == 'ACPC' else inputs.dseg_t1w
            if metric_path is None or dseg_path is None:
                detail = (
                    f'{space_label} metric not found'
                    if metric_path is None
                    else f'{space_label} {ATLAS_DISPLAY} dseg not found'
                )
                status(
                    statuses,
                    key,
                    section,
                    metric.label,
                    False,
                    detail,
                )
                continue
            try:
                coverage = parcel_coverage_values(metric_path, dseg_path)
                if not coverage:
                    raise RuntimeError(f'No nonzero {ATLAS_DISPLAY} parcels found')
                records.append((key, metric.label, coverage))
                values = np.asarray(list(coverage.values()), dtype=float)
                status(
                    statuses,
                    key,
                    section,
                    metric.label,
                    True,
                    (
                        f'{metric_path}; parcels={len(coverage)}, '
                        f'median={np.median(values):.3f}, '
                        f'minimum={np.min(values):.3f}'
                    ),
                )
            except Exception as error:
                LOGGER.exception(
                    'Failed parcel coverage for %s %s %s',
                    key.subject,
                    key.session,
                    metric.label,
                )
                status(
                    statuses,
                    key,
                    section,
                    metric.label,
                    False,
                    str(error),
                )

    if not records:
        return page_number

    parcel_labels = sorted({label for _, _, coverage in records for label in coverage})
    rows_per_page = 28
    for page_start in range(0, len(records), rows_per_page):
        page_records = records[page_start : page_start + rows_per_page]
        matrix = np.full((len(page_records), len(parcel_labels)), np.nan, dtype=float)
        row_labels: list[str] = []
        for row, (key, metric_label, coverage) in enumerate(page_records):
            for column, parcel_label in enumerate(parcel_labels):
                matrix[row, column] = coverage.get(parcel_label, np.nan)
            values = np.asarray(list(coverage.values()), dtype=float)
            row_labels.append(
                f'{metric_label} | {key.session}  '
                f'(median {100 * np.median(values):.1f}%, '
                f'min {100 * np.min(values):.1f}%)'
            )

        figure = plt.figure(figsize=PAGE_SIZE, facecolor='white')
        add_page_header(
            figure,
            f'{subject} | {space_label} gray-matter parcel coverage',
            (
                f'{space_label} {ATLAS_DISPLAY} parcels. Coverage is the percentage of parcel '
                'voxels with finite, nonzero metric values; corpus callosum labels '
                'are excluded.'
            ),
        )
        axis = figure.add_axes([0.20, 0.20, 0.68, 0.65])
        color_map = plt.get_cmap('viridis').copy()
        color_map.set_bad('#D9D9D9')
        image = axis.imshow(
            np.ma.masked_invalid(matrix * 100),
            aspect='auto',
            interpolation='nearest',
            cmap=color_map,
            vmin=0,
            vmax=100,
        )
        axis.set_yticks(np.arange(len(row_labels)))
        axis.set_yticklabels(row_labels, fontsize=7)
        tick_step = max(1, len(parcel_labels) // 30)
        tick_positions = np.arange(0, len(parcel_labels), tick_step)
        axis.set_xticks(tick_positions)
        axis.set_xticklabels(
            [str(parcel_labels[index]) for index in tick_positions],
            rotation=60,
            ha='right',
            fontsize=6,
        )
        axis.set_xlabel(f'{ATLAS_DISPLAY} parcel label ID', fontsize=8)
        axis.tick_params(length=0)
        color_axis = figure.add_axes([0.90, 0.25, 0.012, 0.55])
        color_bar = figure.colorbar(image, cax=color_axis)
        color_bar.set_label('Coverage (%)', fontsize=8)
        color_bar.ax.tick_params(labelsize=7)
        page_number = save_page(pdf, figure, page_number)
    return page_number


def distribution_limits(
    distributions: Iterable[dict[str, np.ndarray]],
) -> tuple[float, float]:
    arrays = [
        values[np.isfinite(values)]
        for distribution in distributions
        for values in distribution.values()
        if np.any(np.isfinite(values))
    ]
    if not arrays:
        return 0.0, 1.0
    pooled = np.concatenate(arrays)
    low, high = np.percentile(pooled, [1, 99])
    if low == high:
        padding = max(abs(float(low)) * 0.05, 0.5)
        return float(low - padding), float(high + padding)
    padding = 0.04 * (high - low)
    return float(low - padding), float(high + padding)


def kde_values(
    values: np.ndarray,
    x_values: np.ndarray,
    max_samples: int = 50_000,
) -> np.ndarray | None:
    """Evaluate a robust KDE using finite values inside the displayed range."""
    values = values[np.isfinite(values) & (values >= x_values[0]) & (values <= x_values[-1])]
    if values.size < 2 or np.unique(values).size < 2:
        return None
    if values.size > max_samples:
        rng = np.random.default_rng(20260727)
        values = rng.choice(values, size=max_samples, replace=False)
    return np.asarray(gaussian_kde(values)(x_values), dtype=float)


def plot_tissue_distributions(
    figure: plt.Figure,
    spec,
    metric: MetricPattern,
    subject_inputs: Sequence[SessionInputs],
    distributions: dict[SessionKey, dict[str, np.ndarray]],
    errors: dict[SessionKey, str],
    space_label: str = 'T1w',
) -> None:
    """Plot session-coded GM/WM KDEs and their medians for one metric."""
    axis = figure.add_subplot(spec)
    x_limits = distribution_limits(distributions.values())
    x_values = np.linspace(x_limits[0], x_limits[1], 300)

    for session_index, inputs in enumerate(subject_inputs):
        key = inputs.key
        line_style = SESSION_LINESTYLES[session_index % len(SESSION_LINESTYLES)]
        for tissue in ('GM', 'WM'):
            values = distributions.get(key, {}).get(tissue, np.array([]))
            if not values.size:
                continue
            density = kde_values(values, x_values)
            color = TISSUE_COLORS[tissue]
            if density is not None:
                axis.plot(
                    x_values,
                    density,
                    color=color,
                    linestyle=line_style,
                    linewidth=1.7,
                )
            axis.axvline(
                float(np.median(values)),
                color=color,
                linestyle=line_style,
                linewidth=1.1,
                alpha=0.80,
            )

    axis.set_xlim(x_limits)
    axis.set_ylim(bottom=0)
    axis.set_title(f'{metric.label} | {space_label} tissue distributions', fontsize=9)
    axis.set_xlabel('Metric value', fontsize=8)
    axis.set_ylabel('Density', fontsize=8)
    axis.tick_params(labelsize=7)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)

    handles = [
        Line2D([0], [0], color=color, linewidth=2, label=tissue)
        for tissue, color in TISSUE_COLORS.items()
    ]
    handles.extend(
        Line2D(
            [0],
            [0],
            color='#333333',
            linestyle=SESSION_LINESTYLES[index % len(SESSION_LINESTYLES)],
            linewidth=1.7,
            label=inputs.key.session,
        )
        for index, inputs in enumerate(subject_inputs)
    )
    axis.legend(handles=handles, fontsize=7, frameon=False, loc='best')
    if errors:
        detail = '\n'.join(f'{key.session}: {message}' for key, message in errors.items())
        axis.text(
            0.02,
            0.02,
            textwrap.shorten(detail, width=100, placeholder='…'),
            transform=axis.transAxes,
            fontsize=6.5,
            color=MISSING_COLOR,
            va='bottom',
        )
    axis.text(
        0.98,
        0.02,
        'Vertical lines: medians',
        transform=axis.transAxes,
        fontsize=6.5,
        color='#555555',
        ha='right',
        va='bottom',
    )


def myelin_pages(
    pdf: PdfPages,
    derivatives: Path,
    subject_inputs: Sequence[SessionInputs],
    metrics: Sequence[MetricPattern],
    rows_per_page: int,
    display_percentiles: tuple[float, float],
    statuses: list[StatusEntry],
    page_number: int,
) -> int:
    if not subject_inputs:
        return page_number
    subject_inputs = sorted(subject_inputs, key=lambda item: item.key.session)
    subject = subject_inputs[0].key.subject
    metrics_per_page = max(1, rows_per_page // len(subject_inputs))

    for page_start in range(0, len(metrics), metrics_per_page):
        page_metrics = metrics[page_start : page_start + metrics_per_page]
        figure = plt.figure(figsize=PAGE_SIZE, facecolor='white')
        add_page_header(
            figure,
            f'{subject} | Primary T1w scalar maps',
            (
                'For each metric, sessions are consecutive rows. Left: original '
                'T1w-space map. Right: native T1w GM/WM distributions. Slice '
                'coordinates and color limits are shared across sessions; limits '
                f'use the {display_percentiles[0]:g}th–'
                f'{display_percentiles[1]:g}th percentiles of finite nonzero '
                'brain-mask voxels.'
            ),
        )
        row_count = len(page_metrics) * len(subject_inputs)
        grid = figure.add_gridspec(
            row_count,
            2,
            left=0.04,
            right=0.98,
            top=0.86,
            bottom=0.07,
            hspace=0.38,
            wspace=0.14,
            width_ratios=(1.5, 0.72),
        )

        for metric_index, metric in enumerate(page_metrics):
            prepared: dict[SessionKey, ScalarPanelData] = {}
            errors: dict[SessionKey, str] = {}
            scale_arrays: list[np.ndarray] = []
            scale_masks: list[np.ndarray] = []

            for inputs in subject_inputs:
                key = inputs.key
                source_path = find_metric(derivatives, key, metric)
                if source_path is None:
                    errors[key] = 'T1w-space metric not found'
                    continue
                prerequisites = (
                    inputs.t1w,
                    inputs.t1w_brain_mask,
                )
                if any(path is None for path in prerequisites):
                    errors[key] = f'Missing T1w or T1w brain mask; source={source_path}'
                    continue

                try:
                    t1w_reference = load_canonical(inputs.t1w)
                    source_image = resample_image(
                        load_canonical(source_path), t1w_reference, order=1
                    )
                    source_data = np.asarray(source_image.get_fdata(), dtype=np.float32)
                    source_brain_mask = np.asarray(
                        resample_image(
                            load_canonical(inputs.t1w_brain_mask),
                            t1w_reference,
                            order=0,
                        ).get_fdata()
                        > 0,
                        dtype=bool,
                    )
                    prepared[key] = ScalarPanelData(
                        source_path=source_path,
                        t1w_reference=t1w_reference,
                        source_data=source_data,
                        source_brain_mask=source_brain_mask,
                    )
                    scale_arrays.append(source_data)
                    scale_masks.append(source_brain_mask)
                except Exception as error:
                    LOGGER.exception(
                        'Failed primary T1w scalar QC for %s %s %s',
                        key.subject,
                        key.session,
                        metric.label,
                    )
                    errors[key] = str(error)

            limits = scalar_group_limits(scale_arrays, scale_masks, display_percentiles)
            centers = shared_metric_centers(prepared.values())
            distributions: dict[SessionKey, dict[str, np.ndarray]] = {}
            distribution_errors: dict[SessionKey, str] = {}
            for inputs in subject_inputs:
                key = inputs.key
                panel_data = prepared.get(key)
                if panel_data is None:
                    distribution_errors[key] = 'Scalar map unavailable'
                    status(
                        statuses,
                        key,
                        'distribution',
                        metric.label,
                        False,
                        distribution_errors[key],
                    )
                    continue
                if inputs.tissue_dseg is None:
                    distribution_errors[key] = 'sMRIPrep tissue dseg not found'
                    status(
                        statuses,
                        key,
                        'distribution',
                        metric.label,
                        False,
                        distribution_errors[key],
                    )
                    continue
                try:
                    values = tissue_metric_values(panel_data, inputs.tissue_dseg)
                    if not all(values[tissue].size for tissue in ('GM', 'WM')):
                        raise RuntimeError(
                            'sMRIPrep dseg contained no GM (label 1) or WM '
                            '(label 2) voxels on the scalar grid'
                        )
                    distributions[key] = values
                    status(
                        statuses,
                        key,
                        'distribution',
                        metric.label,
                        True,
                        (
                            f'{inputs.tissue_dseg}; '
                            f'GM n={values["GM"].size}, WM n={values["WM"].size}'
                        ),
                    )
                except Exception as error:
                    LOGGER.exception(
                        'Failed tissue distribution for %s %s %s',
                        key.subject,
                        key.session,
                        metric.label,
                    )
                    distribution_errors[key] = str(error)
                    status(
                        statuses,
                        key,
                        'distribution',
                        metric.label,
                        False,
                        str(error),
                    )

            for session_index, inputs in enumerate(subject_inputs):
                key = inputs.key
                row = metric_index * len(subject_inputs) + session_index
                panel_data = prepared.get(key)
                if panel_data is None:
                    detail = errors.get(key, 'Scalar map could not be prepared')
                    missing_panel(
                        figure,
                        grid[row, 0],
                        f'{metric.label} | {key.session} | T1w',
                        detail,
                    )
                    status(statuses, key, 'primary_t1w', metric.label, False, detail)
                    continue

                plot_orthogonal_montage(
                    figure,
                    grid[row, 0],
                    panel_data.t1w_reference,
                    np.where(
                        panel_data.source_brain_mask,
                        panel_data.source_data,
                        np.nan,
                    ),
                    overlay_kind='scalar',
                    title=f'{metric.label} | {key.session} | T1w',
                    limits=limits,
                    note=panel_data.source_path.name,
                    center_world=centers.get('T1w'),
                )
                status(
                    statuses,
                    key,
                    'primary_t1w',
                    metric.label,
                    True,
                    (
                        f'{panel_data.source_path}; '
                        f'display percentiles={display_percentiles}, limits={limits}'
                    ),
                )
            distribution_start = metric_index * len(subject_inputs)
            distribution_stop = distribution_start + len(subject_inputs)
            plot_tissue_distributions(
                figure,
                grid[distribution_start:distribution_stop, 1],
                metric,
                subject_inputs,
                distributions,
                distribution_errors,
            )
        page_number = save_page(pdf, figure, page_number)
    return page_number


def dwi_pages(
    pdf: PdfPages,
    derivatives: Path,
    subject_inputs: Sequence[SessionInputs],
    metrics: Sequence[MetricPattern],
    rows_per_page: int,
    display_percentiles: tuple[float, float],
    work_dir: Path,
    ants_command: str | None,
    overwrite_warps: bool,
    statuses: list[StatusEntry],
    page_number: int,
) -> int:
    """Plot native ACPC primary dMRI maps with GM/WM distributions."""
    if not subject_inputs or not metrics:
        return page_number
    subject_inputs = sorted(subject_inputs, key=lambda item: item.key.session)
    subject = subject_inputs[0].key.subject
    metrics_per_page = max(1, rows_per_page // len(subject_inputs))

    for page_start in range(0, len(metrics), metrics_per_page):
        page_metrics = metrics[page_start : page_start + metrics_per_page]
        figure = plt.figure(figsize=PAGE_SIZE, facecolor='white')
        add_page_header(
            figure,
            f'{subject} | Primary dMRI scalar maps',
            (
                'Sessions are consecutive rows. Left: native ACPC scalar over '
                'the ACPC T1w. Right: ACPC GM/WM distributions. Slice coordinates '
                'and color limits are shared across sessions; limits use the '
                f'{display_percentiles[0]:g}th–{display_percentiles[1]:g}th '
                'percentiles of finite nonzero ACPC brain-mask voxels.'
            ),
        )
        row_count = len(page_metrics) * len(subject_inputs)
        grid = figure.add_gridspec(
            row_count,
            2,
            left=0.04,
            right=0.98,
            top=0.86,
            bottom=0.07,
            hspace=0.38,
            wspace=0.14,
            width_ratios=(1.5, 0.72),
        )

        for metric_index, metric in enumerate(page_metrics):
            prepared: dict[SessionKey, DwiPanelData] = {}
            distributions: dict[SessionKey, dict[str, np.ndarray]] = {}
            errors: dict[SessionKey, str] = {}
            distribution_errors: dict[SessionKey, str] = {}
            scale_arrays: list[np.ndarray] = []
            scale_masks: list[np.ndarray] = []
            world_centers: list[np.ndarray] = []

            for inputs in subject_inputs:
                key = inputs.key
                source_path = find_metric(derivatives, key, metric)
                prerequisites = (
                    source_path,
                    inputs.acpc_t1w,
                    inputs.acpc_brain_mask,
                    inputs.tissue_dseg,
                    inputs.t1w_to_acpc_xfm,
                )
                if any(path is None for path in prerequisites):
                    errors[key] = (
                        'Missing dMRI map, ACPC T1w/brain mask, T1w tissue dseg, '
                        'or T1w-to-ACPC transform'
                    )
                    distribution_errors[key] = errors[key]
                    status(
                        statuses,
                        key,
                        'dMRI',
                        metric.label,
                        False,
                        errors[key],
                    )
                    continue

                tissue_acpc_path = (
                    work_dir
                    / key.subject
                    / key.session
                    / 'tissue_acpc'
                    / f'{key.subject}_{key.session}_space-ACPC_desc-tissue_dseg.nii.gz'
                )
                try:
                    warp_scalar_to_acpc(
                        inputs.tissue_dseg,
                        inputs.acpc_t1w,
                        inputs.t1w_to_acpc_xfm,
                        tissue_acpc_path,
                        ants_command=ants_command,
                        overwrite=overwrite_warps,
                        interpolation='GenericLabel',
                    )
                    reference = load_canonical(inputs.acpc_t1w)
                    data = np.asarray(
                        resample_image(load_canonical(source_path), reference, order=1).get_fdata(),
                        dtype=np.float32,
                    )
                    brain_mask = np.asarray(
                        resample_image(
                            load_canonical(inputs.acpc_brain_mask),
                            reference,
                            order=0,
                        ).get_fdata()
                        > 0,
                        dtype=bool,
                    )
                    tissue_labels = np.rint(
                        resample_image(
                            load_canonical(tissue_acpc_path),
                            reference,
                            order=0,
                        ).get_fdata()
                    ).astype(np.int16)
                    tissue_values = tissue_values_from_labels(data, tissue_labels)
                    if not all(tissue_values[tissue].size for tissue in ('GM', 'WM')):
                        raise RuntimeError(
                            'Warped tissue dseg contained no GM (label 1) or WM (label 2) voxels'
                        )
                    prepared[key] = DwiPanelData(
                        source_path=source_path,
                        acpc_reference=reference,
                        data=data,
                        brain_mask=brain_mask,
                    )
                    distributions[key] = tissue_values
                    scale_arrays.append(data)
                    scale_masks.append(brain_mask)
                    background = np.asarray(reference.get_fdata(), dtype=np.float32)
                    voxel_center = foreground_center(np.where(brain_mask, data, np.nan), background)
                    world_centers.append(
                        np.asarray(
                            apply_affine(reference.affine, voxel_center),
                            dtype=float,
                        )
                    )
                    status(
                        statuses,
                        key,
                        'dMRI',
                        metric.label,
                        True,
                        (f'{source_path}; display brain mask={inputs.acpc_brain_mask}'),
                    )
                    status(
                        statuses,
                        key,
                        'dMRI distribution',
                        metric.label,
                        True,
                        (
                            f'{tissue_acpc_path}; '
                            f'GM n={tissue_values["GM"].size}, '
                            f'WM n={tissue_values["WM"].size}'
                        ),
                    )
                except Exception as error:
                    LOGGER.exception(
                        'Failed dMRI QC for %s %s %s',
                        key.subject,
                        key.session,
                        metric.label,
                    )
                    errors[key] = str(error)
                    distribution_errors[key] = str(error)
                    status(
                        statuses,
                        key,
                        'dMRI',
                        metric.label,
                        False,
                        str(error),
                    )

            limits = scalar_group_limits(scale_arrays, scale_masks, display_percentiles)
            center_world = np.median(np.vstack(world_centers), axis=0) if world_centers else None
            for session_index, inputs in enumerate(subject_inputs):
                key = inputs.key
                row = metric_index * len(subject_inputs) + session_index
                panel_data = prepared.get(key)
                if panel_data is None:
                    missing_panel(
                        figure,
                        grid[row, 0],
                        f'{metric.label} | {key.session} | ACPC',
                        errors.get(key, 'dMRI scalar map could not be prepared'),
                    )
                    continue
                plot_orthogonal_montage(
                    figure,
                    grid[row, 0],
                    panel_data.acpc_reference,
                    np.where(panel_data.brain_mask, panel_data.data, np.nan),
                    overlay_kind='scalar',
                    title=f'{metric.label} | {key.session} | ACPC',
                    limits=limits,
                    note=panel_data.source_path.name,
                    center_world=center_world,
                )

            distribution_start = metric_index * len(subject_inputs)
            distribution_stop = distribution_start + len(subject_inputs)
            plot_tissue_distributions(
                figure,
                grid[distribution_start:distribution_stop, 1],
                metric,
                subject_inputs,
                distributions,
                distribution_errors,
                space_label='ACPC',
            )
        page_number = save_page(pdf, figure, page_number)
    return page_number


def cover_page(
    pdf: PdfPages,
    output: Path,
    derivatives: Path,
    session_keys: Sequence[SessionKey],
    bundle_names: Sequence[str],
    metrics: Sequence[MetricPattern],
    dwi_metrics: Sequence[MetricPattern],
    page_number: int,
) -> int:
    figure = plt.figure(figsize=PAGE_SIZE, facecolor='white')
    axis = figure.add_axes([0, 0, 1, 1])
    axis.set_axis_off()
    axis.text(
        0.07,
        0.80,
        'MIRROR Spatial Quality Report',
        fontsize=30,
        fontweight='bold',
        va='top',
    )
    axis.text(
        0.07,
        0.72,
        'ACPC/T1w bundle, parcellation, and primary scalar QC',
        fontsize=15,
        color='#444444',
        va='top',
    )
    details = [
        ('Generated', datetime.now().astimezone().isoformat(timespec='seconds')),
        ('Derivatives', str(derivatives)),
        ('Output', str(output)),
        ('Subject/session pairs', str(len(session_keys))),
        ('Bundles', ', '.join(bundle_names)),
        ('T1w primary maps', ', '.join(metric.label for metric in metrics)),
        ('ACPC dMRI primary maps', ', '.join(metric.label for metric in dwi_metrics)),
    ]
    y = 0.59
    for label, value in details:
        axis.text(0.08, y, label, fontsize=10, fontweight='bold', va='top')
        axis.text(
            0.25,
            y,
            textwrap.fill(value, 110),
            fontsize=10,
            color='#333333',
            va='top',
        )
        y -= 0.075 if len(value) < 100 else 0.10
    axis.text(
        0.07,
        0.12,
        'This report is a visual QC aid. It does not assign automated pass/fail labels.',
        fontsize=10,
        color='#666666',
        style='italic',
    )
    return save_page(pdf, figure, page_number)


def summary_pages(
    pdf: PdfPages,
    entries: Sequence[StatusEntry],
    page_number: int,
) -> int:
    failures = [entry for entry in entries if entry.status != 'OK']
    rows_per_page = 28
    pages = max(1, (len(failures) + rows_per_page - 1) // rows_per_page)
    for page_index in range(pages):
        subset = failures[page_index * rows_per_page : (page_index + 1) * rows_per_page]
        figure = plt.figure(figsize=PAGE_SIZE, facecolor='white')
        add_page_header(
            figure,
            'Missing or failed inputs',
            (
                f'{len(failures)} issue(s) across {len(entries)} checks'
                if failures
                else f'No missing inputs across {len(entries)} checks'
            ),
        )
        axis = figure.add_axes([0.04, 0.07, 0.92, 0.80])
        axis.set_axis_off()
        if not subset:
            axis.text(
                0.5,
                0.55,
                'No missing or failed inputs',
                ha='center',
                va='center',
                fontsize=20,
                color='#2A9D8F',
                transform=axis.transAxes,
            )
        else:
            column_x = (0.00, 0.10, 0.20, 0.34, 0.52)
            headers = ('Subject', 'Session', 'Section', 'Item', 'Detail')
            for x, header in zip(column_x, headers):
                axis.text(
                    x,
                    1.0,
                    header,
                    fontsize=8,
                    fontweight='bold',
                    va='top',
                    transform=axis.transAxes,
                )
            for row, entry in enumerate(subset, start=1):
                y = 1.0 - row * 0.034
                values = (
                    entry.subject,
                    entry.session,
                    entry.section,
                    entry.item,
                    textwrap.shorten(entry.detail, width=92, placeholder='…'),
                )
                for x, value in zip(column_x, values):
                    axis.text(
                        x,
                        y,
                        value,
                        fontsize=6.5,
                        color=MISSING_COLOR if x < 0.52 else '#333333',
                        va='top',
                        transform=axis.transAxes,
                    )
        page_number = save_page(pdf, figure, page_number)
    return page_number


def parse_metrics(
    requested: Sequence[str] | None,
    patterns_file: Path = DEFAULT_PATTERNS_FILE,
) -> list[MetricPattern]:
    metric_by_label = {
        metric.label: metric for metric in primary_metric_patterns(patterns_file, None, 'T1w')
    }
    if not requested:
        return list(metric_by_label.values())
    unknown = sorted(set(requested).difference(metric_by_label))
    if unknown:
        raise ValueError(
            f'Unknown T1w primary metric(s): {unknown}. Choices: {sorted(metric_by_label)}'
        )
    return [metric_by_label[label] for label in requested]


def parse_dwi_metrics(
    requested: Sequence[str] | None,
    patterns_file: Path = DEFAULT_PATTERNS_FILE,
) -> list[MetricPattern]:
    metric_by_label = {
        metric.label: metric
        for metric in primary_metric_patterns(
            patterns_file,
            'dMRI',
            'ACPC',
            include_gm_noddi_icvf=True,
        )
    }
    if not requested:
        return list(metric_by_label.values())
    unknown = sorted(set(requested).difference(metric_by_label))
    if unknown:
        raise ValueError(
            f'Unknown dMRI primary metric(s): {unknown}. Choices: {sorted(metric_by_label)}'
        )
    return [metric_by_label[label] for label in requested]


def build_session_keys(
    derivatives: Path,
    requested_subjects: Sequence[str] | None,
    requested_sessions: Sequence[str] | None,
) -> list[SessionKey]:
    subjects = (
        sorted({normalize_subject(value) for value in requested_subjects})
        if requested_subjects
        else discover_subjects(derivatives)
    )
    session_filter = (
        {normalize_session(value) for value in requested_sessions} if requested_sessions else None
    )
    keys: list[SessionKey] = []
    for subject in subjects:
        sessions = discover_sessions(derivatives, subject)
        if session_filter is not None:
            sessions = sorted(session_filter)
        keys.extend(SessionKey(subject, session) for session in sessions)
    return keys


def generate_report(args: argparse.Namespace) -> tuple[Path, list[StatusEntry]]:
    if args.derivatives_root:
        source_derivatives = Path(args.derivatives_root).expanduser().resolve()
        run_derivatives = source_derivatives
    else:
        source_derivatives = Path(args.source_derivatives_root).expanduser().resolve()
        run_derivatives = Path(args.run_derivatives_root).expanduser().resolve()
    derivatives = DerivativeOverlay(source_derivatives, run_derivatives)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else run_derivatives / 'qc_report' / 'qc_report.pdf'
    )
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else run_derivatives / 'qc_report' / 'work'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    patterns_file = Path(args.patterns_file).expanduser().resolve()
    metrics = parse_metrics(args.myelin_metric, patterns_file)
    dwi_metrics = parse_dwi_metrics(args.dwi_metric, patterns_file)
    bundle_names = args.bundle or [
        'Association_ArcuateFasciculusL',
        'Association_ArcuateFasciculusR',
    ]
    session_keys = build_session_keys(derivatives, args.subject_id, args.session_id)
    if not session_keys:
        raise RuntimeError(
            f'No subject/session pairs discovered under {derivatives}. '
            'Use --subject-id and --session-id to specify them explicitly.'
        )
    ants_command = resolve_ants_command(args.ants_apply_transforms)
    if ants_command is None:
        LOGGER.warning(
            'antsApplyTransforms was not found on PATH; antspyx will be used if available.'
        )

    statuses: list[StatusEntry] = []
    page_number = 1
    LOGGER.info('Writing %s', output)
    with PdfPages(output) as pdf:
        metadata = pdf.infodict()
        metadata['Title'] = 'MIRROR Spatial Quality Report'
        metadata['Author'] = 'MIRROR generate_qc_report.py'
        metadata['Subject'] = 'ACPC and T1w spatial registration QC'
        metadata['CreationDate'] = datetime.now()
        page_number = cover_page(
            pdf,
            output,
            derivatives,
            session_keys,
            bundle_names,
            metrics,
            dwi_metrics,
            page_number,
        )
        subjects = sorted({key.subject for key in session_keys})
        for subject_index, subject in enumerate(subjects, start=1):
            subject_keys = [key for key in session_keys if key.subject == subject]
            subject_inputs = [collect_session_inputs(derivatives, key) for key in subject_keys]
            LOGGER.info(
                '[%d/%d] Processing %s (%d session(s))',
                subject_index,
                len(subjects),
                subject,
                len(subject_inputs),
            )
            for inputs in subject_inputs:
                page_number = bundle_page(
                    pdf,
                    inputs,
                    bundle_names,
                    max_streamlines=args.max_streamlines,
                    work_dir=work_dir,
                    statuses=statuses,
                    page_number=page_number,
                )
                page_number = parcellation_page(
                    pdf, inputs, statuses=statuses, page_number=page_number
                )
            page_number = myelin_pages(
                pdf,
                derivatives,
                subject_inputs,
                metrics,
                rows_per_page=args.metrics_per_page,
                display_percentiles=tuple(args.display_percentiles),
                statuses=statuses,
                page_number=page_number,
            )
            page_number = parcel_coverage_pages(
                pdf,
                derivatives,
                subject_inputs,
                metrics,
                statuses=statuses,
                page_number=page_number,
                space_label='T1w',
            )
            page_number = dwi_pages(
                pdf,
                derivatives,
                subject_inputs,
                dwi_metrics,
                rows_per_page=args.metrics_per_page,
                display_percentiles=tuple(args.display_percentiles),
                work_dir=work_dir,
                ants_command=ants_command,
                overwrite_warps=args.overwrite_warps,
                statuses=statuses,
                page_number=page_number,
            )
            page_number = parcel_coverage_pages(
                pdf,
                derivatives,
                subject_inputs,
                dwi_metrics,
                statuses=statuses,
                page_number=page_number,
                space_label='ACPC',
            )
        summary_pages(pdf, statuses, page_number)

    failures = [entry for entry in statuses if entry.status != 'OK']
    LOGGER.info(
        'Wrote %s (%d checks, %d missing/failed)',
        output,
        len(statuses),
        len(failures),
    )
    if args.strict and failures:
        raise RuntimeError(
            f'Report completed with {len(failures)} missing or failed checks: {output}'
        )
    return output, statuses


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--project-root',
        default=PROJECT_ROOT,
        help='MIRROR project root containing code/, derivatives/, and apptainer/.',
    )
    parser.add_argument(
        '--derivatives-root',
        help='Deprecated single-root compatibility mode.',
    )
    parser.add_argument(
        '--source-derivatives-root',
        default=SOURCE_DERIVATIVES_ROOT,
        help='Read-only reusable preprocessing and source-scalar derivatives.',
    )
    parser.add_argument(
        '--run-derivatives-root',
        default=OUTPUT_DERIVATIVES_ROOT,
        help='Run-specific registration, regional-statistic, and QC products.',
    )
    parser.add_argument(
        '--patterns-file',
        default=DEFAULT_PATTERNS_FILE,
        type=Path,
        help='Scalar path pattern registry used to identify primary metrics.',
    )
    parser.add_argument(
        '--subject-id',
        action='append',
        help='Subject to include, with or without sub-. Repeat to include several.',
    )
    parser.add_argument(
        '--session-id',
        action='append',
        help='Session to include, with or without ses-. Repeat to include several.',
    )
    parser.add_argument(
        '--output',
        help='Output PDF. Defaults to <run-derivatives>/qc_report/qc_report.pdf.',
    )
    parser.add_argument(
        '--work-dir',
        help='Cache directory for temporary tractograms and warped tissue label maps.',
    )
    parser.add_argument(
        '--bundle',
        action='append',
        help=(
            'Bundle name to render. Repeat for several. Defaults to left and right '
            'Association_ArcuateFasciculus.'
        ),
    )
    parser.add_argument(
        '--myelin-metric',
        action='append',
        choices=sorted(METRIC_BY_LABEL),
        help=(
            'T1w-space primary metric to render. Repeat; default is every '
            'non-dMRI primary metric. Kept as --myelin-metric for compatibility.'
        ),
    )
    parser.add_argument(
        '--dwi-metric',
        action='append',
        choices=sorted(DWI_METRIC_BY_LABEL),
        help=(
            'ACPC-space dMRI primary metric to render. Repeat; default is every '
            'primary dMRI metric. Kept as --dwi-metric for compatibility.'
        ),
    )
    parser.add_argument(
        '--metrics-per-page',
        type=int,
        default=4,
        help=(
            'Maximum metric-session rows on each scalar-map page. With two '
            'sessions, the default displays two metrics per page.'
        ),
    )
    parser.add_argument(
        '--display-percentiles',
        type=float,
        nargs=2,
        metavar=('LOW', 'HIGH'),
        default=(5.0, 95.0),
        help=(
            'Robust scalar-map display percentiles, calculated jointly across '
            'all available sessions for each metric, using only voxels inside '
            'the corresponding brain mask.'
        ),
    )
    parser.add_argument(
        '--max-streamlines',
        type=int,
        default=4000,
        help='Maximum streamlines rasterized per bundle and space.',
    )
    parser.add_argument(
        '--ants-apply-transforms',
        help='Path or command name for antsApplyTransforms.',
    )
    parser.add_argument(
        '--overwrite-warps',
        action='store_true',
        help='Regenerate cached ACPC tissue label maps used for dMRI distributions.',
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit nonzero after writing the report if any input/check failed.',
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
    if args.metrics_per_page < 1:
        parser.error('--metrics-per-page must be at least 1')
    if args.max_streamlines < 1:
        parser.error('--max-streamlines must be at least 1')
    low_percentile, high_percentile = args.display_percentiles
    if not 0 <= low_percentile < high_percentile <= 100:
        parser.error('--display-percentiles must satisfy 0 <= LOW < HIGH <= 100')
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    generate_report(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
