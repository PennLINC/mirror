"""Compute DKT parcel-wise summary statistics and coverage for scalar maps.

Runs per subject, writing one statistics CSV and one long-format coverage CSV
per subject/session/run for aparc.DKTatlas parcels. Scalar maps remain in their
native grids; the corresponding label image is resampled to each map with
generic-label interpolation.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

import ants
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.metrics import build_metric_specs, flatten_metric_patterns, metric_specs_for_analysis  # noqa: E402
from utils.paths import OUTPUT_DERIVATIVES_ROOT, SOURCE_DERIVATIVES_ROOT  # noqa: E402

STATS = ('mean', 'median', 'std', 'min', 'max')
KEY_RE = re.compile(r'(ses-[A-Za-z0-9]+)|(run-[A-Za-z0-9]+)')
ATLAS_DESC = 'DKTatlas'
DKT_LABELS: tuple[tuple[int, str, str], ...] = (
    (1002, 'caudal anterior cingulate', 'lh'),
    (1003, 'caudal middle frontal', 'lh'),
    (1005, 'cuneus', 'lh'),
    (1006, 'entorhinal', 'lh'),
    (1007, 'fusiform', 'lh'),
    (1008, 'inferior parietal', 'lh'),
    (1009, 'inferior temporal', 'lh'),
    (1010, 'isthmus cingulate', 'lh'),
    (1011, 'lateral occipital', 'lh'),
    (1012, 'lateral orbitofrontal', 'lh'),
    (1013, 'lingual', 'lh'),
    (1014, 'medial orbitofrontal', 'lh'),
    (1015, 'middle temporal', 'lh'),
    (1016, 'parahippocampal', 'lh'),
    (1017, 'paracentral', 'lh'),
    (1018, 'pars opercularis', 'lh'),
    (1019, 'pars orbitalis', 'lh'),
    (1020, 'pars triangularis', 'lh'),
    (1021, 'pericalcarine', 'lh'),
    (1022, 'postcentral', 'lh'),
    (1023, 'posterior cingulate', 'lh'),
    (1024, 'precentral', 'lh'),
    (1025, 'precuneus', 'lh'),
    (1026, 'rostral anterior cingulate', 'lh'),
    (1027, 'rostral middle frontal', 'lh'),
    (1028, 'superior frontal', 'lh'),
    (1029, 'superior parietal', 'lh'),
    (1030, 'superior temporal', 'lh'),
    (1031, 'supramarginal', 'lh'),
    (1034, 'transverse temporal', 'lh'),
    (1035, 'insula', 'lh'),
    (2002, 'caudal anterior cingulate', 'rh'),
    (2003, 'caudal middle frontal', 'rh'),
    (2005, 'cuneus', 'rh'),
    (2006, 'entorhinal', 'rh'),
    (2007, 'fusiform', 'rh'),
    (2008, 'inferior parietal', 'rh'),
    (2009, 'inferior temporal', 'rh'),
    (2010, 'isthmus cingulate', 'rh'),
    (2011, 'lateral occipital', 'rh'),
    (2012, 'lateral orbitofrontal', 'rh'),
    (2013, 'lingual', 'rh'),
    (2014, 'medial orbitofrontal', 'rh'),
    (2015, 'middle temporal', 'rh'),
    (2016, 'parahippocampal', 'rh'),
    (2017, 'paracentral', 'rh'),
    (2018, 'pars opercularis', 'rh'),
    (2019, 'pars orbitalis', 'rh'),
    (2020, 'pars triangularis', 'rh'),
    (2021, 'pericalcarine', 'rh'),
    (2022, 'postcentral', 'rh'),
    (2023, 'posterior cingulate', 'rh'),
    (2024, 'precentral', 'rh'),
    (2025, 'precuneus', 'rh'),
    (2026, 'rostral anterior cingulate', 'rh'),
    (2027, 'rostral middle frontal', 'rh'),
    (2028, 'superior frontal', 'rh'),
    (2029, 'superior parietal', 'rh'),
    (2030, 'superior temporal', 'rh'),
    (2031, 'supramarginal', 'rh'),
    (2034, 'transverse temporal', 'rh'),
    (2035, 'insula', 'rh'),
)


def _dkt_parcel_table() -> pd.DataFrame:
    rows = [
        {
            'parcel_intensity': intensity,
            'parcel_name': name,
            'parcel_hemi': hemi,
        }
        for intensity, name, hemi in DKT_LABELS
    ]
    return pd.DataFrame(rows).sort_values('parcel_intensity').reset_index(drop=True)


def _parse_ses_run(path: str) -> tuple[str, str]:
    matches = [m.group(0) for m in KEY_RE.finditer(os.path.basename(path))]
    ses = 'ses-unknown'
    run = 'run-01'
    for token in matches:
        if token.startswith('ses-'):
            ses = token
        elif token.startswith('run-'):
            run = token
    return ses, run


def _pattern_lookup(patterns_file: Path) -> dict[str, str]:
    return flatten_metric_patterns(patterns_file)


def _space_for_spec_group(group: str) -> str:
    return 'ACPC' if group == 'dMRI' else 'T1w'


def _build_metric_files(
    subject: str,
    deriv_dir: str,
    patterns_file: Path,
) -> tuple[dict[tuple[str, str], dict[str, str]], list[dict[str, object]]]:
    metric_files_by_key: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    inventory_rows: list[dict[str, object]] = []
    subject_tok = f'sub-{subject}'
    patterns = _pattern_lookup(patterns_file)
    for spec in metric_specs_for_analysis(
        build_metric_specs(patterns_file),
        'full',
        tissue='gm',
    ):
        rel_pattern = patterns[spec.pattern_key]
        subj_pattern = rel_pattern.format(
            subject=subject_tok,
            session='ses-*',
            space=_space_for_spec_group(spec.group),
        )
        matches = sorted(glob(os.path.join(deriv_dir, subj_pattern)))
        inventory_rows.append(
            {
                'subject': subject_tok,
                'metric_key': spec.label,
                'primary_label': spec.primary_label,
                'pattern_key': spec.pattern_key,
                'source_image': spec.source_image,
                'space': _space_for_spec_group(spec.group),
                'glob': os.path.join(deriv_dir, subj_pattern),
                'n_matches': len(matches),
                'selected_file': matches[0] if matches else '',
            }
        )
        if not matches:
            continue
        for map_file in matches:
            ses, run = _parse_ses_run(map_file)
            key = (ses, run)
            if spec.label in metric_files_by_key[key]:
                # Keep the first deterministic match for duplicate paths.
                continue
            metric_files_by_key[key][spec.label] = map_file
    return metric_files_by_key, inventory_rows


def _space_from_path(path: str) -> str:
    fname = os.path.basename(path)
    has_acpc = '_space-ACPC_' in fname
    has_t1w = '_space-T1w_' in fname
    if has_acpc and not has_t1w:
        return 'ACPC'
    if has_t1w and not has_acpc:
        return 'T1w'
    raise ValueError(
        'Could not unambiguously determine space from filename '
        f'(expected _space-ACPC_ or _space-T1w_): {fname}'
    )


def _images_share_grid(image_a: ants.ANTsImage, image_b: ants.ANTsImage) -> bool:
    """Return True when two ANTs images use the same voxel grid."""
    return (
        image_a.shape == image_b.shape
        and np.allclose(image_a.spacing, image_b.spacing)
        and np.allclose(image_a.origin, image_b.origin)
        and np.allclose(image_a.direction, image_b.direction)
    )


def _compute_stats(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
        }
    return {
        'mean': float(np.mean(values)),
        'median': float(np.median(values)),
        'std': float(np.std(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values)),
    }


def process_subject(
    subject: str,
    source_deriv_dir: str,
    run_deriv_dir: str,
    patterns_file: Path,
    out_root: str | None = None,
    zero_is_missing: bool = True,
) -> None:
    t1w_reg_dir = os.path.join(run_deriv_dir, 't1w_registration', f'sub-{subject}', 'anat')
    out_base = out_root or os.path.join(run_deriv_dir, f'{ATLAS_DESC}_myelin_stats')
    out_dir = os.path.join(out_base, f'sub-{subject}')
    os.makedirs(out_dir, exist_ok=True)

    dseg_t1w = os.path.join(t1w_reg_dir, f'sub-{subject}_space-T1w_desc-{ATLAS_DESC}_dseg.nii.gz')
    dseg_acpc = os.path.join(t1w_reg_dir, f'sub-{subject}_space-ACPC_desc-{ATLAS_DESC}_dseg.nii.gz')

    required_files = [dseg_t1w, dseg_acpc]
    for required in required_files:
        if not os.path.exists(required):
            raise FileNotFoundError(required)
    dseg_imgs = {
        'T1w': ants.image_read(dseg_t1w),
        'ACPC': ants.image_read(dseg_acpc),
    }
    dseg_arrays = {space: img.numpy().astype(np.int64) for space, img in dseg_imgs.items()}
    parcel_df = _dkt_parcel_table()
    label_ids = parcel_df['parcel_intensity'].astype(int).to_numpy()
    available_labels = set(np.unique(dseg_arrays['T1w']).astype(int)) | set(
        np.unique(dseg_arrays['ACPC']).astype(int)
    )
    missing_label_ids = [label_id for label_id in label_ids if label_id not in available_labels]
    if missing_label_ids:
        print(
            f'{len(missing_label_ids)} LUT labels absent from subject dseg volumes.',
            flush=True,
        )

    t1w_counts = np.array(
        [int(np.count_nonzero(dseg_arrays['T1w'] == label_id)) for label_id in label_ids],
        dtype=np.int64,
    )
    acpc_counts = np.array(
        [int(np.count_nonzero(dseg_arrays['ACPC'] == label_id)) for label_id in label_ids],
        dtype=np.int64,
    )

    metric_files_by_key, inventory_rows = _build_metric_files(
        subject,
        source_deriv_dir,
        patterns_file,
    )
    inventory_file = os.path.join(out_dir, f'sub-{subject}_desc-{ATLAS_DESC}_metric_inventory.tsv')
    pd.DataFrame(inventory_rows).to_csv(inventory_file, sep='\t', index=False)
    print(f'Wrote {inventory_file}', flush=True)
    if not metric_files_by_key:
        print(f'No scalar maps found for sub-{subject}', flush=True)
        return

    for (ses, run), metric_files in sorted(metric_files_by_key.items()):
        out_df = parcel_df.copy()
        out_df.insert(3, 'parcel_count_t1w', t1w_counts)
        out_df.insert(4, 'parcel_count_acpc', acpc_counts)

        gm_metric_labels = sorted(
            {
                spec.label
                for spec in metric_specs_for_analysis(
                    build_metric_specs(patterns_file),
                    'full',
                    tissue='gm',
                )
            }
        )
        for metric_name in gm_metric_labels:
            for stat in STATS:
                out_df[f'{metric_name}_{stat}'] = np.nan

        # Long-format coverage output: one row per metric and parcel.
        coverage_rows: list[dict[str, object]] = []

        for metric_name, metric_file in metric_files.items():
            actual_space = _space_from_path(metric_file)
            space = actual_space
            dseg_img = dseg_imgs[space]

            # Keep the quantitative scalar map in its native grid. Move only
            # the categorical parcellation to that grid with label-aware
            # interpolation. This avoids smoothing, ringing, and mixing of
            # scalar values across parcel or missing-data boundaries.
            map_img = ants.image_read(metric_file)
            map_data = map_img.numpy()

            if _images_share_grid(dseg_img, map_img):
                metric_dseg_img = dseg_img
            else:
                metric_dseg_img = ants.resample_image_to_target(
                    image=dseg_img,
                    target=map_img,
                    interp_type='genericLabel',
                )

            # genericLabel should preserve integer labels. Rounding before
            # conversion guards against floating-point storage artifacts.
            metric_dseg_data = np.rint(metric_dseg_img.numpy()).astype(np.int64)

            valid_data = np.isfinite(map_data)
            if zero_is_missing:
                valid_data &= map_data != 0

            for label_id in label_ids:
                row_idx = out_df['parcel_intensity'] == label_id
                parcel_row = out_df.loc[row_idx].iloc[0]

                parcel_mask = metric_dseg_data == label_id
                parcel_values = map_data[parcel_mask]
                parcel_valid = valid_data[parcel_mask]
                valid_values = parcel_values[parcel_valid]
                n_total = int(parcel_values.size)
                n_valid = int(np.count_nonzero(parcel_valid))
                coverage = n_valid / n_total if n_total > 0 else np.nan

                stats = _compute_stats(valid_values)
                for stat_name, stat_val in stats.items():
                    out_df.loc[row_idx, f'{metric_name}_{stat_name}'] = stat_val

                coverage_rows.append(
                    {
                        'subject': f'sub-{subject}',
                        'session': ses,
                        'run': run,
                        'metric': metric_name,
                        'space': space,
                        'parcel_intensity': int(label_id),
                        'parcel_name': str(parcel_row['parcel_name']),
                        'parcel_hemi': str(parcel_row['parcel_hemi']),
                        'parcel_count': n_total,
                        'valid_count': n_valid,
                        'coverage': coverage,
                    }
                )

        out_file = os.path.join(
            out_dir,
            f'sub-{subject}_{ses}_{run}_desc-{ATLAS_DESC}_scalarstats.csv',
        )
        out_df.to_csv(out_file, index=False)
        print(f'Wrote {out_file}', flush=True)

        coverage_file = os.path.join(
            out_dir,
            f'sub-{subject}_{ses}_{run}_desc-{ATLAS_DESC}_coverage.csv',
        )
        coverage_df = pd.DataFrame(coverage_rows).sort_values(['metric', 'parcel_intensity'])
        coverage_df.to_csv(coverage_file, index=False)
        print(f'Wrote {coverage_file}', flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--subject-id',
        required=True,
        help='Subject ID without the sub- prefix',
    )
    parser.add_argument(
        '--source-derivatives-dir',
        '--derivatives-dir',
        dest='source_derivatives_dir',
        default=str(SOURCE_DERIVATIVES_ROOT),
        help='Read-only root containing source scalar derivatives.',
    )
    parser.add_argument(
        '--run-derivatives-dir',
        default=str(OUTPUT_DERIVATIVES_ROOT),
        help='Run-specific root containing registration inputs and new outputs.',
    )
    parser.add_argument(
        '--out-root',
        default=None,
        help='Output root. Defaults to <derivatives-dir>/DKTatlas_myelin_stats.',
    )
    parser.add_argument(
        '--include-zero',
        action='store_true',
        help=(
            'Include exact zero values in parcel statistics and coverage. '
            'By default, zero and all nonfinite values are treated as invalid.'
        ),
    )
    parser.add_argument(
        '--patterns-file',
        type=Path,
        default=REPO_ROOT / 'configuration' / 'metrics.yml',
        help='Metric pattern registry.',
    )
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    process_subject(
        args.subject_id,
        args.source_derivatives_dir,
        args.run_derivatives_dir,
        args.patterns_file,
        out_root=args.out_root,
        zero_is_missing=not args.include_zero,
    )
