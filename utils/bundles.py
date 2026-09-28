#!/usr/bin/env python3
"""
Bundle scalar mapping with host or QSIRecon-container execution.

Given:
- a DWI reference image
- one or more .tck bundle files
- one or more scalar NIfTI images

This script:
1. Creates a TDI for each bundle using MRtrix `tckmap`
2. Uses voxels with TDI > 0 as the bundle mask
3. Summarizes each scalar within each bundle
4. Optionally computes TDI-weighted means

Outputs:
- bundle_stats.tsv
- tdi_stats.tsv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.maskers import NiftiMasker

LOGGER = logging.getLogger('bundle_scalar_mapper')


def _resolve_executable(command: str) -> str | None:
    """Resolve a command name or executable path without invoking a shell."""
    expanded = str(Path(command).expanduser())
    if os.path.sep in expanded:
        return expanded if os.path.isfile(expanded) and os.access(expanded, os.X_OK) else None
    return shutil.which(expanded)


def resolve_tckmap_command() -> list[str]:
    """Build the command prefix used to run MRtrix ``tckmap``.

    ``TCKMAP_BIN`` is an explicit host override. Otherwise, a ``tckmap`` found
    on ``PATH`` is preferred, followed by the configured QSIRecon image.
    """
    env_bin = os.environ.get('TCKMAP_BIN')
    if env_bin:
        resolved = _resolve_executable(env_bin)
        if resolved is None:
            raise FileNotFoundError(
                f'TCKMAP_BIN is set to {env_bin!r}, but that executable was not found. '
                'Unset TCKMAP_BIN to use the configured QSIRecon container.'
            )
        return [resolved]

    path_bin = _resolve_executable('tckmap')
    if path_bin is not None:
        return [path_bin]

    container = os.environ.get('MIRROR_APPTAINER_QSIRECON')
    if container:
        container_path = Path(container).expanduser()
        if not container_path.is_file():
            raise FileNotFoundError(
                f'Configured QSIRecon image does not exist: {container_path}. '
                'Update apptainer.qsirecon in the selected MIRROR profile.'
            )

        requested_runtime = os.environ.get('APPTAINER_BIN')
        runtime_candidates = (
            [requested_runtime] if requested_runtime else ['apptainer', 'singularity']
        )
        runtime = next(
            (
                resolved
                for candidate in runtime_candidates
                if candidate and (resolved := _resolve_executable(candidate)) is not None
            ),
            None,
        )
        if runtime is None:
            requested = requested_runtime or 'apptainer (or singularity)'
            raise FileNotFoundError(
                f'Container runtime not found: {requested}. Load Apptainer on the compute nodes '
                'or set APPTAINER_BIN to its executable.'
            )

        command = [runtime, 'exec', '--cleanenv']
        project_root = os.environ.get('MIRROR_PROJECT_ROOT')
        if project_root:
            command.extend(['--bind', f'{project_root}:{project_root}'])
        command.extend([str(container_path), 'tckmap'])
        return command

    raise FileNotFoundError(
        'Could not run tckmap. Load a MIRROR profile with apptainer.qsirecon configured, '
        'or set TCKMAP_BIN to a working host executable.'
    )


def run_tckmap(
    dwiref_image: str,
    tck_file: str,
    output_tdi_file: str,
    tckmap_command: Sequence[str] | None = None,
) -> nib.Nifti1Image:
    """Create a track-density image (TDI) from a .tck file using MRtrix."""
    command_prefix = list(tckmap_command or resolve_tckmap_command())
    cmd = [
        *command_prefix,
        '-template',
        dwiref_image,
        '-contrast',
        'tdi',
        '-force',
        tck_file,
        output_tdi_file,
    ]
    LOGGER.info('Running: %s', shlex.join(cmd))
    subprocess.run(cmd, check=True)
    return nib.load(output_tdi_file)


def safe_nan_stats(values: np.ndarray) -> dict[str, float]:
    """Compute NaN-safe summary statistics."""
    out: dict[str, float] = {}
    if values.size == 0:
        out['mean'] = np.nan
        out['stdev'] = np.nan
        out['median'] = np.nan
        return out

    out['mean'] = float(np.nanmean(values))
    out['stdev'] = float(np.nanstd(values))
    out['median'] = float(np.nanmedian(values))
    return out


def calculate_mask_stats(
    masker: NiftiMasker,
    mask_name: str,
    mask_variable_name: str,
    scalar_img: nib.Nifti1Image,
    variable_name: str,
    source_file: str,
    qsirecon_suffix: str = 'external',
    weighting_vector: np.ndarray | None = None,
) -> dict[str, Any]:
    """
    Compute bundle summary statistics for a scalar image.

    Behavior matches the QSIRecon function closely:
    - 'zero_proportion' treats 0 and non-finite as missing
    - raw mean/stdev/median use all extracted voxels
    - masked_* stats exclude zeros and non-finite voxels
    - weighted stats use the supplied weighting_vector
    """
    voxel_data = masker.fit_transform(scalar_img).squeeze()

    if voxel_data.ndim == 0:
        voxel_data = np.array([voxel_data], dtype=float)

    voxel_data = np.asarray(voxel_data, dtype=float)

    nz_voxel_data = voxel_data.copy()
    nz_voxel_data[nz_voxel_data == 0] = np.nan
    nz_voxel_data[~np.isfinite(voxel_data)] = np.nan

    results: dict[str, Any] = {
        mask_variable_name: mask_name,
        'variable_name': variable_name.replace('_image', '').replace('_file', ''),
        'qsirecon_suffix': qsirecon_suffix,
        'source_file': source_file,
        'zero_proportion': float(np.sum(np.isnan(nz_voxel_data)) / voxel_data.shape[0]),
        'mean': float(np.mean(voxel_data)),
        'stdev': float(np.std(voxel_data)),
        'median': float(np.median(voxel_data)),
        'masked_mean': float(np.nanmean(nz_voxel_data)),
        'masked_median': float(np.nanmedian(nz_voxel_data)),
        'masked_stdev': float(np.nanstd(nz_voxel_data)),
    }

    if weighting_vector is not None:
        weighting_vector = np.asarray(weighting_vector, dtype=float)
        if weighting_vector.shape != voxel_data.shape:
            raise ValueError(
                f'Weighting vector shape {weighting_vector.shape} does not match '
                f'voxel data shape {voxel_data.shape}'
            )

        results['weighted_mean'] = float(np.sum(voxel_data * weighting_vector))

        try:
            nz_weighting_vector = weighting_vector.copy()
            nz_weighting_vector[np.isnan(nz_voxel_data)] = np.nan
            nz_weighting_vector = nz_weighting_vector / np.nansum(nz_weighting_vector)
            results['masked_weighted_mean'] = float(np.nansum(nz_voxel_data * nz_weighting_vector))
        except Exception as exc:
            LOGGER.warning(
                'Error calculating weighted mean of %s in %s: %s',
                variable_name,
                mask_name,
                exc,
            )
            results['masked_weighted_mean'] = np.nan

    return results


def parse_scalar_arg(spec: str) -> dict[str, str]:
    """
    Parse scalar specification.

    Accepted formats:
      /path/to/fa.nii.gz
      FA=/path/to/fa.nii.gz
      FA=/path/to/fa.nii.gz|suffix=DIPYDKI
    """
    parts = spec.split('|')
    first = parts[0]

    if '=' in first:
        variable_name, path = first.split('=', 1)
    else:
        path = first
        variable_name = Path(path).name
        for suf in ['.nii.gz', '.nii']:
            if variable_name.endswith(suf):
                variable_name = variable_name[: -len(suf)]
                break

    meta: dict[str, str] = {
        'variable_name': variable_name,
        'path': path,
        'source_file': path,
        'qsirecon_suffix': 'external',
    }

    for piece in parts[1:]:
        if '=' in piece:
            k, v = piece.split('=', 1)
            meta[k] = v

    return meta


def summarize_bundles(
    dwiref_image: str,
    tck_files: list[str],
    bundle_names: list[str],
    scalar_specs: list[dict[str, str]],
    out_dir: str,
    bundle_source: str | None = None,
    bundle_params_id: str | None = None,
    tckmap_command: Sequence[str] | None = None,
) -> tuple[str, str]:
    """Main bundle summarization routine."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    bundle_rows: list[dict[str, Any]] = []
    tdi_rows: list[dict[str, Any]] = []

    for bundle_name, tck_file in zip(bundle_names, tck_files):
        LOGGER.info('Processing bundle %s', bundle_name)

        tdi_file = str(out_path / f'{bundle_name}_tdi.nii.gz')
        tdi_img = run_tckmap(
            dwiref_image,
            tck_file,
            tdi_file,
            tckmap_command=tckmap_command,
        )

        tdi_data = np.asanyarray(tdi_img.dataobj)
        mask_data = (tdi_data > 0).astype(np.uint8)
        mask_img = nib.Nifti1Image(mask_data, tdi_img.affine, tdi_img.header)

        bundle_masker = NiftiMasker(mask_img=mask_img)

        tdi_weights = bundle_masker.fit_transform(tdi_img).squeeze()
        tdi_weights = np.asarray(tdi_weights, dtype=float)

        if tdi_weights.ndim == 0:
            tdi_weights = np.array([tdi_weights], dtype=float)

        tdi_sum = tdi_weights.sum()
        if tdi_sum <= 0:
            LOGGER.warning('Bundle %s has zero TDI sum; weighted stats will be NaN', bundle_name)
            normalized_weights = np.full_like(tdi_weights, np.nan, dtype=float)
        else:
            normalized_weights = tdi_weights / tdi_sum

        tdi_rows.append(
            calculate_mask_stats(
                masker=bundle_masker,
                mask_name=bundle_name,
                mask_variable_name='bundle',
                scalar_img=tdi_img,
                variable_name='tdi',
                source_file=tck_file,
                qsirecon_suffix=bundle_source or 'external',
                weighting_vector=None,
            )
        )

        for scalar_spec in scalar_specs:
            scalar_img = nib.load(scalar_spec['path'])
            row = calculate_mask_stats(
                masker=bundle_masker,
                mask_name=bundle_name,
                mask_variable_name='bundle',
                scalar_img=scalar_img,
                variable_name=scalar_spec['variable_name'],
                source_file=scalar_spec.get('source_file', scalar_spec['path']),
                qsirecon_suffix=scalar_spec.get('qsirecon_suffix', 'external'),
                weighting_vector=normalized_weights
                if np.all(np.isfinite(normalized_weights))
                else None,
            )
            bundle_rows.append(row)

    bundle_df = pd.DataFrame(bundle_rows)
    tdi_df = pd.DataFrame(tdi_rows)

    if bundle_source is not None:
        bundle_df['bundle_source'] = bundle_source
        tdi_df['bundle_source'] = bundle_source
    if bundle_params_id is not None:
        bundle_df['bundle_params_id'] = bundle_params_id
        tdi_df['bundle_params_id'] = bundle_params_id

    bundle_stats_file = str(out_path / 'bundle_stats.tsv')
    tdi_stats_file = str(out_path / 'tdi_stats.tsv')

    bundle_df.to_csv(bundle_stats_file, sep='\t', index=False)
    tdi_df.to_csv(tdi_stats_file, sep='\t', index=False)

    return bundle_stats_file, tdi_stats_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Standalone bundle scalar mapping')

    parser.add_argument('--dwiref', required=True, help='Reference DWI image')
    parser.add_argument(
        '--tck',
        nargs='+',
        required=True,
        help='Bundle .tck files',
    )
    parser.add_argument(
        '--bundle-names',
        nargs='+',
        required=True,
        help='Names for each bundle, same order as --tck',
    )
    parser.add_argument(
        '--scalar',
        nargs='+',
        required=True,
        help=(
            'Scalar image specs. Examples:\n'
            '  FA=/path/to/fa.nii.gz\n'
            '  MD=/path/to/md.nii.gz|qsirecon_suffix=DIPYDKI\n'
            '  /path/to/rd.nii.gz'
        ),
    )
    parser.add_argument('--out-dir', required=True, help='Output directory')
    parser.add_argument('--bundle-source', default=None, help='Optional bundle source label')
    parser.add_argument('--bundle-params-id', default=None, help='Optional bundle parameter ID')
    parser.add_argument(
        '--scalar-json',
        default=None,
        help=(
            'Optional JSON file containing a list of scalar metadata dicts. '
            'If provided, overrides --scalar.'
        ),
    )
    parser.add_argument('--verbose', action='store_true', help='Enable INFO logging')

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    if len(args.tck) != len(args.bundle_names):
        raise ValueError('--tck and --bundle-names must have the same length')

    if args.scalar_json is not None:
        with open(args.scalar_json, 'r') as f:
            scalar_specs = json.load(f)
    else:
        scalar_specs = [parse_scalar_arg(s) for s in args.scalar]

    bundle_stats_file, tdi_stats_file = summarize_bundles(
        dwiref_image=args.dwiref,
        tck_files=args.tck,
        bundle_names=args.bundle_names,
        scalar_specs=scalar_specs,
        out_dir=args.out_dir,
        bundle_source=args.bundle_source,
        bundle_params_id=args.bundle_params_id,
    )

    print(f'Wrote: {bundle_stats_file}')
    print(f'Wrote: {tdi_stats_file}')


if __name__ == '__main__':
    main()
