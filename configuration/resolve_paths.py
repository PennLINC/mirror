#!/usr/bin/env python3
"""Expose a MIRROR YAML profile to shell launchers.

The script deliberately lives inside the checkout and resolves imports from
that checkout, so changing the repository directory name does not affect it.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from configuration import load_config  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--profile',
        default=None,
        help='Profile name or YAML path. Defaults to MIRROR_CONFIG, then hpc.',
    )
    parser.add_argument('--shell', action='store_true', help='Emit shell-safe assignments.')
    return parser


def main() -> None:
    """Print the selected profile's shared roots."""

    args = build_parser().parse_args()
    config = load_config(args.profile)
    values = {
        'MIRROR_PROJECT_ROOT': config['project_root'],
        'MIRROR_CODE_ROOT': config['code_dir'],
        'MIRROR_BIDS_DIR': config['bids_dir'],
        'MIRROR_SOURCE_DERIVATIVES': config['source_derivatives_dir'],
        'MIRROR_OUTPUT_DERIVATIVES': config['output_derivatives_dir'],
        'MIRROR_LOGS_DIR': config['logs_dir'],
        'MIRROR_WORK_DIR': config['work_dir'],
        'MIRROR_DATA_DIR': config['data_dir'],
        'MIRROR_RUN_NAME': config['run_name'],
    }
    optional_paths = {
        'MIRROR_FREESURFER_LICENSE': config.get('freesurfer', {}).get('license'),
        'MIRROR_APPTAINER_FMRIPREP': config.get('apptainer', {}).get('fmriprep'),
        'MIRROR_APPTAINER_QSIPREP': config.get('apptainer', {}).get('qsiprep'),
        'MIRROR_APPTAINER_QSIRECON': config.get('apptainer', {}).get('qsirecon'),
        'MIRROR_APPTAINER_SYNTHSTRIP': config.get('apptainer', {}).get('synthstrip'),
    }
    values.update({key: value for key, value in optional_paths.items() if value})
    if args.shell:
        for name, value in values.items():
            print(f'export {name}={shlex.quote(str(value))}')
        return
    for name, value in values.items():
        print(f'{name}={value}')


if __name__ == '__main__':
    main()
