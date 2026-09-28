"""Aggregate splenium sidecars and estimate g-ratio scaling factors."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.processing import load_config  # noqa: E402

CFG = load_config()
CODE_DIR = CFG['code_dir']


def compute_scaling_factor(ICVF, MVF, ISOVF, g=0.7):
    """
    Compute the scaling factor such that:
        g = sqrt(FVF / (FVF + MVFs))
    given:
        FVF = (1 - MVFs) * (1 - ISOVF) * ICVF
        MVFs = MVF * scaling_factor
    """
    g2 = np.square(g)
    numerator = (1 - ISOVF) * ICVF * (1 - g2)
    denominator = MVF * (g2 + (1 - ISOVF) * ICVF * (1 - g2))
    return numerator / denominator


def collect_splenium_sidecars(json_dir):
    """Collect per-run splenium scalar sidecar JSON files."""
    pattern = os.path.join(json_dir, 'sub-*', '**', '*_desc-splenium_scalarstats.json')
    sidecar_files = sorted(glob.glob(pattern, recursive=True))
    if not sidecar_files:
        raise FileNotFoundError(f'No splenium scalar sidecars found with pattern: {pattern}')

    rows = []
    for sidecar_file in sidecar_files:
        with open(sidecar_file) as fobj:
            sidecar = json.load(fobj)

        splenium_values = sidecar.get('SpleniumValues', {})
        if 'ihMTsatB1sq' not in splenium_values and 'MTsat' in splenium_values:
            splenium_values['ihMTsatB1sq'] = splenium_values['MTsat']

        rows.append(
            {
                'subject_id': sidecar.get('subject_id'),
                'session_id': sidecar.get('session_id'),
                'run': sidecar.get('run'),
                'ISOVF': splenium_values.get('ISOVF'),
                'ICVF': splenium_values.get('ICVF'),
                'ihMTsatB1sq': splenium_values.get('ihMTsatB1sq'),
                'ihMTR': splenium_values.get('ihMTR'),
            }
        )

    splenium_df = pd.DataFrame(rows)
    splenium_df = splenium_df.sort_values(['subject_id', 'session_id', 'run'])
    return splenium_df


def estimate_scaling_factors(splenium_df, target_g=0.7):
    """Estimate scaling factors from aggregated splenium means."""
    required_columns = ['ISOVF', 'ICVF', 'ihMTsatB1sq', 'ihMTR']
    missing_columns = [col for col in required_columns if col not in splenium_df.columns]
    if missing_columns:
        raise ValueError(f'Missing required columns: {missing_columns}')

    if splenium_df[required_columns].isna().any().any():
        raise ValueError('Some splenium sidecars are missing required scalar values.')

    # Drop PILOT* participants
    splenium_df = splenium_df.loc[~splenium_df['subject_id'].str.contains('PILOT')]

    MTsat_ISOVF_ICVF_scalar = compute_scaling_factor(
        ICVF=splenium_df['ICVF'].mean(),
        MVF=splenium_df['ihMTsatB1sq'].mean(),
        ISOVF=splenium_df['ISOVF'].mean(),
        g=target_g,
    )
    ihMTR_ISOVF_ICVF_scalar = compute_scaling_factor(
        ICVF=splenium_df['ICVF'].mean(),
        MVF=splenium_df['ihMTR'].mean(),
        ISOVF=splenium_df['ISOVF'].mean(),
        g=target_g,
    )

    return {
        'MTsat_ISOVF_ICVF_scalar': MTsat_ISOVF_ICVF_scalar,
        'ihMTR_ISOVF_ICVF_scalar': ihMTR_ISOVF_ICVF_scalar,
    }


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--json-dir',
        default=CFG['derivatives']['g_ratio'],
        help='Directory containing per-run splenium scalar sidecar JSON files.',
    )
    parser.add_argument(
        '--out-file',
        default=os.path.join(CODE_DIR, 'data', 'splenium_values.tsv'),
        help='Output TSV file for aggregated splenium values.',
    )
    parser.add_argument(
        '--target-g',
        type=float,
        default=0.7,
        help='Target mean splenium g-ratio.',
    )
    return parser


def _main(argv=None):
    options = _get_parser().parse_args(argv)
    main(**vars(options))


def main(json_dir, out_file, target_g):
    splenium_df = collect_splenium_sidecars(json_dir)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    splenium_df.to_csv(out_file, sep='\t', index=False)
    print(f'Wrote {len(splenium_df)} rows to {out_file}', flush=True)

    scaling_factors = estimate_scaling_factors(splenium_df, target_g=target_g)
    for name, value in scaling_factors.items():
        print(f'{name}: {value}', flush=True)

    print('DONE!', flush=True)


if __name__ == '__main__':
    _main()
