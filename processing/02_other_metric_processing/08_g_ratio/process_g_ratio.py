"""Calculate g ratio maps.

Steps:

1.  Calculate g ratio maps from various combinations.
2.  Warp g ratio maps to MNI152NLin2009cAsym using normalization transform from sMRIPrep.

Notes:

- sMRIPrep's preprocessed T1w image is used as the "native T1w space".
- This must be run after sMRIPrep, QSIRecon, process_ihmt.py, and process_g_ratio_scaling_factors.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pformat

import ants
import numpy as np
from bids.layout import BIDSLayout, Query
from nilearn import masking
from nireports.assembler.report import Report

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.processing import get_filename, load_config, plot_scalar_map  # noqa: E402

CFG = load_config()
CODE_DIR = CFG['code_dir']
# Scaling factors to be adjusted so that mean g-ratios in splenium are 0.7 across the sample.
MTsat_ISOVF_ICVF_scalar = 0.09696293901605979
ihMTR_ISOVF_ICVF_scalar = 2.2137054628974355


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect T1w-space myelin and microstructure maps for g-ratio computation.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths.
    """
    queries = {
        # T1w-space ISOVF and ICVF maps produced by process_g_ratio_scaling_factors.py
        'isovf_t1w': {
            'datatype': 'dwi',
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'model': 'noddi',
            'param': 'isovf',
            'desc': Query.NONE,
            'suffix': 'dwimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'icvf_t1w': {
            'datatype': 'dwi',
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'model': 'noddi',
            'param': 'icvf',
            'desc': Query.NONE,
            'suffix': 'dwimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep preprocessed T1w in MNI space (warping target and report underlay)
        't1w_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-to-MNI normalization transform from sMRIPrep
        't1w2mni_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'from': 'T1w',
            'to': 'MNI152NLin2009cAsym',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
        },
        # T1w-space MTsat and ihMTR maps from process_ihmt.py
        'mtsat_t1w': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'suffix': 'ihMTsatB1sq',
            'extension': ['.nii', '.nii.gz'],
        },
        'ihmtr_t1w': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'suffix': 'ihMTR',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space dseg from sMRIPrep (for reports)
        'dseg_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'dseg',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep MNI-space brain mask (for reports)
        'mni_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    run_data = {}
    for key, query in queries.items():
        # I have no clue why, but BIDSLayout refuses to index 'param'
        param = None
        if 'param' in query:
            param = query.pop('param')

        query = {**bids_filters, **query}
        files = layout.get(**query)
        if param is not None:
            print(files)
            files = [f for f in files if f'_param-{param}_' in f.filename]
            if param == 'fa':
                # Both DIPYDKI and DSIStudio have 'fa' as a param. Use DIPYDKI.
                files = [f for f in files if 'qsirecon-DIPYDKI' in f.path]
            query['param'] = param

        if len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')

        file = files[0]
        run_data[key] = file.path

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    """Process a single g-ratio run.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object.
    run_data : dict
        Dictionary of run data.
    out_dir : str
        Directory to write output files.
    temp_dir : str
        Directory to write temporary files.
    """
    # T1w-space ISOVF and ICVF are pre-computed by process_g_ratio_scaling_factors.py.
    isovf = ants.image_read(run_data['isovf_t1w'])
    icvf = ants.image_read(run_data['icvf_t1w'])

    # Resample T1w-space MTsat and ihMTR to match the ISOVF/ICVF resolution.
    # Eq. 3 in Berg et al. (2022)
    mtsat_mvf = (
        ants.image_read(run_data['mtsat_t1w']).resample_image_to_target(isovf, interp_type='linear')
        * MTsat_ISOVF_ICVF_scalar
    )
    # Eq. 4 in Berg et al. (2022)
    ihmtr_mvf = (
        ants.image_read(run_data['ihmtr_t1w']).resample_image_to_target(isovf, interp_type='linear')
        * ihMTR_ISOVF_ICVF_scalar
    )

    # Eq 6 in Berg et al. (2022)
    mtsat_fvf = (1 - mtsat_mvf) * (1 - isovf) * icvf
    ihmtr_fvf = (1 - ihmtr_mvf) * (1 - isovf) * icvf

    # G = sqrt(1 - (MVF / (MVF + AVF))) [Eq. 1 in Newman et al. (2024)]
    # Same as sqrt(AVF / (AVF + MVF)) [Eq. 1 in Berg et al. (2022)]
    imgs = {}
    imgs['MTsat+ISOVF+ICVF'] = (mtsat_fvf / (mtsat_fvf + mtsat_mvf)) ** 0.5
    imgs['ihMTR+ISOVF+ICVF'] = (ihmtr_fvf / (ihmtr_fvf + ihmtr_mvf)) ** 0.5

    mni_t1w_img = ants.image_read(run_data['t1w_mni'])

    for desc, img in imgs.items():
        # Write T1w-space g-ratio map.
        t1w_file = get_filename(
            name_source=run_data['mtsat_t1w'],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w', 'desc': desc, 'suffix': 'gratio'},
            dismiss_entities=['reconstruction', 'acquisition'],
        )
        ants.image_write(img, t1w_file)

        # Warp T1w-space g-ratio map to MNI space.
        mni_img = ants.apply_transforms(
            fixed=mni_t1w_img,
            moving=img,
            transformlist=[run_data['t1w2mni_xfm']],
            interpolator='linear',
        )
        mni_file = get_filename(
            name_source=run_data['mtsat_t1w'],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym', 'desc': desc, 'suffix': 'gratio'},
            dismiss_entities=['reconstruction', 'acquisition'],
        )
        ants.image_write(mni_img, mni_file)

        scalar_desc = 'scalar'
        if desc:
            scalar_desc = f'{desc}{scalar_desc}'

        resampled_mni_mask = ants.image_read(run_data['mni_mask']).resample_image_to_target(
            mni_img, interp_type='nearestNeighbor'
        )
        mni_mask_file = os.path.join(temp_dir, 'resampled_mni_mask.nii.gz')
        ants.image_write(resampled_mni_mask, mni_mask_file)
        resampled_mni_t1w = mni_t1w_img.resample_image_to_target(mni_img, interp_type='linear')
        mni_t1w_file = os.path.join(temp_dir, 'resampled_mni_t1w.nii.gz')
        ants.image_write(resampled_mni_t1w, mni_t1w_file)
        resampled_mni_dseg = ants.image_read(run_data['dseg_mni']).resample_image_to_target(
            mni_img, interp_type='nearestNeighbor'
        )
        mni_dseg_file = os.path.join(temp_dir, 'resampled_mni_dseg.nii.gz')
        ants.image_write(resampled_mni_dseg, mni_dseg_file)

        data = masking.apply_mask(mni_file, mni_mask_file)
        vmin = np.percentile(data, 2)
        vmin = np.minimum(vmin, 0)
        vmax = np.percentile(data, 98)

        scalar_report = get_filename(
            name_source=mni_file,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'figures', 'desc': scalar_desc, 'extension': '.svg'},
        )
        plot_scalar_map(
            underlay=mni_t1w_file,
            overlay=mni_file,
            mask=mni_mask_file,
            dseg=mni_dseg_file,
            out_file=scalar_report,
            vmin=vmin,
            vmax=vmax,
        )


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        default=None,
        help='Subject to process. If not provided, all subjects are processed.',
    )
    return parser


def _main(argv=None):
    """Run the process_g_ratio workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    ihmt_dir = CFG['derivatives']['ihmt']
    out_dir = CFG['derivatives']['g_ratio']
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(CFG['work_dir'], 'g_ratio')
    os.makedirs(temp_dir, exist_ok=True)
    print(f'in_dir: {in_dir}')
    print(f'out_dir: {out_dir}')
    print(f'ihmt_dir: {ihmt_dir}')
    print(f'smriprep_dir: {smriprep_dir}')

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports', 'g_ratio.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'configuration', 'bids', 'layout.json'),
        validate=False,
        # out_dir is included so that T1w-space ISOVF/ICVF written by
        # process_g_ratio_scaling_factors.py can be queried directly.
        derivatives=[smriprep_dir, ihmt_dir, out_dir],
    )

    base_query = {
        'space': 'T1w',
        'suffix': 'ihMTR',
        'extension': ['.nii', '.nii.gz'],
    }

    if subject_id:
        subjects = [subject_id]
    else:
        subjects = layout.get_subjects(**base_query)

    for subject_id in subjects:
        print(f'Processing subject {subject_id}')
        sessions = layout.get_sessions(subject=subject_id, **base_query)
        for session in sessions:
            print(f'Processing session {session}')
            base_files = layout.get(
                subject=subject_id,
                session=session,
                **base_query,
            )
            if not base_files:
                print(f'No base files found for subject {subject_id} and session {session}')
                continue

            for base_file in base_files:
                entities = base_file.get_entities()
                try:
                    run_data = collect_run_data(layout, entities)
                except ValueError as e:
                    print(f'Failed {base_file}')
                    print(e)
                    continue

                fname = os.path.basename(base_file.path).split('.')[0]
                run_temp_dir = os.path.join(temp_dir, fname.replace('-', '').replace('_', ''))
                os.makedirs(run_temp_dir, exist_ok=True)
                process_run(layout, run_data, out_dir, run_temp_dir)

            report_dir = os.path.join(out_dir, f'sub-{subject_id}', f'ses-{session}')
            robj = Report(
                report_dir,
                run_uuid=None,
                bootstrap_file=bootstrap_file,
                out_filename=f'sub-{subject_id}_ses-{session}.html',
                reportlets_dir=out_dir,
                plugins=None,
                plugin_meta=None,
                subject=subject_id,
                session=session,
            )
            robj.generate_report()

    # Write out dataset_description.json
    dataset_description_file = os.path.join(out_dir, 'dataset_description.json')
    if not os.path.isfile(dataset_description_file):
        dataset_description = {
            'Name': 'MIRROR G-Ratio Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'DatasetLinks': {
                'raw': in_dir,
                'smriprep': smriprep_dir,
                'ihmt': ihmt_dir,
            },
            'GeneratedBy': [
                {
                    'Name': 'Custom code',
                    'Description': 'Custom Python code.',
                    'CodeURL': 'https://github.com/PennLINC/MIRROR',
                }
            ],
        }
        with open(dataset_description_file, 'w') as fobj:
            json.dump(dataset_description, fobj, sort_keys=True, indent=4)

    print('DONE!')


if __name__ == '__main__':
    _main()
