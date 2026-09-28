"""Generate myelin reports."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nb
import numpy as np
import yaml
from bids.layout import BIDSLayout, Query
from nilearn import maskers
from nireports.assembler.report import Report

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.processing import get_filename, load_config, plot_scalar_map  # noqa: E402

CFG = load_config()
CODE_DIR = CFG['code_dir']
QUERY_LOOKUP = {
    'Query.NONE': Query.NONE,
    'Query.ANY': Query.ANY,
}


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect myelin derivative images for report generation.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, space) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths.
    """
    with open(Path(__file__).resolve().parent / 'report_queries.yml', encoding='utf-8') as fobj:
        queries = yaml.safe_load(fobj)['queries']

    run_data = {}
    for key, query in queries['myelin'].items():
        for k, v in query.items():
            if isinstance(v, list):
                new_v = []
                for item in v:
                    new_v.append(QUERY_LOOKUP.get(item, item))
                query[k] = new_v
            else:
                query[k] = QUERY_LOOKUP.get(v, v)

        # I have no clue why, but BIDSLayout refuses to index 'param'
        param = None
        if 'param' in query:
            param = query.pop('param')

        query = {**bids_filters, **query}
        files = layout.get(**query)
        if param is not None:
            files = [f for f in files if f'_param-{param}_' in f.filename]
            if param == 'fa':
                # Both DIPYDKI and DSIStudio have 'fa' as a param. Use DIPYDKI.
                files = [f for f in files if 'qsirecon-DIPYDKI' in f.path]
            query['param'] = param

        if len(files) != 1:
            print(f'Expected 1 file for {key}, got {len(files)}: {query}')
            run_data[key] = None
            continue

        file = files[0]
        run_data[key] = file.path

    for key, query in queries['other'].items():
        query = {**bids_filters, **query}
        for k, v in query.items():
            if isinstance(v, list):
                new_v = []
                for item in v:
                    new_v.append(QUERY_LOOKUP.get(item, item))
                query[k] = new_v
            else:
                query[k] = QUERY_LOOKUP.get(v, v)

        files = layout.get(**query)
        if len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')
        run_data[key] = files[0].path

    return run_data


def process_run(layout, run_data, out_dir):
    """Process a single subject/session's data.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object.
    run_data : dict
        Dictionary of run data.
    out_dir : str
        Directory to write output files.
    """
    mask_img = nb.load(run_data['brainmask'])
    mask_img.header.set_data_dtype(np.int32)
    dseg_img = nb.load(run_data['dseg'])
    dseg_img.header.set_data_dtype(np.int32)

    for name, file_ in run_data.items():
        if name in ['t1w', 'brainmask', 'dseg']:
            continue

        if file_ is None:
            print(f'No {name} file found for {run_data["t1w"]}')
            continue

        print(f'Processing {name}: {file_}')
        masker = maskers.NiftiMasker(mask_img=mask_img)
        data = masker.fit_transform(file_)
        vmin = np.percentile(data, 2)
        vmin = np.minimum(vmin, 0)
        vmax = np.percentile(data, 98)

        scalar_report = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={
                'datatype': 'figures',
                'desc': name,
                'suffix': 'scalar',
                'extension': '.svg',
            },
            dismiss_entities=['model', 'param'],
        )
        plot_scalar_map(
            underlay=run_data['t1w'],
            overlay=file_,
            mask=mask_img,
            dseg=dseg_img,
            out_file=scalar_report,
            vmin=vmin,
            vmax=vmax,
        )


def _get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--subject-id',
        type=lambda label: label.removeprefix('sub-'),
        required=True,
    )
    return parser


def _main(argv=None):
    """Run the generate_myelin_reports workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    ihmt_dir = CFG['derivatives']['ihmt']
    mp2rage_dir = CFG['derivatives']['pymp2rage']
    qsm_dir = CFG['derivatives']['qsm']
    t1wt2w_dir = CFG['derivatives']['t1wt2w_ratio']
    qsirecon_dki_dir = CFG['derivatives']['qsirecon_dipydki']
    qsirecon_dsi_dir = CFG['derivatives']['qsirecon_dsistudio']

    out_dir = os.path.join(
        CFG['output_derivatives_dir'],
        'quality_control',
        'scalar_reports',
    )
    os.makedirs(out_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports', 'myelin.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    config = os.path.join(CODE_DIR, 'configuration', 'bids', 'layout.json')
    layout = BIDSLayout(
        in_dir,
        config=[config],
        validate=False,
        derivatives=[
            smriprep_dir,
            ihmt_dir,
            mp2rage_dir,
            qsm_dir,
            t1wt2w_dir,
            qsirecon_dki_dir,
            qsirecon_dsi_dir,
        ],
    )

    print(f'Processing subject {subject_id}')
    sessions = layout.get_sessions(subject=subject_id)
    for session in sessions:
        print(f'Processing session {session}')
        try:
            run_data = collect_run_data(
                layout,
                {
                    'subject': subject_id,
                    'session': session,
                    'space': 'MNI152NLin2009cAsym',
                    'extension': '.nii.gz',
                },
            )
        except ValueError as e:
            print(e)
            continue

        process_run(layout, run_data, out_dir)

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
            'Name': 'MIRROR Scalar Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
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
