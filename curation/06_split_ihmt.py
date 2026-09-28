"""Split 4D IHMT files into 3D BIDS format.

One corrupt subject/session ihMTRAGE series was removed before this step.
before running.
"""

import json
import os
import re
import sys
from glob import glob

import nibabel as nb


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from configuration.config import load_config

    _cfg = load_config()

    in_dir = _cfg['bids_dir']

    patterns = [
        '_acq-nosat_run-{run}_mt-off_ihMTRAGE',
        '_acq-singlepos_run-{run}_mt-on_ihMTRAGE',
        '_acq-dual1_run-{run}_mt-on_ihMTRAGE',
        '_acq-singleneg_run-{run}_mt-on_ihMTRAGE',
        '_acq-dual2_run-{run}_mt-on_ihMTRAGE',
    ]

    ihmt_files = sorted(
        glob(
            os.path.join(
                in_dir,
                'sub-*',
                'ses-*',
                'anat',
                'sub-*_ihMTRAGE.nii.gz',
            )
        )
    )
    for ihmt_file in ihmt_files:
        print(ihmt_file)
        img = nb.load(ihmt_file)
        out_dir = os.path.dirname(ihmt_file)
        fname_base = os.path.basename(ihmt_file)
        # Get run entity
        run_entity = re.search(r'run-(\d+)', fname_base).group(1)

        fname_base = fname_base.split('_run')[0]

        in_json = ihmt_file.replace('.nii.gz', '.json')
        if img.ndim == 4:
            print(f'Splitting {ihmt_file}')
            for i_vol, pattern in enumerate(patterns):
                pattern_str = pattern.format(run=run_entity)
                img3d = img.slicer[..., i_vol]
                out_fname = os.path.join(out_dir, f'{fname_base}{pattern_str}.nii.gz')
                with open(in_json, 'r') as f_obj:
                    metadata = json.load(f_obj)

                out_json = os.path.join(out_dir, f'{fname_base}{pattern_str}.json')
                if 'mt-off' in pattern_str:
                    metadata['MTState'] = False
                else:
                    metadata['MTState'] = True

                img3d.to_filename(out_fname)
                with open(out_json, 'w') as f_obj:
                    json.dump(metadata, f_obj, indent=4, sort_keys=True)

            os.remove(ihmt_file)
            os.remove(in_json)

        else:
            print(f'Skipping {ihmt_file}')
