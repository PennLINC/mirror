"""Convert MP2RAGE DICOMs to NIfTI."""

import json
import os
import re
import sys
from glob import glob


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from configuration.config import load_config
    from utils.processing import run_command

    _cfg = load_config()

    in_dir = _cfg['sourcedata']['scitran']
    out_dir = _cfg['bids_dir']

    subses_dirs = sorted(glob(os.path.join(in_dir, '*_*')))
    for subses_dir in subses_dirs:
        subses_id = os.path.basename(subses_dir)
        sub_id, ses_id = subses_id.split('_')
        sub_dir = os.path.join(out_dir, f'sub-{sub_id}', f'ses-{ses_id}')
        sub_out_dir = os.path.join(sub_dir, 'anat')

        dicom_dirs = glob(
            os.path.join(
                subses_dir,
                'CAMRIS^Satterthwaite',
                'anat-MP2RAGE_RR_INV1',
                '*.dicom',
            ),
        )
        dicom_dirs += glob(
            os.path.join(
                subses_dir,
                'CAMRIS^Satterthwaite',
                'anat-MP2RAGE_RR_INV2',
                '*.dicom',
            ),
        )
        dicom_dirs = sorted(dicom_dirs)
        if len(dicom_dirs) != 2:
            print(f'Expected 2 dicom dirs, got {len(dicom_dirs)}: {dicom_dirs}')
            continue

        for dicom_dir in dicom_dirs:
            dicom_dir_id = os.path.basename(dicom_dir)
            dicom_dir_id = dicom_dir_id.split('_')[0]
            inv = re.search(r'INV(\d)', dicom_dir).group(1)
            nii_file = os.path.join(
                f'sub-{sub_id}_ses-{ses_id}_rec-defaced_run-01_inv-{inv}_part-phase_MP2RAGE',
            )
            cmd = f'dcm2niix -b y -z y -f {nii_file} -o {sub_out_dir} {dicom_dir}'
            run_command(cmd)

            out_file = os.path.join(sub_out_dir, nii_file + '.nii.gz')
            if not os.path.exists(out_file):
                out_file = os.path.join(sub_out_dir, nii_file + '_ph.nii.gz')

            out_json = out_file.replace('.nii.gz', '.json')
            print(out_json)
            with open(out_json, 'r') as f:
                metadata = json.load(f)

            if 'P' not in metadata['ImageType']:
                print(f'Image not phase: {out_file}')
                os.remove(out_file)
                os.remove(out_json)
                continue

            if f'INV{inv}' not in metadata['SeriesDescription']:
                print(f'SeriesDescription not INV{inv}: {out_file}')
                os.remove(out_file)
                os.remove(out_json)
                continue

            if 'DIS3D' not in metadata['ImageType']:
                print(f'Image not DIS3D: {out_file}')
                os.remove(out_file)
                os.remove(out_json)
                continue

            if 'DIS2D' not in metadata['ImageType']:
                print(f'Image not DIS2D: {out_file}')
                os.remove(out_file)
                os.remove(out_json)
                continue
