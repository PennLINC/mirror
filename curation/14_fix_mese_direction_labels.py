"""Rename bad MESE files."""

import json
import os
import sys
from glob import glob


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from configuration.config import load_config

    _cfg = load_config()

    dset_dir = _cfg['bids_dir']
    subject_dirs = sorted(glob(os.path.join(dset_dir, 'sub-*')))
    for subject_dir in subject_dirs:
        subject_id = os.path.basename(subject_dir)
        print(f'Processing subject {subject_id}')
        session_dirs = sorted(glob(os.path.join(subject_dir, 'ses-*')))
        for session_dir in session_dirs:
            session_id = os.path.basename(session_dir)
            print(f'\tProcessing session {session_id}')
            mese_dir = os.path.join(session_dir, 'anat')
            ap_mese_files = sorted(glob(os.path.join(mese_dir, '*_dir-AP_run-*_echo-1_MESE.json')))
            for ap_mese_file in ap_mese_files:
                print(f'\t\tProcessing MESE file: {os.path.basename(ap_mese_file)}')
                with open(ap_mese_file, 'r') as f_obj:
                    data = json.load(f_obj)

                if data['PhaseEncodingDirection'] != 'j-':
                    print(f'\t\t\tPhaseEncodingDirection is not j-: {ap_mese_file}')
                    continue

                pa_mese_file = ap_mese_file.replace('dir-AP', 'dir-PA')
                if not os.path.isfile(pa_mese_file):
                    print(f'\t\t\tPA MESE file not found: {pa_mese_file}')
                    continue

                with open(pa_mese_file, 'r') as f_obj:
                    data = json.load(f_obj)

                if data['PhaseEncodingDirection'] != 'j':
                    print(
                        f'\t\t\tPhaseEncodingDirection is not j: {os.path.basename(pa_mese_file)} '
                        f'({data["PhaseEncodingDirection"]})'
                    )
                    nii_file = pa_mese_file.replace('.json', '.nii.gz')
                    if not os.path.isfile(nii_file):
                        print(f'\t\t\tNII file not found: {nii_file}')
                        continue

                    new_filename = pa_mese_file.replace('dir-PA', 'dir-AP').replace(
                        'run-01', 'run-02'
                    )
                    if os.path.isfile(new_filename):
                        print(f'\t\t\tFile already exists: {new_filename}')
                        continue

                    new_nii_file = new_filename.replace('.json', '.nii.gz')
                    if os.path.isfile(new_nii_file):
                        print(f'\t\t\tFile already exists: {new_nii_file}')
                        continue

                    os.rename(pa_mese_file, new_filename)
                    os.rename(nii_file, new_nii_file)
