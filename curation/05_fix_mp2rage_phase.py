"""Fix MP2RAGE phase image conversions."""

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
        session_dirs = sorted(glob(os.path.join(subject_dir, 'ses-*')))
        for session_dir in session_dirs:
            mp2rage_dir = os.path.join(session_dir, 'anat')
            inv1_files = sorted(
                glob(os.path.join(mp2rage_dir, '*inv-1_part-phase_MP2RAGE*.nii.gz')),
            )
            for inv1_file in inv1_files:
                inv1_json = inv1_file.replace('.nii.gz', '.json')
                inv2_file = inv1_file.replace('inv-1', 'inv-2')
                inv2_json = inv2_file.replace('.nii.gz', '.json')
                if 'MP2RAGE_ph' not in inv1_file:
                    print(f'Misnamed magnitude image: {inv1_file}')
                    os.remove(inv1_file)
                    os.remove(inv1_json)
                    os.remove(inv2_file)
                    os.remove(inv2_json)
                    continue

                if not os.path.exists(inv2_file):
                    print(f'inv-2 file not found: {inv2_file}')
                    os.remove(inv1_file)
                    os.remove(inv1_json)
                    continue

                new_inv1_file = inv1_file.replace('MP2RAGE_ph.', 'MP2RAGE.')
                os.rename(inv1_file, new_inv1_file)
                new_inv1_json = new_inv1_file.replace('.nii.gz', '.json')
                os.rename(inv1_json, new_inv1_json)
                new_inv2_file = inv2_file.replace('MP2RAGE_ph.', 'MP2RAGE.')
                os.rename(inv2_file, new_inv2_file)
                new_inv2_json = new_inv2_file.replace('.nii.gz', '.json')
                os.rename(inv2_json, new_inv2_json)

            inv2_files = sorted(
                glob(os.path.join(mp2rage_dir, '*inv-2_part-phase_MP2RAGE*.nii.gz')),
            )
            for inv2_file in inv2_files:
                inv2_json = inv2_file.replace('.nii.gz', '.json')
                inv1_file = inv2_file.replace('inv-2', 'inv-1')
                inv1_json = inv1_file.replace('.nii.gz', '.json')

                if not os.path.exists(inv1_file):
                    print(f'inv-1 file not found: {inv1_file}')
                    os.remove(inv2_file)
                    os.remove(inv2_json)
                    continue
