"""Expand dicom zip files in order to heudiconv.

Rename the folders before running this.
"""

import os
import sys
import zipfile
from glob import glob

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from configuration.config import load_config

    _cfg = load_config()
    _sourcedata_scitran = _cfg['sourcedata']['scitran']

    status_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'status',
        'extracted_dicom_archives.txt',
    )
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            unzipped_subjects = f.read().splitlines()
    else:
        unzipped_subjects = []

    subjects = sorted(glob(os.path.join(_sourcedata_scitran, '*')))
    subjects = [os.path.basename(subject) for subject in subjects]

    for subject in subjects:
        if subject in unzipped_subjects:
            print(f'Subject {subject} already processed, skipping...')
            continue

        zip_files = sorted(
            glob(os.path.join(_sourcedata_scitran, subject, '*', '*', '*.dicom.zip'))
        )
        print(f'Processing {subject}...')
        for zip_file in zip_files:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(zip_file))

            os.remove(zip_file)

        with open(status_file, 'a') as f:
            f.write(f'{subject}\n')
