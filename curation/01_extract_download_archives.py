"""Expand dicom zip files in order to heudiconv."""

import os
import sys
import zipfile
from glob import glob

if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from configuration.config import load_config

    _cfg = load_config()

    status_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'status',
        'extracted_download_archives.txt',
    )
    if os.path.exists(status_file):
        with open(status_file, 'r') as f:
            unzipped_subjects = f.read().splitlines()
    else:
        unzipped_subjects = []

    zip_files = sorted(glob(os.path.join(_cfg['sourcedata']['root'], '*.zip')))
    for zip_file in zip_files:
        subject = os.path.basename(zip_file).split('.')[0]
        if subject in unzipped_subjects:
            print(f'Subject {subject} already processed, skipping...')
        else:
            print(f'Processing {subject}...')
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(zip_file))
            print(f'Subject {subject} processed successfully.')
            with open(status_file, 'a') as f:
                f.write(f'{subject}\n')
