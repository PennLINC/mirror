"""Remove unneeded fields from bottom-level JSON files."""

import json
import os
import sys
from glob import glob


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from configuration.config import load_config

    _cfg = load_config()

    dset_dir = _cfg['bids_dir']
    drop_keys = [
        'AcquisitionTime',
        'AcquisitionDateTime',
        'CogAtlasID',
        'EchoTime1',
        'EchoTime2',
        'InstitutionAddress',
        'TaskName',
        'ImageComments',
    ]

    json_files = sorted(glob(os.path.join(dset_dir, 'sub-*/ses-*/*/*.json')))
    for json_file in json_files:
        print(json_file)
        with open(json_file, 'r') as f_obj:
            json_data = json.load(f_obj)

        for drop_key in drop_keys:
            if drop_key in json_data.keys():
                json_data.pop(drop_key)

        with open(json_file, 'w') as f_obj:
            json.dump(json_data, f_obj, indent=4, sort_keys=True)
