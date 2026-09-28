"""Anonymize acquisition datetimes for a dataset.

Extract the date and time of the first scan for each subject/session,
round the date to the nearest 15th and the time to the nearest hour,
then put that information in the `acq_time` column of the participants.tsv file.
Then remove the scans.tsv files.
"""

import os
import sys
from glob import glob

import pandas as pd
from dateutil import parser


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from configuration.config import load_config

    _cfg = load_config()

    dset_dir = _cfg['bids_dir']

    subject_dirs = sorted(glob(os.path.join(dset_dir, 'sub-*')))
    for subject_dir in subject_dirs:
        sub_id = os.path.basename(subject_dir)
        print(f'Processing {sub_id}')

        # Assumes sessions are named 01, 02, etc.
        scans_files = sorted(glob(os.path.join(subject_dir, 'ses-*/*_scans.tsv')))
        if len(scans_files) == 0:
            print(f'\tNo scans files found for {sub_id}')
            continue

        n_sessions = len(scans_files)

        sessions_tsv = os.path.join(subject_dir, f'{sub_id}_sessions.tsv')
        if not os.path.exists(sessions_tsv):
            sessions_df = pd.DataFrame(
                columns=['session_id', 'acq_time'],
                index=range(n_sessions),
            )
        else:
            sessions_df = pd.read_table(sessions_tsv)

        for i_ses, scans_file in enumerate(scans_files):
            ses_dir = os.path.dirname(scans_file)
            ses_name = os.path.basename(ses_dir)
            print(f'\t{ses_name}')

            scans_df = pd.read_table(scans_file)

            # Anonymize in terms of first scan for subject.
            first_scan = scans_df['acq_time'].min()
            ses_start = parser.parse(first_scan)

            # Round to the nearest 15th and the time to the nearest hour.
            ses_start = ses_start.replace(day=15, minute=0, second=0)
            ses_acqtime = str(ses_start).replace(' ', 'T')

            if ses_name not in sessions_df['session_id'].values:
                next_row = sessions_df.shape[0]
                sessions_df.loc[next_row, 'session_id'] = ses_name

            sessions_df.loc[sessions_df['session_id'] == ses_name, 'acq_time'] = ses_acqtime

        sessions_df.to_csv(sessions_tsv, sep='\t', lineterminator='\n', na_rep='n/a', index=False)

        # Remove scans files.
        for scans_file in scans_files:
            os.remove(scans_file)

    if os.path.isfile(os.path.join(dset_dir, 'scans.json')):
        os.remove(os.path.join(dset_dir, 'scans.json'))
