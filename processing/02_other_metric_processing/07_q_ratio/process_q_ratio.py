"""Calculate Q-Ratio maps from precomputed derivatives.

Steps:

1. Collect T1w-space R1 and B1-corrected R1 maps from pymp2rage derivatives.
2. Collect T1w-space R2*-E12345 and R2*-E2345 maps from megre derivatives.
3. Calculate Q-Ratio variants from these derivatives and warp them to
   MNI152NLin2009cAsym space.

Follows Kawaguchi et al. (2024), "Anisotropy of the R1/T2* value dependent on
white matter fiber orientation with respect to the B0 field", Magn Reson Imaging
https://doi.org/10.1016/j.mri.2024.02.010

That study is the closer match to this dataset than the original q-Ratio paper
(Shim et al., 2022): it is 3T rather than 7T, and it derives R1 from MP2RAGE and
R2* from a separate multi-echo GRE, exactly as done here, rather than from a
single ME-MP2RAGE acquisition.

Notes:

- The quantity is R1/T2*, not R1/R2*. Since 1/T2* == R2*, it is computed here as
  R1 * R2*. This is worth stating explicitly because "q-Ratio" invites the
  reading R1/R2*, which would be dimensionless, roughly 0.05, and nearly flat
  between gray and white matter.
- Both inputs are already in s^-1 (process_mp2rage.py converts T1 to seconds
  before taking the reciprocal, and fit_monoexponential is given echo times in
  seconds), so the Q-Ratio is in s^-2.
- No physiological range masking is applied. Kawaguchi et al. impose none, and
  the T1/T2* bounds published by Shim et al. were 7T values that do not transfer
  to 3T (their T2* <= 60 ms ceiling sits on this dataset's gray matter median of
  ~58 ms and would discard roughly half of gray matter).
- Caveat from the source paper: R1/T2* in white matter is orientation-dependent
  with respect to B0, and more strongly so than R1 or R2* alone. Kawaguchi et al.
  report the minimum near the magic angle to be 18.86% below the maximum at
  perpendicular orientations. Bundle-wise or region-wise comparisons of these
  maps will therefore carry a fiber-orientation confound.
- The B1-corrected R1 map is optional, matching process_mp2rage.py, which skips
  B1 correction when no B1+ map is available. Without it only the uncorrected
  Q-Ratio variants are written.
- This must be run after sMRIPrep, process_mp2rage.py, and process_megre.py,
  which produce the T1w-space R1 and R2* maps.
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
ECHO_SETS = ('E12345', 'E2345')


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect T1w-space R1 and R2* maps for Q-Ratio computation.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths. ``r1_b1c`` is
        ``None`` when no B1-corrected R1 map is available.
    """
    # Session is deliberately not set on the pymp2rage/megre queries below: those
    # derivatives are session-specific, so the session must come through from
    # bids_filters rather than being widened to any/none. The sMRIPrep references
    # are subject-level, so they do widen it.
    queries = {
        # T1w-space R1 map from process_mp2rage.py
        'r1': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'acquisition': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'res': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'R1map',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-space B1-corrected R1 map from process_mp2rage.py (optional)
        'r1_b1c': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'acquisition': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'res': Query.NONE,
            'desc': 'B1corrected',
            'suffix': 'R1map',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-space T1w image from sMRIPrep
        't1w': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space T1w image from sMRIPrep
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
        # Normalization transform from sMRIPrep
        't1w2mni_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'from': 'T1w',
            'to': 'MNI152NLin2009cAsym',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.h5',
        },
        # MNI-space dseg from sMRIPrep
        'dseg_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'dseg',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep MNI-space brain mask
        'mni_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    # T1w-space R2* maps from process_megre.py
    for echo_set in ECHO_SETS:
        queries[f'r2s_{echo_set.lower()}'] = {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'acquisition': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'res': Query.NONE,
            'desc': f'MEGRE+{echo_set}',
            'suffix': 'R2starmap',
            'extension': ['.nii', '.nii.gz'],
        }

    run_data = {}
    for key, query in queries.items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if key == 'r1_b1c' and len(files) == 0:
            print('No B1-corrected R1 map found. Only uncorrected Q-Ratios will be calculated.')
            run_data[key] = None
            continue
        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')

        run_data[key] = files[0].path

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def calculate_q_ratio(r1_file: str, r2s_file: str) -> object:
    """Calculate the Q-Ratio from an R1 map and an R2* map.

    Both maps must be in the same space and on the same grid. Voxels outside the
    brain are already 0 in both inputs and so stay 0 in the product.

    Parameters
    ----------
    r1_file : str
        Path to an R1 map in s^-1.
    r2s_file : str
        Path to an R2* map in s^-1.

    Returns
    -------
    q_ratio_img : ants.ANTsImage
        Q-Ratio map (R1/T2*, computed as R1 * R2*) in s^-2.
    """
    r1_img = ants.image_read(r1_file)
    r2s = ants.image_read(r2s_file).numpy()

    return r1_img.new_image_like(r1_img.numpy() * r2s)


def process_run(layout, run_data, out_dir):
    """Calculate the Q-Ratio variants for a single run.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary containing the paths to the R1 and R2* maps.
    out_dir : str
        Path to the output directory.

    Notes
    -----
    When ``run_data['r1_b1c']`` is None, the B1-corrected variants are skipped.
    """
    # (desc suffix, R1 map) pairs. The B1-corrected R1 is optional.
    r1_variants = [('', run_data['r1'])]
    if run_data['r1_b1c'] is not None:
        r1_variants.append(('+B1corrected', run_data['r1_b1c']))
    else:
        print('No B1-corrected R1 map; calculating uncorrected Q-Ratios only.', flush=True)

    for echo_set in ECHO_SETS:
        r2s_file = run_data[f'r2s_{echo_set.lower()}']

        for desc_suffix, r1_file in r1_variants:
            desc = f'{echo_set}{desc_suffix}'
            print(f'Calculating Q-Ratio for {desc}', flush=True)

            q_ratio_img = calculate_q_ratio(r1_file, r2s_file)
            t1w_file = get_filename(
                name_source=run_data['r1'],
                layout=layout,
                out_dir=out_dir,
                entities={'space': 'T1w', 'desc': desc, 'suffix': 'qratio'},
                dismiss_entities=['inv', 'part', 'echo', 'reconstruction', 'acquisition'],
            )
            ants.image_write(q_ratio_img, t1w_file)

            # The Q-Ratio is already in T1w space, so the coregistration leg of
            # the warp is the identity and only the normalization is applied.
            mni_file = get_filename(
                name_source=t1w_file,
                layout=layout,
                out_dir=out_dir,
                entities={'space': 'MNI152NLin2009cAsym'},
            )
            mni_img = ants.apply_transforms(
                fixed=ants.image_read(run_data['t1w_mni']),
                moving=q_ratio_img,
                transformlist=[run_data['t1w2mni_xfm']],
                interpolator='linear',
            )
            ants.image_write(mni_img, mni_file)

            plot_q_ratio(layout, mni_file, run_data, out_dir)


def plot_q_ratio(layout, mni_file, run_data, out_dir):
    """Write the scalar reportlet for an MNI-space Q-Ratio map.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    mni_file : str
        Path to the MNI-space Q-Ratio map.
    run_data : dict
        Dictionary containing the sMRIPrep MNI references.
    out_dir : str
        Path to the output directory.
    """
    # nireports indexes figures with its own config, whose desc pattern only
    # captures [a-zA-Z0-9], so a '+' in the desc truncates it on indexing (e.g.
    # 'E12345+B1correctedscalar' -> 'E12345') and the report's '.*scalar' query
    # never matches. Strip '+' so the 'scalar' token survives parsing.
    desc = 'scalar'
    if 'desc-' in mni_file:
        raw_desc = mni_file.split('desc-')[-1].split('_')[0]
        desc = raw_desc.replace('+', '') + 'scalar'

    scalar_report = get_filename(
        name_source=mni_file,
        layout=layout,
        out_dir=out_dir,
        entities={'datatype': 'figures', 'desc': desc, 'extension': '.svg'},
    )
    data = masking.apply_mask(mni_file, run_data['mni_mask'])
    vmin = np.percentile(data, 2)
    vmin = np.minimum(vmin, 0)
    vmax = np.percentile(data, 98)
    plot_scalar_map(
        underlay=run_data['t1w_mni'],
        overlay=mni_file,
        mask=run_data['mni_mask'],
        dseg=run_data['dseg_mni'],
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
    """Run the process_q_ratio workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    code_dir = CFG['code_dir']
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    mp2rage_dir = CFG['derivatives']['pymp2rage']
    megre_dir = CFG['derivatives']['megre']
    out_dir = CFG['derivatives']['q_ratio']
    os.makedirs(out_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports', 'q_ratio.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    # Write the dataset_description.json before building the layout. Without it,
    # pybids silently refuses to index this directory as a derivative.
    dataset_description_file = os.path.join(out_dir, 'dataset_description.json')
    if not os.path.isfile(dataset_description_file):
        dataset_description = {
            'Name': 'MIRROR Q-Ratio Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'DatasetLinks': {
                'raw': in_dir,
                'smriprep': smriprep_dir,
                'pymp2rage': mp2rage_dir,
                'megre': megre_dir,
            },
            'GeneratedBy': [
                {
                    'Name': 'Custom code',
                    'Description': (
                        'Custom Python code computing the Q-Ratio (R1/T2*) following '
                        'Kawaguchi et al. (2024), https://doi.org/10.1016/j.mri.2024.02.010'
                    ),
                    'CodeURL': 'https://github.com/PennLINC/MIRROR',
                }
            ],
        }
        with open(dataset_description_file, 'w') as fobj:
            json.dump(dataset_description, fobj, sort_keys=True, indent=4)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(code_dir, 'configuration', 'bids', 'layout.json'),
        validate=False,
        derivatives=[smriprep_dir, mp2rage_dir, megre_dir],
    )

    base_query = {
        'space': 'T1w',
        'desc': Query.NONE,
        'suffix': 'R1map',
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
            base_files = layout.get(subject=subject_id, session=session, **base_query)
            if not base_files:
                print(f'No R1 maps found for subject {subject_id} and session {session}')
                continue

            for base_file in base_files:
                entities = base_file.get_entities()
                # The R1 map's own desc/suffix/space must not narrow the queries
                # for the R2* maps and sMRIPrep references.
                for entity in ('space', 'desc', 'suffix', 'extension', 'datatype'):
                    entities.pop(entity, None)

                try:
                    run_data = collect_run_data(layout, entities)
                except ValueError as e:
                    print(f'Failed {base_file}')
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

    print('DONE!')


if __name__ == '__main__':
    _main()
