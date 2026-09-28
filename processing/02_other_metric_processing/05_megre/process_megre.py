"""Process MEGRE data for later QSM modeling.

Steps:

1.  Compute the RMS across the magnitude images as the MEGRE reference.
2.  Calculate R2* map with nonlinear model fit using monoexponential decay model.
3.  Coregister the reference to the preprocessed T1w image from sMRIPrep.
4.  Warp T1w mask from T1w space into the MEGREref space by applying the inverse of the coregistration
    transform.
5.  Apply the mask in MEGREref space to magnitude images.
6.  Warp the R2* and R2' maps to T1w and MNI152NLin2009cAsym spaces and plot them.

Notes:

- The R2* map is calculated using a magnitude-only nonlinear least squares fit.
- The MESE-derived R2 map is optional. When it is missing, the R2' maps
  (R2* - R2) are not calculated and only the R2* maps are written out.
- This must be run after sMRIPrep and process_mese.py.
- Runs processed before the warping step was added may need their incomplete
  outputs removed before rerunning this workflow; the former one-off backfill
  script has been retired.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from pprint import pformat

import ants
import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from nireports.assembler.report import Report

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.processing import (  # noqa: E402
    coregister_to_t1,
    fit_monoexponential,
    get_filename,
    load_config,
    plot_brain_mask_contour,
    plot_coregistration,
    run_synthstrip,
    warp_scalar_to_t1w_and_mni,
)

CFG = load_config()
CODE_DIR = CFG['code_dir']

MEGRE_FIT_N_THREADS = int(os.environ.get('MEGRE_FIT_N_THREADS', str(os.cpu_count() or 1)))


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect required input files for MEGRE preparation processing.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths. ``r2_map`` is
        ``None`` when no MESE-derived R2 map is available.
    """
    queries = {
        # MEGRE images from raw BIDS dataset
        'megre_mag': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'reconstruction': [Query.NONE, Query.ANY],
            'part': 'mag',
            'echo': Query.ANY,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        'megre_phase': {
            'datatype': 'anat',
            'acquisition': 'QSM',
            'reconstruction': [Query.NONE, Query.ANY],
            'part': 'phase',
            'echo': Query.ANY,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        # T1w-space R2 map from MESE pipeline
        'r2_map': {
            'datatype': 'anat',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'desc': 'MESE',
            'suffix': 'R2map',
            'extension': '.nii.gz',
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
        # sMRIPrep T1w-space brain mask
        't1w_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'res': Query.NONE,
            'desc': 'brain',
            'suffix': 'mask',
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
        'mni2t1w_xfm': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'from': 'MNI152NLin2009cAsym',
            'to': 'T1w',
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
        # T1w-space WM segmentation, created once by process_mp2rage.py and reused here
        'wm_seg_t1w': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'acquisition': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'res': Query.NONE,
            'desc': 'wm',
            'suffix': 'mask',
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

    run_data = {}
    for key, query in queries.items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        if key.startswith('megre_'):
            if len(files) != 5:
                raise ValueError(f'Expected 5 files for {key}, got {len(files)}')
            else:
                run_data[key] = sorted([f.path for f in files])
                continue

        elif key == 'r2_map' and len(files) == 0:
            print(f"No MESE R2 map found with query {query}. R2' maps will not be calculated.")
            run_data[key] = None
            continue
        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')

        file = files[0]
        run_data[key] = file.path

    if len(run_data['megre_mag']) != len(run_data['megre_phase']):
        raise ValueError('Expected same number of magnitude and phase images')

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def process_run(layout, run_data, out_dir, temp_dir, n_threads=4):
    """Process a single run of MEGRE data.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary containing the paths to the QSM data.
    out_dir : str
        Path to the output directory.
    temp_dir : str
        Path to the working directory for temporary files.
        Currently unused.
    n_threads : int
        Number of threads to use for R2*fitting.

    Notes
    -----
    When ``run_data['r2_map']`` is None, the R2' maps (R2* - R2) are skipped and
    only the R2* maps are written out.
    """
    name_source = run_data['megre_mag'][0]

    # Build name of last file generated
    mask_qsm_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'brain', 'suffix': 'mask'},
    )
    if os.path.isfile(mask_qsm_filename):
        print(f'Skipping {os.path.basename(name_source)}')
        return

    megre_metadata = [layout.get_metadata(f) for f in run_data['megre_mag']]
    echo_times = [m['EchoTime'] for m in megre_metadata]  # TEs in seconds

    # The MESE-derived R2 map is optional; without it R2' cannot be calculated.
    has_r2 = run_data['r2_map'] is not None
    if not has_r2:
        print("No MESE R2 map; calculating R2* maps only (no R2').", flush=True)

    # Collect the T1w-space WM segmentation created once by process_mp2rage.py.
    wm_seg_t1w_file = run_data['wm_seg_t1w']

    # Create the MNI-space WM segmentation from the sMRIPrep dseg for the MNI plots.
    wm_seg_img = nb.load(run_data['dseg_mni'])
    wm_seg = wm_seg_img.get_fdata()
    wm_seg = (wm_seg == 2).astype(int)
    wm_seg_file = get_filename(
        name_source=run_data['dseg_mni'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MNI152NLin2009cAsym', 'desc': 'wm', 'suffix': 'mask'},
        dismiss_entities=['reconstruction'],
    )
    wm_seg_img = nb.Nifti1Image(wm_seg, wm_seg_img.affine, wm_seg_img.header)
    wm_seg_img.to_filename(wm_seg_file)
    del wm_seg_img, wm_seg

    # Compute RMS of the magnitude images, to use for coregistration
    ref_img = nb.load(run_data['megre_mag'][0])
    arrs = [nb.load(f).get_fdata() ** 2 for f in run_data['megre_mag']]
    rms_data = np.sqrt(np.mean(np.stack(arrs, axis=-1), axis=-1))
    rms_img = nb.Nifti1Image(rms_data, ref_img.affine, ref_img.header)

    megre_ref_filename = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'rms', 'suffix': 'MEGRE'},
        dismiss_entities=['echo'],
    )
    rms_img.to_filename(megre_ref_filename)

    # Skullstrip the reference image before coregistration
    brain_mask = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'desc': 'brain', 'suffix': 'mask'},
        dismiss_entities=['inv', 'part', 'reconstruction'],
    )
    megre_ref_skullstripped_file = get_filename(
        name_source=name_source,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MEGRE', 'suffix': 'MEGRE', 'desc': 'rmsbrain'},
        dismiss_entities=['inv', 'part', 'reconstruction'],
    )
    run_synthstrip(
        in_file=megre_ref_filename,
        out_file=megre_ref_skullstripped_file,
        mask_file=brain_mask,
        args=['--no-csf'],
    )

    # QC reportlet: SynthStrip brain mask boundary over the un-skull-stripped reference image.
    brain_mask_report = get_filename(
        name_source=megre_ref_skullstripped_file,
        layout=layout,
        out_dir=out_dir,
        entities={
            'datatype': 'figures',
            'space': 'MP2RAGE',
            'desc': 'brain',
            'suffix': 'mask',
            'extension': '.svg',
        },
        dismiss_entities=['inv', 'part', 'reconstruction'],
    )
    plot_brain_mask_contour(
        underlay=megre_ref_filename,
        mask=brain_mask,
        out_file=brain_mask_report,
    )

    # Coregister MEGRE data to preprocessed T1w
    coreg_transform = coregister_to_t1(
        name_source=name_source,
        layout=layout,
        in_file=megre_ref_skullstripped_file,
        t1_file=run_data['t1w'],
        t1_mask=run_data['t1w_mask'],
        source_space='MEGRE',
        target_space='T1w',
        out_dir=out_dir,
    )
    t1_megre_ref_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=ants.image_read(megre_ref_skullstripped_file),
        transformlist=[coreg_transform],
        interpolator='linear',
    )
    t1_megre_ref_skullstripped_file = get_filename(
        name_source=megre_ref_filename,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'rmsbrain', 'suffix': 'MEGRE'},
        dismiss_entities=['echo', 'part', 'reconstruction'],
    )
    ants.image_write(t1_megre_ref_img, t1_megre_ref_skullstripped_file)
    plot_coregistration(
        name_source=t1_megre_ref_skullstripped_file,
        layout=layout,
        in_file=t1_megre_ref_skullstripped_file,
        t1_file=run_data['t1w'],
        out_dir=out_dir,
        source_space='MEGRE',
        target_space='T1w',
        wm_seg=wm_seg_t1w_file,
    )

    mni_megre_ref_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w_mni']),
        moving=ants.image_read(megre_ref_filename),
        transformlist=[run_data['t1w2mni_xfm'], coreg_transform],
        interpolator='linear',
    )
    mni_megre_ref_filename = get_filename(
        name_source=t1_megre_ref_skullstripped_file,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'MNI152NLin2009cAsym', 'desc': 'rmsbrain', 'suffix': 'MEGRE'},
        dismiss_entities=['echo', 'part', 'reconstruction'],
    )
    ants.image_write(mni_megre_ref_img, mni_megre_ref_filename)
    plot_coregistration(
        name_source=mni_megre_ref_filename,
        layout=layout,
        in_file=mni_megre_ref_filename,
        t1_file=run_data['t1w_mni'],
        out_dir=out_dir,
        source_space='MEGRE',
        target_space='MNI152NLin2009cAsym',
        wm_seg=wm_seg_file,
    )

    if has_r2:
        # Warp R2 map from T1w space to MEGRE space
        r2_qsm_filename = get_filename(
            name_source=run_data['r2_map'],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MEGRE'},
            dismiss_entities=['echo', 'part'],
        )
        r2_qsm_img = ants.apply_transforms(
            fixed=ants.image_read(megre_ref_filename),
            moving=ants.image_read(run_data['r2_map']),
            transformlist=[coreg_transform],
            whichtoinvert=[True],
            interpolator='linear',
        )
        ants.image_write(r2_qsm_img, r2_qsm_filename)

        # The MEGRE-space R2 map carries the entities the R2*/R2' outputs need.
        r2s_name_source = r2_qsm_filename
        r2s_dismiss_entities = []
    else:
        # Without an R2 map, name the R2* outputs from the raw MEGRE echoes.
        r2s_name_source = name_source
        r2s_dismiss_entities = ['echo', 'part', 'acquisition']

    # Calculate R2* maps, and R2' maps when an R2 map is available.
    # Each map is written in MEGRE space and then warped to T1w and MNI space.
    # Only the 'space' entity changes, so the desc ('MEGRE+E12345'/'MEGRE+E2345')
    # carries through to the warped outputs.
    for desc, mag_files, tes in (
        ('MEGRE+E12345', run_data['megre_mag'], echo_times),
        # From echo 2 onwards
        ('MEGRE+E2345', run_data['megre_mag'][1:], echo_times[1:]),
    ):
        _, r2s_hz_img, _, _ = fit_monoexponential(mag_files, tes, n_threads=n_threads)
        r2s_hz_filename = get_filename(
            name_source=r2s_name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MEGRE', 'desc': desc, 'suffix': 'R2starmap'},
            dismiss_entities=r2s_dismiss_entities,
        )
        r2s_hz_img.to_filename(r2s_hz_filename)

        warp_scalar_to_t1w_and_mni(
            in_file=r2s_hz_filename,
            layout=layout,
            out_dir=out_dir,
            coreg_transform=coreg_transform,
            t1w=run_data['t1w'],
            t1w_mni=run_data['t1w_mni'],
            t1w2mni_xfm=run_data['t1w2mni_xfm'],
            mni_mask=run_data['mni_mask'],
            dseg_mni=run_data['dseg_mni'],
        )

        if not has_r2:
            continue

        r2prime_hz_filename = get_filename(
            name_source=r2s_name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MEGRE', 'desc': desc, 'suffix': 'R2primemap'},
            dismiss_entities=r2s_dismiss_entities,
        )
        r2s_hz_img = ants.image_read(r2s_hz_filename)
        r2prime_hz_img = r2s_hz_img - r2_qsm_img
        ants.image_write(r2prime_hz_img, r2prime_hz_filename)

        # R2' = R2* - R2 is signed, so plot it on a symmetric diverging scale.
        warp_scalar_to_t1w_and_mni(
            in_file=r2prime_hz_filename,
            layout=layout,
            out_dir=out_dir,
            coreg_transform=coreg_transform,
            t1w=run_data['t1w'],
            t1w_mni=run_data['t1w_mni'],
            t1w2mni_xfm=run_data['t1w2mni_xfm'],
            mni_mask=run_data['mni_mask'],
            dseg_mni=run_data['dseg_mni'],
            cmap='RdBu_r',
            symmetric=True,
        )

    # Warp the QSM mask
    mask_qsm_img = ants.apply_transforms(
        fixed=ants.image_read(megre_ref_filename),
        moving=ants.image_read(run_data['t1w_mask']),
        transformlist=[coreg_transform],
        whichtoinvert=[True],
        interpolator='nearestNeighbor',
    )
    ants.image_write(mask_qsm_img, mask_qsm_filename)


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
    """Run the process_megre workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    code_dir = CFG['code_dir']
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    mp2rage_dir = CFG['derivatives']['pymp2rage']
    mese_dir = CFG['derivatives']['mese']
    out_dir = CFG['derivatives']['megre']
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(CFG['work_dir'], 'megre')
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports', 'megre.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    n_threads = 28  # hardcoded for my PC. Change for other situations.

    # Write the QSM derivatives dataset_description.json before building the
    # layout. Without it, pybids silently refuses to index this directory as a
    # derivative, so process_qsm.py cannot find the prep outputs (brain mask,
    # R2*/R2' maps) written here.
    dataset_description_file = os.path.join(out_dir, 'dataset_description.json')
    if not os.path.isfile(dataset_description_file):
        dataset_description = {
            'Name': 'MIRROR MEGRE Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'DatasetLinks': {
                'raw': in_dir,
                'smriprep': smriprep_dir,
                'mese': mese_dir,
            },
            'GeneratedBy': [
                {
                    'Name': 'Custom code',
                    'Description': 'Custom Python code combining ANTsPy and a nonlinear R2* fit.',
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
        derivatives=[mp2rage_dir, smriprep_dir, mese_dir],
    )

    if subject_id:
        subjects = [subject_id]
    else:
        subjects = layout.get_subjects(suffix='MEGRE')

    for subject_id in subjects:
        print(f'Processing subject {subject_id}')
        sessions = layout.get_sessions(subject=subject_id, suffix='MEGRE')
        for session in sessions:
            print(f'Processing session {session}')
            megre_files = layout.get(
                subject=subject_id,
                session=session,
                acquisition='QSM',
                echo=1,
                part='mag',
                suffix='MEGRE',
                extension=['.nii', '.nii.gz'],
            )
            for megre_file in megre_files:
                entities = megre_file.get_entities()
                entities.pop('echo')
                entities.pop('part')
                entities.pop('acquisition')
                try:
                    run_data = collect_run_data(layout, entities)
                except ValueError as e:
                    print(f'Failed {megre_file}')
                    print(e)
                    continue

                fname = os.path.basename(megre_file.path).split('.')[0]
                run_temp_dir = os.path.join(temp_dir, fname.replace('-', '').replace('_', ''))
                os.makedirs(run_temp_dir, exist_ok=True)
                process_run(layout, run_data, out_dir, run_temp_dir, n_threads=n_threads)

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
