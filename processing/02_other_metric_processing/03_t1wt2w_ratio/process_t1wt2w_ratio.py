"""Calculate basic T1w/T2w ratio maps.

Steps:

1.  Coregister SPACE T1w, SPACE T2w, and MPRAGE to sMRIPrep's preprocessed T1w image.
2.  Calculate SPACE T1w/SPACE T2w ratio map.
3.  Calculate MPRAGE T1w/SPACE T2w ratio map.
4.  Warp T1w/T2w ratio maps to MNI152NLin2009cAsym using normalization transform from sMRIPrep.

Notes:

- sMRIPrep's preprocessed T1w image is used as the "native T1w space".
- The MPRAGE T1w image is optional. When it is missing, steps 3 and the MPRAGE
  part of step 4 are skipped and only the SPACE T1w/SPACE T2w ratio maps are
  calculated.
- This must be run after sMRIPrep and process_mp2rage.py.
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
from nilearn import masking
from nireports.assembler.report import Report

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.processing import (  # noqa: E402
    coregister_to_t1,
    get_filename,
    load_config,
    plot_coregistration,
    plot_scalar_map,
    run_synthstrip,
)

CFG = load_config()
CODE_DIR = CFG['code_dir']


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect required input files for T1w/T2w ratio processing.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths. ``mprage_t1w`` and
        ``mprage2t1w_xfm`` are ``None`` when the MPRAGE T1w image is missing.
    """
    queries = {
        'space_t1w': {
            'part': Query.NONE,
            'acquisition': 'SPACE',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        'space_t2w': {
            'part': Query.NONE,
            'acquisition': 'SPACE',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'T2w',
            'extension': ['.nii', '.nii.gz'],
        },
        'mprage_t1w': {
            'part': Query.NONE,
            'acquisition': 'MPRAGE',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'T1w',
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
        # Coregistration transform for MPRAGE, from sMRIPrep
        'mprage2t1w_xfm': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'reconstruction': [Query.NONE, Query.ANY],
            'from': 'orig',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.txt',
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
        # Normalization transform from sMRIPrep
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
        if key == 'mprage_t1w' and len(files) == 0:
            print(
                f'No MPRAGE T1w image found for {query}. '
                'Only SPACE T1w/SPACE T2w ratio maps will be calculated.'
            )
            run_data[key] = None
            continue
        elif key == 'mprage2t1w_xfm' and len(files) == 0:
            print(
                f'No MPRAGE T1w coregistration transform found for {query}. Using identity transform.'
            )
            run_data[key] = None
            continue
        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')

        file = files[0]
        run_data[key] = file.path

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    """Process a single T1w/T2w ratio run.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object.
    run_data : dict
        Dictionary of run data.
    out_dir : str
        Directory to write output files.
    temp_dir : str
        Directory to write temporary files (skull-stripped moving images used for
        coregistration).

    Notes
    -----
    When ``run_data['mprage_t1w']`` is None, the MPRAGE T1w/SPACE T2w ratio maps
    are skipped and only the SPACE T1w/SPACE T2w ratio maps are produced.
    """
    # The MPRAGE T1w image is optional; without it only the SPACE ratio maps are made.
    has_mprage = run_data['mprage_t1w'] is not None
    if not has_mprage:
        print('No MPRAGE T1w image; only calculating SPACE T1w/SPACE T2w ratio maps.', flush=True)

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

    # Create scaled versions of the original T1w and T2w images
    space_t1w_img = ants.image_read(run_data['space_t1w'])
    space_t2w_img = ants.image_read(run_data['space_t2w'])
    space_t1w_data = space_t1w_img.numpy()
    space_t2w_data = space_t2w_img.numpy()
    # Scale the images to 0 - 100
    scaled_space_t1w_data = space_t1w_data - space_t1w_data.min()
    scaled_space_t1w_data = 100 * (scaled_space_t1w_data / scaled_space_t1w_data.max())
    scaled_space_t2w_data = space_t2w_data - space_t2w_data.min()
    scaled_space_t2w_data = 100 * (scaled_space_t2w_data / scaled_space_t2w_data.max())
    scaled_space_t1w_img = space_t1w_img.new_image_like(scaled_space_t1w_data)
    scaled_space_t2w_img = space_t2w_img.new_image_like(scaled_space_t2w_data)

    if has_mprage:
        mprage_t1w_img = ants.image_read(run_data['mprage_t1w'])
        mprage_t1w_data = mprage_t1w_img.numpy()
        scaled_mprage_t1w_data = mprage_t1w_data - mprage_t1w_data.min()
        scaled_mprage_t1w_data = 100 * (scaled_mprage_t1w_data / scaled_mprage_t1w_data.max())
        scaled_mprage_t1w_img = mprage_t1w_img.new_image_like(scaled_mprage_t1w_data)

    # Coregister SPACE T1w to sMRIPrep T1w using the shared brain-extracted rigid
    # approach. Skull-strip the moving image first so it is a brain-to-brain fit
    # against the brain-extracted T1w target that coregister_to_t1 builds internally.
    fixed_img = ants.image_read(run_data['t1w'])
    space_t1w_brain_file = os.path.join(temp_dir, 'space-SPACE_desc-brain_T1w.nii.gz')
    run_synthstrip(
        in_file=run_data['space_t1w'],
        out_file=space_t1w_brain_file,
        cfg=CFG,
    )
    space_t1w_to_t1w_xfm = coregister_to_t1(
        name_source=run_data['space_t1w'],
        layout=layout,
        in_file=space_t1w_brain_file,
        t1_file=run_data['t1w'],
        t1_mask=run_data['t1w_mask'],
        out_dir=out_dir,
        source_space='SPACET1w',
        target_space='T1w',
    )

    # Apply the transform to SPACE T1w
    space_t1w_img = ants.image_read(run_data['space_t1w'])
    t1w_space_t1w_img = ants.apply_transforms(
        fixed=fixed_img,
        moving=space_t1w_img,
        transformlist=[space_t1w_to_t1w_xfm],
        interpolator='linear',
    )
    t1w_space_t1w_file = get_filename(
        name_source=run_data['space_t1w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'T1w'},
        dismiss_entities=['reconstruction'],
    )
    ants.image_write(t1w_space_t1w_img, t1w_space_t1w_file)

    # Apply the transform to scaled SPACE T1w
    t1w_scaled_space_t1w_img = ants.apply_transforms(
        fixed=fixed_img,
        moving=scaled_space_t1w_img,
        transformlist=[space_t1w_to_t1w_xfm],
        interpolator='linear',
    )
    t1w_scaled_space_t1w_file = get_filename(
        name_source=run_data['space_t1w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'SPACEscaled', 'suffix': 'T1w'},
        dismiss_entities=['reconstruction'],
    )
    ants.image_write(t1w_scaled_space_t1w_img, t1w_scaled_space_t1w_file)
    del scaled_space_t1w_img

    # Coregister SPACE T2w to sMRIPrep T1w using the shared brain-extracted rigid
    # approach. Skull-strip the moving image first so it is a brain-to-brain fit
    # against the brain-extracted T1w target that coregister_to_t1 builds internally.
    space_t2w_brain_file = os.path.join(temp_dir, 'space-SPACE_desc-brain_T2w.nii.gz')
    run_synthstrip(
        in_file=run_data['space_t2w'],
        out_file=space_t2w_brain_file,
        cfg=CFG,
    )
    space_t2w_to_t1w_xfm = coregister_to_t1(
        name_source=run_data['space_t2w'],
        layout=layout,
        in_file=space_t2w_brain_file,
        t1_file=run_data['t1w'],
        t1_mask=run_data['t1w_mask'],
        out_dir=out_dir,
        source_space='SPACET2w',
        target_space='T1w',
    )

    # Apply the transform to SPACE T2w
    t1w_space_t2w_img = ants.apply_transforms(
        fixed=fixed_img,
        moving=space_t2w_img,
        transformlist=[space_t2w_to_t1w_xfm],
        interpolator='linear',
    )
    t1w_space_t2w_file = get_filename(
        name_source=run_data['space_t2w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'T2w'},
        dismiss_entities=['reconstruction'],
    )
    ants.image_write(t1w_space_t2w_img, t1w_space_t2w_file)
    del space_t2w_img

    # Apply the transform to scaled SPACE T2w
    t1w_scaled_space_t2w_img = ants.apply_transforms(
        fixed=fixed_img,
        moving=scaled_space_t2w_img,
        transformlist=[space_t2w_to_t1w_xfm],
        interpolator='linear',
    )
    t1w_scaled_space_t2w_file = get_filename(
        name_source=run_data['space_t2w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'SPACEscaled', 'suffix': 'T2w'},
        dismiss_entities=['reconstruction'],
    )
    ants.image_write(t1w_scaled_space_t2w_img, t1w_scaled_space_t2w_file)
    del scaled_space_t2w_img

    if has_mprage:
        # Apply the sMRIPrep coregistration transform to MPRAGE T1w
        mprage_t1w_img = ants.image_read(run_data['mprage_t1w'])
        if run_data['mprage2t1w_xfm'] is None:
            t1w_mprage_t1w_img = mprage_t1w_img
            t1w_scaled_mprage_t1w_img = scaled_mprage_t1w_img
        else:
            fwd_transform = run_data['mprage2t1w_xfm']
            t1w_mprage_t1w_img = ants.apply_transforms(
                fixed=fixed_img,
                moving=mprage_t1w_img,
                transformlist=fwd_transform,
                interpolator='linear',
            )
            t1w_scaled_mprage_t1w_img = ants.apply_transforms(
                fixed=fixed_img,
                moving=scaled_mprage_t1w_img,
                transformlist=fwd_transform,
                interpolator='linear',
            )

        t1w_mprage_t1w_file = get_filename(
            name_source=run_data['mprage_t1w'],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w', 'suffix': 'T1w'},
        )
        ants.image_write(t1w_mprage_t1w_img, t1w_mprage_t1w_file)
        del mprage_t1w_img

        t1w_scaled_mprage_t1w_file = get_filename(
            name_source=run_data['mprage_t1w'],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w', 'desc': 'MPRAGEscaled', 'suffix': 'T1w'},
        )
        ants.image_write(t1w_scaled_mprage_t1w_img, t1w_scaled_mprage_t1w_file)
        del scaled_mprage_t1w_img

    # Calculate SPACE T1w/SPACE T2w ratio map for unscaled images
    t1w_unscaled_space_ratio_file = get_filename(
        name_source=run_data['space_t1w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'SPACEunscaled', 'suffix': 'myelinw'},
        dismiss_entities=['reconstruction', 'acquisition'],
    )
    t1w_unscaled_space_ratio_img = t1w_space_t1w_img / t1w_space_t2w_img
    ants.image_write(t1w_unscaled_space_ratio_img, t1w_unscaled_space_ratio_file)
    del t1w_space_t1w_img, t1w_unscaled_space_ratio_img

    # Calculate SPACE T1w/SPACE T2w ratio map for scaled images
    t1w_scaled_space_ratio_file = get_filename(
        name_source=run_data['space_t1w'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'SPACEscaled', 'suffix': 'myelinw'},
        dismiss_entities=['reconstruction', 'acquisition'],
    )
    t1w_scaled_space_ratio_img = t1w_scaled_space_t1w_img / t1w_scaled_space_t2w_img
    ants.image_write(t1w_scaled_space_ratio_img, t1w_scaled_space_ratio_file)
    del t1w_scaled_space_t1w_img, t1w_scaled_space_ratio_img

    if has_mprage:
        # Calculate MPRAGE T1w/SPACE T2w ratio map for unscaled images
        t1w_unscaled_mprage_ratio_file = get_filename(
            name_source=run_data['mprage_t1w'],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w', 'desc': 'MPRAGEunscaled', 'suffix': 'myelinw'},
            dismiss_entities=['reconstruction', 'acquisition'],
        )
        t1w_unscaled_mprage_ratio_img = t1w_mprage_t1w_img / t1w_space_t2w_img
        ants.image_write(t1w_unscaled_mprage_ratio_img, t1w_unscaled_mprage_ratio_file)
        del t1w_mprage_t1w_img, t1w_unscaled_mprage_ratio_img

        # Calculate MPRAGE T1w/SPACE T2w ratio map for scaled images
        t1w_scaled_mprage_ratio_file = get_filename(
            name_source=run_data['mprage_t1w'],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w', 'desc': 'MPRAGEscaled', 'suffix': 'myelinw'},
            dismiss_entities=['reconstruction', 'acquisition'],
        )
        t1w_scaled_mprage_ratio_img = t1w_scaled_mprage_t1w_img / t1w_scaled_space_t2w_img
        ants.image_write(t1w_scaled_mprage_ratio_img, t1w_scaled_mprage_ratio_file)
        del t1w_scaled_mprage_t1w_img, t1w_scaled_mprage_ratio_img

    del t1w_space_t2w_img, t1w_scaled_space_t2w_img

    # Plot coregistration of SPACE and MPRAGE files to sMRIPrep T1w
    coreg_files = [t1w_space_t1w_file, t1w_space_t2w_file]
    descs = ['SPACE', 'SPACE']
    if has_mprage:
        coreg_files.insert(1, t1w_mprage_t1w_file)
        descs.insert(1, 'MPRAGE')

    for i_file, file_ in enumerate(coreg_files):
        plot_coregistration(
            name_source=file_,
            layout=layout,
            in_file=file_,
            t1_file=run_data['t1w'],
            out_dir=out_dir,
            source_space=descs[i_file],
            target_space='T1w',
            wm_seg=wm_seg_t1w_file,
        )

        mni_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym'},
            dismiss_entities=['reconstruction'],
        )
        mni_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm']],
            interpolator='linear',
        )
        ants.image_write(mni_img, mni_file)

        plot_coregistration(
            name_source=mni_file,
            layout=layout,
            in_file=mni_file,
            t1_file=run_data['t1w_mni'],
            out_dir=out_dir,
            source_space=descs[i_file],
            target_space='MNI152NLin2009cAsym',
            wm_seg=wm_seg_file,
        )
        del mni_img, mni_file

    del coreg_files, t1w_space_t1w_file, t1w_space_t2w_file

    # Warp T1w-space SPACE T1w/SPACE T2w and MPRAGE T1w/SPACE T2w ratio maps to MNI152NLin2009cAsym
    # using normalization transform from sMRIPrep.
    files = [t1w_unscaled_space_ratio_file, t1w_scaled_space_ratio_file]
    descs = ['SPACEunscaled', 'SPACEscaled']
    if has_mprage:
        files = [
            t1w_unscaled_space_ratio_file,
            t1w_unscaled_mprage_ratio_file,
            t1w_scaled_space_ratio_file,
            t1w_scaled_mprage_ratio_file,
        ]
        descs = ['SPACEunscaled', 'MPRAGEunscaled', 'SPACEscaled', 'MPRAGEscaled']

    for i_file, file_ in enumerate(files):
        desc = descs[i_file]
        mni_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym'},
            dismiss_entities=['reconstruction', 'acquisition'],
        )
        mni_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm']],
            interpolator='linear',
        )
        ants.image_write(mni_img, mni_file)

        # Plot the ratio maps
        scalar_desc = 'scalar'
        if desc:
            scalar_desc = f'{desc}{scalar_desc}'

        data = masking.apply_mask(mni_file, run_data['mni_mask'])
        vmin = np.percentile(data, 2)
        vmin = np.minimum(vmin, 0)
        vmax = np.percentile(data, 98)

        scalar_report = get_filename(
            name_source=mni_file,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'figures', 'desc': scalar_desc, 'extension': '.svg'},
        )
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
    """Run the process_t1wt2w_ratio workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    mp2rage_dir = CFG['derivatives']['pymp2rage']
    out_dir = CFG['derivatives']['t1wt2w_ratio']
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(CFG['work_dir'], 't1wt2w_ratio')
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports', 't1wt2w_ratio.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    # Write out dataset_description.json
    dataset_description_file = os.path.join(out_dir, 'dataset_description.json')
    if not os.path.isfile(dataset_description_file):
        dataset_description = {
            'Name': 'MIRROR T1w/T2w Ratio Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'DatasetLinks': {
                'raw': in_dir,
                'smriprep': smriprep_dir,
            },
            'GeneratedBy': [
                {
                    'Name': 'Custom code',
                    'Description': 'Custom Python code using ANTsPy.',
                    'CodeURL': 'https://github.com/PennLINC/MIRROR',
                }
            ],
        }
        with open(dataset_description_file, 'w') as fobj:
            json.dump(dataset_description, fobj, sort_keys=True, indent=4)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'configuration', 'bids', 'layout.json'),
        validate=False,
        derivatives=[mp2rage_dir, smriprep_dir],
    )

    if subject_id:
        subjects = [subject_id]
    else:
        subjects = layout.get_subjects(acquisition='SPACE', suffix='T2w')

    for subject_id in subjects:
        print(f'Processing subject {subject_id}')
        sessions = layout.get_sessions(subject=subject_id, acquisition='SPACE', suffix='T2w')
        for session in sessions:
            print(f'Processing session {session}')
            space_t2w_files = layout.get(
                subject=subject_id,
                session=session,
                acquisition='SPACE',
                suffix='T2w',
                extension=['.nii', '.nii.gz'],
            )
            if not space_t2w_files:
                print(f'No SPACE T2w files found for subject {subject_id} and session {session}')
                continue

            for space_t2w_file in space_t2w_files:
                entities = space_t2w_file.get_entities()
                entities.pop('acquisition')
                entities.pop('reconstruction')
                try:
                    run_data = collect_run_data(layout, entities)
                except ValueError as e:
                    print(f'Failed {space_t2w_file}')
                    print(e)
                    continue

                fname = os.path.basename(space_t2w_file.path).split('.')[0]
                run_temp_dir = os.path.join(temp_dir, fname.replace('-', '').replace('_', ''))
                os.makedirs(run_temp_dir, exist_ok=True)
                process_run(layout, run_data, out_dir, run_temp_dir)

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
