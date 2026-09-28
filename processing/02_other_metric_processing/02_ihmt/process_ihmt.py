"""Calculate derivatives for ihMTRAGE files.

Steps:

1.  Concatenate ihMTRAGE files into one 4D file (has to be right order).
3.  Run dwidenoise on concatenated data.
4.  Motion correct ihMTRAGE files using iterative motion correction algorithm.
5.  Coregister ihMTRAGE reference image to T1w image from sMRIPrep.
6.  Apply motion correction and coregistration transforms to ihMTRAGE files.
7.  Calculate subset of ihMT derivatives manually.
8.  Concatenate T1w-space ihMTRAGE files into one 4D file (has to be right order)
9.  Calculate T1w-space ihMT derivatives with ihmt_proc.
10. Warp T1w-space ihMT derivatives to MNI152NLin2009cAsym using normalization transform from
    sMRIPrep.

Notes:

- sMRIPrep's preprocessed T1w image is used as the "native T1w space".
- This must be run after sMRIPrep and process_mp2rage.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from pprint import pformat

import ants
import nibabel as nb
import numpy as np
from bids.layout import BIDSLayout, Query
from ihmt_proc import cli
from nilearn import image, masking
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
    run_command,
    run_synthstrip,
)

CFG = load_config()
CODE_DIR = CFG['code_dir']


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect required input files for ihMTRAGE processing.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths.
    """
    queries = {
        # ihMTRAGE files from raw BIDS dataset
        'm0': {
            'datatype': 'anat',
            'acquisition': 'nosat',
            'mt': 'off',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mtplus': {
            'datatype': 'anat',
            'acquisition': 'singlepos',
            'mt': 'on',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mtminus': {
            'datatype': 'anat',
            'acquisition': 'singleneg',
            'mt': 'on',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mtdual1': {
            'datatype': 'anat',
            'acquisition': 'dual1',
            'mt': 'on',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        'mtdual2': {
            'datatype': 'anat',
            'acquisition': 'dual2',
            'mt': 'on',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'ihMTRAGE',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep T1w-space B1 map from MP2RAGE derivatives
        'b1map': {
            'datatype': 'fmap',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            'suffix': 'TB1map',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep T1w-space T1 map from MP2RAGE derivatives
        't1map': {
            'datatype': 'anat',
            'reconstruction': [Query.NONE, Query.ANY],
            'space': 'T1w',
            # 'desc': 'B1corrected',
            'desc': Query.NONE,
            'suffix': 'T1map',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep T1w image
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
            'from': 'MNI152NLin2009cAsym',
            'reconstruction': [Query.NONE, Query.ANY],
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
        if len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)}: {query}')

        file = files[0]
        run_data[key] = file.path

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def process_run(layout, run_data, out_dir, temp_dir):
    name_base = os.path.basename(run_data['m0'])

    # ihMTRAGE parameters
    interpulse_delay = 100  # ms
    n_pulses_per_burst = 1
    n_bursts = 10
    burst_tr = 100  # ms
    burst_tr_final = 100  # ms, MT is unsure
    ihmt_params = f'{interpulse_delay},{n_pulses_per_burst},{n_bursts},{burst_tr},{burst_tr_final}'

    # TB1TFL parameters
    echo_spacing = 4.4  # ms
    flip_angle = 80  # deg
    n_repetitions = 1
    t_r = 9500  # ms
    tfl_params = f'{echo_spacing},{flip_angle},{n_repetitions},{t_r}'

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

    # Concatenate ihMTRAGE files into one 4D file (has to be right order)
    # nosat_mt-off, singlepos_mt-on, dual1_mt-on, singleneg_mt-on, dual2_mt-on
    filetypes = ['m0', 'mtplus', 'mtdual1', 'mtminus', 'mtdual2']
    in_files = [run_data[filetype] for filetype in filetypes]
    concat_ihmt_img = image.concat_imgs(in_files)
    concat_ihmt_file = os.path.join(temp_dir, f'concat_{name_base}')
    concat_ihmt_img.to_filename(concat_ihmt_file)

    os.makedirs(os.path.join(temp_dir, 'dwidenoise'), exist_ok=True)
    denoised_ihmt_file = os.path.join(temp_dir, 'dwidenoise', 'concat_ihmt.nii.gz')
    cmd = f'dwidenoise {concat_ihmt_file} {denoised_ihmt_file} -force --extent 3,3,3'
    run_command(cmd)

    denoised_ihmt_imgs = list(image.iter_img(denoised_ihmt_file))
    denoised_ihmt_files = []
    for i_img, img in enumerate(denoised_ihmt_imgs):
        base_filename = os.path.basename(in_files[i_img])
        denoised_ihmt_file = os.path.join(temp_dir, 'dwidenoise', base_filename)
        img.to_filename(denoised_ihmt_file)
        denoised_ihmt_files.append(denoised_ihmt_file)

    # Motion correct ihMTRAGE files
    ihmt_template, hmc_transforms, brain_mask = iterative_motion_correction(
        name_sources=in_files,
        layout=layout,
        in_files=denoised_ihmt_files,
        filetypes=filetypes,
        out_dir=out_dir,
        temp_dir=temp_dir,
    )

    # Coregister ihMTRAGE reference image to sMRIPrep T1w image
    ihmtrage_to_smriprep_xfm = coregister_to_t1(
        name_source=run_data['m0'],
        layout=layout,
        in_file=ihmt_template,
        t1_file=run_data['t1w'],
        t1_mask=run_data['t1w_mask'],
        out_dir=out_dir,
        source_space='ihMTRAGEref',
        target_space='T1w',
    )

    # Apply motion correction and coregistration to ihMTRAGE files
    ihmt_files_t1space = []
    for i_file, ihmt_file in enumerate(denoised_ihmt_files):
        in_file = in_files[i_file]
        ihmt_file = denoised_ihmt_files[i_file]
        if i_file == 0:
            # XXX: For some reason, including the identity transform here
            # led to misregistration, so we're not using it.
            transform_list = [ihmtrage_to_smriprep_xfm]
        else:
            hmc_transform = hmc_transforms[i_file]
            transform_list = [ihmtrage_to_smriprep_xfm, hmc_transform]

        ihmt_img = ants.image_read(ihmt_file)
        ihmt_img_t1space = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w']),
            moving=ihmt_img,
            transformlist=transform_list,
            interpolator='linear',
        )
        ihmt_file_t1space = get_filename(
            name_source=in_file,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w'},
        )
        ants.image_write(ihmt_img_t1space, ihmt_file_t1space)
        ihmt_files_t1space.append(ihmt_file_t1space)

    # Create brain mask in T1w space
    brain_mask_t1space = get_filename(
        name_source=brain_mask,
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'desc': 'brain', 'suffix': 'mask'},
    )
    brain_mask_t1w_img = ants.apply_transforms(
        fixed=ants.image_read(run_data['t1w']),
        moving=ants.image_read(brain_mask),
        transformlist=[ihmtrage_to_smriprep_xfm],
        interpolator='nearestNeighbor',
    )
    ants.image_write(brain_mask_t1w_img, brain_mask_t1space)

    # Calculate ihMTw
    ihmtw_img = image.math_img(
        'mtplus + mtminus - (mtdual1 + mtdual2)',
        mtplus=ihmt_files_t1space[1],
        mtminus=ihmt_files_t1space[3],
        mtdual1=ihmt_files_t1space[2],
        mtdual2=ihmt_files_t1space[4],
    )
    ihmtw_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'ihMTw'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ihmtw_img.to_filename(ihmtw_file)

    # Calculate ihMTR
    ihmtr_img = image.math_img(
        '(ihmt / m0) * mask',
        ihmt=ihmtw_img,
        m0=ihmt_files_t1space[0],
        mask=brain_mask_t1space,
    )
    ihmtr_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'ihMTR'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ihmtr_img.to_filename(ihmtr_file)

    # Calculate MTR
    mtr_img = image.math_img(
        '(1 - mtplus / m0) * mask',
        mtplus=ihmt_files_t1space[1],
        m0=ihmt_files_t1space[0],
        mask=brain_mask_t1space,
    )
    mtr_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MTRmap'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtr_img.to_filename(mtr_file)

    # Concatenate ihMTRAGE files into one 4D file (has to be right order)
    # nosat_mt-off, singlepos_mt-on, dual1_mt-on, singleneg_mt-on, dual2_mt-on
    concat_ihmt_t1space = os.path.join(temp_dir, f'concat_t1space_{name_base}')
    concat_ihmt_img = image.concat_imgs(ihmt_files_t1space)
    concat_ihmt_img = image.math_img(
        'img * mask[..., None]',
        img=concat_ihmt_img,
        mask=brain_mask_t1space,
    )
    concat_ihmt_img.to_filename(concat_ihmt_t1space)

    # Run ihmt_proc to calculate T1w-space ihMT derivatives
    ihmtsat_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'ihMTsat'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtdsat_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MTdsat'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtssat_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MTssat'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ihmtsatb1sq_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'ihMTsatB1sq'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtdsatb1sq_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MTdsatB1sq'},
        dismiss_entities=['acquisition', 'mt'],
    )
    mtssatb1sq_file = get_filename(
        name_source=run_data['m0'],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'T1w', 'suffix': 'MTssatB1sq'},
        dismiss_entities=['acquisition', 'mt'],
    )

    # PROVENANCE ONLY: do not use MTssat or MTssatB1sq from this workflow.
    # The historical modified ihmt_proc API incorrectly populated both from
    # MTdsat. They remain outputs to preserve the manuscript-era processing
    # record but are excluded from configuration/metrics.yml. See issue #22:
    # https://github.com/PennLINC/mirror/issues/22

    cli.main(
        ihmt=concat_ihmt_t1space,
        mask=brain_mask_t1space,
        t1=run_data['t1map'],
        ihmtparx=ihmt_params,
        tflparx=tfl_params,
        b1=run_data['b1map'],
        ihmtsat_file=ihmtsat_file,
        mtdsat_file=mtdsat_file,
        mtssat_file=mtssat_file,
        ihmtsatb1sq_file=ihmtsatb1sq_file,
        mtdsatb1sq_file=mtdsatb1sq_file,
        mtssatb1sq_file=mtssatb1sq_file,
    )

    # Warp T1w-space ihMT derivatives to MNI152NLin2009cAsym using normalization transform from
    # sMRIPrep
    for file_ in [
        ihmtw_file,
        ihmtr_file,
        mtr_file,
        ihmtsat_file,
        mtdsat_file,
        mtssat_file,
        ihmtsatb1sq_file,
        mtdsatb1sq_file,
        mtssatb1sq_file,
    ]:
        suffix = os.path.basename(file_).split('_')[-1].split('.')[0]
        mni_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym'},
        )
        reg_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm']],
            interpolator='linear',
        )
        ants.image_write(reg_img, mni_file)

        # Get vmin (2nd percentile) and vmax (98th percentile) from image
        reg_data = masking.apply_mask(mni_file, run_data['mni_mask'])
        vmin = np.percentile(reg_data, 2)
        vmin = np.minimum(vmin, 0)
        vmax = np.percentile(reg_data, 98)

        scalar_report = get_filename(
            name_source=mni_file,
            layout=layout,
            out_dir=out_dir,
            entities={'datatype': 'figures', 'desc': 'scalar', 'extension': '.svg'},
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

        if suffix == 'ihMTw':
            plot_coregistration(
                name_source=file_,
                layout=layout,
                in_file=file_,
                t1_file=run_data['t1w'],
                out_dir=out_dir,
                source_space='ihMTRAGEref',
                target_space='T1w',
                wm_seg=wm_seg_t1w_file,
            )
            plot_coregistration(
                name_source=mni_file,
                layout=layout,
                in_file=mni_file,
                t1_file=run_data['t1w_mni'],
                out_dir=out_dir,
                source_space='ihMTRAGEref',
                target_space='MNI152NLin2009cAsym',
                wm_seg=wm_seg_file,
            )


def iterative_motion_correction(name_sources, layout, in_files, filetypes, out_dir, temp_dir):
    """Apply iterative motion correction to a list of images.

    This method is based on the method described in https://doi.org/10.1101/2020.09.11.292649.

    Parameters
    ----------
    name_sources : list of str
        List of names of the source files to use for output file names.
    layout : BIDSLayout
        BIDSLayout object.
    in_files : list of str
        List of input image files.
    filetypes : list of str
        List of filetypes of the input images.
    out_dir : str
        Directory to write output files.
    temp_dir : str
        Directory to write temporary files.

    Returns
    -------
    template_file : str
        Path to the template image in ihMTRAGEref space.
    transforms : list of str
        List of transform files.
    brain_mask_file : str
        Brain mask from sum image.
    """
    # Step 1: Skull-strip each image.
    skullstripped_files = []
    brain_masks = []
    n4_dir = os.path.join(temp_dir, 'n4')
    os.makedirs(n4_dir, exist_ok=True)
    skullstripped_dir = os.path.join(temp_dir, 'skullstripped')
    os.makedirs(skullstripped_dir, exist_ok=True)
    for i_file, in_file in enumerate(in_files):
        print(f'Skull-stripping {in_file}')
        # Bias field correction
        in_img = ants.image_read(in_file)
        n4_img = ants.n4_bias_field_correction(in_img)
        n4_file = os.path.join(n4_dir, os.path.basename(in_file))
        ants.image_write(n4_img, n4_file)

        # Step 1a: Create a brain mask with SynthStrip.
        brain_mask = get_filename(
            name_source=in_file,
            layout=layout,
            out_dir=out_dir,
            entities={'desc': 'brain', 'suffix': 'mask'},
        )
        print(f'Creating brain mask: {brain_mask}')
        skullstripped_file = os.path.join(skullstripped_dir, os.path.basename(in_file))
        run_synthstrip(
            in_file=n4_file,
            out_file=skullstripped_file,
            mask_file=brain_mask,
            cfg=CFG,
        )

        skullstripped_files.append(skullstripped_file)
        brain_masks.append(brain_mask)

    # Step 2: Define template image, then register each image to the template.
    # Update the template image with the registered images.
    transforms = []
    for i_file, brain_n4_file in enumerate(skullstripped_files):
        in_file = in_files[i_file]
        filetype = filetypes[i_file]
        if i_file == 0:
            template_img = ants.image_read(brain_n4_file)
            transform = os.path.join(
                CODE_DIR,
                'data',
                'resources',
                'itkIdentityTransform.txt',
            )
        else:
            reg = ants.registration(
                fixed=template_img,
                moving=ants.image_read(brain_n4_file),
                type_of_transform='Rigid',
            )
            transform = reg['fwdtransforms'][0]
            warped_img = reg['warpedmovout']

            # The updated template is the mean of the template and the registered image.
            template_img = (template_img + warped_img) / 2

        # Step 2a: Write out individual transforms to ihMTRAGEref space.
        transform_file = get_filename(
            name_source=name_sources[i_file],
            layout=layout,
            out_dir=out_dir,
            entities={
                'from': filetype,
                'to': 'ihMTRAGEref',
                'mode': 'image',
                'suffix': 'xfm',
                'extension': 'txt' if transform.endswith('.txt') else 'mat',
            },
            dismiss_entities=['acquisition', 'mt'],
        )
        shutil.copyfile(transform, transform_file)
        transforms.append(transform_file)

    # Step 3: Write out template image as ihMTRAGEref.nii.gz
    template_file = get_filename(
        name_source=name_sources[0],
        layout=layout,
        out_dir=out_dir,
        entities={'suffix': 'ihMTRAGEref'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ants.image_write(template_img, template_file)

    # Step 4: Apply transforms to original images.
    for i_file, in_file in enumerate(in_files):
        transform_file = transforms[i_file]
        out_file = get_filename(
            name_source=name_sources[i_file],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'ihMTRAGEref'},
        )
        out_img = ants.apply_transforms(
            fixed=ants.image_read(template_file),
            moving=ants.image_read(in_file),
            transformlist=[transform_file],
            interpolator='linear',
        )
        ants.image_write(out_img, out_file)

        plot_coregistration(
            name_source=out_file,
            layout=layout,
            in_file=out_file,
            t1_file=template_file,
            out_dir=out_dir,
            source_space=filetypes[i_file],
            target_space='ihMTRAGEref',
        )

        # Also apply transform to brain masks
        brain_mask = brain_masks[i_file]
        out_brain_mask = get_filename(
            name_source=name_sources[i_file],
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'ihMTRAGEref', 'desc': 'brain', 'suffix': 'mask'},
        )
        reg_img = ants.apply_transforms(
            fixed=ants.image_read(template_file),
            moving=ants.image_read(brain_mask),
            transformlist=[transform_file],
            interpolator='nearestNeighbor',
        )
        ants.image_write(reg_img, out_brain_mask)

        if i_file == 0:
            sum_mask_img = reg_img
        else:
            sum_mask_img = sum_mask_img + reg_img

    # Step 5: Create sum image
    sum_mask = get_filename(
        name_source=name_sources[0],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'ihMTRAGEref', 'desc': 'sum', 'suffix': 'mask'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ants.image_write(sum_mask_img, sum_mask)

    # Step 6: Create mask from sum image
    brain_mask_img = sum_mask_img > 4
    brain_mask_file = get_filename(
        name_source=name_sources[0],
        layout=layout,
        out_dir=out_dir,
        entities={'space': 'ihMTRAGEref', 'desc': 'brain', 'suffix': 'mask'},
        dismiss_entities=['acquisition', 'mt'],
    )
    ants.image_write(brain_mask_img, brain_mask_file)

    return template_file, transforms, brain_mask_file


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
    """Run the process_ihmt workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = CFG['bids_dir']
    mp2rage_dir = CFG['derivatives']['pymp2rage']
    smriprep_dir = CFG['derivatives']['smriprep']
    out_dir = CFG['derivatives']['ihmt']
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(CFG['work_dir'], 'ihmt')
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports', 'ihmt.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    # Write out dataset_description.json
    dataset_description_file = os.path.join(out_dir, 'dataset_description.json')
    if not os.path.isfile(dataset_description_file):
        dataset_description = {
            'Name': 'MIRROR ihMT Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'DatasetLinks': {
                'raw': in_dir,
                'mp2rage': mp2rage_dir,
                'smriprep': smriprep_dir,
            },
            'GeneratedBy': [
                {
                    'Name': 'Custom code',
                    'Description': 'Custom Python code to calculate ihMTw and MTR.',
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
        subjects = layout.get_subjects(suffix='MP2RAGE')

    for subject_id in subjects:
        print(f'Processing subject {subject_id}')
        sessions = layout.get_sessions(subject=subject_id, suffix='ihMTRAGE')
        for session in sessions:
            print(f'Processing session {session}')
            m0_files = layout.get(
                subject=subject_id,
                session=session,
                acquisition='nosat',
                mt='off',
                suffix='ihMTRAGE',
                extension=['.nii', '.nii.gz'],
            )
            if not m0_files:
                print(f'No m0 files found for subject {subject_id} and session {session}')
                continue

            for m0_file in m0_files:
                entities = m0_file.get_entities()
                entities.pop('acquisition')
                entities.pop('mt')
                try:
                    run_data = collect_run_data(layout, entities)
                except ValueError as e:
                    print(f'Failed {m0_file}')
                    print(e)
                    continue
                fname = os.path.basename(m0_file.path).split('.')[0]
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
