"""Process QSM data.

Steps:

1.  Rename chi-separation QSM outputs to BIDS-compliant filenames.
2.  Warp QSM derivatives to T1w and MNI152NLin2009cAsym spaces.
3.  Generate scalar reports for QSM derivatives.

Notes:

- This must be run after sMRIPrep, process_mese.py, and process_qsm.py.
- Every SEPIA/chi-separation input is optional, so a run whose R2' inputs were
  missing (only the ``chisep+r2s`` and ``chisep+r2primenet`` variants ran in
  process_qsm.py) is handled by skipping the absent ``chisep+r2p`` maps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
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
    get_filename,
    load_config,
    plot_scalar_map,
)

CFG = load_config()
CODE_DIR = CFG['code_dir']


def rename_qsm_outputs(subject_id: str, session: str) -> None:
    """Rename local chi-separation outputs to BIDS-compliant filenames.

    Reads the chi-separation NIfTIs that process_qsm.py writes to per-combination
    working directories and writes compressed, BIDS-named files to the QSM
    derivatives directory. SEPIA Chimaps are written directly to the derivatives
    by process_qsm.py and are not handled here.

    Variants that process_qsm.py did not run (e.g. ``chisep+r2p`` when no R2'
    input was available) have no working directory and are skipped.

    Parameters
    ----------
    subject_id : str
        BIDS subject label (without 'sub-' prefix).
    session : str
        BIDS session label (without 'ses-' prefix).
    """
    work_dir = CFG['work_dir']
    out_dir = CFG['derivatives']['qsm']
    ses_out_dir = os.path.join(out_dir, f'sub-{subject_id}', f'ses-{session}', 'anat')
    os.makedirs(ses_out_dir, exist_ok=True)

    suffix_map = {
        'paramagnetic': 'para',
        'diamagnetic': 'dia',
        'total': 'Chimap',
        'r2prime': 'R2primemap',
        'r2s': 'R2starmap',
    }

    # process_qsm.py writes one working directory per echo set and R2 variant:
    #   work_dir/qsm-<version>+chisep+<map_label>/sub-*/ses-*/anat/
    #   sub-*_ses-*_<contrast>_<map_label>.nii
    chisep_variant_dirs = [
        (
            f'{version}+chisep+{map_label}',
            os.path.join(
                work_dir,
                f'qsm-{version}+chisep+{map_label}',
                f'sub-{subject_id}',
                f'ses-{session}',
                'anat',
            ),
            map_label,
        )
        for version in ('E12345', 'E2345')
        for map_label in ('r2p', 'r2primenet', 'r2s')
    ]

    for desc, variant_dir, map_label in chisep_variant_dirs:
        if not os.path.isdir(variant_dir):
            print(f'Chi-sep output directory not found: {variant_dir}')
            continue
        for contrast, bids_suffix in suffix_map.items():
            in_file = os.path.join(
                variant_dir,
                f'sub-{subject_id}_ses-{session}_{contrast}_{map_label}.nii',
            )
            out_file = os.path.join(
                ses_out_dir,
                f'sub-{subject_id}_ses-{session}_run-01_space-MEGRE_desc-{desc}_{bids_suffix}.nii.gz',
            )
            if not os.path.isfile(in_file):
                print(f'Chi-sep output not found: {in_file}')
                continue
            nb.load(in_file).to_filename(out_file)


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect SEPIA chi maps, GRE images, and masks for QSM post-processing.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths. All SEPIA,
        chi-separation, and process_megre reference maps are optional and map to
        ``None`` when absent, so echo sets whose chi-separation variants did not
        run (e.g. ``chisep+r2p`` without an R2' input) are simply skipped.
    """
    queries = {
        # SEPIA Chimap
        'sepia_chimap_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+sepia',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'sepia_chimap_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+sepia',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep chi map with R2'
        'chisep_chimap_r2p_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2p',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_chimap_r2p_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2p',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep iron map with R2'
        'chisep_iron_r2p_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2p',
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_iron_r2p_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2p',
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2'
        'chisep_myelin_r2p_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2p',
            'suffix': 'dia',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_myelin_r2p_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2p',
            'suffix': 'dia',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep chi map with R2'
        'chisep_chimap_r2primenet_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2primenet',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_chimap_r2primenet_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2primenet',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep iron map with R2'
        'chisep_iron_r2primenet_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2primenet',
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_iron_r2primenet_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2primenet',
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2'
        'chisep_myelin_r2primenet_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2primenet',
            'suffix': 'dia',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_myelin_r2primenet_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2primenet',
            'suffix': 'dia',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep chi map with R2*
        'chisep_chimap_r2s_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2s',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_chimap_r2s_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2s',
            'suffix': 'Chimap',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep iron map with R2*
        'chisep_iron_r2s_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2s',
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_iron_r2s_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2s',
            'suffix': 'para',
            'extension': ['.nii', '.nii.gz'],
        },
        # Chisep myelin map with R2*
        'chisep_myelin_r2s_e12345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E12345+chisep+r2s',
            'suffix': 'dia',
            'extension': ['.nii', '.nii.gz'],
        },
        'chisep_myelin_r2s_e2345': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'space': 'MEGRE',
            'res': Query.NONE,
            'desc': 'E2345+chisep+r2s',
            'suffix': 'dia',
            'extension': ['.nii', '.nii.gz'],
        },
        # Coregistration transform from process_megre.py
        'megre2t1w_xfm': {
            'datatype': 'anat',
            'run': [Query.NONE, Query.ANY],
            'from': 'MEGRE',
            'to': 'T1w',
            'mode': 'image',
            'suffix': 'xfm',
            'extension': '.mat',
        },
        # T1w-space T1w image from sMRIPrep
        't1w': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': Query.NONE,
            'desc': 'preproc',
            'suffix': 'T1w',
            'extension': ['.nii', '.nii.gz'],
        },
        # MNI-space T1w image from sMRIPrep
        't1w_mni': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
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
            'space': 'MNI152NLin2009cAsym',
            'suffix': 'dseg',
            'extension': ['.nii', '.nii.gz'],
        },
        # sMRIPrep MNI-space brain mask
        'mni_mask': {
            'datatype': 'anat',
            'session': [Query.NONE, Query.ANY],
            'run': [Query.NONE, Query.ANY],
            'space': 'MNI152NLin2009cAsym',
            'desc': 'brain',
            'suffix': 'mask',
            'extension': ['.nii', '.nii.gz'],
        },
    }

    # Chi-sep R2*/R2' maps (renamed above) and the process_megre reference
    # R2*/R2' maps, per echo set and R2 variant, plus the MEGRE-space brain mask.
    # Used for the scalar plots and the head-to-head comparisons; all optional.
    for version in ('E12345', 'E2345'):
        vl = version.lower()
        for ref_suffix in ('R2starmap', 'R2primemap'):
            queries[f'megreref_{ref_suffix.lower()}_{vl}'] = {
                'datatype': 'anat',
                'run': [Query.NONE, Query.ANY],
                'space': 'MEGRE',
                'res': Query.NONE,
                'desc': f'MEGRE+{version}',
                'suffix': ref_suffix,
                'extension': ['.nii', '.nii.gz'],
            }
            for map_label in ('r2p', 'r2primenet', 'r2s'):
                queries[f'chisep_{ref_suffix.lower()}_{map_label}_{vl}'] = {
                    'datatype': 'anat',
                    'run': [Query.NONE, Query.ANY],
                    'space': 'MEGRE',
                    'res': Query.NONE,
                    'desc': f'{version}+chisep+{map_label}',
                    'suffix': ref_suffix,
                    'extension': ['.nii', '.nii.gz'],
                }

    optional_prefixes = ('chisep_', 'sepia_', 'megreref_')
    run_data = {}
    for key, query in queries.items():
        query = {**bids_filters, **query}
        files = layout.get(**query)
        optional = key.startswith(optional_prefixes)
        if optional and len(files) == 0:
            print(f'No files found for {key} with query {query}')
            run_data[key] = None
            continue
        elif len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')

        file = files[0]
        run_data[key] = file.path

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def process_run(layout, run_data, out_dir):
    """Process a single run of QSM data.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary containing the paths to the QSM data.
    out_dir : str
        Path to the output directory.
    """
    coreg_transform = run_data['megre2t1w_xfm']

    # Warp MEGREref-space QSM outputs to MNI152NLin2009cAsym using normalization
    # transform from sMRIPrep and coregistration transform to sMRIPrep's T1w space.
    keys = [
        # SEPIA
        'sepia_chimap_e12345',
        'sepia_chimap_e2345',
        # Chi-separation with input R2' map
        'chisep_chimap_r2p_e12345',
        'chisep_chimap_r2p_e2345',
        'chisep_iron_r2p_e12345',
        'chisep_iron_r2p_e2345',
        'chisep_myelin_r2p_e12345',
        'chisep_myelin_r2p_e2345',
        # Chi-separation with R2' map from r2primenet
        'chisep_chimap_r2primenet_e12345',
        'chisep_chimap_r2primenet_e2345',
        'chisep_iron_r2primenet_e12345',
        'chisep_iron_r2primenet_e2345',
        'chisep_myelin_r2primenet_e12345',
        'chisep_myelin_r2primenet_e2345',
        # Chi-separation with R2* map from ARLO
        'chisep_chimap_r2s_e12345',
        'chisep_chimap_r2s_e2345',
        'chisep_iron_r2s_e12345',
        'chisep_iron_r2s_e2345',
        'chisep_myelin_r2s_e12345',
        'chisep_myelin_r2s_e2345',
    ]
    # Also warp and scalar-plot the chi-sep R2* and R2' maps.
    for _version in ('E12345', 'E2345'):
        _vl = _version.lower()
        for _map_label in ('r2p', 'r2primenet', 'r2s'):
            keys.append(f'chisep_r2starmap_{_map_label}_{_vl}')
            keys.append(f'chisep_r2primemap_{_map_label}_{_vl}')

    for key in keys:
        file_ = run_data[key]
        if file_ is None:
            continue

        # Warp to T1w space
        t1w_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'T1w'},
            dismiss_entities=['echo', 'part'],
        )
        reg_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w']),
            moving=ants.image_read(file_),
            transformlist=[coreg_transform],
            interpolator='linear',
        )
        ants.image_write(reg_img, t1w_file)

        # Warp to MNI152NLin2009cAsym
        mni_file = get_filename(
            name_source=file_,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MNI152NLin2009cAsym'},
            dismiss_entities=['echo', 'part'],
        )
        reg_img = ants.apply_transforms(
            fixed=ants.image_read(run_data['t1w_mni']),
            moving=ants.image_read(file_),
            transformlist=[run_data['t1w2mni_xfm'], coreg_transform],
            interpolator='linear',
        )
        ants.image_write(reg_img, mni_file)

        # Plot scalar map. nireports indexes figures with its own config, whose
        # desc pattern only captures [a-zA-Z0-9], so a '+' in the desc truncates
        # it on indexing (e.g. 'E12345+chisep+r2pscalar' -> 'E12345') and the
        # report's '.*scalar' query never matches. Strip '+' so the 'scalar'
        # token survives parsing and the reportlet is picked up.
        desc = 'scalar'
        if 'desc-' in mni_file:
            # Append the desc to the target desc
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
    """Run the process_qsm_post workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def _get_sessions_to_rename(subject_id: str) -> list[str]:
    """Find sessions with chi-separation work directories for a subject.

    process_qsm.py writes per-combination working directories at
    work_dir/qsm-<version>+chisep+<map_label>/sub-<id>/ses-<session>/. This
    enumerates the ses-* directories there so the chi-sep outputs can be
    renamed before the processing layout is built.

    Parameters
    ----------
    subject_id : str
        BIDS subject label (without 'sub-' prefix).

    Returns
    -------
    list of str
        Sorted session labels (without 'ses-' prefix) that have chi-sep outputs.
    """
    session_search_patterns = [
        os.path.join(
            CFG['work_dir'],
            f'qsm-{version}+chisep+{map_label}',
            f'sub-{subject_id}',
            'ses-*',
        )
        for version in ('E12345', 'E2345')
        for map_label in ('r2p', 'r2primenet', 'r2s')
    ]
    print(f'searching: {session_search_patterns}', flush=True)

    sessions_to_rename = set()
    for search_pattern in session_search_patterns:
        sessions_to_rename.update(
            os.path.basename(d).removeprefix('ses-')
            for d in glob(search_pattern)
            if os.path.isdir(d)
        )
    return sorted(sessions_to_rename)


def process_subject(layout, subject_id: str, out_dir: str, bootstrap_file: str) -> None:
    """Warp QSM derivatives and build reports for a single subject.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout indexing the dataset and derivatives (built after renaming).
    subject_id : str
        BIDS subject label (without 'sub-' prefix).
    out_dir : str
        Path to the QSM derivatives output directory.
    bootstrap_file : str
        Path to the nireports bootstrap YAML used to assemble the report.
    """
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
        if not megre_files:
            print(f'No MEGRE files found for subject {subject_id} and session {session}')
            continue

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


def main(subject_id=None):
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    mese_dir = CFG['derivatives']['mese']
    megre_dir = CFG['derivatives']['megre']
    out_dir = CFG['derivatives']['qsm']
    os.makedirs(out_dir, exist_ok=True)
    temp_dir = os.path.join(CFG['work_dir'], 'qsm')
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports', 'qsm.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    bids_config = os.path.join(CODE_DIR, 'configuration', 'bids', 'layout.json')

    # Write out dataset_description.json
    dataset_description_file = os.path.join(out_dir, 'dataset_description.json')
    if not os.path.isfile(dataset_description_file):
        dataset_description = {
            'Name': 'MIRROR QSM Derivatives',
            'BIDSVersion': '1.10.0',
            'DatasetType': 'derivative',
            'DatasetLinks': {
                'raw': in_dir,
                'smriprep': smriprep_dir,
                'mese': mese_dir,
                'megre': megre_dir,
            },
            'GeneratedBy': [
                {
                    'Name': 'Custom code',
                    'Description': 'Custom Python code.',
                    'CodeURL': 'https://github.com/PennLINC/MIRROR',
                }
            ],
        }
        with open(dataset_description_file, 'w') as fobj:
            json.dump(dataset_description, fobj, sort_keys=True, indent=4)

    if subject_id:
        subjects = [subject_id]
    else:
        # MEGRE subjects come from the raw dataset, so a lightweight layout
        # (no derivatives) is enough to enumerate them.
        subjects = BIDSLayout(
            in_dir,
            config=bids_config,
            validate=False,
        ).get_subjects(suffix='MEGRE')

    print(f'Processing subjects: {subjects}', flush=True)

    # Rename chi-separation outputs first, for every subject/session, so they
    # are indexed when the processing layout is built below.
    for subject_id in subjects:
        sessions_to_rename = _get_sessions_to_rename(subject_id)
        print(f'sessions to rename for sub-{subject_id}: {sessions_to_rename}', flush=True)
        for session in sessions_to_rename:
            print(f'Renaming QSM outputs for sub-{subject_id} ses-{session}')
            rename_qsm_outputs(subject_id, session)

    layout = BIDSLayout(
        in_dir,
        config=bids_config,
        validate=False,
        derivatives=[smriprep_dir, megre_dir, out_dir],
    )

    for subject_id in subjects:
        process_subject(layout, subject_id, out_dir, bootstrap_file)

    print('DONE!')


if __name__ == '__main__':
    _main()
