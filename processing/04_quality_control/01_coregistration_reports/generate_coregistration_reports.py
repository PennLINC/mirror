"""Create coregistration plots showing ses-01/ses-02 T1w coregistration for various derivatives."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import ants
from bids.layout import BIDSLayout, Query
from nireports.assembler.report import Report

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.processing import (  # noqa: E402
    get_filename,
    load_config,
    plot_coregistration,
)

CFG = load_config()
CODE_DIR = CFG['code_dir']


def get_tissue_segmentation(layout: object, subject_id: str) -> str | None:
    """Find a subject's sMRIPrep tissue segmentation in native T1w space.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the sMRIPrep derivatives.
    subject_id : str
        Subject label, without the ``sub-`` prefix.

    Returns
    -------
    dseg : str or None
        Path to the segmentation, or None if it could not be resolved
        unambiguously.

    Notes
    -----
    sMRIPrep writes a single subject-level segmentation on the same grid as the
    preprocessed T1w image, which is the grid every session's derivatives are
    resampled onto, so one segmentation applies to both sessions.
    """
    dseg_files = layout.get(
        subject=subject_id,
        datatype='anat',
        session=[Query.NONE, Query.ANY],
        run=[Query.NONE, Query.ANY],
        space=Query.NONE,
        desc=Query.NONE,
        suffix='dseg',
        extension=['.nii', '.nii.gz'],
    )
    if len(dseg_files) != 1:
        print(
            f'Expected 1 T1w-space dseg for subject {subject_id}, got {len(dseg_files)}. '
            'Coregistration plots will omit the tissue overlay.'
        )
        return None

    return dseg_files[0].path


def get_preproc_t1w(layout: object, subject_id: str) -> str | None:
    """Find a subject's sMRIPrep preprocessed T1w image in native T1w space.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the sMRIPrep derivatives.
    subject_id : str
        Subject label, without the ``sub-`` prefix.

    Returns
    -------
    t1w : str or None
        Path to the preprocessed T1w image, or None if it could not be resolved
        unambiguously. This is the grid every T1w-space derivative sits on, so it
        is the resampling target for :func:`warp_megre_reference`.
    """
    t1w_files = layout.get(
        subject=subject_id,
        datatype='anat',
        session=[Query.NONE, Query.ANY],
        run=[Query.NONE, Query.ANY],
        reconstruction=[Query.NONE, Query.ANY],
        space=Query.NONE,
        res=Query.NONE,
        desc='preproc',
        suffix='T1w',
        extension=['.nii', '.nii.gz'],
    )
    if len(t1w_files) != 1:
        print(f'Expected 1 preprocessed T1w for subject {subject_id}, got {len(t1w_files)}.')
        return None

    return t1w_files[0].path


def warp_megre_reference(
    layout: object,
    subject_id: str,
    session: str,
    t1w_file: str,
    work_dir: str,
) -> list[str]:
    """Warp the un-skull-stripped MEGRE reference into T1w space.

    process_megre.py only writes the *skull-stripped* reference (``desc-rmsbrain``)
    in T1w space; the un-stripped RMS reference (``desc-rms``) stays in MEGRE
    space. The intersession plots want the un-stripped image, so it is resampled
    here with the MEGRE-to-T1w transform process_megre.py already estimated.

    The result goes to ``work_dir``, never to the report output directory:
    nireports indexes the whole reportlets directory and derives its section
    ordering from every entity it finds there, so a stray ``acq-QSM``/``run-01``
    NIfTI beside the figures makes it split the section on acquisition and run,
    match no figure, and emit an empty report.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the MEGRE derivatives.
    subject_id, session : str
        BIDS subject and session labels, without their prefixes.
    t1w_file : str
        Preprocessed T1w image defining the output grid.
    work_dir : str
        Scratch directory the warped image is written to.

    Returns
    -------
    list of str
        Single-element list holding the T1w-space image, or an empty list when
        the reference or the transform could not be resolved unambiguously. The
        list shape matches ``layout.get`` so callers can treat both alike.
    """
    rms_files = layout.get(
        subject=subject_id,
        session=session,
        datatype='anat',
        run=[Query.NONE, Query.ANY],
        space='MEGRE',
        desc='rms',
        suffix='MEGRE',
        extension=['.nii', '.nii.gz'],
    )
    if len(rms_files) != 1:
        print(
            f'Expected 1 MEGRE-space RMS reference for subject {subject_id} and session '
            f'{session}, got {len(rms_files)}.'
        )
        return []

    xfm_files = layout.get(
        subject=subject_id,
        session=session,
        datatype='anat',
        run=[Query.NONE, Query.ANY],
        **{'from': 'MEGRE', 'to': 'T1w'},
        mode='image',
        suffix='xfm',
        extension='.mat',
    )
    if len(xfm_files) != 1:
        print(
            f'Expected 1 MEGRE-to-T1w transform for subject {subject_id} and session '
            f'{session}, got {len(xfm_files)}.'
        )
        return []

    rms_file = rms_files[0].path
    out_file = get_filename(
        name_source=rms_file,
        layout=layout,
        out_dir=work_dir,
        entities={'space': 'T1w', 'desc': 'rms', 'suffix': 'MEGRE'},
    )
    if os.path.isfile(out_file):
        return [out_file]

    warped_img = ants.apply_transforms(
        fixed=ants.image_read(t1w_file),
        moving=ants.image_read(rms_file),
        transformlist=[xfm_files[0].path],
        interpolator='linear',
    )
    ants.image_write(warped_img, out_file)
    return [out_file]


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
    """Run the coregistration report workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = CFG['bids_dir']
    out_dir = os.path.join(
        CFG['output_derivatives_dir'],
        'quality_control',
        'coregistration_reports',
    )
    os.makedirs(out_dir, exist_ok=True)
    # Intermediate images must stay out of out_dir: nireports indexes it to build
    # the report, and any non-figure file there skews its section ordering.
    temp_dir = os.path.join(CFG['work_dir'], 'coregistration_reports')
    os.makedirs(temp_dir, exist_ok=True)

    bootstrap_file = os.path.join(CODE_DIR, 'configuration', 'reports', 'coregistration.yml')
    assert os.path.isfile(bootstrap_file), f'Bootstrap file {bootstrap_file} not found'

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'configuration', 'bids', 'layout.json'),
        validate=False,
        derivatives=[
            CFG['derivatives']['smriprep'],
            CFG['derivatives']['pymp2rage'],
            CFG['derivatives']['ihmt'],
            CFG['derivatives']['mese'],
            CFG['derivatives']['megre'],
            CFG['derivatives']['t1wt2w_ratio'],
            # CFG['derivatives']['g_ratio'],
            # CFG['derivatives']['qsiprep'],
        ],
    )

    # Define images to look for
    query_dict = {
        'mp2rage': {
            'space': 'T1w',
            'desc': 'B1correctedbrain',
            'suffix': 'UNIT1',
            'extension': '.nii.gz',
        },
        'ihmt': {
            'acquisition': 'nosat',
            'mt': 'off',
            'space': 'T1w',
            'suffix': 'ihMTRAGE',
            'extension': '.nii.gz',
        },
        'mese': {
            'space': 'T1w',
            'suffix': 'MESEref',
            'extension': '.nii.gz',
        },
        # The un-stripped MEGRE reference is not written in T1w space by
        # process_megre.py, so warp_megre_reference() produces it below rather than
        # querying the layout. The entries here describe that output.
        'megre': {
            'space': 'T1w',
            'desc': 'rms',
            'suffix': 'MEGRE',
            'extension': '.nii.gz',
        },
        # The T1w/T2w outputs come in unscaled and scaled (desc-MPRAGEscaled /
        # desc-SPACEscaled) flavors, so desc must be pinned to NONE to select the
        # unscaled image. Modality keys must stay alphanumeric: they end up in the
        # reportlet's desc entity, and a '_' truncates it when the name is parsed.
        't1wt2wmprage': {
            'space': 'T1w',
            'acquisition': 'MPRAGE',
            'desc': Query.NONE,
            'suffix': 'T1w',
            'extension': '.nii.gz',
        },
        't1wt2wspacet1w': {
            'space': 'T1w',
            'acquisition': 'SPACE',
            'desc': Query.NONE,
            'suffix': 'T1w',
            'extension': '.nii.gz',
        },
        't1wt2wspacet2w': {
            'space': 'T1w',
            'acquisition': 'SPACE',
            'desc': Query.NONE,
            'suffix': 'T2w',
            'extension': '.nii.gz',
        },
    }

    if subject_id:
        subjects = [subject_id]
    else:
        subjects = layout.get_subjects()

    for subject_id in subjects:
        print(f'Processing subject {subject_id}')
        dseg = get_tissue_segmentation(layout, subject_id)
        t1w = get_preproc_t1w(layout, subject_id)
        for modality, query in query_dict.items():
            if modality == 'megre' and t1w is None:
                print(f'No preprocessed T1w for subject {subject_id}; skipping {modality}')
                continue

            files = []
            for session in ['01', '02']:
                if modality == 'megre':
                    session_file = warp_megre_reference(
                        layout=layout,
                        subject_id=subject_id,
                        session=session,
                        t1w_file=t1w,
                        work_dir=temp_dir,
                    )
                else:
                    session_file = [
                        f.path
                        for f in layout.get(
                            subject=subject_id,
                            session=session,
                            **query,
                        )
                    ]

                if not session_file:
                    print(f'No files found for subject {subject_id} and session {session}: {query}')
                    continue
                elif len(session_file) > 1:
                    print(
                        f'Multiple files found for subject {subject_id} and session {session}: {query}'
                    )
                    continue

                files.append(session_file[0])

            if len(files) != 2:
                # At least one session not present, no need to plot
                print(f'At least one session missing for subject {subject_id}')
                continue

            out_file = get_filename(
                name_source=f'sub-{subject_id}/anat/sub-{subject_id}_space-T1w_desc-{modality}_{query["suffix"]}.nii.gz',
                layout=layout,
                out_dir=out_dir,
                entities={'space': 'T1w'},
            )
            plot_coregistration(
                name_source=out_file,
                layout=layout,
                in_file=files[1],
                t1_file=files[0],
                out_dir=out_dir,
                # plot_coregistration labels t1_file (the fixed panel) with
                # target_space and in_file (the moving panel) with source_space,
                # so ses-01 is the target here and ses-02 the source.
                source_space='ses-02',
                target_space='ses-01',
                dseg=dseg,
                output_space='T1w',
            )

        report_dir = os.path.join(out_dir, f'sub-{subject_id}')
        robj = Report(
            report_dir,
            run_uuid=None,
            bootstrap_file=bootstrap_file,
            out_filename=f'sub-{subject_id}.html',
            reportlets_dir=out_dir,
            plugins=None,
            plugin_meta=None,
            subject=subject_id,
        )
        robj.generate_report()

    print('DONE!')


if __name__ == '__main__':
    _main()
