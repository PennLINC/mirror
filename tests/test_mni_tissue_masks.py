"""Focused tests for deterministic MNI tissue-mask construction."""

from pathlib import Path

import nibabel as nib
import numpy as np

from utils import tissue_masks


def save_image(path: Path, data: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.int16), np.eye(4)), path)
    return path


def test_build_template_tissue_masks_uses_explicit_aseg_labels(tmp_path):
    labels = np.array(
        [
            0,
            2,
            3,
            10,
            26,
            28,
            41,
            42,
            49,
            58,
            60,
            8,
            47,
            251,
        ],
        dtype=np.int16,
    ).reshape(14, 1, 1)
    template = save_image(tmp_path / 'template_dseg.nii.gz', labels)

    _, masks = tissue_masks.build_template_tissue_masks(template)

    assert np.flatnonzero(masks['cortical_gm']).tolist() == [2, 7]
    assert np.flatnonzero(masks['deep_gm']).tolist() == [3, 4, 8, 9]
    assert np.flatnonzero(masks['all_gm']).tolist() == [2, 3, 4, 7, 8, 9]
    assert np.flatnonzero(masks['wm']).tolist() == [1, 6, 13]
    assert not masks['deep_gm'][5]
    assert not masks['deep_gm'][10]
    assert not masks['all_gm'][11]
    assert not masks['all_gm'][12]


def test_build_subject_masks_combines_ribbon_dseg_and_template(tmp_path):
    dseg = save_image(
        tmp_path / 'subject_dseg.nii.gz',
        np.array([1, 1, 1, 2, 2, 0], dtype=np.int16).reshape(6, 1, 1),
    )
    ribbon = save_image(
        tmp_path / 'ribbon.nii.gz',
        np.array([1, 0, 0, 0, 1, 0], dtype=np.int16).reshape(6, 1, 1),
    )
    template = save_image(
        tmp_path / 'template_dseg.nii.gz',
        np.array([3, 10, 28, 2, 41, 49], dtype=np.int16).reshape(6, 1, 1),
    )

    _, masks = tissue_masks.build_subject_tissue_masks(dseg, ribbon, template)

    assert np.flatnonzero(masks['cortical_gm']).tolist() == [0, 4]
    assert np.flatnonzero(masks['deep_gm']).tolist() == [1]
    assert np.flatnonzero(masks['all_gm']).tolist() == [0, 1, 2]
    assert np.flatnonzero(masks['wm']).tolist() == [3, 4]


def test_smriprep_discovery_supports_subject_and_session_anat(tmp_path):
    derivatives = tmp_path / 'derivatives'
    subject = 'sub-001'
    session = 'ses-01'
    session_anat = derivatives / 'smriprep' / subject / session / 'anat'
    session_dseg = session_anat / (
        f'{subject}_{session}_acq-MPRAGE_rec-refaced_run-01_'
        f'space-{tissue_masks.SPACE}_dseg.nii.gz'
    )
    session_ribbon = session_anat / (
        f'{subject}_{session}_acq-MPRAGE_rec-refaced_run-01_desc-ribbon_mask.nii.gz'
    )
    session_dseg.parent.mkdir(parents=True)
    session_dseg.touch()
    session_ribbon.touch()

    assert tissue_masks.find_smriprep_dseg(derivatives, subject, session) == session_dseg
    assert tissue_masks.find_native_ribbon(derivatives, subject, session) == session_ribbon

    subject_anat = derivatives / 'smriprep' / subject / 'anat'
    subject_ribbon = subject_anat / (
        f'{subject}_acq-MPRAGE_rec-refaced_run-01_desc-ribbon_mask.nii.gz'
    )
    subject_ribbon.parent.mkdir(parents=True)
    subject_ribbon.touch()
    assert tissue_masks.find_native_ribbon(derivatives, subject, session) == subject_ribbon


def test_mni_ribbon_path_adds_space_entity(tmp_path):
    derivatives = tmp_path / 'derivatives'
    subject = 'sub-001'
    anat = derivatives / 'smriprep' / subject / 'anat'
    native = anat / (f'{subject}_acq-MPRAGE_rec-refaced_run-01_desc-ribbon_mask.nii.gz')
    anat.mkdir(parents=True)
    native.touch()

    assert tissue_masks.mni_ribbon_path(native).name == (
        f'{subject}_acq-MPRAGE_rec-refaced_run-01_'
        f'space-{tissue_masks.SPACE}_desc-ribbon_mask.nii.gz'
    )
