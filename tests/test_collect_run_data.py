"""Tier 2 mock tests for collect_run_data across processing modules."""

import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Module-level mocks for heavy dependencies
# ---------------------------------------------------------------------------
# These modules import optional heavy neuroimaging libraries at the top level.
# Mock packages that are not part of the test environment; nibabel is a real CI
# dependency because the tissue-mask tests exercise it directly.
for _mod_name in [
    'ants',
    'antspynet',
    'antspynet.utilities',
    'nilearn',
    'nilearn.image',
    'nilearn.masking',
    'nilearn.maskers',
    'nilearn.plotting',
    'nireports',
    'nireports.assembler',
    'nireports.assembler.report',
    'nireports.interfaces',
    'nireports.interfaces.reporting',
    'nireports.interfaces.reporting.base',
    'nireports.reportlets',
    'nireports.reportlets.utils',
    'pymp2rage',
]:
    sys.modules.setdefault(_mod_name, MagicMock())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bids_file(path):
    """Create a mock BIDSFile-like object with .path and .filename attrs."""
    f = MagicMock()
    f.path = path
    f.filename = os.path.basename(path)
    return f


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR_BIDS_CONFIG = os.path.join(REPO_ROOT, 'configuration', 'bids', 'layout.json')

# (acquisition, mt) pairs of the five raw ihMTRAGE images, as named by curation/heuristic.py
IHMTRAGE_ACQS = [
    ('nosat', 'off'),
    ('singlepos', 'on'),
    ('singleneg', 'on'),
    ('dual1', 'on'),
    ('dual2', 'on'),
]


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()


def _write_dataset_description(root, name, derivative=True):
    os.makedirs(root, exist_ok=True)
    desc = {'Name': name, 'BIDSVersion': '1.10.0'}
    if derivative:
        desc['DatasetType'] = 'derivative'
        desc['GeneratedBy'] = [{'Name': name}]
    with open(os.path.join(root, 'dataset_description.json'), 'w') as fobj:
        json.dump(desc, fobj)


def _build_scaling_factors_dataset(tmp_path, subject, sessions):
    """Build a real (empty-file) BIDS tree with every input collect_run_data queries.

    The sMRIPrep T1w and brain mask carry ``acq``/``rec`` entities inherited from the raw
    T1w, matching what sMRIPrep actually writes for this dataset.
    """
    from bids.layout import BIDSLayout

    raw_dir = str(tmp_path / 'raw')
    smriprep_dir = str(tmp_path / 'smriprep')
    qsiprep_dir = str(tmp_path / 'qsiprep')
    noddi_dir = str(tmp_path / 'qsirecon-NODDI')
    ihmt_dir = str(tmp_path / 'ihmt')
    _write_dataset_description(raw_dir, 'NIBS', derivative=False)
    _write_dataset_description(smriprep_dir, 'sMRIPrep')
    _write_dataset_description(qsiprep_dir, 'QSIPrep')
    _write_dataset_description(noddi_dir, 'qsirecon-NODDI')
    _write_dataset_description(ihmt_dir, 'ihMT')

    for session in sessions:
        prefix = f'sub-{subject}_ses-{session}'
        raw_anat = os.path.join(raw_dir, f'sub-{subject}', f'ses-{session}', 'anat')
        ihmt_anat = os.path.join(ihmt_dir, f'sub-{subject}', f'ses-{session}', 'anat')
        for acq, mt in IHMTRAGE_ACQS:
            stem = f'{prefix}_acq-{acq}_mt-{mt}_run-01'
            _touch(os.path.join(raw_anat, f'{stem}_ihMTRAGE.nii.gz'))
            # process_ihmt.py writes one SynthStrip mask per raw ihMTRAGE image
            _touch(os.path.join(ihmt_anat, f'{stem}_desc-brain_mask.nii.gz'))

        _touch(os.path.join(ihmt_anat, f'{prefix}_run-01_space-T1w_ihMTR.nii.gz'))
        _touch(os.path.join(ihmt_anat, f'{prefix}_run-01_space-T1w_ihMTsatB1sq.nii.gz'))

        noddi_dwi = os.path.join(noddi_dir, f'sub-{subject}', f'ses-{session}', 'dwi')
        for param in ['isovf', 'icvf']:
            _touch(
                os.path.join(
                    noddi_dwi, f'{prefix}_space-ACPC_model-noddi_param-{param}_dwimap.nii.gz'
                )
            )

    smriprep_anat = os.path.join(smriprep_dir, f'sub-{subject}', 'anat')
    t1w_stem = f'sub-{subject}_acq-MPRAGE_rec-norm'
    t1w = os.path.join(smriprep_anat, f'{t1w_stem}_desc-preproc_T1w.nii.gz')
    t1w_mask = os.path.join(smriprep_anat, f'{t1w_stem}_desc-brain_mask.nii.gz')
    fs2t1w_xfm = os.path.join(smriprep_anat, f'{t1w_stem}_from-fsnative_to-T1w_mode-image_xfm.txt')
    _touch(t1w)
    _touch(t1w_mask)
    _touch(fs2t1w_xfm)

    fs_mri = os.path.join(smriprep_dir, 'sourcedata', 'freesurfer', f'sub-{subject}', 'mri')
    _touch(os.path.join(fs_mri, 'aseg.mgz'))
    _touch(os.path.join(fs_mri, 'brain.mgz'))

    t1w_acpc = os.path.join(
        qsiprep_dir, f'sub-{subject}', 'anat', f'sub-{subject}_space-ACPC_desc-preproc_T1w.nii.gz'
    )
    _touch(t1w_acpc)

    layout = BIDSLayout(
        raw_dir,
        config=MIRROR_BIDS_CONFIG,
        validate=False,
        derivatives=[smriprep_dir, qsiprep_dir, noddi_dir, ihmt_dir],
    )
    return {
        'layout': layout,
        'smriprep_dir': smriprep_dir,
        't1w': t1w,
        't1w_mask': t1w_mask,
        't1w_acpc': t1w_acpc,
        'fs2t1w_xfm': fs2t1w_xfm,
    }


# ===================================================================
# process_mp2rage.collect_run_data
# ===================================================================


class TestCollectRunDataMp2rage:
    """Mock-based tests for process_mp2rage.collect_run_data."""

    @pytest.fixture(autouse=True)
    def _import_collect(self):
        """Import collect_run_data, patching load_config for module-level CFG."""
        # Force a fresh import each time
        sys.modules.pop('process_mp2rage', None)
        from process_mp2rage import collect_run_data

        self.collect_run_data = collect_run_data

    @pytest.fixture()
    def bids_filters(self):
        return {
            'subject': '01',
            'session': '01',
            'run': '01',
            'datatype': 'anat',
        }

    @pytest.fixture()
    def expected_keys(self):
        return [
            'inv1_magnitude',
            'inv1_phase',
            'inv2_magnitude',
            'inv2_phase',
            'b1_famp',
            'b1_anat',
            't1w',
            't1w_mask',
            't1w_mni',
            't1w2mni_xfm',
            'mni2t1w_xfm',
            'dseg_mni',
            'mni_mask',
        ]

    def test_returns_all_keys(self, bids_filters, expected_keys):
        """When layout.get returns exactly 1 file per query, all keys are present."""
        layout = MagicMock()
        layout.get.return_value = [_make_bids_file('/fake/file.nii.gz')]

        run_data = self.collect_run_data(layout, bids_filters)

        for key in expected_keys:
            assert key in run_data, f'Missing key: {key}'

    def test_missing_phase_skipped(self, bids_filters):
        """Phase images are optional; missing phase should be silently skipped."""
        layout = MagicMock()

        def _get_side_effect(**kwargs):
            if kwargs.get('part') == 'phase':
                return []
            return [_make_bids_file('/fake/file.nii.gz')]

        layout.get.side_effect = _get_side_effect

        run_data = self.collect_run_data(layout, bids_filters)

        assert 'inv1_phase' not in run_data
        assert 'inv2_phase' not in run_data
        assert 'inv1_magnitude' in run_data

    def test_multiple_files_raises(self, bids_filters):
        """More than one file for a non-phase query should raise ValueError."""
        layout = MagicMock()
        layout.get.return_value = [
            _make_bids_file('/fake/a.nii.gz'),
            _make_bids_file('/fake/b.nii.gz'),
        ]

        with pytest.raises(ValueError, match='Expected 1 file'):
            self.collect_run_data(layout, bids_filters)


# ===================================================================
# process_g_ratio_scaling_factors.collect_run_data
# ===================================================================


class TestCollectRunDataScalingFactors:
    """Mock-based tests for process_g_ratio_scaling_factors.collect_run_data."""

    @pytest.fixture(autouse=True)
    def _import_collect(self):
        """Import collect_run_data, patching load_config for module-level CFG."""
        with patch(
            'utils.processing.load_config',
            return_value={
                'code_dir': '/fake/code',
                'bids_dir': '/fake/bids',
                'work_dir': '/fake/work',
                'derivatives': {'smriprep': '/fake/smriprep'},
            },
        ):
            sys.modules.pop('process_g_ratio_scaling_factors', None)
            from process_g_ratio_scaling_factors import collect_run_data
        self.collect_run_data = collect_run_data

    @pytest.fixture()
    def bids_filters(self):
        return {
            'subject': '01',
            'session': '01',
            'run': '01',
            'datatype': 'anat',
        }

    @pytest.fixture()
    def expected_keys(self):
        return [
            'mtsat_t1w',
            'ihmtr_t1w',
            't1w',
            't1w_mask',
            'isovf_acpc',
            'icvf_acpc',
            't1w_acpc',
            'fs2t1w_xfm',
            'aseg_fsnative',
            'brain_fsnative',
        ]

    def test_returns_all_keys(self, bids_filters, expected_keys, tmp_path):
        """When layout.get returns one file per query, all keys are present."""
        # Create fake freesurfer files that the function checks with os.path.isfile
        fs_dir = tmp_path / 'smriprep' / 'sourcedata' / 'freesurfer' / 'sub-01' / 'mri'
        fs_dir.mkdir(parents=True)
        (fs_dir / 'aseg.mgz').touch()
        (fs_dir / 'brain.mgz').touch()

        smriprep_dir = str(tmp_path / 'smriprep')

        layout = MagicMock()

        # The function iterates queries in dict insertion order.  The NODDI
        # queries have a ``param`` key that gets popped before calling
        # layout.get, then used to filter files by filename.
        _responses = iter(
            [
                [_make_bids_file('/fake/sub-01_ihMTsatB1sq.nii.gz')],
                [_make_bids_file('/fake/sub-01_ihMTR.nii.gz')],
                [_make_bids_file('/fake/sub-01_desc-preproc_T1w.nii.gz')],
                [_make_bids_file('/fake/sub-01_desc-brain_mask.nii.gz')],
                [_make_bids_file('/fake/sub-01_param-isovf_dwimap.nii.gz')],
                [_make_bids_file('/fake/sub-01_param-icvf_dwimap.nii.gz')],
                [_make_bids_file('/fake/sub-01_space-ACPC_desc-preproc_T1w.nii.gz')],
                [_make_bids_file('/fake/sub-01_xfm.txt')],
            ]
        )

        layout.get.side_effect = lambda **kw: next(_responses)

        run_data = self.collect_run_data(layout, bids_filters, smriprep_dir=smriprep_dir)

        for key in expected_keys:
            assert key in run_data, f'Missing key: {key}'

    def test_missing_file_raises(self, bids_filters):
        """Zero files for a query should raise ValueError."""
        layout = MagicMock()
        layout.get.return_value = []

        with pytest.raises(ValueError, match='Expected 1 file'):
            self.collect_run_data(layout, bids_filters, smriprep_dir='/fake/smriprep')

    def test_ihmt_brain_masks_do_not_shadow_smriprep_mask(self, bids_filters, tmp_path):
        """The ihMT derivatives contain per-source-image ``desc-brain_mask`` files.

        Those masks live in ``anat``, carry no ``space`` entity, and are indexed by this
        script's layout, so the ``t1w_mask`` query has to distinguish them from the single
        sMRIPrep brain mask.
        """
        dset = _build_scaling_factors_dataset(tmp_path, subject='22449', sessions=['01', '02'])

        run_data = self.collect_run_data(
            dset['layout'],
            {'subject': '22449', 'session': '01', 'run': '01', 'datatype': 'anat'},
            smriprep_dir=dset['smriprep_dir'],
        )

        assert run_data['t1w_mask'] == dset['t1w_mask']
