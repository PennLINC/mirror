"""Tier 3 smoke tests for all _get_parser() functions across processing modules.

All the parsers are structurally identical (single --subject-id arg with a
sub- prefix stripper), so we parametrize across all of them.
"""

import sys
from unittest.mock import patch

import pytest


# List of (module_name, function_name) pairs.  Every processing module that
# defines a ``_get_parser`` is covered here.
_PARSER_MODULES = [
    'process_mp2rage',
    'process_ihmt',
    'process_mese',
    'process_t1wt2w_ratio',
    'process_megre',
    'process_qsm_post',
    'process_qsm',
    'process_g_ratio',
    'process_q_ratio',
    'generate_scalar_reports',
]

# The processing modules loop over every subject when --subject-id is omitted,
# so the argument is optional there.  Report generation targets one subject and
# still requires it.
_SUBJECT_ID_REQUIRED = {'generate_scalar_reports'}


def _import_get_parser(module_name):
    """Import ``_get_parser`` from *module_name*, patching heavy side-effects.

    Several modules execute ``load_config()`` at import time or import heavy
    libraries (ANTs, tensorflow).  We patch ``load_config`` and let missing
    optional deps fail gracefully so we can test the pure-argparse code.
    """
    # Patch load_config to avoid real YAML / filesystem access at import time
    fake_cfg = {
        'project_root': '/fake',
        'bids_dir': '/fake/dset',
        'code_dir': '/fake/code',
        'work_dir': '/fake/work',
        'derivatives': {},
        'apptainer': {'synthstrip': '/fake/apptainer/synthstrip-1.7.sif'},
        'freesurfer': {
            'subjects_dir': '/fake/derivatives/smriprep/sourcedata/freesurfer',
            'license': '/fake/tokens/freesurfer_license.txt',
        },
    }

    with patch('utils.processing.load_config', return_value=fake_cfg):
        # Remove cached module so the fresh import picks up the patch
        if module_name in sys.modules:
            del sys.modules[module_name]

        try:
            mod = __import__(module_name)
        except ImportError as exc:
            pytest.skip(f'Cannot import {module_name}: {exc}')

    return mod._get_parser


@pytest.mark.parametrize('module_name', _PARSER_MODULES)
class TestGetParser:
    """Smoke tests for every _get_parser in the processing suite."""

    def test_parse_subject_id_plain(self, module_name):
        """--subject-id 01 should parse successfully."""
        get_parser = _import_get_parser(module_name)
        parser = get_parser()
        args = parser.parse_args(['--subject-id', '01'])
        assert args.subject_id == '01'

    def test_parse_subject_id_with_prefix(self, module_name):
        """--subject-id sub-01 should strip the 'sub-' prefix."""
        get_parser = _import_get_parser(module_name)
        parser = get_parser()
        args = parser.parse_args(['--subject-id', 'sub-01'])
        assert args.subject_id == '01'

    def test_missing_subject_id(self, module_name):
        """Omitting --subject-id defaults to None, or exits where it is required."""
        get_parser = _import_get_parser(module_name)
        parser = get_parser()

        if module_name in _SUBJECT_ID_REQUIRED:
            with pytest.raises(SystemExit):
                parser.parse_args([])
        else:
            assert parser.parse_args([]).subject_id is None
