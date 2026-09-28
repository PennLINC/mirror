"""Shared fixtures for the MIRROR test suite."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESSING_SCRIPT_DIRS = sorted(
    path.parent
    for path in (REPO_ROOT / 'processing').glob('**/*.py')
    if path.name != '__init__.py'
)
for script_dir in PROCESSING_SCRIPT_DIRS:
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


# ---------------------------------------------------------------------------
# Tier 1 fixtures
# ---------------------------------------------------------------------------


def _fake_config_dir(tmp_path, monkeypatch, name, yaml_text):
    """Write ``paths_<name>.yml`` into *tmp_path* and point ``load_config`` at it.

    ``load_config`` takes a config *name* and resolves it against its own
    directory, so the only way to feed it a temporary file is to make that
    directory temporary too.
    """
    import configuration.config as config_module

    (tmp_path / f'paths_{name}.yml').write_text(yaml_text)
    monkeypatch.setattr(config_module, '__file__', str(tmp_path / 'config.py'))
    return name


@pytest.fixture()
def minimal_config_name(tmp_path, monkeypatch):
    """Write a small paths_test.yml and return the config name to load it by."""
    yaml_text = """\
project_root: /tmp/nibs_test

bids_dir: dset
sourcedata:
  root: sourcedata
  scitran: sourcedata/scitran/bbl/NIBS_857664
code_dir: code
work_dir: work

derivatives:
  smriprep: derivatives/smriprep
  pymp2rage: derivatives/pymp2rage

apptainer:
  synthstrip: apptainer/synthstrip-1.7.sif

freesurfer:
  subjects_dir: derivatives/smriprep/sourcedata/freesurfer
  license: tokens/freesurfer_license.txt
"""
    return _fake_config_dir(tmp_path, monkeypatch, 'test', yaml_text)


@pytest.fixture()
def minimal_config_no_sourcedata(tmp_path, monkeypatch):
    """A config that lacks the optional ``sourcedata`` section."""
    yaml_text = """\
project_root: /tmp/nibs_test

bids_dir: dset
code_dir: code
work_dir: work

derivatives:
  smriprep: derivatives/smriprep

apptainer:
  synthstrip: apptainer/synthstrip-1.7.sif

freesurfer:
  subjects_dir: derivatives/smriprep/sourcedata/freesurfer
  license: tokens/freesurfer_license.txt
"""
    return _fake_config_dir(tmp_path, monkeypatch, 'nosourcedata', yaml_text)


@pytest.fixture()
def synthetic_monoexp_data():
    """Generate perfect monoexponential decay data.

    Returns (data, echo_times, s0, t2s) where data == s0 * exp(-TE / t2s)
    exactly, so R-squared should be 1.0.
    """
    rng = np.random.default_rng(42)
    n_voxels = 50
    echo_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # ms

    s0 = rng.uniform(500, 2000, size=n_voxels)
    t2s = rng.uniform(20, 80, size=n_voxels)  # ms

    echo_times_rep = np.tile(echo_times, (n_voxels, 1))
    data = s0[:, np.newaxis] * np.exp(-echo_times_rep / t2s[:, np.newaxis])

    return data, echo_times, s0, t2s


# ---------------------------------------------------------------------------
# Tier 2 fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_bids_layout(tmp_path):
    """Return a ``MagicMock`` that quacks like a ``BIDSLayout``."""
    layout = MagicMock()
    layout.root = str(tmp_path / 'bids')
    os.makedirs(layout.root, exist_ok=True)

    # build_path returns a path rooted at layout.root
    def _build_path(entities, **kwargs):
        parts = '_'.join(f'{k}-{v}' for k, v in sorted(entities.items()) if k != 'extension')
        ext = entities.get('extension', '.nii.gz')
        if not ext.startswith('.'):
            ext = '.' + ext
        datatype = entities.get('datatype', 'anat')
        return os.path.join(layout.root, datatype, parts + ext)

    layout.build_path.side_effect = _build_path
    return layout
