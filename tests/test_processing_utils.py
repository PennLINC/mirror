"""Tests for shared processing utilities -- Tier 1 (unit) and Tier 2 (mock)."""

import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from configuration import load_config
from utils.processing import calculate_r_squared


# ===================================================================
# Tier 1: calculate_r_squared
# ===================================================================


class TestCalculateRSquared:
    """Unit tests for the pure-NumPy R-squared calculation."""

    def test_perfect_monoexponential(self, synthetic_monoexp_data):
        """Data generated from the exact model should yield R^2 ~ 1.0."""
        data, echo_times, s0, t2s = synthetic_monoexp_data
        r2 = calculate_r_squared(data, echo_times, s0, t2s)
        np.testing.assert_allclose(r2, 1.0, atol=1e-10)

    def test_output_shape(self, synthetic_monoexp_data):
        """Output should have shape (n_voxels,)."""
        data, echo_times, s0, t2s = synthetic_monoexp_data
        r2 = calculate_r_squared(data, echo_times, s0, t2s)
        assert r2.shape == (data.shape[0],)

    def test_random_noise_low_r_squared(self):
        """Pure noise data should yield R^2 well below 1."""
        rng = np.random.default_rng(99)
        n_voxels, n_echos = 100, 5
        echo_times = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        data = rng.normal(1000, 500, size=(n_voxels, n_echos))
        s0 = rng.uniform(500, 2000, size=n_voxels)
        t2s = rng.uniform(20, 80, size=n_voxels)
        r2 = calculate_r_squared(data, echo_times, s0, t2s)
        # Most voxels should have R^2 far from 1
        assert np.mean(r2) < 0.5

    def test_constant_signal_no_nan(self):
        """Constant signal across echoes (ss_total == 0) must not produce NaN."""
        n_voxels, n_echos = 10, 4
        echo_times = np.array([10.0, 20.0, 30.0, 40.0])
        data = np.full((n_voxels, n_echos), 1000.0)
        s0 = np.full(n_voxels, 1000.0)
        t2s = np.full(n_voxels, 50.0)
        r2 = calculate_r_squared(data, echo_times, s0, t2s)
        assert not np.any(np.isnan(r2))
        assert not np.any(np.isinf(r2))

    def test_two_voxels_analytical(self):
        """Hand-verified two-voxel case with a known analytical answer."""
        echo_times = np.array([0.0, 1.0])
        # Voxel 0: s0=100, t2s=1 => predicted = [100, 100*exp(-1)] = [100, 36.788]
        # Voxel 1: perfect fit
        s0 = np.array([100.0, 200.0])
        t2s = np.array([1.0, 2.0])
        predicted_0 = s0[0] * np.exp(-echo_times / t2s[0])
        predicted_1 = s0[1] * np.exp(-echo_times / t2s[1])
        # Use exact predicted values as data => R^2 = 1
        data = np.stack([predicted_0, predicted_1])
        r2 = calculate_r_squared(data, echo_times, s0, t2s)
        np.testing.assert_allclose(r2, [1.0, 1.0], atol=1e-12)


# ===================================================================
# Tier 1: load_config
# ===================================================================


class TestLoadConfig:
    """Tests for the YAML config loader."""

    def test_top_level_keys_resolved(self, minimal_config_name):
        cfg = load_config(minimal_config_name)
        root = '/tmp/nibs_test'
        assert cfg['project_root'] == root
        assert cfg['bids_dir'] == os.path.join(root, 'dset')
        assert cfg['code_dir'] == os.path.join(root, 'code')
        assert cfg['work_dir'] == os.path.join(root, 'work')

    def test_derivatives_resolved(self, minimal_config_name):
        cfg = load_config(minimal_config_name)
        root = '/tmp/nibs_test'
        assert cfg['derivatives']['smriprep'] == os.path.join(root, 'derivatives/smriprep')
        assert cfg['derivatives']['pymp2rage'] == os.path.join(root, 'derivatives/pymp2rage')

    def test_sourcedata_resolved(self, minimal_config_name):
        cfg = load_config(minimal_config_name)
        root = '/tmp/nibs_test'
        assert cfg['sourcedata']['root'] == os.path.join(root, 'sourcedata')
        assert cfg['sourcedata']['scitran'] == os.path.join(
            root, 'sourcedata/scitran/bbl/NIBS_857664'
        )

    def test_apptainer_resolved(self, minimal_config_name):
        cfg = load_config(minimal_config_name)
        root = '/tmp/nibs_test'
        assert cfg['apptainer']['synthstrip'] == os.path.join(root, 'apptainer/synthstrip-1.7.sif')

    def test_freesurfer_resolved(self, minimal_config_name):
        cfg = load_config(minimal_config_name)
        root = '/tmp/nibs_test'
        assert cfg['freesurfer']['license'] == os.path.join(root, 'tokens/freesurfer_license.txt')

    def test_missing_sourcedata_section(self, minimal_config_no_sourcedata):
        """Config without the optional sourcedata section must not crash."""
        cfg = load_config(minimal_config_no_sourcedata)
        assert 'sourcedata' not in cfg
        # Other sections should still be present
        assert 'bids_dir' in cfg
        assert 'derivatives' in cfg


# ===================================================================
# Tier 2: get_filename
# ===================================================================


class TestGetFilename:
    """Mock-based tests for the BIDS filename builder."""

    def test_entity_merging(self, mock_bids_layout, tmp_path):
        from utils.processing import get_filename

        name_source = 'sub-01_ses-01_run-01_T1w.nii.gz'
        out_dir = str(tmp_path / 'out')

        with patch('bids.layout.parse_file_entities') as mock_parse:
            mock_parse.return_value = {
                'subject': '01',
                'session': '01',
                'run': '01',
                'suffix': 'T1w',
            }
            result = get_filename(
                name_source=name_source,
                layout=mock_bids_layout,
                out_dir=out_dir,
                entities={'space': 'MNI', 'desc': 'preproc'},
            )

        # build_path should have been called with merged entities
        call_args = mock_bids_layout.build_path.call_args
        merged = call_args[0][0]
        assert merged['subject'] == '01'
        assert merged['space'] == 'MNI'
        assert merged['desc'] == 'preproc'

    def test_dismiss_entities(self, mock_bids_layout, tmp_path):
        from utils.processing import get_filename

        name_source = 'sub-01_ses-01_echo-1_run-01_T1w.nii.gz'
        out_dir = str(tmp_path / 'out')

        with patch('bids.layout.parse_file_entities') as mock_parse:
            mock_parse.return_value = {
                'subject': '01',
                'session': '01',
                'echo': '1',
                'run': '01',
                'suffix': 'T1w',
            }
            get_filename(
                name_source=name_source,
                layout=mock_bids_layout,
                out_dir=out_dir,
                entities={'space': 'MNI'},
                dismiss_entities=['echo'],
            )

        merged = mock_bids_layout.build_path.call_args[0][0]
        assert 'echo' not in merged

    def test_override_entities(self, mock_bids_layout, tmp_path):
        """Caller-supplied entities override source entities."""
        from utils.processing import get_filename

        name_source = 'sub-01_ses-01_T1w.nii.gz'
        out_dir = str(tmp_path / 'out')

        with patch('bids.layout.parse_file_entities') as mock_parse:
            mock_parse.return_value = {'subject': '01', 'suffix': 'T1w'}
            get_filename(
                name_source=name_source,
                layout=mock_bids_layout,
                out_dir=out_dir,
                entities={'suffix': 'bold'},
            )

        merged = mock_bids_layout.build_path.call_args[0][0]
        assert merged['suffix'] == 'bold'

    def test_output_dir_substitution(self, mock_bids_layout, tmp_path):
        """Layout root should be replaced with out_dir in the returned path."""
        from utils.processing import get_filename

        out_dir = str(tmp_path / 'derivatives')

        with patch('bids.layout.parse_file_entities') as mock_parse:
            mock_parse.return_value = {'subject': '01', 'suffix': 'T1w'}
            result = get_filename(
                name_source='sub-01_T1w.nii.gz',
                layout=mock_bids_layout,
                out_dir=out_dir,
                entities={},
            )

        assert result.startswith(os.path.abspath(out_dir))
        assert os.path.abspath(mock_bids_layout.root) not in result


# ===================================================================
# Tier 2: run_command
# ===================================================================


class TestRunCommand:
    """Mock-based tests for the shell command runner."""

    def test_env_not_mutated(self):
        """Regression: run_command must not permanently alter os.environ."""
        from utils.processing import run_command

        original_env = os.environ.copy()

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout.readline.side_effect = [b'\n', b'']
            mock_process.poll.side_effect = [None, 0]
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            run_command('echo hello', env={'MY_CUSTOM_VAR': '1'})

        assert 'MY_CUSTOM_VAR' not in os.environ
        assert os.environ == original_env

    def test_env_merged_into_subprocess(self):
        """Custom env vars should be passed to subprocess."""
        from utils.processing import run_command

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout.readline.side_effect = [b'\n', b'']
            mock_process.poll.side_effect = [None, 0]
            mock_process.returncode = 0
            mock_popen.return_value = mock_process

            run_command('echo hello', env={'MY_VAR': 'value'})

        call_kwargs = mock_popen.call_args[1]
        assert call_kwargs['env']['MY_VAR'] == 'value'
        # PATH should still be present from the base environment
        assert 'PATH' in call_kwargs['env']

    def test_nonzero_returncode_raises(self):
        """Non-zero exit codes must raise RuntimeError."""
        from utils.processing import run_command

        with patch('subprocess.Popen') as mock_popen:
            mock_process = MagicMock()
            mock_process.stdout.readline.side_effect = [b'\n', b'']
            mock_process.poll.side_effect = [None, 1]
            mock_process.returncode = 1
            mock_process.stdout.read.return_value = b'error output'
            mock_popen.return_value = mock_process

            with pytest.raises(RuntimeError, match='Non zero return code: 1'):
                run_command('false')


# ===================================================================
# Tier 2: run_synthstrip
# ===================================================================


class TestRunSynthstrip:
    """Mock-based tests for the SynthStrip container wrapper."""

    APPTAINER_CFG = {
        'synthstrip_runtime': 'apptainer',
        'apptainer': {'synthstrip': '/img/synthstrip-1.7.sif'},
    }
    DOCKER_CFG = {
        'synthstrip_runtime': 'docker',
        'docker': {'synthstrip': 'freesurfer/synthstrip:1.7'},
    }

    def _run(self, cfg, **kwargs):
        """Call run_synthstrip with run_command patched; return the command string."""
        from utils import processing

        with patch.object(processing, 'run_command') as mock_run:
            processing.run_synthstrip(cfg=cfg, **kwargs)
        return mock_run.call_args[0][0]

    def test_apptainer_binds_all_file_directories(self):
        """Apptainer only auto-mounts $HOME, so every path needs an explicit bind.

        On CUBIC the working directory (/cbica/comp_space) lives outside the project
        home, so SynthStrip could not create its output file inside the container.
        """
        cmd = self._run(
            self.APPTAINER_CFG,
            in_file='/cbica/projects/nibs/derivatives/qsiprep/sub-01/anat/t1w.nii.gz',
            out_file='/cbica/comp_space/nibs/g_ratio/sub01/brain.nii.gz',
            mask_file='/cbica/comp_space/nibs/g_ratio/sub01/mask.nii.gz',
        )

        assert (
            '--bind /cbica/comp_space/nibs/g_ratio/sub01:/cbica/comp_space/nibs/g_ratio/sub01'
            in cmd
        )
        assert (
            '--bind /cbica/projects/nibs/derivatives/qsiprep/sub-01/anat:'
            '/cbica/projects/nibs/derivatives/qsiprep/sub-01/anat' in cmd
        )
        # Binds must precede the image, and the image must precede SynthStrip's args.
        assert cmd.index('--bind') < cmd.index('/img/synthstrip-1.7.sif') < cmd.index('-i ')

    def test_docker_binds_all_file_directories(self):
        """The docker runtime keeps its existing identical-path bind mounts."""
        cmd = self._run(
            self.DOCKER_CFG,
            in_file='/data/in/t1w.nii.gz',
            out_file='/work/out/brain.nii.gz',
        )

        assert '-v /data/in:/data/in' in cmd
        assert '-v /work/out:/work/out' in cmd

    def test_repeated_calls_do_not_accumulate_args(self):
        """Regression: the default ``args`` list must not persist between calls."""
        first = self._run(self.APPTAINER_CFG, in_file='/in/a.nii.gz', out_file='/out/a.nii.gz')
        second = self._run(self.APPTAINER_CFG, in_file='/in/b.nii.gz', out_file='/out/b.nii.gz')

        assert first.count('-i ') == 1
        assert second.count('-i ') == 1
        assert 'a.nii.gz' not in second
        assert '/in/a:/in/a' not in second

    def test_extra_args_are_passed_through(self):
        """Caller-supplied SynthStrip flags are preserved."""
        cmd = self._run(
            self.APPTAINER_CFG,
            in_file='/in/a.nii.gz',
            out_file='/out/a.nii.gz',
            args=['--no-csf'],
        )

        assert '--no-csf' in cmd

    def test_unknown_runtime_raises(self):
        """An unsupported runtime should raise ValueError."""
        from utils import processing

        with pytest.raises(ValueError, match='Unknown synthstrip_runtime'):
            processing.run_synthstrip(
                in_file='/in/a.nii.gz',
                cfg={'synthstrip_runtime': 'podman'},
            )
