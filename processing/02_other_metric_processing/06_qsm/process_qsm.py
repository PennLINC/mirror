"""Process QSM data with SEPIA and chi-separation.

Steps:

0.  Locate MATLAB. This pipeline runs under WSL and launches the Windows
    ``matlab.exe`` (newest under ``C:/Program Files/MATLAB``, or ``MATLAB_CMD``),
    translating WSL paths to Windows paths for MATLAB.
1.  For each echo set (E12345, E2345):
    a.  Concatenate the magnitude and phase MEGRE echoes and build the SEPIA header.
    b.  Run SEPIA QSM estimation once on the concatenated data.
    c.  Run chi-separation once for each parameter combination, using the
        processed magnitude and phase images from SEPIA.

Notes:

- Must be run after process_megre.py (R2*/R2' maps).
- The R2*/R2' maps are optional. When they are missing for an echo set, the
  chi-separation R2' variant (``chisep+r2p``) is skipped for that echo set and
  only the R2* (``chisep+r2s``) and R2'-net (``chisep+r2primenet``) variants are
  run. SEPIA does not use them and always runs.
- Requires SEPIA and the chi-sep MATLAB toolbox along with their dependencies.
- Chimap outputs are in parts per million (ppm).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from pathlib import Path
from pprint import pformat

import nibabel as nb
from bids.layout import BIDSLayout, Query
from nilearn import image
from scipy.io import loadmat, savemat

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.processing import get_filename, load_config, run_command  # noqa: E402

CFG = load_config()
CODE_DIR = CFG['code_dir']

# This pipeline runs under WSL but MATLAB is a Windows install, so MATLAB is
# launched as matlab.exe and every path handed to it (script paths, data files,
# toolbox roots) is translated from the WSL mount (/mnt/<drive>/...) to a
# Windows path (<DRIVE>:/...). MATLAB on Windows accepts forward slashes.
# Override the MATLAB executable with MATLAB_CMD and the toolbox roots with
# SEPIA_HOME / MIRROR_QSM_SOFTWARE_ROOT (give WSL paths when using Windows MATLAB).
QSM_SOFTWARE_ROOT = os.environ.get(
    'MIRROR_QSM_SOFTWARE_ROOT',
    os.environ.get('NIBS_SOFTWARE_ROOT', CFG.get('software', {}).get('qsm_root', '')),
)
SEPIA_HOME = os.environ.get(
    'SEPIA_HOME',
    CFG.get('software', {}).get('sepia_home', os.path.join(QSM_SOFTWARE_ROOT, 'sepia-1.2.2.6')),
)

_WSL_MOUNT_RE = re.compile(r'^/mnt/([a-zA-Z])/(.*)$')


def to_windows_path(path: str, sep: str = '/') -> str:
    """Translate a WSL mount path (``/mnt/c/...``) to a Windows path (``C:/...``).

    Empty strings and paths not under ``/mnt/<drive>/`` are returned unchanged,
    so it is safe to call on optional/empty arguments and already-Windows paths.

    ``sep`` selects the path separator in the result. Use the default ``'/'``
    for general MATLAB use; use ``'\\'`` for SEPIA, whose ``sepiaIO`` parses the
    output path with ``filesep`` (a backslash on Windows) and fails on forward
    slashes. The chi-sep script, in contrast, requires forward slashes because
    it parses paths with ``regexp(..., 'sub-([^/]+)')`` and ``sprintf('%s/...')``.
    """
    if not path:
        return path
    match = _WSL_MOUNT_RE.match(path)
    if not match:
        return path
    win_path = f'{match.group(1).upper()}:/{match.group(2)}'
    if sep != '/':
        win_path = win_path.replace('/', sep)
    return win_path


def find_matlab() -> str:
    """Resolve the MATLAB executable to launch from WSL.

    Honors ``MATLAB_CMD`` if set, otherwise picks the newest
    ``matlab.exe`` under the standard Windows install location, falling back to
    a bare ``matlab`` on PATH.
    """
    cmd = os.environ.get('MATLAB_CMD')
    if cmd:
        return cmd
    matches = sorted(glob.glob('/mnt/c/Program Files/MATLAB/*/bin/matlab.exe'))
    if matches:
        return matches[-1]
    return 'matlab'


# Each echo set selects which MEGRE echoes feed SEPIA and chi-separation:
# E12345 uses all five echoes, E2345 drops the first echo.
ECHO_SETS = ['E2345', 'E12345']

# Chi-separation parameter combinations run within each echo set.
# (label, is_scaling, have_r2prime) -- the R2*/R2' paths are filled in per echo
# set because only the R2' variant reads precomputed maps.
CHISEP_COMBOS = [
    ('chisep+r2p', 0, 1),
    ('chisep+r2primenet', 0, 0),
    ('chisep+r2s', 1, 0),
]


def collect_run_data(layout: object, bids_filters: dict) -> dict[str, str]:
    """Collect inputs for SEPIA and chi-separation processing.

    Parameters
    ----------
    layout : bids.BIDSLayout
        BIDSLayout indexing the dataset and derivatives.
    bids_filters : dict
        BIDS entity filters (e.g., subject, session, run) to narrow the query.

    Returns
    -------
    run_data : dict
        Mapping of descriptive keys to resolved file paths. ``megre_mag`` and
        ``megre_phase`` map to sorted lists of five echo paths; all other keys
        map to a single path, or to ``None`` when the optional R2*/R2' map is
        missing.
    """
    queries = {
        # Multi-echo GRE magnitude/phase from the raw BIDS dataset (for SEPIA).
        'megre_mag': {
            'datatype': 'anat',
            'acquisition': 'QSM',
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
            'part': 'phase',
            'echo': Query.ANY,
            'space': Query.NONE,
            'desc': Query.NONE,
            'suffix': 'MEGRE',
            'extension': ['.nii', '.nii.gz'],
        },
        # R2*/R2' maps per echo set (for the chi-separation R2' variant).
        'r2s_e12345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E12345',
            'suffix': 'R2starmap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2p_e12345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E12345',
            'suffix': 'R2primemap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2s_e2345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E2345',
            'suffix': 'R2starmap',
            'extension': ['.nii', '.nii.gz'],
        },
        'r2p_e2345': {
            'datatype': 'anat',
            'space': 'MEGRE',
            'desc': 'MEGRE+E2345',
            'suffix': 'R2primemap',
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
            run_data[key] = sorted([f.path for f in files])
            continue

        # The R2*/R2' maps only feed the chi-separation R2' variant, which is
        # skipped when they are absent.
        if key.startswith(('r2s_', 'r2p_')) and len(files) == 0:
            print(f'No file found for {key} with query {query}')
            run_data[key] = None
            continue

        if len(files) != 1:
            raise ValueError(f'Expected 1 file for {key}, got {len(files)} with query {query}')

        run_data[key] = files[0].path

    if len(run_data['megre_mag']) != len(run_data['megre_phase']):
        raise ValueError('Expected same number of magnitude and phase images')

    print(f'Collected run data:\n{pformat(run_data, indent=4)}', flush=True)
    return run_data


def run_sepia(
    subject_id: str,
    session: str,
    version: str,
    sepia_work_dir: str,
    mag_file: str,
    phase_file: str,
    header_file: str,
) -> tuple[str, str]:
    """Run SEPIA QSM estimation on one echo set and return output paths.

    Parameters
    ----------
    subject_id, session, version : str
        BIDS subject/session labels and echo-set label (e.g., ``E12345``).
    sepia_work_dir : str
        Directory holding the concatenated inputs and receiving SEPIA outputs.
    mag_file, phase_file, header_file : str
        Concatenated magnitude/phase NIfTIs and SEPIA header.

    Returns
    -------
    sepia_chimap_file : str
        Path to the SEPIA Chimap output.
    sepia_mask_file : str
        Path to the BET brain mask produced by SEPIA.
    """
    # SEPIA treats this as an output basename prefix and appends '_<map>.nii.gz'
    # (e.g. '_Chimap.nii.gz'), so it must not end in '_' or the outputs get a
    # doubled underscore.
    out_prefix = os.path.join(sepia_work_dir, f'sub-{subject_id}_ses-{session}_desc-{version}')

    sepia_script = os.path.join(
        CODE_DIR,
        'processing',
        '02_other_metric_processing',
        '06_qsm',
        'process_qsm_sepia.m',
    )
    with open(sepia_script) as fobj:
        base_sepia_script = fobj.read()

    # Paths baked into the MATLAB script must be Windows paths for matlab.exe.
    # SEPIA's sepiaIO derives the output directory by searching the output path
    # for filesep (a backslash on Windows), so use backslash-separated Windows
    # paths here; forward slashes make sepiaIO fail with an empty index.
    modified_sepia_script = (
        base_sepia_script.replace('{{ phase_file }}', to_windows_path(phase_file, sep='\\'))
        .replace('{{ mag_file }}', to_windows_path(mag_file, sep='\\'))
        .replace('{{ output_dir }}', to_windows_path(out_prefix, sep='\\'))
        .replace('{{ header_file }}', to_windows_path(header_file, sep='\\'))
    )

    out_sepia_script = os.path.join(sepia_work_dir, f'process_qsm_sepia_{version}.m')
    with open(out_sepia_script, 'w') as fobj:
        fobj.write(modified_sepia_script)

    # Set SEPIA_HOME and MIRROR_QSM_SOFTWARE_ROOT inside the MATLAB
    # session so the script's getenv() calls resolve without hitting their WSL
    # session. The software root lets the script add the MEDI toolbox, which
    # provides the BET brain extraction used by SEPIA's isBET option. WSL does
    # not forward env vars to Windows processes, so this is done via setenv in
    # the command rather than the subprocess environment.
    sepia_home_win = to_windows_path(SEPIA_HOME, sep='\\')
    software_root_win = to_windows_path(QSM_SOFTWARE_ROOT, sep='\\')
    out_sepia_script_win = to_windows_path(out_sepia_script, sep='\\')
    result = subprocess.run(
        [
            find_matlab(),
            '-batch',
            f"setenv('SEPIA_HOME','{sepia_home_win}'); "
            f"setenv('MIRROR_QSM_SOFTWARE_ROOT','{software_root_win}'); "
            f"run('{out_sepia_script_win}')",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'MATLAB exited with code {result.returncode}\n'
            f'stdout:\n{result.stdout}\n'
            f'stderr:\n{result.stderr}'
        )

    sepia_chimap_file = f'{out_prefix}_Chimap.nii.gz'
    sepia_mask_file = f'{out_prefix}_mask_brain.nii.gz'
    expected_outputs = {
        'SEPIA QSM output': sepia_chimap_file,
        'SEPIA BET brain mask': sepia_mask_file,
    }
    for label, path in expected_outputs.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{label} file {path} not found')

    return sepia_chimap_file, sepia_mask_file


def run_chisep(
    run_data: dict,
    version: str,
    example_nifti: str,
    sepia_work_dir: str,
    brain_mask_file: str,
    subject_id: str,
    session: str,
) -> None:
    """Run chi-separation for every parameter combination of one echo set.

    Parameters
    ----------
    run_data : dict
        Input file paths from :func:`collect_run_data`.
    version : str
        Echo-set label (e.g., ``E12345``); selects the matching R2*/R2' maps.
        The R2' variant is skipped when those maps are missing.
    example_nifti : str
        Raw MEGRE echo-1 magnitude NIfTI, used for the output NIfTI header.
    sepia_work_dir : str
        SEPIA working directory holding the concatenated MEGRE data and header
        that chi-separation reads as input.
    brain_mask_file : str
        BET brain mask produced by SEPIA for this echo set.
    subject_id, session : str
        BIDS subject/session labels.
    """
    work_dir = CFG['work_dir']
    matlab_script_dir = os.path.dirname(__file__)

    version_key = version.lower()
    r2s_path = run_data[f'r2s_{version_key}']
    r2p_path = run_data[f'r2p_{version_key}']

    # Set MIRROR_QSM_SOFTWARE_ROOT inside the MATLAB session.
    # WSL does not forward env vars to Windows processes, so this is done via
    # setenv in the command rather than the subprocess environment.
    matlab_exe = find_matlab()
    software_root_win = to_windows_path(QSM_SOFTWARE_ROOT)

    for label, is_scaling, have_r2prime in CHISEP_COMBOS:
        if have_r2prime and (r2s_path is None or r2p_path is None):
            # Only this variant reads the precomputed maps, so the R2* and
            # R2'-net variants still run without them.
            print(f"Skipping chi-sep {version} {label}: no R2' input available.", flush=True)
            continue

        r2s = r2s_path if have_r2prime else ''
        r2p = r2p_path if have_r2prime else ''

        out_dir = os.path.join(
            work_dir, f'qsm-{version}+{label}', f'sub-{subject_id}', f'ses-{session}', 'anat'
        )
        os.makedirs(out_dir, exist_ok=True)

        # All paths handed to matlab.exe are translated to Windows paths.
        # Keep forward slashes here: process_qsm_chisep.m parses the input dir
        # with regexp(..., 'sub-([^/]+)') and builds paths with sprintf('%s/...'),
        # both of which require '/' (unlike SEPIA, which needs backslashes).
        cmd = [
            matlab_exe,
            '-batch',
            (
                'try; '
                f"setenv('MIRROR_QSM_SOFTWARE_ROOT','{software_root_win}'); "
                f"addpath(genpath('{to_windows_path(matlab_script_dir)}')); "
                f"process_qsm_chisep('{to_windows_path(example_nifti)}',"
                f"'{to_windows_path(sepia_work_dir)}','{to_windows_path(out_dir)}',"
                f'{is_scaling},{have_r2prime},'
                f"'{to_windows_path(r2s)}','{to_windows_path(r2p)}',"
                f"'{to_windows_path(brain_mask_file)}','{version}'); "
                "catch ME; disp(getReport(ME, 'extended', 'hyperlinks', 'off')); exit(1); end;"
            ),
        ]
        print(f'Running chi-sep: {version} {label}', flush=True)
        run_command(cmd)
        print(f'Finished chi-sep: {version} {label}', flush=True)


def process_run(layout, run_data, out_dir, subject_id, session):
    """Run SEPIA and chi-separation for each echo set of a single run.

    Parameters
    ----------
    layout : BIDSLayout
        BIDSLayout object for the dataset.
    run_data : dict
        Dictionary of input file paths from :func:`collect_run_data`.
    out_dir : str
        Path to the QSM derivatives directory (for the copied Chimap).
    subject_id : str
        BIDS subject label (without 'sub-' prefix).
    session : str
        BIDS session label (without 'ses-' prefix).
    """
    name_source = run_data['megre_mag'][0]
    example_nifti = run_data['megre_mag'][0]

    header_file = os.path.join(os.path.dirname(__file__), 'sepia_header.mat')
    header_struct = loadmat(header_file)
    header_struct['B0_dir'] = header_struct['B0_dir'].astype(float)
    header_struct['B0'] = header_struct['B0'].astype(float)
    # Keep the full echo-time vector; the per-version TE is derived from this each
    # iteration so the slice does not accumulate across echo sets (which broke
    # E12345 whenever E2345 ran first and left TE with only 4 elements).
    full_te = header_struct['TE']

    for version in ECHO_SETS:
        if version == 'E12345':
            mag_imgs = run_data['megre_mag']
            phase_imgs = run_data['megre_phase']
            header_struct['TE'] = full_te
        else:
            # Drop the first echo for both the images and the header TE vector.
            mag_imgs = run_data['megre_mag'][1:]
            phase_imgs = run_data['megre_phase'][1:]
            header_struct['TE'] = full_te[:, 1:]

        sepia_work_dir = os.path.join(
            CFG['work_dir'], f'qsm-{version}+sepia', f'sub-{subject_id}', f'ses-{session}', 'anat'
        )
        os.makedirs(sepia_work_dir, exist_ok=True)

        # Write the concatenated MEGRE inputs and header with the names that both
        # SEPIA and chi-separation read from the SEPIA working directory.
        prefix = os.path.join(sepia_work_dir, f'sub-{subject_id}_ses-{session}_')
        mag_concat_file = f'{prefix}part-mag_desc-concat_MEGRE.nii.gz'
        phase_concat_file = f'{prefix}part-phase_desc-concat_MEGRE.nii.gz'
        header_concat_file = f'{prefix}header.mat'
        image.concat_imgs(mag_imgs).to_filename(mag_concat_file)
        image.concat_imgs(phase_imgs).to_filename(phase_concat_file)
        savemat(header_concat_file, header_struct)

        # Run SEPIA once for this echo set.
        sepia_chimap_file, sepia_mask_file = run_sepia(
            subject_id=subject_id,
            session=session,
            version=version,
            sepia_work_dir=sepia_work_dir,
            mag_file=mag_concat_file,
            phase_file=phase_concat_file,
            header_file=header_concat_file,
        )

        # Copy the SEPIA Chimap into the QSM derivatives.
        sepia_chimap_filename = get_filename(
            name_source=name_source,
            layout=layout,
            out_dir=out_dir,
            entities={'space': 'MEGRE', 'desc': f'{version}+sepia', 'suffix': 'Chimap'},
            dismiss_entities=['acquisition', 'echo', 'part', 'inv', 'reconstruction'],
        )
        nb.load(sepia_chimap_file).to_filename(sepia_chimap_filename)

        # Run chi-separation for every parameter combination, using the SEPIA
        # working directory (concatenated MEGRE data + header) as input.
        run_chisep(
            run_data=run_data,
            version=version,
            example_nifti=example_nifti,
            sepia_work_dir=sepia_work_dir,
            brain_mask_file=sepia_mask_file,
            subject_id=subject_id,
            session=session,
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
    """Run the process_qsm workflow."""
    options = _get_parser().parse_args(argv)
    kwargs = vars(options)
    main(**kwargs)


def main(subject_id):
    in_dir = CFG['bids_dir']
    smriprep_dir = CFG['derivatives']['smriprep']
    mese_dir = CFG['derivatives']['mese']
    megre_dir = CFG['derivatives']['megre']
    out_dir = CFG['derivatives']['qsm']
    os.makedirs(out_dir, exist_ok=True)

    layout = BIDSLayout(
        in_dir,
        config=os.path.join(CODE_DIR, 'configuration', 'bids', 'layout.json'),
        validate=False,
        derivatives=[smriprep_dir, mese_dir, megre_dir],
    )

    if subject_id:
        subjects = [subject_id]
    else:
        subjects = layout.get_subjects(suffix='MEGRE')

    for subject_id in subjects:
        print(f'Processing subject {subject_id}', flush=True)
        sessions = layout.get_sessions(subject=subject_id, suffix='MEGRE')
        for session in sessions:
            print(f'Processing session {session}', flush=True)
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
                    print(f'Failed {megre_file}', flush=True)
                    print(e, flush=True)
                    continue

                process_run(layout, run_data, out_dir, subject_id, session)

    print('DONE!', flush=True)


if __name__ == '__main__':
    _main()
