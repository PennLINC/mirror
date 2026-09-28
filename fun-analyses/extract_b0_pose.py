#!/usr/bin/env python3
"""Extract the scanner B0 direction in QSIPrep ACPC space for each participant and session.

Step 0 of the B0-orientation analysis (see ``b0_orientation_fa_tractometry_analysis_plan.md``,
Section 4). For head-first-supine Siemens acquisitions, B0 is parallel to the world z-axis of the
raw dcm2niix NIfTI. This script estimates the rigid rotation relating each raw acquisition to the
subject's ACPC space and expresses B0 there.

Sources (one row each per participant x session):

``dwi-<dir>``
    Rigid registration (ANTs, Mattes MI) of the first raw b0 of each phase-encoding run to the
    session's ``space-ACPC_dwiref``.
``dwi``
    Mean of the ``dwi-<dir>`` B0 vectors. Opposite phase-encoding directions have opposite
    distortions, so averaging cancels most distortion-induced bias in the rigid fit.
``anat-chain``
    Composition of QSIPrep's ``from-orig_to-anat`` (session MPRAGE -> unbiased template) and
    ``from-anat_to-ACPC`` transforms. No registration; reflects pose during the MPRAGE.
``megre``
    Rigid registration of the skull-stripped MEGRE RMS reference (``process_megre.py`` output,
    in raw MEGRE scanner space) to the skull-stripped ACPC T1w, initialized from the anatomical
    chain. **Diagnostic only**: in pilot data this cross-contrast registration was under-
    constrained (seed-dependent, a median of 1.5 degrees and up to 7 degrees from the chain pose,
    without improving NMI), while DWI and the anatomical chain agreed to a median of 0.7 degrees.
    Use the ``anat-chain`` pose for the R2* control unless ``sim_fwd`` clearly exceeds
    ``sim_at_anat_chain``.

Every transform is checked by resampling the raw image into ACPC space with the transform and
with its inverse and comparing with the ACPC reference (correlation for same-contrast pairs,
normalized mutual information for MEGRE vs. T1w, whose tissue contrast is inverted). The forward
direction must win by a margin, which guards against LPS/RAS or direction-convention mistakes and
gross registration failures.

Conventions
-----------
All matrices are 4x4 point maps in RAS world coordinates. ``acpc_to_scanner`` maps a point in
ACPC world coordinates to the corresponding point in raw scanner world coordinates (the "pull"
direction used for resampling a raw image onto the ACPC grid, which is also what ITK/ANTs
transforms encode). A direction ``u`` in ACPC maps to ``R @ u`` in the scanner, where ``R`` is the
rotation part, so B0 in ACPC is ``R.T @ [0, 0, 1]``.

B0 vectors are sign-normalized so that their ACPC z-component is non-negative. Pose angles:

- ``tilt_deg``: angle between B0 and ACPC z (superior);
- ``pitch_deg``: signed angle of B0 projected onto the ACPC y-z (sagittal) plane, positive toward
  +y (anterior);
- ``roll_deg``: signed angle of B0 projected onto the ACPC x-z (coronal) plane, positive toward
  +x (right).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import warnings
from glob import glob

import nibabel as nb
import numpy as np
import pandas as pd
from scipy import ndimage

B0_SCANNER = np.array([0.0, 0.0, 1.0])
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])
# Forward-vs-inverse similarity margin required to accept a transform, per metric.
MIN_MARGIN = {'corr': 0.05, 'nmi': 0.01}
# Fixed seed so ANTs' random metric sampling is reproducible.
RANDOM_SEED = 1
# Maximum b-value treated as a b0 volume.
B0_THRESHOLD = 50


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def polar_rotation(linear: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the closest proper rotation to a 3x3 linear map and its deviation from rigidity.

    Parameters
    ----------
    linear : (3, 3) array
        Linear part of an affine transform.

    Returns
    -------
    rotation : (3, 3) array
        Closest rotation matrix (polar decomposition).
    scale_dev : float
        Maximum absolute deviation of the singular values from 1. Near 0 for rigid transforms.
    """
    u, s, vt = np.linalg.svd(linear)
    d = np.sign(np.linalg.det(u @ vt))
    rotation = u @ np.diag([1.0, 1.0, d]) @ vt
    return rotation, float(np.max(np.abs(s - 1.0)))


def b0_in_reference(ref_to_scanner: np.ndarray) -> np.ndarray:
    """Express the scanner B0 axis as a unit vector in reference-space world coordinates.

    Parameters
    ----------
    ref_to_scanner : (4, 4) array
        RAS point map from reference world coordinates to scanner world coordinates.

    Returns
    -------
    b0 : (3,) array
        Unit vector, sign-normalized so that its z-component is non-negative.
    """
    rotation, _ = polar_rotation(ref_to_scanner[:3, :3])
    return normalize_axial(rotation.T @ B0_SCANNER)


def normalize_axial(vec: np.ndarray) -> np.ndarray:
    """Normalize an axial vector to unit length with a non-negative z-component."""
    vec = np.asarray(vec, dtype=float)
    vec = vec / np.linalg.norm(vec)
    return -vec if vec[2] < 0 else vec


def mean_axial(vectors: list[np.ndarray]) -> np.ndarray:
    """Average axial unit vectors (principal eigenvector of the summed outer products)."""
    scatter = sum(np.outer(v, v) for v in vectors)
    _, eigvecs = np.linalg.eigh(scatter)
    return normalize_axial(eigvecs[:, -1])


def axial_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned acute angle between two axial vectors, in degrees."""
    cos = abs(np.dot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))


def pose_angles(b0: np.ndarray) -> dict[str, float]:
    """Summarize a B0 unit vector in ACPC space as tilt, pitch, and roll (degrees)."""
    b0 = normalize_axial(b0)
    return {
        'tilt_deg': float(np.degrees(np.arccos(np.clip(b0[2], -1.0, 1.0)))),
        'pitch_deg': float(np.degrees(np.arctan2(b0[1], b0[2]))),
        'roll_deg': float(np.degrees(np.arctan2(b0[0], b0[2]))),
    }


# ---------------------------------------------------------------------------
# ITK transform I/O
# ---------------------------------------------------------------------------


def point_map_to_matrix(apply_to_point) -> np.ndarray:
    """Recover the 4x4 matrix of a linear point map by probing it at the origin and unit axes."""
    origin = np.asarray(apply_to_point((0.0, 0.0, 0.0)), dtype=float)
    matrix = np.eye(4)
    for axis in range(3):
        probe = [0.0, 0.0, 0.0]
        probe[axis] = 1.0
        matrix[:3, axis] = np.asarray(apply_to_point(tuple(probe)), dtype=float) - origin
    matrix[:3, 3] = origin
    return matrix


def load_itk_affine(path: str) -> np.ndarray:
    """Load an ITK/ANTs linear transform of any type (affine, Euler, ...) as a RAS 4x4.

    The transform is applied by ANTs itself to probe points, so parameterization details
    (Euler angle order, center of rotation) are handled by ITK. The returned matrix maps points
    in the transform's fixed (reference) space to its moving space, i.e. the direction used when
    resampling the moving image onto the fixed grid.
    """
    import ants

    transform = ants.read_transform(path)
    lps = point_map_to_matrix(transform.apply_to_point)
    return LPS_TO_RAS @ lps @ LPS_TO_RAS


# ---------------------------------------------------------------------------
# Transform validation
# ---------------------------------------------------------------------------


def normalized_mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Studholme NMI, (H(a) + H(b)) / H(a, b); 1 for independent, 2 for identical."""
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    joint = joint / joint.sum()

    def _entropy(p):
        p = p[p > 0]
        return -np.sum(p * np.log(p))

    return float((_entropy(joint.sum(1)) + _entropy(joint.sum(0))) / _entropy(joint))


def resampled_similarity(
    ref_img: nb.Nifti1Image,
    mov_img: nb.Nifti1Image,
    ref_to_mov: np.ndarray,
    mask: np.ndarray | None = None,
    metric: str = 'corr',
    step: int = 3,
) -> float:
    """Compare the reference image with the moving image pulled through ``ref_to_mov``.

    Parameters
    ----------
    ref_img, mov_img : Nifti1Image
        3D images. Only every ``step``-th reference voxel is sampled.
    ref_to_mov : (4, 4) array
        RAS point map from reference world coordinates to moving world coordinates.
    mask : array, optional
        Boolean mask on the reference grid restricting the comparison.
    metric : {'corr', 'nmi'}
        Pearson correlation (same contrast) or normalized mutual information (cross contrast,
        e.g. GRE vs. MPRAGE, where tissue contrast is inverted).
    step : int
        Subsampling stride on the reference grid.
    """
    ref = np.asarray(ref_img.dataobj, dtype=np.float32)[::step, ::step, ::step]
    mov = np.asarray(mov_img.dataobj, dtype=np.float32)

    ijk = np.stack(
        np.meshgrid(*[np.arange(0, n, step) for n in ref_img.shape[:3]], indexing='ij'), -1
    ).reshape(-1, 3)
    xyz = nb.affines.apply_affine(ref_img.affine, ijk)
    mov_ijk = nb.affines.apply_affine(np.linalg.inv(mov_img.affine) @ ref_to_mov, xyz)
    sampled = ndimage.map_coordinates(mov, mov_ijk.T, order=1, cval=np.nan).reshape(ref.shape)

    keep = np.isfinite(sampled) & (ref > 0)
    if mask is not None:
        keep &= np.asarray(mask, bool)[::step, ::step, ::step]
    if keep.sum() < 1000:
        return float('nan')
    if metric == 'nmi':
        return normalized_mutual_information(ref[keep], sampled[keep])
    return float(np.corrcoef(ref[keep], sampled[keep])[0, 1])


def validate_direction(
    ref_img, mov_img, ref_to_mov, mask=None, metric: str = 'corr'
) -> dict[str, float | bool | str]:
    """Compare forward vs inverse resampling similarity for a candidate transform."""
    fwd = resampled_similarity(ref_img, mov_img, ref_to_mov, mask, metric)
    inv = resampled_similarity(ref_img, mov_img, np.linalg.inv(ref_to_mov), mask, metric)
    ok = bool(np.isfinite(fwd) and fwd - np.nan_to_num(inv) >= MIN_MARGIN[metric])
    return {'check_metric': metric, 'sim_fwd': fwd, 'sim_inv': inv, 'direction_ok': ok}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def rigid_register(
    fixed_img: nb.Nifti1Image,
    moving_img: nb.Nifti1Image,
    init: np.ndarray | None = None,
    fixed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Rigidly register ``moving_img`` to ``fixed_img`` with ANTs; return fixed->moving RAS map.

    Parameters
    ----------
    init : (4, 4) array, optional
        Initial fixed->moving RAS point map. Defaults to ANTs' center-of-mass initialization,
        which fails when the two fields of view differ a lot (e.g. MEGRE vs. whole-head T1w).
    fixed_mask : array, optional
        Boolean mask on the fixed grid restricting the similarity metric.
    """
    import ants

    with tempfile.TemporaryDirectory() as tmpdir:
        fixed_path = os.path.join(tmpdir, 'fixed.nii.gz')
        moving_path = os.path.join(tmpdir, 'moving.nii.gz')
        nb.save(fixed_img, fixed_path)
        nb.save(moving_img, moving_path)
        kwargs = {}
        if init is not None:
            lps = LPS_TO_RAS @ init @ LPS_TO_RAS
            init_path = os.path.join(tmpdir, 'init.mat')
            ants.write_transform(
                ants.create_ants_transform(
                    transform_type='AffineTransform',
                    dimension=3,
                    matrix=lps[:3, :3],
                    translation=lps[:3, 3],
                ),
                init_path,
            )
            kwargs['initial_transform'] = [init_path]
        if fixed_mask is not None:
            mask_path = os.path.join(tmpdir, 'mask.nii.gz')
            nb.save(nb.Nifti1Image(fixed_mask.astype(np.uint8), fixed_img.affine), mask_path)
            kwargs['mask'] = ants.image_read(mask_path)
        reg = ants.registration(
            fixed=ants.image_read(fixed_path),
            moving=ants.image_read(moving_path),
            type_of_transform='Rigid',
            aff_metric='mattes',
            random_seed=RANDOM_SEED,
            outprefix=os.path.join(tmpdir, 'reg_'),
            **kwargs,
        )
        if len(reg['fwdtransforms']) != 1:
            raise RuntimeError(f'Expected one collapsed transform, got {reg["fwdtransforms"]}')
        return load_itk_affine(reg['fwdtransforms'][0])


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def _single(pattern: str, required: bool = True) -> str | None:
    matches = sorted(glob(pattern))
    if len(matches) > 1:
        raise ValueError(f'Expected one match for {pattern}, found {len(matches)}')
    if not matches:
        if required:
            raise FileNotFoundError(pattern)
        return None
    return matches[0]


def _patient_position(nifti_path: str) -> str:
    sidecar = re.sub(r'\.nii(\.gz)?$', '.json', nifti_path)
    with open(sidecar) as fobj:
        return json.load(fobj).get('PatientPosition', 'n/a')


def _first_b0(dwi_path: str) -> nb.Nifti1Image:
    """Return the first b0 volume of a raw 4D DWI as a 3D image."""
    bval_path = re.sub(r'_part-mag_dwi\.nii(\.gz)?$', '_dwi.bval', dwi_path)
    if not os.path.isfile(bval_path):
        bval_path = re.sub(r'\.nii(\.gz)?$', '.bval', dwi_path)
    bvals = np.loadtxt(bval_path)
    idx = int(np.flatnonzero(bvals <= B0_THRESHOLD)[0])
    img = nb.load(dwi_path)
    return nb.Nifti1Image(np.asarray(img.dataobj[..., idx], dtype=np.float32), img.affine)


def _hmc_rotation_range_deg(confounds_path: str | None) -> float:
    """Largest within-scan rotation range (degrees) across axes from the QSIPrep confounds."""
    if confounds_path is None:
        return float('nan')
    conf = pd.read_table(confounds_path)
    rots = conf[['rot_x', 'rot_y', 'rot_z']].to_numpy(float)
    return float(np.degrees(np.nanmax(np.nanmax(rots, 0) - np.nanmin(rots, 0))))


def _row(participant, session, source, ref_to_scanner, check, extra=None) -> dict:
    b0 = b0_in_reference(ref_to_scanner)
    _, scale_dev = polar_rotation(ref_to_scanner[:3, :3])
    # Location of the ACPC origin in scanner coordinates (approximately relative to isocenter).
    origin = ref_to_scanner @ np.array([0.0, 0.0, 0.0, 1.0])
    row = {
        'participant_id': participant,
        'session_id': session,
        'source': source,
        'b0_x': b0[0],
        'b0_y': b0[1],
        'b0_z': b0[2],
        **pose_angles(b0),
        'scale_dev': scale_dev,
        'acpc_origin_scanner_x': origin[0],
        'acpc_origin_scanner_y': origin[1],
        'acpc_origin_scanner_z': origin[2],
        **check,
    }
    row.update(extra or {})
    return row


# ---------------------------------------------------------------------------
# Per-session extraction
# ---------------------------------------------------------------------------


def extract_session(
    bids_dir: str,
    qsiprep_dir: str,
    participant: str,
    session: str,
    register: bool = True,
    megre_dir: str | None = None,
) -> list[dict]:
    """Estimate B0-in-ACPC for one participant/session from all available sources."""
    sub, ses = f'sub-{participant}', f'ses-{session}'
    rows: list[dict] = []

    acpc_t1w_path = _single(f'{qsiprep_dir}/{sub}/anat/{sub}_space-ACPC_desc-preproc_T1w.nii.gz')
    acpc_t1w = nb.load(acpc_t1w_path)
    brain_mask = (
        nb.load(
            _single(f'{qsiprep_dir}/{sub}/anat/{sub}_space-ACPC_desc-brain_mask.nii.gz')
        ).get_fdata()
        > 0
    )

    # --- anatomical chain -------------------------------------------------------------------
    anat_chain = None
    orig_to_anat = _single(
        f'{qsiprep_dir}/{sub}/{ses}/anat/{sub}_{ses}_*MPRAGE*_from-orig_to-anat_mode-image_xfm.mat',
        required=False,
    )
    raw_t1w_path = _single(
        f'{bids_dir}/{sub}/{ses}/anat/{sub}_{ses}_acq-MPRAGE_*_T1w.nii.gz', required=False
    )
    if orig_to_anat and raw_t1w_path:
        anat_to_acpc = _single(
            f'{qsiprep_dir}/{sub}/anat/{sub}_from-anat_to-ACPC_mode-image_xfm.mat'
        )
        # Both files map fixed -> moving points: ACPC -> anat, then anat -> orig.
        acpc_to_scanner = load_itk_affine(orig_to_anat) @ load_itk_affine(anat_to_acpc)
        raw_t1w = nb.load(raw_t1w_path)
        check = validate_direction(acpc_t1w, raw_t1w, acpc_to_scanner, brain_mask)
        if check['direction_ok']:
            anat_chain = acpc_to_scanner
        rows.append(
            _row(
                participant,
                session,
                'anat-chain',
                acpc_to_scanner,
                check,
                {'patient_position': _patient_position(raw_t1w_path)},
            )
        )

    if not register:
        return rows

    # --- DWI registration -------------------------------------------------------------------
    dwiref_path = _single(
        f'{qsiprep_dir}/{sub}/{ses}/dwi/{sub}_{ses}_*_space-ACPC_dwiref.nii.gz', required=False
    )
    confounds_path = _single(
        f'{qsiprep_dir}/{sub}/{ses}/dwi/{sub}_{ses}_*_desc-confounds_timeseries.tsv',
        required=False,
    )
    dwi_b0s = []
    if dwiref_path:
        dwiref = nb.load(dwiref_path)
        for dwi_path in sorted(
            glob(f'{bids_dir}/{sub}/{ses}/dwi/{sub}_{ses}_*_part-mag_dwi.nii.gz')
        ):
            pe_dir = re.search(r'_dir-([A-Za-z]+)_', os.path.basename(dwi_path)).group(1)
            raw_b0 = _first_b0(dwi_path)
            acpc_to_scanner = rigid_register(dwiref, raw_b0)
            check = validate_direction(dwiref, raw_b0, acpc_to_scanner)
            row = _row(
                participant,
                session,
                f'dwi-{pe_dir}',
                acpc_to_scanner,
                check,
                {
                    'patient_position': _patient_position(dwi_path),
                    'hmc_rot_range_deg': _hmc_rotation_range_deg(confounds_path),
                },
            )
            rows.append(row)
            if check['direction_ok']:
                dwi_b0s.append(np.array([row['b0_x'], row['b0_y'], row['b0_z']]))

    if dwi_b0s:
        b0 = mean_axial(dwi_b0s)
        pe_rows = [r for r in rows if r['source'].startswith('dwi-')]
        rows.append(
            {
                'participant_id': participant,
                'session_id': session,
                'source': 'dwi',
                'b0_x': b0[0],
                'b0_y': b0[1],
                'b0_z': b0[2],
                **pose_angles(b0),
                'direction_ok': True,
                'n_runs': len(dwi_b0s),
                'pe_disagreement_deg': (
                    axial_angle_deg(*dwi_b0s[:2]) if len(dwi_b0s) == 2 else float('nan')
                ),
                'patient_position': ','.join(sorted({r['patient_position'] for r in pe_rows})),
                'hmc_rot_range_deg': _hmc_rotation_range_deg(confounds_path),
            }
        )

    # --- MEGRE registration (for the R2* positive control) -----------------------------------
    # Use the skull-stripped RMS reference from process_megre.py. It shares the raw MEGRE
    # (scanner) affine, as do the space-MEGRE R2* maps this transform will later resample.
    megre_path = megre_dir and _single(
        f'{megre_dir}/{sub}/{ses}/anat/{sub}_{ses}_*_space-MEGRE_desc-rmsbrain_MEGRE.nii.gz',
        required=False,
    )
    raw_megre_path = _single(
        f'{bids_dir}/{sub}/{ses}/anat/{sub}_{ses}_acq-QSM_*echo-1_part-mag_MEGRE.nii.gz',
        required=False,
    )
    if megre_path and raw_megre_path:
        megre = nb.load(megre_path)
        if not np.allclose(megre.affine, nb.load(raw_megre_path).affine, atol=1e-3):
            raise ValueError(f'{megre_path} is not in the raw MEGRE scanner frame')
        acpc_brain = nb.Nifti1Image(
            np.asarray(acpc_t1w.dataobj, dtype=np.float32) * brain_mask, acpc_t1w.affine
        )
        # Start from the same-session MPRAGE pose; the MEGRE slab covers far less than the T1w.
        acpc_to_scanner = rigid_register(acpc_brain, megre, init=anat_chain, fixed_mask=brain_mask)
        # GRE tissue contrast is inverted relative to MPRAGE, so compare with NMI.
        check = validate_direction(acpc_brain, megre, acpc_to_scanner, brain_mask, metric='nmi')
        extra = {'patient_position': _patient_position(raw_megre_path)}
        if anat_chain is not None:
            # This registration is under-constrained at the precision needed (see module
            # docstring). Report both so the R2* analysis can use the chain pose unless
            # registration clearly wins.
            rot, _ = polar_rotation((np.linalg.inv(anat_chain) @ acpc_to_scanner)[:3, :3])
            extra['sim_at_anat_chain'] = resampled_similarity(
                acpc_brain, megre, anat_chain, brain_mask, 'nmi'
            )
            extra['rot_from_anat_chain_deg'] = float(
                np.degrees(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1.0, 1.0)))
            )
        rows.append(_row(participant, session, 'megre', acpc_to_scanner, check, extra))

    return rows


def summarize_between_sessions(pose: pd.DataFrame) -> pd.DataFrame:
    """Per participant and source: angle between session B0 vectors, and pose differences."""
    records = []
    for (participant, source), grp in pose.groupby(['participant_id', 'source']):
        grp = grp.sort_values('session_id')
        if len(grp) != 2:
            continue
        v1, v2 = grp[['b0_x', 'b0_y', 'b0_z']].to_numpy(float)
        records.append(
            {
                'participant_id': participant,
                'source': source,
                'delta_b0_deg': axial_angle_deg(v1, v2),
                'delta_pitch_deg': float(np.diff(grp['pitch_deg'])[0]),
                'delta_roll_deg': float(np.diff(grp['roll_deg'])[0]),
                'delta_acpc_origin_scanner_z': (
                    float(np.diff(grp['acpc_origin_scanner_z'])[0])
                    if 'acpc_origin_scanner_z' in grp
                    else float('nan')
                ),
            }
        )
    return pd.DataFrame(records)


def summarize_source_agreement(pose: pd.DataFrame, a: str = 'dwi', b: str = 'anat-chain'):
    """Per participant and session: angle between the B0 vectors from two sources."""
    cols = ['participant_id', 'session_id', 'b0_x', 'b0_y', 'b0_z']
    merged = pose.loc[pose['source'] == a, cols].merge(
        pose.loc[pose['source'] == b, cols],
        on=['participant_id', 'session_id'],
        suffixes=('_a', '_b'),
    )
    merged[f'{a}_vs_{b}_deg'] = [
        axial_angle_deg(
            r[['b0_x_a', 'b0_y_a', 'b0_z_a']].to_numpy(float),
            r[['b0_x_b', 'b0_y_b', 'b0_z_b']].to_numpy(float),
        )
        for _, r in merged.iterrows()
    ]
    return merged[['participant_id', 'session_id', f'{a}_vs_{b}_deg']]


def default_out_dir(cfg: dict) -> str:
    """B0-analysis output directory: ``derivatives.b0_orientation`` from the path config,
    falling back to ``<project_root>/derivatives/b0_orientation``."""
    return cfg['derivatives'].get(
        'b0_orientation', os.path.join(cfg['project_root'], 'derivatives', 'b0_orientation')
    )


def _get_parser():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='pc', choices=['pc', 'cubic'])
    parser.add_argument('--participant', nargs='*', help='Participant labels (no "sub-").')
    parser.add_argument(
        '--out-dir', help='Output directory (default: derivatives.b0_orientation in the config).'
    )
    parser.add_argument(
        '--no-registration', action='store_true', help='Only compute the anatomical-chain estimate.'
    )
    return parser


def main(argv=None):
    args = _get_parser().parse_args(argv)
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from configuration.config import load_config

    cfg = load_config(args.config)
    bids_dir, qsiprep_dir = cfg['bids_dir'], cfg['derivatives']['qsiprep']
    out_dir = args.out_dir or default_out_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    participants = args.participant or sorted(
        os.path.basename(p)[4:] for p in glob(f'{qsiprep_dir}/sub-*') if os.path.isdir(p)
    )
    rows = []
    for participant in participants:
        for ses_dir in sorted(glob(f'{qsiprep_dir}/sub-{participant}/ses-*')):
            session = os.path.basename(ses_dir)[4:]
            print(f'[INFO] sub-{participant} ses-{session}', flush=True)
            try:
                rows += extract_session(
                    bids_dir,
                    qsiprep_dir,
                    participant,
                    session,
                    register=not args.no_registration,
                    megre_dir=cfg['derivatives'].get('megre'),
                )
            except Exception as exc:  # keep going; report at the end
                warnings.warn(f'sub-{participant} ses-{session} failed: {exc}', stacklevel=1)

    pose = pd.DataFrame(rows)
    pose.to_csv(os.path.join(out_dir, 'b0_pose.tsv'), sep='\t', index=False, na_rep='n/a')
    summarize_between_sessions(pose).to_csv(
        os.path.join(out_dir, 'b0_pose_between_sessions.tsv'),
        sep='\t',
        index=False,
        na_rep='n/a',
    )
    if {'dwi', 'anat-chain'} <= set(pose['source']):
        summarize_source_agreement(pose).to_csv(
            os.path.join(out_dir, 'b0_pose_source_agreement.tsv'),
            sep='\t',
            index=False,
            na_rep='n/a',
        )

    bad = pose.loc[~pose['direction_ok'].astype(bool)]
    if len(bad):
        print(f'[WARNING] {len(bad)} transform(s) failed the direction check:')
        print(bad[['participant_id', 'session_id', 'source', 'check_metric', 'sim_fwd', 'sim_inv']])
    non_hfs = pose.loc[~pose['patient_position'].astype(str).str.fullmatch('HFS')]
    if len(non_hfs):
        print('[WARNING] Non-HFS acquisitions (B0 assumption invalid):')
        print(non_hfs[['participant_id', 'session_id', 'source', 'patient_position']])
    print(f'[INFO] Wrote outputs to {out_dir}')


if __name__ == '__main__':
    main()
