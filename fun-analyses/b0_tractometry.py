#!/usr/bin/env python3
"""Node-wise tractometry for the B0-orientation analysis (plan Sections 3, 5, 8).

For every participant x session x AutoTrack bundle, streamlines are resampled to ``N_NODES``
equally spaced nodes and summarized per node with Gaussian (Mahalanobis) weights. Per node the
output holds:

- weighted means and medians of tensor metrics (TORTOISE inner-shell and full-shell tensors, DKI)
  and of R2* (MEGRE, sampled through a rigid sMRIPrep-T1w -> ACPC registration);
- weighted orientation scatter tensors ``M = sum(w v v^T) / sum(w)`` for three per-point fiber
  direction estimators: the MSMT-CSD fixel best aligned with the streamline tangent (primary),
  the streamline tangent, and the DTI principal eigenvector. For any B0 unit vector ``b`` the
  node's weighted mean ``sin^2(theta)`` is then ``1 - b^T M b``, so angles for any head pose
  (actual, reference, or permuted) can be computed later without revisiting the images;
- weighted fourth-order orientation moments ``Q = sum(w v_x^a v_y^b v_z^c) / sum(w)`` with
  ``a + b + c = 4`` for the same estimators. They give the node's weighted mean
  ``cos^4(theta)``, and with ``M`` the exact weighted mean ``sin^4(theta)``, for any B0;
- fiber-coherence measures (second/first fixel amplitude ratio, fixel-tangent alignment),
  node support, and the mean node position (used to align node order across sessions).

MSMT fixel peaks come from QSIRecon's ``model-msmt_dwimap.fib.gz`` (DSI Studio format). Its
voxel grid matches the ACPC DWI NIfTI grid without flips, and its peak directions are expressed
in voxel-axis coordinates; both were verified empirically (FA correlation 0.98; peak-tangent
median |cos| 0.99) and the latter is re-checked for every session (``qc_fixel_tangent_cos``).
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import warnings
from glob import glob
from math import factorial

import nibabel as nb
import numpy as np
import pandas as pd
from nibabel.streamlines import TckFile
from scipy import ndimage
from scipy.io import loadmat

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_b0_pose import default_out_dir, rigid_register, validate_direction  # noqa: E402

N_NODES = 50
MAX_STREAMLINES = 1000
MIN_STREAMLINES = 5
N_PEAKS = 5
SEED = 20260928
# Per-session QC floor for median |cos| between chosen fixel and streamline tangent.
MIN_FIXEL_TANGENT_COS = 0.85
R2STAR_RANGE = (0.0, 100.0)

TENSOR_PARAMS = ('fa', 'ad', 'rd', 'md')
# name -> (recon directory, file pattern with {param})
METRIC_SOURCES = {
    'inner': ('qsirecon-TORTOISE_model-MAPMRI', '*_space-ACPC_model-tensor_param-{param}_dwimap.nii.gz'),
    'full': ('qsirecon-TORTOISE_model-tensor', '*_space-ACPC_model-tensor_param-{param}_dwimap.nii.gz'),
    'dki': ('qsirecon-DIPYDKI', '*_space-ACPC_model-dki_param-{param}_dwimap.nii.gz'),
}
DIRECTION_ESTIMATORS = ('fixel', 'tangent', 'v1')
SCATTER_KEYS = ('xx', 'xy', 'xz', 'yy', 'yz', 'zz')
# Exponents (a, b, c) of the 15 fourth-order monomials x^a y^b z^c, their names, and their
# multinomial coefficients 4! / (a! b! c!) in the expansion of (b . v)^4.
MOMENT4_EXPONENTS = tuple(
    (a, b, 4 - a - b) for a in range(4, -1, -1) for b in range(4 - a, -1, -1)
)
MOMENT4_KEYS = tuple('x' * a + 'y' * b + 'z' * c for a, b, c in MOMENT4_EXPONENTS)
MOMENT4_COEFS = tuple(
    factorial(4) // (factorial(a) * factorial(b) * factorial(c)) for a, b, c in MOMENT4_EXPONENTS
)


# ---------------------------------------------------------------------------
# Streamline geometry
# ---------------------------------------------------------------------------


def resample_streamline(streamline: np.ndarray, n_nodes: int = N_NODES) -> np.ndarray | None:
    """Resample a streamline to ``n_nodes`` points equally spaced in arc length."""
    streamline = np.asarray(streamline, dtype=float)
    if len(streamline) < 2:
        return None
    cum = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(streamline, axis=0), axis=1))]
    if cum[-1] <= 0:
        return None
    t = np.linspace(0.0, cum[-1], n_nodes)
    return np.column_stack([np.interp(t, cum, streamline[:, k]) for k in range(3)])


def orient_streamlines(points: np.ndarray, n_iter: int = 2) -> np.ndarray:
    """Flip resampled streamlines (S, N, 3) so they all run in the same direction.

    Each streamline is compared with a reference, as is and reversed, and flipped if reversed is
    closer. The reference starts as the first streamline and is then replaced by the mean.
    """
    points = points.copy()
    ref = points[0]
    for _ in range(n_iter):
        d_same = np.linalg.norm(points - ref, axis=2).mean(1)
        d_flip = np.linalg.norm(points[:, ::-1] - ref, axis=2).mean(1)
        flip = d_flip < d_same
        points[flip] = points[flip, ::-1]
        ref = points.mean(0)
    return points


def streamline_tangents(points: np.ndarray) -> np.ndarray:
    """Unit tangents (S, N, 3) of resampled streamlines by central differences."""
    tangents = np.gradient(points, axis=1)
    norms = np.linalg.norm(tangents, axis=2, keepdims=True)
    return tangents / np.where(norms > 0, norms, 1.0)


def node_weights(points: np.ndarray) -> np.ndarray:
    """Gaussian Mahalanobis weights (S, N), normalized to sum to 1 at each node."""
    n_sl, n_nodes, _ = points.shape
    if n_sl < MIN_STREAMLINES:
        return np.full((n_sl, n_nodes), 1.0 / n_sl)
    weights = np.empty((n_sl, n_nodes))
    for node in range(n_nodes):
        x = points[:, node, :]
        centered = x - x.mean(0)
        cov = np.cov(centered, rowvar=False) + 1e-3 * np.eye(3)
        d2 = np.einsum('ij,jk,ik->i', centered, np.linalg.inv(cov), centered)
        weights[:, node] = np.exp(-0.5 * d2)
    weights /= weights.sum(0, keepdims=True)
    return weights


# ---------------------------------------------------------------------------
# Weighted node summaries
# ---------------------------------------------------------------------------


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean over streamlines (axis 0), ignoring NaNs."""
    ok = np.isfinite(values)
    w = np.where(ok, weights, 0.0)
    total = w.sum(0)
    return np.where(total > 0, (w * np.where(ok, values, 0.0)).sum(0) / np.where(total > 0, total, 1), np.nan)


def weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted median over streamlines (axis 0) for each node, ignoring NaNs."""
    out = np.full(values.shape[1], np.nan)
    for node in range(values.shape[1]):
        v, w = values[:, node], weights[:, node]
        ok = np.isfinite(v) & (w > 0)
        if not ok.any():
            continue
        order = np.argsort(v[ok])
        cw = np.cumsum(w[ok][order])
        out[node] = v[ok][order][np.searchsorted(cw, 0.5 * cw[-1])]
    return out


def _direction_weights(directions: np.ndarray, weights: np.ndarray):
    """Weights renormalized over finite directions (S, N), zero-filled directions, and a
    per-node flag for nodes with at least one finite direction."""
    ok = np.all(np.isfinite(directions), axis=2)
    w = np.where(ok, weights, 0.0)
    total = w.sum(0)
    w = w / np.where(total > 0, total, 1.0)
    return w, np.where(ok[..., None], directions, 0.0), total > 0


def weighted_scatter(directions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted orientation scatter tensor per node, as (N, 6) in ``SCATTER_KEYS`` order.

    Nodes without any finite direction are NaN.
    """
    w, d, has_dir = _direction_weights(directions, weights)
    outer = np.einsum('sn,sni,snj->nij', w, d, d)
    out = outer[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]
    return np.where(has_dir[:, None], out, np.nan)


def weighted_moment4(directions: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted fourth-order orientation moments per node, (N, 15) in ``MOMENT4_KEYS`` order.

    Nodes without any finite direction are NaN.
    """
    w, d, has_dir = _direction_weights(directions, weights)
    out = np.stack(
        [(w * d[..., 0] ** a * d[..., 1] ** b * d[..., 2] ** c).sum(0)
         for a, b, c in MOMENT4_EXPONENTS],
        axis=1,
    )
    return np.where(has_dir[:, None], out, np.nan)


def sin2_from_scatter(scatter: np.ndarray, b0: np.ndarray) -> np.ndarray:
    """Weighted mean ``sin^2(theta)`` = 1 - b^T M b for scatter rows (..., 6) and B0 (..., 3)."""
    xx, xy, xz, yy, yz, zz = np.moveaxis(np.asarray(scatter, float), -1, 0)
    bx, by, bz = np.moveaxis(np.asarray(b0, float), -1, 0)
    quad = xx * bx**2 + yy * by**2 + zz * bz**2 + 2 * (xy * bx * by + xz * bx * bz + yz * by * bz)
    return 1.0 - quad


def cos4_from_moment4(moment4: np.ndarray, b0: np.ndarray) -> np.ndarray:
    """Weighted mean ``cos^4(theta)`` for moment rows (..., 15) and B0 unit vectors (..., 3)."""
    q = np.moveaxis(np.asarray(moment4, float), -1, 0)
    bx, by, bz = np.moveaxis(np.asarray(b0, float), -1, 0)
    return sum(
        coef * bx**a * by**b * bz**c * q[k]
        for k, ((a, b, c), coef) in enumerate(zip(MOMENT4_EXPONENTS, MOMENT4_COEFS))
    )


def sin4_from_moments(scatter: np.ndarray, moment4: np.ndarray, b0: np.ndarray) -> np.ndarray:
    """Weighted mean ``sin^4(theta)`` = 1 - 2 <cos^2> + <cos^4> for any B0 unit vector."""
    return 2.0 * sin2_from_scatter(scatter, b0) - 1.0 + cos4_from_moment4(moment4, b0)


# ---------------------------------------------------------------------------
# Image sampling
# ---------------------------------------------------------------------------


class Volume:
    """A 3D image loaded once: float64 data (non-finite -> NaN) and its affine."""

    def __init__(self, img: nb.Nifti1Image):
        data = np.asarray(img.dataobj, dtype=np.float64)
        self.data = np.where(np.isfinite(data), data, np.nan)
        self.affine = img.affine
        self.shape = self.data.shape


def sample_image(img, points_world: np.ndarray, world_to_img=None) -> np.ndarray:
    """Trilinearly sample a 3D image at world points (..., 3); NaN outside the field of view."""
    if not isinstance(img, Volume):
        img = Volume(img)
    mat = np.linalg.inv(img.affine)
    if world_to_img is not None:
        mat = mat @ world_to_img
    ijk = nb.affines.apply_affine(mat, points_world.reshape(-1, 3))
    vals = ndimage.map_coordinates(img.data, ijk.T, order=1, mode='constant', cval=np.nan)
    return vals.reshape(points_world.shape[:-1])


def nearest_voxel(affine: np.ndarray, shape, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest voxel indices (..., 3) and an in-bounds mask for world points."""
    ijk = np.rint(nb.affines.apply_affine(np.linalg.inv(affine), points_world)).astype(int)
    inside = np.all((ijk >= 0) & (ijk < np.asarray(shape)), axis=-1)
    return np.clip(ijk, 0, np.asarray(shape) - 1), inside


def voxel_to_world_rotation(affine: np.ndarray) -> np.ndarray:
    """Rotation (with possible reflection) mapping voxel-axis directions to world directions."""
    lin = affine[:3, :3]
    return lin / np.linalg.norm(lin, axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# MSMT fib (DSI Studio) peaks and tensor
# ---------------------------------------------------------------------------


class FibPeaks:
    """Fixel peaks and tensor from a DSI Studio ``.fib.gz`` on a known NIfTI grid."""

    def __init__(self, fib_path: str, ref_img: nb.Nifti1Image):
        names = ['dimension', 'odf_vertices'] + [f'fa{k}' for k in range(N_PEAKS)]
        names += [f'index{k}' for k in range(N_PEAKS)]
        names += ['txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz']
        with gzip.open(fib_path) as fobj:
            mat = loadmat(io.BytesIO(fobj.read()), variable_names=names)
        dim = tuple(int(d) for d in mat['dimension'].ravel())
        if dim != tuple(ref_img.shape[:3]):
            raise ValueError(f'fib dimension {dim} != reference grid {ref_img.shape[:3]}')

        def vol(key):
            return mat[key].ravel(order='F')[: np.prod(dim)].reshape(dim, order='F')

        self.shape = dim
        self.affine = ref_img.affine
        self.rot = voxel_to_world_rotation(ref_img.affine)
        vertices = mat['odf_vertices'].T.astype(float)  # (n_vertices, 3), voxel frame
        self.vertices_world = vertices @ self.rot.T
        self.amp = np.stack([vol(f'fa{k}') for k in range(N_PEAKS)], -1).astype(np.float32)
        self.index = np.stack([vol(f'index{k}') for k in range(N_PEAKS)], -1).astype(np.int32)
        self.tensor = np.stack([vol(k) for k in ('txx', 'txy', 'txz', 'tyy', 'tyz', 'tzz')], -1)

    def lookup(self, points_world: np.ndarray, tangents: np.ndarray) -> dict[str, np.ndarray]:
        """Per point: best-aligned fixel direction, amplitude ratio, alignment, and V1."""
        ijk, inside = nearest_voxel(self.affine, self.shape, points_world)
        amp = self.amp[ijk[..., 0], ijk[..., 1], ijk[..., 2]]  # (..., K)
        idx = self.index[ijk[..., 0], ijk[..., 1], ijk[..., 2]]
        dirs = self.vertices_world[idx]  # (..., K, 3)
        cos = np.abs(np.einsum('...kj,...j->...k', dirs, tangents))
        cos = np.where(amp > 0, cos, -1.0)
        best = cos.argmax(-1)
        has_peak = (amp[..., 0] > 0) & inside
        fixel = np.take_along_axis(dirs, best[..., None, None], axis=-2)[..., 0, :]
        fixel = np.where(has_peak[..., None], fixel, np.nan)
        align = np.where(has_peak, np.take_along_axis(cos, best[..., None], -1)[..., 0], np.nan)
        ratio = np.where(has_peak, amp[..., 1] / np.where(amp[..., 0] > 0, amp[..., 0], 1), np.nan)

        t = self.tensor[ijk[..., 0], ijk[..., 1], ijk[..., 2]]
        tens = np.empty(t.shape[:-1] + (3, 3))
        tens[..., 0, 0], tens[..., 0, 1], tens[..., 0, 2] = t[..., 0], t[..., 1], t[..., 2]
        tens[..., 1, 0], tens[..., 1, 1], tens[..., 1, 2] = t[..., 1], t[..., 3], t[..., 4]
        tens[..., 2, 0], tens[..., 2, 1], tens[..., 2, 2] = t[..., 2], t[..., 4], t[..., 5]
        _, vecs = np.linalg.eigh(tens)
        v1 = vecs[..., :, -1] @ self.rot.T
        v1 = np.where((np.abs(t).sum(-1) > 0)[..., None] & inside[..., None], v1, np.nan)
        return {'fixel': fixel, 'fixel_align': align, 'peak_ratio': ratio, 'v1': v1}


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


def load_tck_gz(path: str):
    """Load streamlines from a gzipped .tck file (world RAS mm)."""
    with gzip.open(path) as fobj:
        return TckFile.load(io.BytesIO(fobj.read())).streamlines


def bundle_name(path: str) -> str:
    return os.path.basename(path).split('_bundle-')[1].split('_streamlines')[0]


def acpc_to_t1w(smriprep_dir, qsiprep_dir, participant, cache_dir) -> tuple[np.ndarray, dict]:
    """Rigid ACPC -> sMRIPrep T1w point map (for R2* maps in space-T1w), cached per subject."""
    sub = f'sub-{participant}'
    cache = os.path.join(cache_dir, f'{sub}_from-ACPC_to-T1w.json')
    if os.path.isfile(cache):
        with open(cache) as fobj:
            payload = json.load(fobj)
        return np.array(payload['matrix']), payload['check']

    acpc = nb.load(f'{qsiprep_dir}/{sub}/anat/{sub}_space-ACPC_desc-preproc_T1w.nii.gz')
    acpc_mask = nb.load(f'{qsiprep_dir}/{sub}/anat/{sub}_space-ACPC_desc-brain_mask.nii.gz')
    # Native sMRIPrep T1w (no space- entity), the target of process_megre.py's space-T1w maps.
    prefix = f'{smriprep_dir}/{sub}/anat/{sub}_acq-MPRAGE_rec-refaced_run-01'
    t1w_path = _single(f'{prefix}_desc-preproc_T1w.nii.gz')
    t1w_mask_path = _single(f'{prefix}_desc-brain_mask.nii.gz')
    t1w = nb.load(t1w_path)
    fixed_mask = acpc_mask.get_fdata() > 0
    fixed = nb.Nifti1Image(np.asarray(acpc.dataobj, np.float32) * fixed_mask, acpc.affine)
    moving = nb.Nifti1Image(
        np.asarray(t1w.dataobj, np.float32) * (nb.load(t1w_mask_path).get_fdata() > 0), t1w.affine
    )
    matrix = rigid_register(fixed, moving, fixed_mask=fixed_mask)
    check = validate_direction(fixed, moving, matrix, fixed_mask)
    check = {k: (v if isinstance(v, str) else float(v)) for k, v in check.items()}
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache, 'w') as fobj:
        json.dump({'matrix': matrix.tolist(), 'check': check, 't1w': t1w_path}, fobj, indent=2)
    return matrix, check


# ---------------------------------------------------------------------------
# Per-session extraction
# ---------------------------------------------------------------------------


def summarize_bundle(streamlines, maps, fib, r2star=None, rng=None) -> pd.DataFrame | None:
    """Node table for one bundle. ``maps`` is {column: image}; ``r2star`` is (image, matrix)."""
    rng = rng or np.random.default_rng(SEED)
    n_total = len(streamlines)
    keep = np.arange(n_total)
    if n_total > MAX_STREAMLINES:
        keep = np.sort(rng.choice(n_total, MAX_STREAMLINES, replace=False))
    resampled = [resample_streamline(streamlines[i]) for i in keep]
    resampled = [r for r in resampled if r is not None]
    if len(resampled) < MIN_STREAMLINES:
        return None

    points = orient_streamlines(np.stack(resampled))
    tangents = streamline_tangents(points)
    weights = node_weights(points)
    fixels = fib.lookup(points, tangents)

    out = {
        'node': np.arange(N_NODES),
        'n_streamlines': np.full(N_NODES, len(resampled)),
        'n_streamlines_total': np.full(N_NODES, n_total),
    }
    centroid = np.einsum('sn,sni->ni', weights, points)
    out.update({'pos_x': centroid[:, 0], 'pos_y': centroid[:, 1], 'pos_z': centroid[:, 2]})
    for name, img in maps.items():
        vals = sample_image(img, points)
        out[f'{name}_mean'] = weighted_mean(vals, weights)
        out[f'{name}_median'] = weighted_median(vals, weights)
    if r2star is not None:
        img, matrix = r2star
        vals = sample_image(img, points, world_to_img=matrix)
        vals = np.where((vals > R2STAR_RANGE[0]) & (vals < R2STAR_RANGE[1]), vals, np.nan)
        out['r2star_mean'] = weighted_mean(vals, weights)
        out['r2star_median'] = weighted_median(vals, weights)

    out['peak_ratio'] = weighted_mean(fixels['peak_ratio'], weights)
    out['fixel_align'] = weighted_mean(fixels['fixel_align'], weights)
    out['fixel_coverage'] = weighted_mean(np.isfinite(fixels['fixel'][..., 0]).astype(float), weights)
    directions = {'fixel': fixels['fixel'], 'tangent': tangents, 'v1': fixels['v1']}
    for est, dirs in directions.items():
        scatter = weighted_scatter(dirs, weights)
        for k, key in enumerate(SCATTER_KEYS):
            out[f'M_{est}_{key}'] = scatter[:, k]
        moment4 = weighted_moment4(dirs, weights)
        for k, key in enumerate(MOMENT4_KEYS):
            out[f'Q_{est}_{key}'] = moment4[:, k]
    return pd.DataFrame(out)


def extract_session(recon_dir, qsiprep_dir, smriprep_dir, megre_dir, participant, session, cache_dir):
    """Build the node table for one participant/session; returns (DataFrame, qc dict)."""
    sub, ses = f'sub-{participant}', f'ses-{session}'
    maps = {}
    for source, (rdir, pattern) in METRIC_SOURCES.items():
        for param in TENSOR_PARAMS:
            path = _single(f'{recon_dir}/{rdir}/{sub}/{ses}/dwi/{pattern.format(param=param)}',
                           required=False)
            if path:
                maps[f'{source}_{param}'] = Volume(nb.load(path))
    if 'inner_fa' not in maps:
        raise FileNotFoundError(f'{sub} {ses}: inner-shell FA missing')

    fib = FibPeaks(
        _single(f'{recon_dir}/qsirecon-MSMTAutoTrack/{sub}/{ses}/dwi/*_model-msmt_dwimap.fib.gz'),
        maps['inner_fa'],
    )

    r2star, r2_check = None, None
    r2_path = megre_dir and _single(
        f'{megre_dir}/{sub}/{ses}/anat/{sub}_{ses}_*_space-T1w_desc-MEGRE+E12345_R2starmap.nii.gz',
        required=False,
    )
    if r2_path:
        try:
            matrix, r2_check = acpc_to_t1w(smriprep_dir, qsiprep_dir, participant, cache_dir)
        except FileNotFoundError as exc:  # R2* is optional; keep the DWI metrics
            warnings.warn(f'{sub}: no T1w->ACPC transform, skipping R2* ({exc})', stacklevel=1)
            r2_check = {'direction_ok': False}
        if r2_check['direction_ok']:
            r2star = (Volume(nb.load(r2_path)), matrix)

    rng = np.random.default_rng([SEED, int(participant) if participant.isdigit() else 0,
                                 int(session)])
    frames = []
    tck_paths = sorted(glob(
        f'{recon_dir}/qsirecon-MSMTAutoTrack/{sub}/{ses}/dwi/*_bundle-*_streamlines.tck.gz'
    ))
    for tck_path in tck_paths:
        frame = summarize_bundle(load_tck_gz(tck_path), maps, fib, r2star, rng)
        if frame is not None:
            frame.insert(0, 'bundle', bundle_name(tck_path))
            frames.append(frame)
    nodes = pd.concat(frames, ignore_index=True)
    nodes.insert(0, 'session_id', session)
    nodes.insert(0, 'participant_id', participant)

    qc = {
        'participant_id': participant,
        'session_id': session,
        'n_bundles': int(nodes['bundle'].nunique()),
        'qc_fixel_tangent_cos': float(np.nanmedian(nodes['fixel_align'])),
        'r2star_available': r2star is not None,
        'r2star_t1w_sim_fwd': (r2_check or {}).get('sim_fwd', np.nan),
        'r2star_t1w_sim_inv': (r2_check or {}).get('sim_inv', np.nan),
    }
    if qc['qc_fixel_tangent_cos'] < MIN_FIXEL_TANGENT_COS:
        warnings.warn(f'{sub} {ses}: fixel-tangent alignment {qc["qc_fixel_tangent_cos"]:.3f} '
                      'is low; check fib orientation conventions', stacklevel=1)
    return nodes, qc


def _run_one(args):
    (recon_dir, qsiprep_dir, smriprep_dir, megre_dir, participant, session, out_dir,
     overwrite) = args
    out_file = os.path.join(out_dir, 'nodes', f'sub-{participant}_ses-{session}_nodes.parquet')
    qc_file = out_file.replace('_nodes.parquet', '_qc.json')
    if os.path.isfile(out_file) and os.path.isfile(qc_file) and not overwrite:
        return 'cached'
    try:
        nodes, qc = extract_session(recon_dir, qsiprep_dir, smriprep_dir, megre_dir,
                                    participant, session, os.path.join(out_dir, 'xfm'))
    except Exception as exc:  # report and continue with other sessions
        return f'FAILED sub-{participant} ses-{session}: {exc!r}'
    nodes.to_parquet(out_file, index=False)
    with open(qc_file, 'w') as fobj:
        json.dump(qc, fobj, indent=2)
    return f'sub-{participant} ses-{session}: {qc["n_bundles"]} bundles'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--config', default='pc', choices=['pc', 'cubic'])
    parser.add_argument('--participant', nargs='*')
    parser.add_argument('--out-dir')
    parser.add_argument('--n-jobs', type=int, default=6)
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute sessions that already have outputs.')
    args = parser.parse_args(argv)

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    from configuration.config import load_config
    from joblib import Parallel, delayed

    cfg = load_config(args.config)
    recon_dir = os.path.dirname(cfg['derivatives']['qsirecon_dsistudio'])
    qsiprep_dir = cfg['derivatives']['qsiprep']
    out_dir = args.out_dir or default_out_dir(cfg)
    os.makedirs(os.path.join(out_dir, 'nodes'), exist_ok=True)

    jobs = []
    for tck_dir in sorted(glob(f'{recon_dir}/qsirecon-MSMTAutoTrack/sub-*/ses-*/dwi')):
        participant = tck_dir.split('sub-')[1].split(os.sep)[0].split('/')[0]
        session = tck_dir.split('ses-')[1].split('/')[0]
        if args.participant and participant not in args.participant:
            continue
        jobs.append((recon_dir, qsiprep_dir, cfg['derivatives']['smriprep'],
                     cfg['derivatives'].get('megre'), participant, session, out_dir,
                     args.overwrite))

    # Per-subject T1w -> ACPC registrations first, so sessions don't race on the cache.
    def _register(participant):
        try:
            acpc_to_t1w(cfg['derivatives']['smriprep'], qsiprep_dir, participant,
                        os.path.join(out_dir, 'xfm'))
        except Exception as exc:
            print(f'[WARNING] T1w->ACPC registration failed for sub-{participant}: {exc!r}')

    Parallel(n_jobs=args.n_jobs)(delayed(_register)(p) for p in sorted({j[4] for j in jobs}))
    for msg in Parallel(n_jobs=args.n_jobs, verbose=0)(delayed(_run_one)(j) for j in jobs):
        print(f'[INFO] {msg}', flush=True)


if __name__ == '__main__':
    main()
