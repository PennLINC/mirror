"""Tests for extract_b0_pose geometry and transform handling."""

import os
import sys

import nibabel as nb
import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import extract_b0_pose as ebp  # noqa: E402


def _rot_x(deg):
    t = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])


def _rot_y(deg):
    t = np.radians(deg)
    return np.array([[np.cos(t), 0, np.sin(t)], [0, 1, 0], [-np.sin(t), 0, np.cos(t)]])


def _affine(rot, trans=(0, 0, 0)):
    mat = np.eye(4)
    mat[:3, :3] = rot
    mat[:3, 3] = trans
    return mat


def test_b0_identity():
    b0 = ebp.b0_in_reference(np.eye(4))
    np.testing.assert_allclose(b0, [0, 0, 1])
    assert ebp.pose_angles(b0) == pytest.approx({'tilt_deg': 0, 'pitch_deg': 0, 'roll_deg': 0})


def test_b0_pitch_and_roll():
    # ACPC->scanner rotation of +10 deg about x: B0 in ACPC is R^T z.
    b0 = ebp.b0_in_reference(_affine(_rot_x(10), (5, -3, 40)))
    angles = ebp.pose_angles(b0)
    assert angles['tilt_deg'] == pytest.approx(10)
    assert abs(angles['pitch_deg']) == pytest.approx(10)
    assert angles['roll_deg'] == pytest.approx(0, abs=1e-9)
    # Rotation about y changes roll only.
    angles = ebp.pose_angles(ebp.b0_in_reference(_affine(_rot_y(-7))))
    assert abs(angles['roll_deg']) == pytest.approx(7)
    assert angles['pitch_deg'] == pytest.approx(0, abs=1e-9)


def test_b0_uses_transpose_not_rotation():
    # For a rotation about x, R^T z and R z differ in the sign of y.
    rot = _rot_x(20)
    b0 = ebp.b0_in_reference(_affine(rot))
    np.testing.assert_allclose(b0, ebp.normalize_axial(rot.T @ [0, 0, 1]))
    assert not np.allclose(b0, ebp.normalize_axial(rot @ [0, 0, 1]))


def test_polar_rotation_strips_scale():
    rot = _rot_x(15) @ _rot_y(5)
    recovered, scale_dev = ebp.polar_rotation(rot @ np.diag([1.02, 0.99, 1.01]))
    np.testing.assert_allclose(recovered, rot, atol=0.02)
    assert scale_dev == pytest.approx(0.02, abs=0.005)
    _, scale_dev = ebp.polar_rotation(rot)
    assert scale_dev == pytest.approx(0, abs=1e-12)


def test_axial_helpers():
    a = np.array([0, 0.1, 1.0])
    assert ebp.axial_angle_deg(a, -a) == pytest.approx(0, abs=1e-6)
    assert ebp.axial_angle_deg([1, 0, 0], [0, 0, 1]) == pytest.approx(90)
    mean = ebp.mean_axial([ebp.normalize_axial(_rot_x(2) @ [0, 0, 1]),
                           -ebp.normalize_axial(_rot_x(-2) @ [0, 0, 1])])
    np.testing.assert_allclose(mean, [0, 0, 1], atol=1e-12)


def test_point_map_to_matrix():
    mat = _affine(_rot_x(12) @ _rot_y(-4), (3, -8, 21))
    recovered = ebp.point_map_to_matrix(lambda p: (mat @ np.r_[p, 1])[:3])
    np.testing.assert_allclose(recovered, mat, atol=1e-12)


def test_load_itk_affine_lps_to_ras(tmp_path):
    ants = pytest.importorskip('ants')
    lps = _affine(_rot_x(9) @ _rot_y(3), (2, -5, 11))
    tx = ants.create_ants_transform(
        transform_type='AffineTransform', dimension=3,
        matrix=lps[:3, :3], translation=lps[:3, 3],
    )
    path = str(tmp_path / 'xfm.mat')
    ants.write_transform(tx, path)
    np.testing.assert_allclose(
        ebp.load_itk_affine(path), ebp.LPS_TO_RAS @ lps @ ebp.LPS_TO_RAS, atol=1e-6
    )


def _blob_image(shape=(48, 48, 48)):
    rng = np.random.default_rng(0)
    data = ndimage.gaussian_filter(rng.random(shape), 3)
    data[data < np.percentile(data, 30)] = 0
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[:3, 3] = -np.array(shape) + 1
    return nb.Nifti1Image(data.astype(np.float32), affine)


def test_validate_direction_prefers_correct_transform():
    ref = _blob_image()
    ref_to_mov = _affine(_rot_x(25) @ _rot_y(10), (4, -2, 6))
    # Build the moving image so that ref_to_mov is exactly the true pull-back map: the moving
    # image's world frame is the reference frame pushed through ref_to_mov.
    mov = nb.Nifti1Image(np.asarray(ref.dataobj), ref_to_mov @ ref.affine)
    check = ebp.validate_direction(ref, mov, ref_to_mov)
    assert check['sim_fwd'] == pytest.approx(1, abs=1e-3)
    assert check['direction_ok']
    flipped = ebp.validate_direction(ref, mov, np.linalg.inv(ref_to_mov))
    assert not flipped['direction_ok']
