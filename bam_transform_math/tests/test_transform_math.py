"""
Tests for bam_transform_math.transform_math
============================================
All tests use only numpy and pytest; no external fixtures.
"""

import math
import numpy as np
import pytest

from bam_transform_math import (
    make_transform,
    compose_transforms,
    invert_transform,
    quat_to_rotmat,
    rotmat_to_quat,
    pixel_to_ray,
    ray_to_pixel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quat_axis_angle(axis, angle_rad):
    """Return quaternion [w, x, y, z] for a rotation of angle_rad about axis."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([math.cos(half), *(math.sin(half) * axis)])


# ---------------------------------------------------------------------------
# Identity tests
# ---------------------------------------------------------------------------

class TestIdentity:
    def test_identity_transform(self):
        """make_transform with zero translation and identity quaternion → I4."""
        T = make_transform([0, 0, 0], [1, 0, 0, 0])
        assert T.shape == (4, 4)
        assert T.dtype == np.float64
        np.testing.assert_array_equal(T, np.eye(4))

    def test_identity_inversion(self):
        """Inverting I4 yields I4."""
        T_inv = invert_transform(np.eye(4))
        np.testing.assert_allclose(T_inv, np.eye(4), atol=1e-10)

    def test_identity_composition(self):
        """T @ T_inv yields I4 within 1e-10."""
        q = _quat_axis_angle([0, 0, 1], math.radians(37))
        T = make_transform([1.5, -2.3, 0.7], q)
        T_inv = invert_transform(T)
        product = compose_transforms(T, T_inv)
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)


# ---------------------------------------------------------------------------
# Golden rotation tests (analytically known outputs, tolerance < 1e-6)
# ---------------------------------------------------------------------------

class TestGoldenRotations:
    """Analytically known rotation matrices for 90° rotations about principal axes."""

    # 90° roll (rotation about X): [0,1,0] → [0,0,1], [0,0,1] → [0,-1,0]
    R_roll_90 = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ], dtype=np.float64)

    # 90° pitch (rotation about Y): [1,0,0] → [0,0,-1], [0,0,1] → [1,0,0]
    R_pitch_90 = np.array([
        [0,  0,  1],
        [0,  1,  0],
        [-1, 0,  0],
    ], dtype=np.float64)

    # 90° yaw (rotation about Z): [1,0,0] → [0,1,0], [0,1,0] → [-1,0,0]
    R_yaw_90 = np.array([
        [0, -1,  0],
        [1,  0,  0],
        [0,  0,  1],
    ], dtype=np.float64)

    def test_roll_90(self):
        """Quaternion for 90° roll about X → known rotation matrix."""
        q = _quat_axis_angle([1, 0, 0], math.radians(90))
        R = quat_to_rotmat(q)
        np.testing.assert_allclose(R, self.R_roll_90, atol=1e-6)

    def test_pitch_90(self):
        """Quaternion for 90° pitch about Y → known rotation matrix."""
        q = _quat_axis_angle([0, 1, 0], math.radians(90))
        R = quat_to_rotmat(q)
        np.testing.assert_allclose(R, self.R_pitch_90, atol=1e-6)

    def test_yaw_90(self):
        """Quaternion for 90° yaw about Z → known rotation matrix."""
        q = _quat_axis_angle([0, 0, 1], math.radians(90))
        R = quat_to_rotmat(q)
        np.testing.assert_allclose(R, self.R_yaw_90, atol=1e-6)

    def test_rotmat_roundtrip(self):
        """R → quat → R' should have ||R - R'|| < 1e-6 for all golden cases."""
        for R_golden in (self.R_roll_90, self.R_pitch_90, self.R_yaw_90):
            q = rotmat_to_quat(R_golden)
            R_recovered = quat_to_rotmat(q)
            diff = np.linalg.norm(R_golden - R_recovered)
            assert diff < 1e-6, f"Roundtrip error {diff} exceeds 1e-6"


# ---------------------------------------------------------------------------
# Inversion correctness
# ---------------------------------------------------------------------------

class TestInversion:
    def test_inversion_general(self):
        """T @ invert_transform(T) = I4 within 1e-6 for a non-trivial transform."""
        q = _quat_axis_angle([0, 0, 1], math.radians(45))
        T = make_transform([3.0, -1.0, 2.5], q)
        T_inv = invert_transform(T)
        product = compose_transforms(T, T_inv)
        np.testing.assert_allclose(product, np.eye(4), atol=1e-6)

    def test_inversion_vs_numpy(self):
        """Analytical inverse matches np.linalg.inv(T) within 1e-10."""
        q = _quat_axis_angle([1, 1, 0], math.radians(60))
        T = make_transform([0.5, 1.5, -2.0], q)
        T_inv_analytical = invert_transform(T)
        T_inv_numpy = np.linalg.inv(T)
        np.testing.assert_allclose(T_inv_analytical, T_inv_numpy, atol=1e-10)


# ---------------------------------------------------------------------------
# Composition correctness
# ---------------------------------------------------------------------------

class TestComposition:
    def test_composition_associativity(self):
        """(T_AB @ T_BC) @ T_CD == T_AB @ (T_BC @ T_CD) within 1e-10."""
        q_ab = _quat_axis_angle([0, 0, 1], math.radians(30))
        q_bc = _quat_axis_angle([1, 0, 0], math.radians(45))
        q_cd = _quat_axis_angle([0, 1, 0], math.radians(60))

        T_AB = make_transform([1, 0, 0], q_ab)
        T_BC = make_transform([0, 2, 0], q_bc)
        T_CD = make_transform([0, 0, 3], q_cd)

        lhs = compose_transforms(compose_transforms(T_AB, T_BC), T_CD)
        rhs = compose_transforms(T_AB, compose_transforms(T_BC, T_CD))
        np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_composition_known(self):
        """Identity composed with pure translation [1,0,0] → T_WC has t=[1,0,0]."""
        T_WB = np.eye(4)
        T_BC = make_transform([1, 0, 0], [1, 0, 0, 0])
        T_WC = compose_transforms(T_WB, T_BC)
        np.testing.assert_allclose(T_WC[:3, 3], [1, 0, 0], atol=1e-15)
        np.testing.assert_allclose(T_WC[:3, :3], np.eye(3), atol=1e-15)


# ---------------------------------------------------------------------------
# Quaternion order detection
# ---------------------------------------------------------------------------

class TestQuatConvention:
    def test_quat_wxyz_vs_xyzw_distinguishable(self):
        """[x,y,z,w] (wrong order) gives a DIFFERENT rotation matrix than [w,x,y,z]."""
        # 90° yaw: correct quaternion is [w, x, y, z] = [cos45°, 0, 0, sin45°]
        q_correct = _quat_axis_angle([0, 0, 1], math.radians(90))  # [w, x, y, z]
        # Feed the components in the wrong order: [x, y, z, w]
        q_wrong = np.array([q_correct[1], q_correct[2], q_correct[3], q_correct[0]])

        R_correct = quat_to_rotmat(q_correct)
        R_wrong = quat_to_rotmat(q_wrong)

        assert not np.allclose(R_correct, R_wrong), (
            "Expected different rotation matrices for [w,x,y,z] vs [x,y,z,w] but got the same."
        )


# ---------------------------------------------------------------------------
# Pixel/ray stubs
# ---------------------------------------------------------------------------

class TestStubs:
    def test_pixel_to_ray_stub(self):
        """pixel_to_ray raises NotImplementedError."""
        K = np.eye(3)
        uv = np.array([320.0, 240.0])
        with pytest.raises(NotImplementedError):
            pixel_to_ray(uv, K)

    def test_ray_to_pixel_stub(self):
        """ray_to_pixel raises NotImplementedError."""
        K = np.eye(3)
        ray = np.array([0.0, 0.0, 1.0])
        with pytest.raises(NotImplementedError):
            ray_to_pixel(ray, K)
