"""
test_transform_math.py
======================
Unit tests for the ml_transforms.transform_math module.

All tests use only numpy and pytest – no ROS runtime required.
Tests are discoverable by both:
  * ``pytest ROS2_ws/ml_pkgs/ml_transforms/test/ -v``  (local / CI)
  * ``colcon test --packages-select ml_transforms``      (ROS2 Jazzy workspace)

Golden tests
------------
At least 5 tests with analytically pre-computed expected values:
  1. test_identity_transform
  2. test_roll_90
  3. test_pitch_90
  4. test_yaw_90
  5. test_rotmat_roundtrip_roll
  6. test_rotmat_roundtrip_pitch
  7. test_rotmat_roundtrip_yaw

Numerical tolerance for golden tests: < 1e-6 (see RTOL / ATOL below).
"""

import math
import numpy as np
import pytest

from ml_transforms.transform_math import (
    make_transform,
    compose_transforms,
    invert_transform,
    quat_to_rotmat,
    rotmat_to_quat,
    pixel_to_ray,
    ray_to_pixel,
)

# Tolerances
ATOL_GOLDEN = 1e-6   # acceptance criterion stated in ticket
ATOL_TIGHT  = 1e-10  # for algebraically exact operations


# ===========================================================================
# Helpers
# ===========================================================================

def _quat_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return unit quaternion [w, x, y, z] for rotation of *angle_rad* about *axis*."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([math.cos(half), *(math.sin(half) * axis)])


# ===========================================================================
# 1. Identity tests
# ===========================================================================

class TestIdentity:
    """Golden test 1: identity transform."""

    def test_identity_transform(self):
        """make_transform([0,0,0], [1,0,0,0]) must equal the 4×4 identity."""
        T = make_transform([0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        assert T.shape == (4, 4)
        np.testing.assert_allclose(T, np.eye(4), atol=ATOL_GOLDEN)

    def test_identity_inversion(self):
        """Inverting the 4×4 identity must return the 4×4 identity."""
        T_inv = invert_transform(np.eye(4))
        np.testing.assert_allclose(T_inv, np.eye(4), atol=ATOL_TIGHT)

    def test_identity_composition(self):
        """T @ T_inv must equal I4 for an arbitrary non-trivial transform."""
        q = _quat_axis_angle([0, 0, 1], math.radians(37.5))
        T = make_transform([1.2, -0.4, 3.1], q)
        T_inv = invert_transform(T)
        result = compose_transforms(T, T_inv)
        np.testing.assert_allclose(result, np.eye(4), atol=ATOL_TIGHT)


# ===========================================================================
# 2. Known-rotation golden tests  (analytically pre-computed)
# ===========================================================================

class TestGoldenRotations:
    """Golden tests 2-4: 90-degree single-axis rotations."""

    # --- Roll: 90° around X-axis -------------------------------------------
    # q = [cos45°, sin45°, 0, 0] = [√2/2, √2/2, 0, 0]
    # R_x90 = [[1, 0,  0],
    #          [0, 0, -1],
    #          [0, 1,  0]]
    R_ROLL_90 = np.array([
        [1,  0,  0],
        [0,  0, -1],
        [0,  1,  0],
    ], dtype=np.float64)

    # --- Pitch: 90° around Y-axis ------------------------------------------
    # q = [cos45°, 0, sin45°, 0]
    # R_y90 = [[ 0, 0, 1],
    #          [ 0, 1, 0],
    #          [-1, 0, 0]]
    R_PITCH_90 = np.array([
        [ 0,  0,  1],
        [ 0,  1,  0],
        [-1,  0,  0],
    ], dtype=np.float64)

    # --- Yaw: 90° around Z-axis --------------------------------------------
    # q = [cos45°, 0, 0, sin45°]
    # R_z90 = [[0, -1, 0],
    #          [1,  0, 0],
    #          [0,  0, 1]]
    R_YAW_90 = np.array([
        [0, -1,  0],
        [1,  0,  0],
        [0,  0,  1],
    ], dtype=np.float64)

    def test_roll_90(self):
        """Golden test 2: 90° roll (X-axis) quaternion → rotation matrix."""
        q = _quat_axis_angle([1, 0, 0], math.radians(90))
        R = quat_to_rotmat(q)
        np.testing.assert_allclose(R, self.R_ROLL_90, atol=ATOL_GOLDEN)

    def test_pitch_90(self):
        """Golden test 3: 90° pitch (Y-axis) quaternion → rotation matrix."""
        q = _quat_axis_angle([0, 1, 0], math.radians(90))
        R = quat_to_rotmat(q)
        np.testing.assert_allclose(R, self.R_PITCH_90, atol=ATOL_GOLDEN)

    def test_yaw_90(self):
        """Golden test 4: 90° yaw (Z-axis) quaternion → rotation matrix."""
        q = _quat_axis_angle([0, 0, 1], math.radians(90))
        R = quat_to_rotmat(q)
        np.testing.assert_allclose(R, self.R_YAW_90, atol=ATOL_GOLDEN)

    def test_rotmat_roundtrip_roll(self):
        """Golden test 5: R_roll90 → quat → R' must have ‖R-R'‖ < 1e-6."""
        q = rotmat_to_quat(self.R_ROLL_90)
        R_prime = quat_to_rotmat(q)
        np.testing.assert_allclose(R_prime, self.R_ROLL_90, atol=ATOL_GOLDEN)

    def test_rotmat_roundtrip_pitch(self):
        """Golden test 6: R_pitch90 → quat → R' must have ‖R-R'‖ < 1e-6."""
        q = rotmat_to_quat(self.R_PITCH_90)
        R_prime = quat_to_rotmat(q)
        np.testing.assert_allclose(R_prime, self.R_PITCH_90, atol=ATOL_GOLDEN)

    def test_rotmat_roundtrip_yaw(self):
        """Golden test 7: R_yaw90 → quat → R' must have ‖R-R'‖ < 1e-6."""
        q = rotmat_to_quat(self.R_YAW_90)
        R_prime = quat_to_rotmat(q)
        np.testing.assert_allclose(R_prime, self.R_YAW_90, atol=ATOL_GOLDEN)


# ===========================================================================
# 3. Inversion correctness
# ===========================================================================

class TestInversion:

    def test_inversion_general(self):
        """T @ invert_transform(T) = I4 within 1e-6 for a non-trivial T."""
        q = _quat_axis_angle([0, 0, 1], math.radians(45))
        T = make_transform([3.0, -1.5, 0.7], q)
        T_inv = invert_transform(T)
        np.testing.assert_allclose(T @ T_inv, np.eye(4), atol=ATOL_GOLDEN)

    def test_inversion_vs_numpy(self):
        """Analytical inverse matches np.linalg.inv(T) within 1e-10."""
        q = _quat_axis_angle([1, 1, 0], math.radians(60))
        T = make_transform([0.5, 2.0, -1.0], q)
        T_inv_analytic = invert_transform(T)
        T_inv_numpy = np.linalg.inv(T)
        np.testing.assert_allclose(T_inv_analytic, T_inv_numpy, atol=ATOL_TIGHT)

    def test_double_inversion_is_original(self):
        """invert(invert(T)) == T."""
        q = _quat_axis_angle([0, 1, 1], math.radians(30))
        T = make_transform([1.0, 2.0, 3.0], q)
        np.testing.assert_allclose(invert_transform(invert_transform(T)), T, atol=ATOL_TIGHT)


# ===========================================================================
# 4. Composition correctness
# ===========================================================================

class TestComposition:

    def test_composition_known_translation(self):
        """Identity · pure-translation T_BC → result has t = [1, 0, 0]."""
        T_WB = np.eye(4)
        T_BC = make_transform([1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        T_WC = compose_transforms(T_WB, T_BC)
        np.testing.assert_allclose(T_WC[:3, 3], [1.0, 0.0, 0.0], atol=ATOL_GOLDEN)
        np.testing.assert_allclose(T_WC[:3, :3], np.eye(3), atol=ATOL_GOLDEN)

    def test_composition_associativity(self):
        """(T_AB @ T_BC) @ T_CD == T_AB @ (T_BC @ T_CD) within 1e-10."""
        q1 = _quat_axis_angle([1, 0, 0], math.radians(30))
        q2 = _quat_axis_angle([0, 1, 0], math.radians(45))
        q3 = _quat_axis_angle([0, 0, 1], math.radians(60))
        T_AB = make_transform([1.0, 0.0, 0.0], q1)
        T_BC = make_transform([0.0, 2.0, 0.0], q2)
        T_CD = make_transform([0.0, 0.0, 3.0], q3)
        lhs = compose_transforms(compose_transforms(T_AB, T_BC), T_CD)
        rhs = compose_transforms(T_AB, compose_transforms(T_BC, T_CD))
        np.testing.assert_allclose(lhs, rhs, atol=ATOL_TIGHT)

    def test_composition_inverse_identity(self):
        """T_AB @ T_BA = I4 where T_BA = invert(T_AB)."""
        q = _quat_axis_angle([1, 2, 3], math.radians(80))
        T_AB = make_transform([4.0, -2.0, 1.5], q)
        T_BA = invert_transform(T_AB)
        np.testing.assert_allclose(
            compose_transforms(T_AB, T_BA), np.eye(4), atol=ATOL_TIGHT
        )


# ===========================================================================
# 5. Quaternion-order mistake detection
# ===========================================================================

class TestQuaternionOrderGuard:
    """
    These tests MUST PASS by asserting that the wrong quaternion order
    produces a DIFFERENT result.  If quat_to_rotmat were order-agnostic,
    these tests would catch the silent bug.
    """

    def test_wxyz_vs_xyzw_are_distinguishable_roll90(self):
        """
        For 90° roll: correct = [w,x,y,z] = [√2/2, √2/2, 0, 0].
        If someone passes [x,y,z,w] = [√2/2, 0, 0, √2/2] by mistake,
        the resulting R must differ from the correct R_x90.
        """
        # Correct [w, x, y, z] for 90° roll
        q_correct = _quat_axis_angle([1, 0, 0], math.radians(90))  # [√2/2, √2/2, 0, 0]
        # Wrong order: treat [x, y, z, w] as if it were [w, x, y, z]
        # i.e. pass [x, y, z, w] = [√2/2, 0, 0, √2/2]
        q_wrong_order = np.array([q_correct[1], q_correct[2], q_correct[3], q_correct[0]])

        R_correct = quat_to_rotmat(q_correct)
        R_wrong   = quat_to_rotmat(q_wrong_order)

        # These MUST differ – assert not allclose
        assert not np.allclose(R_correct, R_wrong, atol=ATOL_GOLDEN), (
            'BUG: quat_to_rotmat is insensitive to quaternion component order! '
            'The [w,x,y,z] and [x,y,z,w] forms produced identical rotation matrices.'
        )

    def test_wxyz_vs_xyzw_are_distinguishable_yaw90(self):
        """Same guard for 90° yaw: [w,x,y,z] vs [x,y,z,w] must differ."""
        q_correct = _quat_axis_angle([0, 0, 1], math.radians(90))  # [√2/2, 0, 0, √2/2]
        q_wrong_order = np.array([q_correct[1], q_correct[2], q_correct[3], q_correct[0]])

        R_correct = quat_to_rotmat(q_correct)
        R_wrong   = quat_to_rotmat(q_wrong_order)

        assert not np.allclose(R_correct, R_wrong, atol=ATOL_GOLDEN), (
            'BUG: quat_to_rotmat produced the same R for both [w,x,y,z] '
            'and [x,y,z,w] on a 90° yaw quaternion.'
        )


# ===========================================================================
# 6. Pixel / ray stubs (WP2 guard)
# ===========================================================================

class TestPixelRayStubs:

    def test_pixel_to_ray_raises(self):
        """pixel_to_ray must raise NotImplementedError until WP2."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        with pytest.raises(NotImplementedError):
            pixel_to_ray([320.0, 240.0], K)

    def test_ray_to_pixel_raises(self):
        """ray_to_pixel must raise NotImplementedError until WP2."""
        K = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float64)
        with pytest.raises(NotImplementedError):
            ray_to_pixel([0.0, 0.0, 1.0], K)


# ===========================================================================
# 7. Input validation
# ===========================================================================

class TestInputValidation:

    def test_bad_position_shape_raises(self):
        with pytest.raises(ValueError):
            make_transform([1, 2], [1, 0, 0, 0])  # position must be (3,)

    def test_bad_quaternion_shape_raises(self):
        with pytest.raises(ValueError):
            make_transform([0, 0, 0], [1, 0, 0])   # quaternion must be (4,)

    def test_zero_quaternion_raises(self):
        with pytest.raises(ValueError):
            quat_to_rotmat([0, 0, 0, 0])

    def test_bad_transform_shape_invert(self):
        with pytest.raises(ValueError):
            invert_transform(np.eye(3))             # must be (4,4)

    def test_bad_transform_shape_compose(self):
        with pytest.raises(ValueError):
            compose_transforms(np.eye(3), np.eye(4))
