// Copyright 2026 Newton Campbell
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

/// @file transform_ops.hpp
/// @brief SE(3) homogeneous transform construction, composition, and inversion.
///
/// Convention (data_contract.md §4 and §3):
///   Transforms are 4×4 float64 matrices:
//
///       T = | R  t |
///           | 0  1 |
///
///   where R ∈ SO(3) is the 3×3 rotation block and t is the 3-vector
///   translation expressed in the *parent* frame.
///
///   makeTransform(position, q_wxyz) builds T_WB:
///
///       T_WB = | R(q_i2b)  p_W |
///              | 0  0  0    1  |
///
///   where p_W is the aircraft position in the World (NED) frame and
///   R(q_i2b) is derived from the scalar-first quaternion per §4.1.

#pragma once

#include <stdexcept>

#include "bam_transform_math/types.hpp"
#include "bam_transform_math/quat_utils.hpp"

namespace bam_transform_math
{

/// @brief Build a 4×4 SE(3) homogeneous transform from position + quaternion.
///
/// @param position   Translation [x, y, z] in the parent frame (metres, NED).
/// @param q_wxyz     Rotation quaternion [w, x, y, z] (scalar-first, BAM convention).
///                   Normalised internally.
/// @returns          4×4 homogeneous transform T.
/// @throws std::invalid_argument on zero-norm quaternion.
[[nodiscard]] inline Mat4 makeTransform(const Vec3 & position, const Vec4 & q_wxyz)
{
  const Mat3 R = quatToRotmat(q_wxyz);

  Mat4 T = Mat4::Identity();
  T.topLeftCorner<3, 3>() = R;
  T.topRightCorner<3, 1>() = position;
  return T;
}

/// @brief Compose two SE(3) transforms: T_AC = T_AB * T_BC.
///
/// Subscript chaining rule: the inner frame label must match.
/// e.g.  T_WC = composeTransforms(T_WB, T_BoC)
///
/// @param T_AB  Transform from frame A to frame B.
/// @param T_BC  Transform from frame B to frame C.
/// @returns     T_AC = T_AB * T_BC.
[[nodiscard]] inline Mat4 composeTransforms(const Mat4 & T_AB, const Mat4 & T_BC)
{
  return T_AB * T_BC;
}

/// @brief Analytically invert an SE(3) homogeneous transform.
///
/// For T = | R  t |, the inverse is T⁻¹ = | Rᵀ  −Rᵀt |
///         | 0  1 |                        | 0     1  |
///
/// This exploits R⁻¹ = Rᵀ for orthogonal matrices and is more numerically
/// stable than a general matrix inverse.
///
/// @param T  Valid SE(3) transform matrix.
/// @returns  T_inv such that T * T_inv ≈ Identity (error < 1e-12 for well-
///           conditioned inputs).
[[nodiscard]] inline Mat4 invertTransform(const Mat4 & T)
{
  const Mat3 R = T.topLeftCorner<3, 3>();
  const Vec3 t = T.topRightCorner<3, 1>();

  const Mat3 Rt = R.transpose();

  Mat4 T_inv = Mat4::Identity();
  T_inv.topLeftCorner<3, 3>() = Rt;
  T_inv.topRightCorner<3, 1>() = -Rt * t;
  return T_inv;
}

/// @brief Apply an SE(3) transform to a 3D point.
///
/// Convenience wrapper that avoids manual homogeneous lifting.
///
/// @param T  4×4 SE(3) transform.
/// @param p  3D point in the source frame.
/// @returns  Transformed 3D point.
[[nodiscard]] inline Vec3 applyTransform(const Mat4 & T, const Vec3 & p)
{
  const Eigen::Vector4d p_h{p[0], p[1], p[2], 1.0};
  return (T * p_h).head<3>();
}

/// @brief Apply only the rotation block of a transform to a 3D vector.
///
/// Equivalent to T.topLeftCorner<3,3>() * v but reads intent more clearly.
///
/// @param T  4×4 SE(3) transform.
/// @param v  3D vector (direction, not a point — translation is NOT applied).
/// @returns  Rotated 3D vector.
[[nodiscard]] inline Vec3 applyRotation(const Mat4 & T, const Vec3 & v)
{
  return T.topLeftCorner<3, 3>() * v;
}

}  // namespace bam_transform_math
