// Copyright 2026 Newton Campbell
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

/// @file quat_utils.hpp
/// @brief Quaternion ↔ rotation-matrix conversion utilities.
///
/// Convention (data_contract.md §4):
///   - Quaternions are represented as Vec4 [w, x, y, z] (scalar-first).
///   - q_i2b: rotates a vector FROM World (NED inertial) INTO body frame.
///   - ZYX aerospace (3-2-1) Euler sequence: ψ yaw → θ pitch → φ roll.
///   - Right-handed, consistent with NED world frame.

#pragma once

#include <stdexcept>
#include <cmath>

#include "bam_transform_math/types.hpp"

namespace bam_transform_math
{

/// @brief Convert a scalar-first [w, x, y, z] quaternion to a 3×3 rotation matrix.
///
/// Uses the direct SU(2)→SO(3) homomorphism formula.  The input quaternion
/// is normalized before use; a zero-norm quaternion throws std::invalid_argument.
///
/// @param q_wxyz  Quaternion as [w, x, y, z] (scalar first, BAM convention).
/// @returns       3×3 rotation matrix R such that v_body = R * v_world when
///                q_wxyz encodes q_i2b.
/// @throws std::invalid_argument if q has zero norm.
[[nodiscard]] inline Mat3 quatToRotmat(const Vec4 & q_wxyz)
{
  const Scalar norm = q_wxyz.norm();
  if (norm < std::numeric_limits<Scalar>::epsilon() * 10.0) {
    throw std::invalid_argument("bam_transform_math::quatToRotmat: quaternion has zero norm");
  }

  // Normalise and unpack — BAM convention: q = [w, x, y, z]
  const Vec4 q = q_wxyz / norm;
  const Scalar w = q[0];
  const Scalar x = q[1];
  const Scalar y = q[2];
  const Scalar z = q[3];

  // Standard formula: R = I + 2w[v]× + 2[v]×²  (Rodrigues form expanded)
  Mat3 R;
  R(0, 0) = w*w + x*x - y*y - z*z;
  R(0, 1) = 2.0*(x*y - w*z);
  R(0, 2) = 2.0*(x*z + w*y);
  R(1, 0) = 2.0*(x*y + w*z);
  R(1, 1) = w*w - x*x + y*y - z*z;
  R(1, 2) = 2.0*(y*z - w*x);
  R(2, 0) = 2.0*(x*z - w*y);
  R(2, 1) = 2.0*(y*z + w*x);
  R(2, 2) = w*w - x*x - y*y + z*z;

  return R;
}

/// @brief Convert a 3×3 rotation matrix to a scalar-first [w, x, y, z] quaternion.
///
/// Uses Shepperd's method with stable branch selection to avoid numerical
/// issues near singular configurations (trace ≈ −1).
///
/// @param R  Valid rotation matrix (orthogonal, det ≈ +1).
/// @returns  Unit quaternion [w, x, y, z] with w ≥ 0 (canonical form).
/// @throws std::invalid_argument if R is not 3×3 (compile-time shape is
///         guaranteed by type, so this is effectively unreachable).
[[nodiscard]] inline Vec4 rotmatToQuat(const Mat3 & R)
{
  Scalar w, x, y, z;
  const Scalar trace = R(0, 0) + R(1, 1) + R(2, 2);

  if (trace > 0.0) {
    const Scalar s = 0.5 / std::sqrt(trace + 1.0);
    w = 0.25 / s;
    x = (R(2, 1) - R(1, 2)) * s;
    y = (R(0, 2) - R(2, 0)) * s;
    z = (R(1, 0) - R(0, 1)) * s;
  } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
    const Scalar s = 2.0 * std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2));
    w = (R(2, 1) - R(1, 2)) / s;
    x = 0.25 * s;
    y = (R(0, 1) + R(1, 0)) / s;
    z = (R(0, 2) + R(2, 0)) / s;
  } else if (R(1, 1) > R(2, 2)) {
    const Scalar s = 2.0 * std::sqrt(1.0 - R(0, 0) + R(1, 1) - R(2, 2));
    w = (R(0, 2) - R(2, 0)) / s;
    x = (R(0, 1) + R(1, 0)) / s;
    y = 0.25 * s;
    z = (R(1, 2) + R(2, 1)) / s;
  } else {
    const Scalar s = 2.0 * std::sqrt(1.0 - R(0, 0) - R(1, 1) + R(2, 2));
    w = (R(1, 0) - R(0, 1)) / s;
    x = (R(0, 2) + R(2, 0)) / s;
    y = (R(1, 2) + R(2, 1)) / s;
    z = 0.25 * s;
  }

  Vec4 q;
  q << w, x, y, z;
  q.normalize();

  // Canonical form: ensure w ≥ 0 (double-cover equivalence q ≡ −q)
  if (q[0] < 0.0) { q = -q; }

  return q;
}

}  // namespace bam_transform_math
