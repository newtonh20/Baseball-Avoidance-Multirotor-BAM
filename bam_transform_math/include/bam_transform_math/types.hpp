// Copyright 2026 Newton Campbell
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// @file types.hpp
/// @brief Canonical type aliases for the bam_transform_math library.
///
/// All library headers use these aliases exclusively.  Callers may also
/// use them directly for interoperability with Eigen.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace bam_transform_math
{

/// Scalar type used throughout the library.  float64 (double) everywhere.
using Scalar = double;

/// 3-vector (translation / position).
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

/// 3×3 rotation matrix.
using Mat3 = Eigen::Matrix<Scalar, 3, 3>;

/// 4×4 homogeneous SE(3) transform.
using Mat4 = Eigen::Matrix<Scalar, 4, 4>;

/// 2-vector (pixel / ray-in-image-plane coordinates).
using Vec2 = Eigen::Matrix<Scalar, 2, 1>;

/// 3×3 camera intrinsic matrix.
using Mat3K = Eigen::Matrix<Scalar, 3, 3>;

/// Quaternion stored as Eigen::Quaternion<double>.
/// Eigen stores quaternions internally as [x, y, z, w] but exposes them
/// through the .w() .x() .y() .z() accessors.  All public API in this
/// library accepts and returns the BAM convention [w, x, y, z] as a Vec4.
using Quat = Eigen::Quaternion<Scalar>;

/// 4-vector used exclusively for scalar-first quaternion representation
/// [w, x, y, z] per data_contract.md §4.1.
using Vec4 = Eigen::Matrix<Scalar, 4, 1>;

}  // namespace bam_transform_math
