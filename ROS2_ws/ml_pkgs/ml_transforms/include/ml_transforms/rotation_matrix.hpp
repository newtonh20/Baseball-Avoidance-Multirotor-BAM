#pragma once

#include <Eigen/Core>

#include "ml_transforms/quaternion.hpp"

namespace ml_transforms {

/**
 * Convert scalar-first quaternion [w, x, y, z] into rotation matrix.
 *
 * Input quaternion is normalized before conversion.
 * Throws std::invalid_argument for near-zero norm.
 */
Eigen::Matrix3d quatToRotmat(const QuaternionWXYZ& q_wxyz);

/**
 * Convert rotation matrix into scalar-first quaternion [w, x, y, z].
 *
 * Uses a numerically stable branch method and enforces w >= 0 for canonical
 * output sign. Throws std::invalid_argument if R is not orthonormal.
 */
QuaternionWXYZ rotmatToQuat(const Eigen::Matrix3d& R);

/**
 * Check whether a matrix is a valid right-handed rotation matrix.
 */
bool isValidRotationMatrix(const Eigen::Matrix3d& R, double orthonormal_tolerance,
                           double determinant_tolerance);

}  // namespace ml_transforms
