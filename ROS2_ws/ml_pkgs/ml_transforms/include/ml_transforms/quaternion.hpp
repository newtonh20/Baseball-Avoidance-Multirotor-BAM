#pragma once

#include <Eigen/Core>

namespace ml_transforms {

/**
 * Quaternion with scalar-first storage order [w, x, y, z].
 *
 * This package intentionally uses [w, x, y, z] to match the BAM synthetic-data
 * contract. This differs from geometry_msgs::msg::Quaternion field order
 * (x, y, z, w).
 */
struct QuaternionWXYZ {
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

/**
 * Return Euclidean norm of quaternion components.
 */
double quaternionNorm(const QuaternionWXYZ& q);

/**
 * Normalize quaternion and return scalar-first [w, x, y, z].
 *
 * Throws std::invalid_argument if norm is near zero.
 */
QuaternionWXYZ normalizeQuaternion(const QuaternionWXYZ& q);

/**
 * Quaternion conjugate in scalar-first convention.
 */
QuaternionWXYZ conjugateQuaternion(const QuaternionWXYZ& q);

/**
 * Compare quaternions up to global sign ambiguity.
 */
bool quaternionsApproxEqual(const QuaternionWXYZ& a, const QuaternionWXYZ& b,
                            double tolerance);

/**
 * Convert quaternion to an Eigen vector [w, x, y, z].
 */
Eigen::Vector4d toVectorWXYZ(const QuaternionWXYZ& q);

}  // namespace ml_transforms
