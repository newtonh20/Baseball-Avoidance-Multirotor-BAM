#pragma once

#include <Eigen/Core>

#include "ml_transforms/quaternion.hpp"
#include "ml_transforms/transform.hpp"

namespace ml_transforms {

/**
 * Build transform from translation and scalar-first quaternion [w, x, y, z].
 */
Transform makeTransform(const Eigen::Vector3d& position,
                        const QuaternionWXYZ& q_wxyz);

/**
 * Compose transforms: T_AC = T_AB * T_BC.
 */
Transform composeTransforms(const Transform& T_AB, const Transform& T_BC);

/**
 * Analytic inverse of rigid transform.
 */
Transform invertTransform(const Transform& T);

}  // namespace ml_transforms
