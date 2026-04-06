#pragma once

#include <Eigen/Core>

#include "ml_transforms/quaternion.hpp"
#include "ml_transforms/transform.hpp"

namespace ml_transforms {

Transform makeTransform(const Eigen::Vector3d & position, const QuaternionWXYZ & q_wxyz);

}  // namespace ml_transforms
