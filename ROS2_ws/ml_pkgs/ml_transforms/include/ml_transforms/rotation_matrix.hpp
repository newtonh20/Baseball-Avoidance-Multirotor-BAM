#pragma once

#include <Eigen/Core>

#include "ml_transforms/quaternion.hpp"

namespace ml_transforms {

Eigen::Matrix3d quatToRotmat(const QuaternionWXYZ& q_wxyz);
QuaternionWXYZ rotmatToQuat(const Eigen::Matrix3d& R);

}  // namespace ml_transforms
