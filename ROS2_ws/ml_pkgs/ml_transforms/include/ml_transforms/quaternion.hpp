#pragma once

#include <Eigen/Core>

namespace ml_transforms {

struct QuaternionWXYZ {
  double w{1.0};
  double x{0.0};
  double y{0.0};
  double z{0.0};
};

QuaternionWXYZ normalizeQuaternion(const QuaternionWXYZ & q);
QuaternionWXYZ conjugateQuaternion(const QuaternionWXYZ & q);

Eigen::Matrix3d quatToRotmat(const QuaternionWXYZ & q_wxyz);
QuaternionWXYZ rotmatToQuat(const Eigen::Matrix3d & R);

}  // namespace ml_transforms
