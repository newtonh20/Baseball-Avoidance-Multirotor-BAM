#pragma once

#include <Eigen/Core>

namespace ml_transforms {

// Scalar-first quaternion storage order [w, x, y, z].
struct QuaternionWXYZ {
  double w;
  double x;
  double y;
  double z;
};

QuaternionWXYZ normalizeQuaternion(const QuaternionWXYZ& q);
QuaternionWXYZ conjugateQuaternion(const QuaternionWXYZ& q);
Eigen::Vector4d toVectorWXYZ(const QuaternionWXYZ& q);

}  // namespace ml_transforms
