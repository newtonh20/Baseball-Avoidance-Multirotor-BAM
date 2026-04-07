#include "ml_transforms/quaternion.hpp"

#include <cmath>
#include <stdexcept>

namespace ml_transforms {

QuaternionWXYZ normalizeQuaternion(const QuaternionWXYZ& q) {
  const double n = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (n < 1e-12) {
    throw std::invalid_argument("normalizeQuaternion: quaternion norm is near zero");
  }

  return QuaternionWXYZ{q.w / n, q.x / n, q.y / n, q.z / n};
}

QuaternionWXYZ conjugateQuaternion(const QuaternionWXYZ& q) {
  return QuaternionWXYZ{q.w, -q.x, -q.y, -q.z};
}

Eigen::Vector4d toVectorWXYZ(const QuaternionWXYZ& q) {
  return Eigen::Vector4d(q.w, q.x, q.y, q.z);
}

}  // namespace ml_transforms
