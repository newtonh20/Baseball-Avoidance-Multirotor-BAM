#include "ml_transforms/quaternion.hpp"

#include <cmath>
#include <stdexcept>

namespace ml_transforms {
namespace {
constexpr double kMinQuatNorm = 1e-12;
}

double quaternionNorm(const QuaternionWXYZ& q) {
  return std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
}

QuaternionWXYZ normalizeQuaternion(const QuaternionWXYZ& q) {
  const double n = quaternionNorm(q);
  if (n < kMinQuatNorm) {
    throw std::invalid_argument("normalizeQuaternion: quaternion norm is near zero");
  }
  return QuaternionWXYZ{q.w / n, q.x / n, q.y / n, q.z / n};
}

QuaternionWXYZ conjugateQuaternion(const QuaternionWXYZ& q) {
  return QuaternionWXYZ{q.w, -q.x, -q.y, -q.z};
}

bool quaternionsApproxEqual(const QuaternionWXYZ& a, const QuaternionWXYZ& b,
                            const double tolerance) {
  const Eigen::Vector4d av = toVectorWXYZ(a);
  const Eigen::Vector4d bv = toVectorWXYZ(b);
  return ((av - bv).norm() <= tolerance) || ((av + bv).norm() <= tolerance);
}

Eigen::Vector4d toVectorWXYZ(const QuaternionWXYZ& q) {
  return Eigen::Vector4d(q.w, q.x, q.y, q.z);
}

}  // namespace ml_transforms
