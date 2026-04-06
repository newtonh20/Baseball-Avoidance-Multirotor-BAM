#include "ml_transforms/quaternion.hpp"

#include <cmath>
#include <stdexcept>

#include "ml_transforms/rotation_matrix.hpp"

namespace ml_transforms {

namespace {
constexpr double kQuaternionNormMin = 1e-12;
}

QuaternionWXYZ normalizeQuaternion(const QuaternionWXYZ & q)
{
  const double norm = std::sqrt(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
  if (norm < kQuaternionNormMin) {
    throw std::invalid_argument("normalizeQuaternion: quaternion norm is near zero");
  }

  return QuaternionWXYZ{q.w / norm, q.x / norm, q.y / norm, q.z / norm};
}

QuaternionWXYZ conjugateQuaternion(const QuaternionWXYZ & q)
{
  return QuaternionWXYZ{q.w, -q.x, -q.y, -q.z};
}

Eigen::Matrix3d quatToRotmat(const QuaternionWXYZ & q_wxyz)
{
  const QuaternionWXYZ q = normalizeQuaternion(q_wxyz);

  const double ww = q.w * q.w;
  const double xx = q.x * q.x;
  const double yy = q.y * q.y;
  const double zz = q.z * q.z;

  const double wx = q.w * q.x;
  const double wy = q.w * q.y;
  const double wz = q.w * q.z;
  const double xy = q.x * q.y;
  const double xz = q.x * q.z;
  const double yz = q.y * q.z;

  Eigen::Matrix3d R;
  R << ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy),
    2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx),
    2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz;

  return R;
}

QuaternionWXYZ rotmatToQuat(const Eigen::Matrix3d & R)
{
  if (!isRotationMatrix(R, 1e-8)) {
    throw std::invalid_argument("rotmatToQuat: input matrix is not a valid rotation matrix");
  }

  QuaternionWXYZ q;
  const double trace = R.trace();

  if (trace > 0.0) {
    const double s = 2.0 * std::sqrt(trace + 1.0);
    q.w = 0.25 * s;
    q.x = (R(2, 1) - R(1, 2)) / s;
    q.y = (R(0, 2) - R(2, 0)) / s;
    q.z = (R(1, 0) - R(0, 1)) / s;
  } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
    const double s = 2.0 * std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2));
    q.w = (R(2, 1) - R(1, 2)) / s;
    q.x = 0.25 * s;
    q.y = (R(0, 1) + R(1, 0)) / s;
    q.z = (R(0, 2) + R(2, 0)) / s;
  } else if (R(1, 1) > R(2, 2)) {
    const double s = 2.0 * std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2));
    q.w = (R(0, 2) - R(2, 0)) / s;
    q.x = (R(0, 1) + R(1, 0)) / s;
    q.y = 0.25 * s;
    q.z = (R(1, 2) + R(2, 1)) / s;
  } else {
    const double s = 2.0 * std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1));
    q.w = (R(1, 0) - R(0, 1)) / s;
    q.x = (R(0, 2) + R(2, 0)) / s;
    q.y = (R(1, 2) + R(2, 1)) / s;
    q.z = 0.25 * s;
  }

  q = normalizeQuaternion(q);
  if (q.w < 0.0) {
    q.w = -q.w;
    q.x = -q.x;
    q.y = -q.y;
    q.z = -q.z;
  }
  return q;
}

}  // namespace ml_transforms
