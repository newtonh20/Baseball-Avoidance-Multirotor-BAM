#include "ml_transforms/rotation_matrix.hpp"

#include <cmath>
#include <stdexcept>

#include "ml_transforms/quaternion.hpp"

namespace ml_transforms {

Eigen::Matrix3d quatToRotmat(const QuaternionWXYZ& q_wxyz) {
  const QuaternionWXYZ q = normalizeQuaternion(q_wxyz);

  const double w = q.w;
  const double x = q.x;
  const double y = q.y;
  const double z = q.z;

  Eigen::Matrix3d R;
  R << 1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
      2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
      2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y);
  return R;
}

QuaternionWXYZ rotmatToQuat(const Eigen::Matrix3d& R) {
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d orth_err = R.transpose() * R - I;
  const double det = R.determinant();
  if (orth_err.norm() > 1e-9 || std::abs(det - 1.0) > 1e-9) {
    throw std::invalid_argument("rotmatToQuat: rotation matrix is not orthonormal with det +1");
  }

  QuaternionWXYZ q{};
  const double trace = R.trace();

  if (trace > 0.0) {
    const double s = std::sqrt(trace + 1.0) * 2.0;
    q.w = 0.25 * s;
    q.x = (R(2, 1) - R(1, 2)) / s;
    q.y = (R(0, 2) - R(2, 0)) / s;
    q.z = (R(1, 0) - R(0, 1)) / s;
  } else if (R(0, 0) > R(1, 1) && R(0, 0) > R(2, 2)) {
    const double s = std::sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;
    q.w = (R(2, 1) - R(1, 2)) / s;
    q.x = 0.25 * s;
    q.y = (R(0, 1) + R(1, 0)) / s;
    q.z = (R(0, 2) + R(2, 0)) / s;
  } else if (R(1, 1) > R(2, 2)) {
    const double s = std::sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;
    q.w = (R(0, 2) - R(2, 0)) / s;
    q.x = (R(0, 1) + R(1, 0)) / s;
    q.y = 0.25 * s;
    q.z = (R(1, 2) + R(2, 1)) / s;
  } else {
    const double s = std::sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;
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
