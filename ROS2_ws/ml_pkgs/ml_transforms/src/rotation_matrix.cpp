#include "ml_transforms/rotation_matrix.hpp"

#include <cmath>
#include <stdexcept>

namespace ml_transforms {
namespace {
constexpr double kDefaultOrthonormalTolerance = 1e-9;
constexpr double kDefaultDeterminantTolerance = 1e-9;
}  // namespace

bool isValidRotationMatrix(const Eigen::Matrix3d& R,
                           const double orthonormal_tolerance,
                           const double determinant_tolerance) {
  const Eigen::Matrix3d should_be_I = R.transpose() * R;
  const double ortho_error = (should_be_I - Eigen::Matrix3d::Identity()).norm();
  const double det_error = std::abs(R.determinant() - 1.0);
  return ortho_error <= orthonormal_tolerance && det_error <= determinant_tolerance;
}

Eigen::Matrix3d quatToRotmat(const QuaternionWXYZ& q_wxyz) {
  const QuaternionWXYZ q = normalizeQuaternion(q_wxyz);

  const double w = q.w;
  const double x = q.x;
  const double y = q.y;
  const double z = q.z;

  Eigen::Matrix3d R;
  R(0, 0) = 1.0 - 2.0 * (y * y + z * z);
  R(0, 1) = 2.0 * (x * y - w * z);
  R(0, 2) = 2.0 * (x * z + w * y);
  R(1, 0) = 2.0 * (x * y + w * z);
  R(1, 1) = 1.0 - 2.0 * (x * x + z * z);
  R(1, 2) = 2.0 * (y * z - w * x);
  R(2, 0) = 2.0 * (x * z - w * y);
  R(2, 1) = 2.0 * (y * z + w * x);
  R(2, 2) = 1.0 - 2.0 * (x * x + y * y);

  return R;
}

QuaternionWXYZ rotmatToQuat(const Eigen::Matrix3d& R) {
  if (!isValidRotationMatrix(R, kDefaultOrthonormalTolerance,
                             kDefaultDeterminantTolerance)) {
    throw std::invalid_argument("rotmatToQuat: input is not a valid rotation matrix");
  }

  QuaternionWXYZ q{};
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
