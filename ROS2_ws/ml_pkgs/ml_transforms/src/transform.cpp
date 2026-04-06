#include "ml_transforms/transform.hpp"

#include <stdexcept>

#include "ml_transforms/rotation_matrix.hpp"

namespace ml_transforms {

Eigen::Matrix4d toHomogeneousMatrix(const Transform& T) {
  Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
  H.block<3, 3>(0, 0) = T.R;
  H.block<3, 1>(0, 3) = T.t;
  return H;
}

Transform fromHomogeneousMatrix(const Eigen::Matrix4d& H) {
  const Eigen::RowVector4d expected_bottom(0.0, 0.0, 0.0, 1.0);
  constexpr double kBottomRowTolerance = 1e-12;
  if ((H.row(3) - expected_bottom).norm() > kBottomRowTolerance) {
    throw std::invalid_argument(
        "fromHomogeneousMatrix: invalid bottom row, expected [0 0 0 1]");
  }

  const Eigen::Matrix3d R = H.block<3, 3>(0, 0);
  if (!isValidRotationMatrix(R, 1e-9, 1e-9)) {
    throw std::invalid_argument(
        "fromHomogeneousMatrix: rotation block is not orthonormal");
  }

  Transform T;
  T.R = R;
  T.t = H.block<3, 1>(0, 3);
  return T;
}

Eigen::Vector3d applyTransform(const Transform& T_AB, const Eigen::Vector3d& p_B) {
  return T_AB.R * p_B + T_AB.t;
}

Eigen::Vector3d applyRotation(const Eigen::Matrix3d& R_AB,
                              const Eigen::Vector3d& v_B) {
  return R_AB * v_B;
}

}  // namespace ml_transforms
