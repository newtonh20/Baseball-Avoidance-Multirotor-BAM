#include "ml_transforms/transform.hpp"

#include <cmath>
#include <stdexcept>

#include "ml_transforms/rotation_matrix.hpp"

namespace ml_transforms {

Transform makeTransform(const Eigen::Vector3d& position, const QuaternionWXYZ& q_wxyz) {
  Transform T;
  T.R = quatToRotmat(q_wxyz);
  T.t = position;
  return T;
}

Transform composeTransforms(const Transform& T_AB, const Transform& T_BC) {
  Transform T_AC;
  T_AC.R = T_AB.R * T_BC.R;
  T_AC.t = T_AB.R * T_BC.t + T_AB.t;
  return T_AC;
}

Transform invertTransform(const Transform& T) {
  Transform T_BA;
  T_BA.R = T.R.transpose();
  T_BA.t = -T_BA.R * T.t;
  return T_BA;
}

Eigen::Vector3d applyTransform(const Transform& T_AB, const Eigen::Vector3d& p_B) {
  return T_AB.R * p_B + T_AB.t;
}

Eigen::Vector3d applyRotation(const Eigen::Matrix3d& R_AB, const Eigen::Vector3d& v_B) {
  return R_AB * v_B;
}

Eigen::Matrix4d toHomogeneousMatrix(const Transform& T) {
  Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
  H.block<3, 3>(0, 0) = T.R;
  H.block<3, 1>(0, 3) = T.t;
  return H;
}

Transform fromHomogeneousMatrix(const Eigen::Matrix4d& H) {
  const Eigen::RowVector4d bottom_expected(0.0, 0.0, 0.0, 1.0);
  if ((H.row(3) - bottom_expected).norm() > 1e-12) {
    throw std::invalid_argument("fromHomogeneousMatrix: bottom row must be [0 0 0 1]");
  }

  Transform T;
  T.R = H.block<3, 3>(0, 0);
  T.t = H.block<3, 1>(0, 3);

  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d orth_err = T.R.transpose() * T.R - I;
  const double det = T.R.determinant();
  if (orth_err.norm() > 1e-9 || std::abs(det - 1.0) > 1e-9) {
    throw std::invalid_argument(
        "fromHomogeneousMatrix: rotation block is not orthonormal with det +1");
  }

  return T;
}

}  // namespace ml_transforms
