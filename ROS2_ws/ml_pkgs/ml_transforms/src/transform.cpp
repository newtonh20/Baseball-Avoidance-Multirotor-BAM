#include "ml_transforms/transform.hpp"

#include <stdexcept>

#include "ml_transforms/rotation_matrix.hpp"

namespace ml_transforms {

Transform composeTransforms(const Transform & T_AB, const Transform & T_BC)
{
  Transform T_AC;
  T_AC.R = T_AB.R * T_BC.R;
  T_AC.t = T_AB.R * T_BC.t + T_AB.t;
  return T_AC;
}

Transform invertTransform(const Transform & T)
{
  Transform T_BA;
  T_BA.R = T.R.transpose();
  T_BA.t = -T_BA.R * T.t;
  return T_BA;
}

Eigen::Vector3d applyTransform(const Transform & T_AB, const Eigen::Vector3d & p_B)
{
  return T_AB.R * p_B + T_AB.t;
}

Eigen::Vector3d applyRotation(const Eigen::Matrix3d & R_AB, const Eigen::Vector3d & v_B)
{
  return R_AB * v_B;
}

Eigen::Matrix4d toHomogeneousMatrix(const Transform & T)
{
  Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
  H.block<3, 3>(0, 0) = T.R;
  H.block<3, 1>(0, 3) = T.t;
  return H;
}

Transform fromHomogeneousMatrix(const Eigen::Matrix4d & H)
{
  const Eigen::RowVector4d expected_row(0.0, 0.0, 0.0, 1.0);
  if ((H.row(3) - expected_row).norm() > 1e-9) {
    throw std::invalid_argument("fromHomogeneousMatrix: bottom row must be [0,0,0,1]");
  }

  const Eigen::Matrix3d R = H.block<3, 3>(0, 0);
  if (!isRotationMatrix(R, 1e-8)) {
    throw std::invalid_argument("fromHomogeneousMatrix: rotation block is not orthonormal");
  }

  Transform T;
  T.R = R;
  T.t = H.block<3, 1>(0, 3);
  return T;
}

}  // namespace ml_transforms
