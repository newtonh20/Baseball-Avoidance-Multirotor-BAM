#pragma once

#include <Eigen/Core>

#include "ml_transforms/quaternion.hpp"

namespace ml_transforms {

struct Transform {
  Eigen::Matrix3d R;
  Eigen::Vector3d t;
};

Transform makeTransform(const Eigen::Vector3d& position, const QuaternionWXYZ& q_wxyz);
Transform composeTransforms(const Transform& T_AB, const Transform& T_BC);
Transform invertTransform(const Transform& T);
Eigen::Vector3d applyTransform(const Transform& T_AB, const Eigen::Vector3d& p_B);
Eigen::Vector3d applyRotation(const Eigen::Matrix3d& R_AB, const Eigen::Vector3d& v_B);

Eigen::Matrix4d toHomogeneousMatrix(const Transform& T);
Transform fromHomogeneousMatrix(const Eigen::Matrix4d& H);

}  // namespace ml_transforms
