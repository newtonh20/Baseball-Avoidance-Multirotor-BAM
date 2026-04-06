#include "ml_transforms/transform_math.hpp"

#include "ml_transforms/rotation_matrix.hpp"

namespace ml_transforms {

Transform makeTransform(const Eigen::Vector3d& position,
                        const QuaternionWXYZ& q_wxyz) {
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
  Transform T_inv;
  T_inv.R = T.R.transpose();
  T_inv.t = -T_inv.R * T.t;
  return T_inv;
}

}  // namespace ml_transforms
