#include "ml_transforms/transform_math.hpp"

namespace ml_transforms {

Transform makeTransform(const Eigen::Vector3d & position, const QuaternionWXYZ & q_wxyz)
{
  Transform T;
  T.R = quatToRotmat(q_wxyz);
  T.t = position;
  return T;
}

}  // namespace ml_transforms
