#include "ml_transforms/rotation_matrix.hpp"

#include <cmath>

namespace ml_transforms {

bool isRotationMatrix(const Eigen::Matrix3d & R, const double tolerance)
{
  const Eigen::Matrix3d should_be_identity = R.transpose() * R;
  const double determinant = R.determinant();
  return (should_be_identity - Eigen::Matrix3d::Identity()).norm() <= tolerance &&
         std::abs(determinant - 1.0) <= tolerance;
}

}  // namespace ml_transforms
