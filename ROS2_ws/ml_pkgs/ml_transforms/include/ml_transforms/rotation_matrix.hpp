#pragma once

#include <Eigen/Core>

namespace ml_transforms {

bool isRotationMatrix(const Eigen::Matrix3d & R, double tolerance = 1e-9);

}  // namespace ml_transforms
