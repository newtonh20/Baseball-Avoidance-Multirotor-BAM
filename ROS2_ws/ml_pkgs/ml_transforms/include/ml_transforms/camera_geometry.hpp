#pragma once

#include <Eigen/Core>

namespace ml_transforms {

Eigen::Vector3d pixelToRay(const Eigen::Vector2d& uv, const Eigen::Matrix3d& K);
Eigen::Vector2d rayToPixel(const Eigen::Vector3d& ray, const Eigen::Matrix3d& K);

}  // namespace ml_transforms
