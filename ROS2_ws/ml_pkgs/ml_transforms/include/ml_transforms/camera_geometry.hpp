#pragma once

#include <Eigen/Core>

namespace ml_transforms {

/**
 * WP2 stub: convert pixel coordinates into camera ray.
 * Throws std::logic_error("pixelToRay: to be completed in WP2").
 */
Eigen::Vector3d pixelToRay(const Eigen::Vector2d& uv, const Eigen::Matrix3d& K);

/**
 * WP2 stub: project camera ray into pixel coordinates.
 * Throws std::logic_error("rayToPixel: to be completed in WP2").
 */
Eigen::Vector2d rayToPixel(const Eigen::Vector3d& ray, const Eigen::Matrix3d& K);

}  // namespace ml_transforms
