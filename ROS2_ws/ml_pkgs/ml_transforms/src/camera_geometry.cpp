#include "ml_transforms/camera_geometry.hpp"

#include <stdexcept>

namespace ml_transforms {

Eigen::Vector3d pixelToRay(const Eigen::Vector2d &, const Eigen::Matrix3d &)
{
  throw std::logic_error("pixelToRay: to be completed in WP2");
}

Eigen::Vector2d rayToPixel(const Eigen::Vector3d &, const Eigen::Matrix3d &)
{
  throw std::logic_error("rayToPixel: to be completed in WP2");
}

}  // namespace ml_transforms
