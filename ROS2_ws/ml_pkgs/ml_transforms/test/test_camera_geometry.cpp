#include <gtest/gtest.h>

#include "ml_transforms/camera_geometry.hpp"

TEST(CameraGeometryTest, PixelRayStubsThrow) {
  EXPECT_THROW((void)ml_transforms::pixelToRay(Eigen::Vector2d(0.0, 0.0), Eigen::Matrix3d::Identity()),
               std::logic_error);

  EXPECT_THROW((void)ml_transforms::rayToPixel(Eigen::Vector3d(1.0, 0.0, 1.0), Eigen::Matrix3d::Identity()),
               std::logic_error);
}
