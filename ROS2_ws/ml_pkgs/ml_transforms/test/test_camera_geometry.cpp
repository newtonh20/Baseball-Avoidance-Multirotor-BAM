#include <gtest/gtest.h>

#include <stdexcept>

#include "ml_transforms/camera_geometry.hpp"

TEST(CameraGeometryTest, PixelToRayThrowsUntilWP2)
{
  EXPECT_THROW(
    ml_transforms::pixelToRay(Eigen::Vector2d(100.0, 50.0), Eigen::Matrix3d::Identity()),
    std::logic_error);
}

TEST(CameraGeometryTest, RayToPixelThrowsUntilWP2)
{
  EXPECT_THROW(
    ml_transforms::rayToPixel(Eigen::Vector3d(0.0, 0.0, 1.0), Eigen::Matrix3d::Identity()),
    std::logic_error);
}
