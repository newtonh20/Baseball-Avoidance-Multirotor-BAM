#include <gtest/gtest.h>

#include <Eigen/Core>

#include "ml_transforms/camera_geometry.hpp"

TEST(CameraGeometryTest, PixelToRayThrowsStubLogicError) {
  EXPECT_THROW(
      (void)ml_transforms::pixelToRay(Eigen::Vector2d(100.0, 200.0),
                                      Eigen::Matrix3d::Identity()),
      std::logic_error);
}

TEST(CameraGeometryTest, RayToPixelThrowsStubLogicError) {
  EXPECT_THROW(
      (void)ml_transforms::rayToPixel(Eigen::Vector3d(1.0, 0.0, 1.0),
                                      Eigen::Matrix3d::Identity()),
      std::logic_error);
}
