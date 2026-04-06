#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>

#include "ml_transforms/quaternion.hpp"

namespace {
constexpr double kTol = 1e-6;

bool matricesNear(const Eigen::Matrix3d & A, const Eigen::Matrix3d & B, const double tol)
{
  return (A - B).norm() < tol;
}

}  // namespace

TEST(QuaternionTest, GoldenRoll90Degrees)
{
  const ml_transforms::QuaternionWXYZ q{std::cos(M_PI / 4.0), std::sin(M_PI / 4.0), 0.0, 0.0};
  Eigen::Matrix3d expected;
  expected << 1.0, 0.0, 0.0,
    0.0, 0.0, -1.0,
    0.0, 1.0, 0.0;

  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q);
  EXPECT_TRUE(matricesNear(R, expected, kTol));
}

TEST(QuaternionTest, GoldenPitch90Degrees)
{
  const ml_transforms::QuaternionWXYZ q{std::cos(M_PI / 4.0), 0.0, std::sin(M_PI / 4.0), 0.0};
  Eigen::Matrix3d expected;
  expected << 0.0, 0.0, 1.0,
    0.0, 1.0, 0.0,
    -1.0, 0.0, 0.0;

  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q);
  EXPECT_TRUE(matricesNear(R, expected, kTol));
}

TEST(QuaternionTest, GoldenYaw90Degrees)
{
  const ml_transforms::QuaternionWXYZ q{std::cos(M_PI / 4.0), 0.0, 0.0, std::sin(M_PI / 4.0)};
  Eigen::Matrix3d expected;
  expected << 0.0, -1.0, 0.0,
    1.0, 0.0, 0.0,
    0.0, 0.0, 1.0;

  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q);
  EXPECT_TRUE(matricesNear(R, expected, kTol));
}

TEST(QuaternionTest, QuaternionRoundTrip)
{
  const ml_transforms::QuaternionWXYZ q_in = ml_transforms::normalizeQuaternion({0.7, 0.2, -0.3, 0.61});
  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q_in);
  const ml_transforms::QuaternionWXYZ q_out = ml_transforms::rotmatToQuat(R);

  EXPECT_NEAR(q_in.w, q_out.w, kTol);
  EXPECT_NEAR(q_in.x, q_out.x, kTol);
  EXPECT_NEAR(q_in.y, q_out.y, kTol);
  EXPECT_NEAR(q_in.z, q_out.z, kTol);
}

TEST(QuaternionTest, RotationMatrixRoundTrip)
{
  const ml_transforms::QuaternionWXYZ q = ml_transforms::normalizeQuaternion({0.35, -0.12, 0.91, 0.16});
  const Eigen::Matrix3d R_in = ml_transforms::quatToRotmat(q);
  const ml_transforms::QuaternionWXYZ q_back = ml_transforms::rotmatToQuat(R_in);
  const Eigen::Matrix3d R_out = ml_transforms::quatToRotmat(q_back);
  EXPECT_TRUE(matricesNear(R_in, R_out, 1e-10));
}

TEST(QuaternionTest, WrongOrderXYZWIsDetected)
{
  const ml_transforms::QuaternionWXYZ q_wxyz{std::cos(M_PI / 4.0), std::sin(M_PI / 4.0), 0.0, 0.0};
  const ml_transforms::QuaternionWXYZ q_xyzw_as_wxyz{q_wxyz.x, q_wxyz.y, q_wxyz.z, q_wxyz.w};

  const Eigen::Matrix3d R_correct = ml_transforms::quatToRotmat(q_wxyz);
  const Eigen::Matrix3d R_wrong = ml_transforms::quatToRotmat(q_xyzw_as_wxyz);

  EXPECT_FALSE(matricesNear(R_correct, R_wrong, 1e-3));
}

TEST(QuaternionTest, ZeroQuaternionThrows)
{
  EXPECT_THROW(ml_transforms::quatToRotmat({0.0, 0.0, 0.0, 0.0}), std::invalid_argument);
  EXPECT_THROW(ml_transforms::normalizeQuaternion({0.0, 0.0, 0.0, 0.0}), std::invalid_argument);
}

TEST(QuaternionTest, InvalidRotationMatrixThrows)
{
  Eigen::Matrix3d bad = Eigen::Matrix3d::Identity();
  bad(0, 1) = 0.2;
  EXPECT_THROW(ml_transforms::rotmatToQuat(bad), std::invalid_argument);
}
