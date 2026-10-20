#include <gtest/gtest.h>

#include <cmath>

#include "ml_transforms/quaternion.hpp"
#include "ml_transforms/rotation_matrix.hpp"

namespace {

constexpr double kTol = 1e-6;

bool matrixNear(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B, double tol) {
  return (A - B).norm() < tol;
}

}  // namespace

TEST(QuaternionTest, ZeroQuaternionThrows) {
  const ml_transforms::QuaternionWXYZ q{0.0, 0.0, 0.0, 0.0};
  EXPECT_THROW((void)ml_transforms::normalizeQuaternion(q), std::invalid_argument);
  EXPECT_THROW((void)ml_transforms::quatToRotmat(q), std::invalid_argument);
}

TEST(QuaternionTest, GoldenRollPitchYawNinetyDegrees) {
  const double c = std::cos(M_PI / 4.0);
  const double s = std::sin(M_PI / 4.0);

  const ml_transforms::QuaternionWXYZ q_roll{c, s, 0.0, 0.0};
  const ml_transforms::QuaternionWXYZ q_pitch{c, 0.0, s, 0.0};
  const ml_transforms::QuaternionWXYZ q_yaw{c, 0.0, 0.0, s};

  Eigen::Matrix3d R_roll_expected;
  R_roll_expected << 1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0;

  Eigen::Matrix3d R_pitch_expected;
  R_pitch_expected << 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0;

  Eigen::Matrix3d R_yaw_expected;
  R_yaw_expected << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;

  EXPECT_TRUE(matrixNear(ml_transforms::quatToRotmat(q_roll), R_roll_expected, kTol));
  EXPECT_TRUE(matrixNear(ml_transforms::quatToRotmat(q_pitch), R_pitch_expected, kTol));
  EXPECT_TRUE(matrixNear(ml_transforms::quatToRotmat(q_yaw), R_yaw_expected, kTol));
}

TEST(QuaternionTest, QuaternionOrderMistakeDetected) {
  const double c = std::cos(M_PI / 4.0);
  const double s = std::sin(M_PI / 4.0);

  // Correct wxyz for +90 deg roll.
  const ml_transforms::QuaternionWXYZ q_wxyz{c, s, 0.0, 0.0};
  // Mistakenly passing xyzw interpreted as wxyz.
  const ml_transforms::QuaternionWXYZ q_wrong{s, 0.0, 0.0, c};

  const Eigen::Matrix3d R_good = ml_transforms::quatToRotmat(q_wxyz);
  const Eigen::Matrix3d R_bad = ml_transforms::quatToRotmat(q_wrong);

  EXPECT_FALSE(matrixNear(R_good, R_bad, kTol));
}

TEST(QuaternionTest, RoundTripQToRToQ) {
  ml_transforms::QuaternionWXYZ q{0.8, -0.1, 0.2, 0.55};
  q = ml_transforms::normalizeQuaternion(q);

  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q);
  const ml_transforms::QuaternionWXYZ q_back = ml_transforms::rotmatToQuat(R);

  const Eigen::Vector4d a = ml_transforms::toVectorWXYZ(q);
  const Eigen::Vector4d b = ml_transforms::toVectorWXYZ(q_back);
  const Eigen::Vector4d b_neg = -b;

  EXPECT_TRUE((a - b).norm() < kTol || (a - b_neg).norm() < kTol);
}

TEST(QuaternionTest, RoundTripRToQToR) {
  ml_transforms::QuaternionWXYZ q{0.67, 0.12, -0.6, 0.42};
  q = ml_transforms::normalizeQuaternion(q);

  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q);
  const ml_transforms::QuaternionWXYZ q_back = ml_transforms::rotmatToQuat(R);
  const Eigen::Matrix3d R_back = ml_transforms::quatToRotmat(q_back);

  EXPECT_TRUE(matrixNear(R, R_back, 1e-10));
}

TEST(QuaternionTest, RotmatValidationThrows) {
  Eigen::Matrix3d bad = Eigen::Matrix3d::Identity();
  bad(0, 0) = 2.0;
  EXPECT_THROW((void)ml_transforms::rotmatToQuat(bad), std::invalid_argument);
}
