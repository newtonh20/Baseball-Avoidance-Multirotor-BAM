#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>

#include "ml_transforms/quaternion.hpp"
#include "ml_transforms/rotation_matrix.hpp"

namespace {
constexpr double kTol = 1e-6;
}

TEST(QuaternionTest, NormalizeQuaternionThrowsOnZero) {
  const ml_transforms::QuaternionWXYZ q{0.0, 0.0, 0.0, 0.0};
  EXPECT_THROW((void)ml_transforms::normalizeQuaternion(q), std::invalid_argument);
}

TEST(QuaternionTest, GoldenRoll90Deg) {
  const double c = std::sqrt(0.5);
  const ml_transforms::QuaternionWXYZ q{c, c, 0.0, 0.0};

  Eigen::Matrix3d expected;
  expected << 1.0, 0.0, 0.0,
              0.0, 0.0, -1.0,
              0.0, 1.0, 0.0;

  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q);
  EXPECT_LT((R - expected).norm(), kTol);
}

TEST(QuaternionTest, GoldenPitch90Deg) {
  const double c = std::sqrt(0.5);
  const ml_transforms::QuaternionWXYZ q{c, 0.0, c, 0.0};

  Eigen::Matrix3d expected;
  expected << 0.0, 0.0, 1.0,
              0.0, 1.0, 0.0,
              -1.0, 0.0, 0.0;

  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q);
  EXPECT_LT((R - expected).norm(), kTol);
}

TEST(QuaternionTest, GoldenYaw90Deg) {
  const double c = std::sqrt(0.5);
  const ml_transforms::QuaternionWXYZ q{c, 0.0, 0.0, c};

  Eigen::Matrix3d expected;
  expected << 0.0, -1.0, 0.0,
              1.0, 0.0, 0.0,
              0.0, 0.0, 1.0;

  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q);
  EXPECT_LT((R - expected).norm(), kTol);
}

TEST(QuaternionTest, RotationRoundtripRtoQtoR) {
  const double c = std::sqrt(0.5);
  const ml_transforms::QuaternionWXYZ q{c, 0.0, c, 0.0};
  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q);
  const ml_transforms::QuaternionWXYZ q2 = ml_transforms::rotmatToQuat(R);
  const Eigen::Matrix3d R2 = ml_transforms::quatToRotmat(q2);
  EXPECT_LT((R - R2).norm(), 1e-10);
}

TEST(QuaternionTest, RotationRoundtripQtoRtoQ) {
  const ml_transforms::QuaternionWXYZ q0{0.35, -0.2, 0.1, 0.9};
  const ml_transforms::QuaternionWXYZ q = ml_transforms::normalizeQuaternion(q0);

  const Eigen::Matrix3d R = ml_transforms::quatToRotmat(q);
  const ml_transforms::QuaternionWXYZ q2 = ml_transforms::rotmatToQuat(R);

  EXPECT_TRUE(ml_transforms::quaternionsApproxEqual(q, q2, 1e-10));
}

TEST(QuaternionTest, DetectsWrongQuaternionOrderUsage) {
  const double c = std::sqrt(0.5);

  // Correct 90 deg roll in [w, x, y, z].
  const ml_transforms::QuaternionWXYZ q_wxyz{c, c, 0.0, 0.0};

  // Wrongly interpreted [x, y, z, w] values fed into [w, x, y, z] API.
  const ml_transforms::QuaternionWXYZ wrong_xyzw_as_wxyz{c, 0.0, 0.0, c};

  const Eigen::Matrix3d R_correct = ml_transforms::quatToRotmat(q_wxyz);
  const Eigen::Matrix3d R_wrong = ml_transforms::quatToRotmat(wrong_xyzw_as_wxyz);

  EXPECT_GT((R_correct - R_wrong).norm(), 1e-3);
}

TEST(QuaternionTest, RotmatToQuatThrowsOnInvalidRotationMatrix) {
  Eigen::Matrix3d bad = Eigen::Matrix3d::Identity();
  bad(0, 0) = 2.0;
  EXPECT_THROW((void)ml_transforms::rotmatToQuat(bad), std::invalid_argument);
}
