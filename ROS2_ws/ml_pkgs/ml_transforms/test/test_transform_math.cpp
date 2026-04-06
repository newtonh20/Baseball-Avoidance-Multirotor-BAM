#include <gtest/gtest.h>

#include <cmath>

#include "ml_transforms/quaternion.hpp"
#include "ml_transforms/transform.hpp"
#include "ml_transforms/transform_math.hpp"

namespace {

constexpr double kTol = 1e-6;
constexpr double kTightTol = 1e-10;

bool transformNear(const ml_transforms::Transform & A, const ml_transforms::Transform & B, const double tol)
{
  return (A.R - B.R).norm() < tol && (A.t - B.t).norm() < tol;
}

}  // namespace

TEST(TransformMathTest, MakeIdentityTransform)
{
  const auto T = ml_transforms::makeTransform(Eigen::Vector3d::Zero(), {1.0, 0.0, 0.0, 0.0});
  EXPECT_TRUE((T.R - Eigen::Matrix3d::Identity()).norm() < kTightTol);
  EXPECT_TRUE((T.t - Eigen::Vector3d::Zero()).norm() < kTightTol);
}

TEST(TransformMathTest, InverseIdentityIsIdentity)
{
  const ml_transforms::Transform I;
  const auto inv = ml_transforms::invertTransform(I);
  EXPECT_TRUE(transformNear(inv, I, kTightTol));
}

TEST(TransformMathTest, TransformTimesInverseIsIdentity)
{
  const auto T = ml_transforms::makeTransform(
    Eigen::Vector3d(1.0, -2.0, 3.0), {std::cos(M_PI / 8.0), 0.0, 0.0, std::sin(M_PI / 8.0)});

  const auto I = ml_transforms::composeTransforms(T, ml_transforms::invertTransform(T));
  EXPECT_TRUE((I.R - Eigen::Matrix3d::Identity()).norm() < kTol);
  EXPECT_TRUE((I.t - Eigen::Vector3d::Zero()).norm() < kTol);
}

TEST(TransformMathTest, InverseMatchesHomogeneousInverse)
{
  const auto T = ml_transforms::makeTransform(
    Eigen::Vector3d(2.3, -0.7, 5.1), {std::cos(M_PI / 8.0), 0.0, 0.0, std::sin(M_PI / 8.0)});

  const auto T_inv = ml_transforms::invertTransform(T);
  const Eigen::Matrix4d H = ml_transforms::toHomogeneousMatrix(T);
  const Eigen::Matrix4d H_inv = H.inverse();

  EXPECT_TRUE((ml_transforms::toHomogeneousMatrix(T_inv) - H_inv).norm() < 1e-10);
}

TEST(TransformMathTest, CompositionAssociativity)
{
  const auto T_AB = ml_transforms::makeTransform(
    Eigen::Vector3d(1.0, 0.0, -0.4), ml_transforms::normalizeQuaternion({0.96, 0.1, -0.2, 0.18}));
  const auto T_BC = ml_transforms::makeTransform(
    Eigen::Vector3d(-0.1, 0.2, 0.3), ml_transforms::normalizeQuaternion({0.88, -0.3, 0.1, 0.34}));
  const auto T_CD = ml_transforms::makeTransform(
    Eigen::Vector3d(0.5, 0.7, -1.1), ml_transforms::normalizeQuaternion({0.71, 0.4, 0.29, -0.5}));

  const auto left = ml_transforms::composeTransforms(ml_transforms::composeTransforms(T_AB, T_BC), T_CD);
  const auto right = ml_transforms::composeTransforms(T_AB, ml_transforms::composeTransforms(T_BC, T_CD));

  EXPECT_TRUE(transformNear(left, right, 1e-10));
}

TEST(TransformMathTest, KnownTranslationComposition)
{
  const auto T_AB = ml_transforms::makeTransform(Eigen::Vector3d(1.0, 2.0, 3.0), {1.0, 0.0, 0.0, 0.0});
  const auto T_BC = ml_transforms::makeTransform(Eigen::Vector3d(-2.0, 1.0, 4.0), {1.0, 0.0, 0.0, 0.0});

  const auto T_AC = ml_transforms::composeTransforms(T_AB, T_BC);
  EXPECT_NEAR(T_AC.t.x(), -1.0, kTightTol);
  EXPECT_NEAR(T_AC.t.y(), 3.0, kTightTol);
  EXPECT_NEAR(T_AC.t.z(), 7.0, kTightTol);
  EXPECT_TRUE((T_AC.R - Eigen::Matrix3d::Identity()).norm() < kTightTol);
}

TEST(TransformMathTest, KnownCameraComposition)
{
  const auto T_WB = ml_transforms::makeTransform(
    Eigen::Vector3d(10.0, 5.0, 2.0), {std::cos(M_PI / 4.0), 0.0, 0.0, std::sin(M_PI / 4.0)});
  const auto T_BoC = ml_transforms::makeTransform(Eigen::Vector3d(1.0, 0.0, 0.0), {1.0, 0.0, 0.0, 0.0});

  const auto T_WC = ml_transforms::composeTransforms(T_WB, T_BoC);
  EXPECT_NEAR(T_WC.t.x(), 10.0, kTol);
  EXPECT_NEAR(T_WC.t.y(), 6.0, kTol);
  EXPECT_NEAR(T_WC.t.z(), 2.0, kTol);
}

TEST(TransformMathTest, HomogeneousRoundTrip)
{
  const auto T = ml_transforms::makeTransform(
    Eigen::Vector3d(-1.0, 2.2, -3.3), ml_transforms::normalizeQuaternion({0.8, 0.2, 0.3, 0.4}));
  const Eigen::Matrix4d H = ml_transforms::toHomogeneousMatrix(T);
  const auto T_back = ml_transforms::fromHomogeneousMatrix(H);

  EXPECT_TRUE(transformNear(T, T_back, 1e-10));
}

TEST(TransformMathTest, InvalidHomogeneousThrows)
{
  Eigen::Matrix4d H = Eigen::Matrix4d::Identity();
  H(3, 0) = 0.2;
  EXPECT_THROW(ml_transforms::fromHomogeneousMatrix(H), std::invalid_argument);
}

TEST(TransformMathTest, RelativePositionBodyContractExample)
{
  const ml_transforms::QuaternionWXYZ q_i2b{std::cos(M_PI / 4.0), 0.0, 0.0, std::sin(M_PI / 4.0)};
  const Eigen::Matrix3d R_i2b = ml_transforms::quatToRotmat(q_i2b);
  const Eigen::Vector3d rel_ned(1.0, 0.0, 0.0);
  const Eigen::Vector3d rel_body = ml_transforms::applyRotation(R_i2b, rel_ned);

  EXPECT_NEAR(rel_body.x(), 0.0, kTol);
  EXPECT_NEAR(rel_body.y(), 1.0, kTol);
  EXPECT_NEAR(rel_body.z(), 0.0, kTol);
}

TEST(TransformMathTest, NEDNegativeUpConventionPassThrough)
{
  const Eigen::Vector3d pos_ned(100.0, 20.0, -50.0);
  const auto T = ml_transforms::makeTransform(pos_ned, {1.0, 0.0, 0.0, 0.0});
  EXPECT_NEAR(T.t.z(), -50.0, kTightTol);
}
