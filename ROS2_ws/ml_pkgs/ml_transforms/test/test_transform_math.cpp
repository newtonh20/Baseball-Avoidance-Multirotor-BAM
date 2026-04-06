#include <gtest/gtest.h>

#include <cmath>
#include <Eigen/Core>

#include "ml_transforms/rotation_matrix.hpp"
#include "ml_transforms/transform.hpp"
#include "ml_transforms/transform_math.hpp"

namespace {
constexpr double kTol = 1e-6;

bool transformApproxEqual(const ml_transforms::Transform& A,
                          const ml_transforms::Transform& B,
                          const double tol) {
  return (A.R - B.R).norm() < tol && (A.t - B.t).norm() < tol;
}
}  // namespace

TEST(TransformMathTest, IdentityFromMakeTransform) {
  const Eigen::Vector3d p = Eigen::Vector3d::Zero();
  const ml_transforms::QuaternionWXYZ q{1.0, 0.0, 0.0, 0.0};

  const auto T = ml_transforms::makeTransform(p, q);
  EXPECT_LT((T.R - Eigen::Matrix3d::Identity()).norm(), 1e-12);
  EXPECT_LT(T.t.norm(), 1e-12);
}

TEST(TransformMathTest, InversionOfIdentityIsIdentity) {
  const ml_transforms::Transform I{};
  const auto I_inv = ml_transforms::invertTransform(I);
  EXPECT_TRUE(transformApproxEqual(I, I_inv, 1e-12));
}

TEST(TransformMathTest, TransformTimesInverseIsIdentity) {
  const double c = std::sqrt(0.5);
  const ml_transforms::Transform T = ml_transforms::makeTransform(
      Eigen::Vector3d(2.0, -1.0, 0.5), ml_transforms::QuaternionWXYZ{c, 0.0, 0.0, c});

  const auto T_inv = ml_transforms::invertTransform(T);
  const auto I = ml_transforms::composeTransforms(T, T_inv);

  EXPECT_LT((I.R - Eigen::Matrix3d::Identity()).norm(), 1e-10);
  EXPECT_LT(I.t.norm(), 1e-10);
}

TEST(TransformMathTest, NonTrivialInverseGolden) {
  const double c = std::sqrt(0.5);
  const ml_transforms::Transform T = ml_transforms::makeTransform(
      Eigen::Vector3d(5.0, -3.0, 1.0), ml_transforms::QuaternionWXYZ{c, 0.0, 0.0, c});

  const auto T_inv = ml_transforms::invertTransform(T);
  const auto I = ml_transforms::composeTransforms(T, T_inv);

  EXPECT_LT((I.R - Eigen::Matrix3d::Identity()).norm(), kTol);
  EXPECT_LT(I.t.norm(), kTol);
}

TEST(TransformMathTest, InverseMatchesHomogeneousInverseGolden) {
  const ml_transforms::QuaternionWXYZ q =
      ml_transforms::normalizeQuaternion(ml_transforms::QuaternionWXYZ{0.8, 0.1, -0.3, 0.4});
  const ml_transforms::Transform T =
      ml_transforms::makeTransform(Eigen::Vector3d(1.5, -2.0, 7.0), q);

  const auto T_inv_analytic = ml_transforms::invertTransform(T);
  const Eigen::Matrix4d H = ml_transforms::toHomogeneousMatrix(T);
  const Eigen::Matrix4d H_inv = H.inverse();
  const auto T_inv_mat = ml_transforms::fromHomogeneousMatrix(H_inv);

  EXPECT_TRUE(transformApproxEqual(T_inv_analytic, T_inv_mat, 1e-10));
}

TEST(TransformMathTest, CompositionAssociativity) {
  const auto T_AB = ml_transforms::makeTransform(
      Eigen::Vector3d(1.0, 2.0, 3.0), ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});
  const auto T_BC = ml_transforms::makeTransform(
      Eigen::Vector3d(-1.0, 0.5, 2.0), ml_transforms::QuaternionWXYZ{0.9238795325, 0.0, 0.0, 0.3826834324});
  const auto T_CD = ml_transforms::makeTransform(
      Eigen::Vector3d(0.0, -2.0, 1.0), ml_transforms::QuaternionWXYZ{0.9659258263, 0.2588190451, 0.0, 0.0});

  const auto left = ml_transforms::composeTransforms(
      ml_transforms::composeTransforms(T_AB, T_BC), T_CD);
  const auto right = ml_transforms::composeTransforms(
      T_AB, ml_transforms::composeTransforms(T_BC, T_CD));

  EXPECT_TRUE(transformApproxEqual(left, right, 1e-10));
}

TEST(TransformMathTest, CompositionWithIdentityAndTranslationGolden) {
  const ml_transforms::Transform I{};
  ml_transforms::Transform T;
  T.R = Eigen::Matrix3d::Identity();
  T.t = Eigen::Vector3d(3.0, 4.0, -5.0);

  const auto out = ml_transforms::composeTransforms(I, T);
  EXPECT_TRUE(transformApproxEqual(out, T, 1e-12));
}

TEST(TransformMathTest, KnownCameraCompositionGolden) {
  const auto T_WB = ml_transforms::makeTransform(
      Eigen::Vector3d(10.0, 2.0, -50.0), ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});

  const auto T_BoC = ml_transforms::makeTransform(
      Eigen::Vector3d(0.2, 0.0, 0.1), ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});

  const auto T_WC = ml_transforms::composeTransforms(T_WB, T_BoC);
  EXPECT_LT((T_WC.t - Eigen::Vector3d(10.2, 2.0, -49.9)).norm(), 1e-12);
  EXPECT_LT((T_WC.R - Eigen::Matrix3d::Identity()).norm(), 1e-12);
}

TEST(TransformMathTest, FromHomogeneousThrowsOnInvalidBottomRow) {
  Eigen::Matrix4d bad = Eigen::Matrix4d::Identity();
  bad(3, 0) = 0.1;
  EXPECT_THROW((void)ml_transforms::fromHomogeneousMatrix(bad), std::invalid_argument);
}

TEST(TransformMathTest, FromHomogeneousThrowsOnInvalidRotationBlock) {
  Eigen::Matrix4d bad = Eigen::Matrix4d::Identity();
  bad(0, 0) = 2.0;
  EXPECT_THROW((void)ml_transforms::fromHomogeneousMatrix(bad), std::invalid_argument);
}

TEST(TransformMathTest, RelativePositionBodyContractExample) {
  // Contract-aligned usage: rel_body = R_i2b * rel_ned.
  const double c = std::sqrt(0.5);
  const Eigen::Matrix3d R_i2b = ml_transforms::quatToRotmat({c, 0.0, 0.0, c});
  const Eigen::Vector3d rel_ned(10.0, 0.0, 0.0);

  const Eigen::Vector3d rel_body = R_i2b * rel_ned;
  EXPECT_LT((rel_body - Eigen::Vector3d(0.0, 10.0, 0.0)).norm(), 1e-6);
}

TEST(TransformMathTest, NEDAltitudeSignHandledAsDataOnly) {
  const auto T = ml_transforms::makeTransform(Eigen::Vector3d(100.0, 5.0, -120.0),
                                              ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});
  EXPECT_DOUBLE_EQ(T.t.z(), -120.0);
}
