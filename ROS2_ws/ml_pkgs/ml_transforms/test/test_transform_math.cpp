#include <gtest/gtest.h>

#include <cmath>

#include "ml_transforms/transform_math.hpp"

namespace {

bool mat3Near(const Eigen::Matrix3d& A, const Eigen::Matrix3d& B, double tol) {
  return (A - B).norm() < tol;
}

bool mat4Near(const Eigen::Matrix4d& A, const Eigen::Matrix4d& B, double tol) {
  return (A - B).norm() < tol;
}

}  // namespace

TEST(TransformMathTest, IdentityTransformConstruction) {
  const Eigen::Vector3d pos = Eigen::Vector3d::Zero();
  const ml_transforms::QuaternionWXYZ q_identity{1.0, 0.0, 0.0, 0.0};

  const ml_transforms::Transform T = ml_transforms::makeTransform(pos, q_identity);

  EXPECT_TRUE(mat3Near(T.R, Eigen::Matrix3d::Identity(), 1e-10));
  EXPECT_TRUE((T.t - Eigen::Vector3d::Zero()).norm() < 1e-10);
}

TEST(TransformMathTest, InversionOfIdentityIsIdentity) {
  ml_transforms::Transform I{Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero()};
  const ml_transforms::Transform I_inv = ml_transforms::invertTransform(I);

  EXPECT_TRUE(mat3Near(I_inv.R, Eigen::Matrix3d::Identity(), 1e-12));
  EXPECT_TRUE((I_inv.t - Eigen::Vector3d::Zero()).norm() < 1e-12);
}

TEST(TransformMathTest, TransformTimesInverseIsIdentity) {
  const double c = std::cos(M_PI / 8.0);
  const double s = std::sin(M_PI / 8.0);
  const ml_transforms::QuaternionWXYZ q{c, 0.0, 0.0, s};

  const ml_transforms::Transform T =
      ml_transforms::makeTransform(Eigen::Vector3d(4.0, -2.0, 5.5), q);
  const ml_transforms::Transform T_inv = ml_transforms::invertTransform(T);
  const ml_transforms::Transform should_be_I = ml_transforms::composeTransforms(T, T_inv);

  EXPECT_TRUE(mat3Near(should_be_I.R, Eigen::Matrix3d::Identity(), 1e-6));
  EXPECT_TRUE(should_be_I.t.norm() < 1e-6);
}

TEST(TransformMathTest, InverseMatchesHomogeneousInverse) {
  const double c = std::cos(M_PI / 8.0);
  const double s = std::sin(M_PI / 8.0);
  const ml_transforms::QuaternionWXYZ q{c, 0.0, 0.0, s};

  const ml_transforms::Transform T =
      ml_transforms::makeTransform(Eigen::Vector3d(1.0, 2.0, -3.0), q);
  const Eigen::Matrix4d H = ml_transforms::toHomogeneousMatrix(T);

  const Eigen::Matrix4d H_inv_numeric = H.inverse();
  const Eigen::Matrix4d H_inv_analytic =
      ml_transforms::toHomogeneousMatrix(ml_transforms::invertTransform(T));

  EXPECT_TRUE(mat4Near(H_inv_numeric, H_inv_analytic, 1e-10));
}

TEST(TransformMathTest, CompositionAssociativity) {
  const ml_transforms::Transform T_AB = ml_transforms::makeTransform(
      Eigen::Vector3d(1.0, 2.0, 3.0), ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});

  const double c = std::cos(M_PI / 4.0);
  const double s = std::sin(M_PI / 4.0);
  const ml_transforms::Transform T_BC =
      ml_transforms::makeTransform(Eigen::Vector3d(0.0, 1.0, 0.0),
                                   ml_transforms::QuaternionWXYZ{c, 0.0, 0.0, s});

  const ml_transforms::Transform T_CD = ml_transforms::makeTransform(
      Eigen::Vector3d(0.2, -0.3, 0.7), ml_transforms::QuaternionWXYZ{0.97, 0.1, 0.2, 0.05});

  const ml_transforms::Transform lhs =
      ml_transforms::composeTransforms(ml_transforms::composeTransforms(T_AB, T_BC), T_CD);
  const ml_transforms::Transform rhs =
      ml_transforms::composeTransforms(T_AB, ml_transforms::composeTransforms(T_BC, T_CD));

  EXPECT_TRUE(mat4Near(ml_transforms::toHomogeneousMatrix(lhs), ml_transforms::toHomogeneousMatrix(rhs),
                       1e-10));
}

TEST(TransformMathTest, KnownTranslationComposition) {
  const ml_transforms::Transform I = ml_transforms::makeTransform(
      Eigen::Vector3d::Zero(), ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});
  const ml_transforms::Transform T = ml_transforms::makeTransform(
      Eigen::Vector3d(3.0, -4.0, 2.5), ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});

  const ml_transforms::Transform out = ml_transforms::composeTransforms(I, T);
  EXPECT_TRUE((out.t - Eigen::Vector3d(3.0, -4.0, 2.5)).norm() < 1e-12);
}

TEST(TransformMathTest, KnownCameraComposition) {
  const ml_transforms::Transform T_WB = ml_transforms::makeTransform(
      Eigen::Vector3d(10.0, 20.0, -30.0), ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});

  const ml_transforms::Transform T_BoC = ml_transforms::makeTransform(
      Eigen::Vector3d(0.5, 0.1, 0.2), ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});

  const ml_transforms::Transform T_WC = ml_transforms::composeTransforms(T_WB, T_BoC);
  EXPECT_TRUE((T_WC.t - Eigen::Vector3d(10.5, 20.1, -29.8)).norm() < 1e-10);
}

TEST(TransformMathTest, HomogeneousRoundTripAndValidation) {
  const ml_transforms::Transform T = ml_transforms::makeTransform(
      Eigen::Vector3d(-1.0, 4.0, -100.0), ml_transforms::QuaternionWXYZ{0.95, 0.1, -0.2, 0.0});

  const Eigen::Matrix4d H = ml_transforms::toHomogeneousMatrix(T);
  const ml_transforms::Transform T_back = ml_transforms::fromHomogeneousMatrix(H);

  EXPECT_TRUE(mat4Near(ml_transforms::toHomogeneousMatrix(T), ml_transforms::toHomogeneousMatrix(T_back),
                       1e-10));

  Eigen::Matrix4d bad = H;
  bad(3, 0) = 1.0;
  EXPECT_THROW((void)ml_transforms::fromHomogeneousMatrix(bad), std::invalid_argument);
}

TEST(TransformMathTest, ContractQi2bAndRelativePositionBody) {
  // q_i2b for +90 deg yaw rotates inertial NED vectors into body frame.
  const double c = std::cos(M_PI / 4.0);
  const double s = std::sin(M_PI / 4.0);
  const ml_transforms::QuaternionWXYZ q_i2b{c, 0.0, 0.0, s};

  const Eigen::Matrix3d R_i2b = ml_transforms::quatToRotmat(q_i2b);
  const Eigen::Vector3d rel_ned(1.0, 0.0, 0.0);  // north in inertial NED
  const Eigen::Vector3d rel_body = R_i2b * rel_ned;

  // With this rotation, inertial north maps to +body y.
  EXPECT_NEAR(rel_body.x(), 0.0, 1e-6);
  EXPECT_NEAR(rel_body.y(), 1.0, 1e-6);
  EXPECT_NEAR(rel_body.z(), 0.0, 1e-6);
}
