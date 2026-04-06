// Copyright 2026 Newton Campbell
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

/// @file test_transform_math.cpp
/// @brief GTest unit tests for bam_transform_math.
///
/// Epic: CM-44
/// Acceptance criteria:
///   - All transform utilities have tests and tests pass in CI/local.
///   - At least 5 "golden tests" exist with analytically known outputs.
///   - Transform composition/inversion numerical error < 1e-6 on golden tests.
///   - Quaternion order-swap tests FAIL when [x,y,z,w] is used as [w,x,y,z].

#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

#include "bam_transform_math/bam_transform_math.hpp"

using namespace bam_transform_math;  // NOLINT(build/namespaces)

constexpr double kTol = 1e-9;   // strict tolerance for golden tests
constexpr double kAccept = 1e-6; // acceptance-criteria tolerance

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Return identity quaternion [w=1, x=0, y=0, z=0].
static Vec4 identityQuat()
{
  return Vec4{1.0, 0.0, 0.0, 0.0};
}

/// Build a quaternion encoding a pure rotation about the NED X axis (North/roll)
/// by angle_rad using the aerospace ZYX formula simplified for φ-only rotation.
static Vec4 quatRoll(double angle_rad)
{
  // ZYX 3-2-1 with ψ=0, θ=0, φ=angle_rad
  const double half = angle_rad / 2.0;
  return Vec4{std::cos(half), std::sin(half), 0.0, 0.0};
}

/// Build a quaternion encoding a pure rotation about the NED Y axis (East/pitch).
static Vec4 quatPitch(double angle_rad)
{
  const double half = angle_rad / 2.0;
  return Vec4{std::cos(half), 0.0, std::sin(half), 0.0};
}

/// Build a quaternion encoding a pure rotation about the NED Z axis (Down/yaw).
static Vec4 quatYaw(double angle_rad)
{
  const double half = angle_rad / 2.0;
  return Vec4{std::cos(half), 0.0, 0.0, std::sin(half)};
}

// ─────────────────────────────────────────────────────────────────────────────
// GOLDEN TEST 1 — Identity quaternion produces identity rotation matrix
// ─────────────────────────────────────────────────────────────────────────────
TEST(QuatToRotmat, IdentityQuatGivesIdentityMatrix)
{
  const Mat3 R = quatToRotmat(identityQuat());
  const double err = (R - Mat3::Identity()).norm();
  EXPECT_LT(err, kTol);
}

// ─────────────────────────────────────────────────────────────────────────────
// GOLDEN TEST 2 — 90° roll (rotation about X)
// Expected: Y axis maps to -Z, Z axis maps to +Y
// ─────────────────────────────────────────────────────────────────────────────
TEST(QuatToRotmat, NinetyDegRollGolden)
{
  const Mat3 R = quatToRotmat(quatRoll(M_PI / 2.0));

  // R * [1,0,0] = [1,0,0]  (X unchanged)
  EXPECT_NEAR(R(0, 0), 1.0, kAccept);
  EXPECT_NEAR(R(1, 0), 0.0, kAccept);
  EXPECT_NEAR(R(2, 0), 0.0, kAccept);

  // R * [0,1,0] = [0,0,1]  (Y → +Z under NED right-hand roll)
  EXPECT_NEAR(R(0, 1), 0.0, kAccept);
  EXPECT_NEAR(R(1, 1), 0.0, kAccept);
  EXPECT_NEAR(R(2, 1), 1.0, kAccept);

  // R * [0,0,1] = [0,-1,0]
  EXPECT_NEAR(R(0, 2), 0.0,  kAccept);
  EXPECT_NEAR(R(1, 2), -1.0, kAccept);
  EXPECT_NEAR(R(2, 2), 0.0,  kAccept);
}

// ─────────────────────────────────────────────────────────────────────────────
// GOLDEN TEST 3 — 90° pitch (rotation about Y)
// ─────────────────────────────────────────────────────────────────────────────
TEST(QuatToRotmat, NinetyDegPitchGolden)
{
  const Mat3 R = quatToRotmat(quatPitch(M_PI / 2.0));

  // R * [0,1,0] = [0,1,0]  (Y unchanged)
  EXPECT_NEAR(R(0, 1), 0.0, kAccept);
  EXPECT_NEAR(R(1, 1), 1.0, kAccept);
  EXPECT_NEAR(R(2, 1), 0.0, kAccept);

  // R * [1,0,0] should map X toward -Z for a +pitch rotation
  EXPECT_NEAR(R(0, 0), 0.0,  kAccept);
  EXPECT_NEAR(R(1, 0), 0.0,  kAccept);
  EXPECT_NEAR(R(2, 0), -1.0, kAccept);

  // R * [0,0,1] = [1,0,0]
  EXPECT_NEAR(R(0, 2), 1.0, kAccept);
  EXPECT_NEAR(R(1, 2), 0.0, kAccept);
  EXPECT_NEAR(R(2, 2), 0.0, kAccept);
}

// ─────────────────────────────────────────────────────────────────────────────
// GOLDEN TEST 4 — 90° yaw (rotation about Z / Down)
// ─────────────────────────────────────────────────────────────────────────────
TEST(QuatToRotmat, NinetyDegYawGolden)
{
  const Mat3 R = quatToRotmat(quatYaw(M_PI / 2.0));

  // R * [1,0,0] = [0,1,0]  (North → East)
  EXPECT_NEAR(R(0, 0), 0.0, kAccept);
  EXPECT_NEAR(R(1, 0), 1.0, kAccept);
  EXPECT_NEAR(R(2, 0), 0.0, kAccept);

  // R * [0,1,0] = [-1,0,0]  (East → South)
  EXPECT_NEAR(R(0, 1), -1.0, kAccept);
  EXPECT_NEAR(R(1, 1), 0.0,  kAccept);
  EXPECT_NEAR(R(2, 1), 0.0,  kAccept);

  // R * [0,0,1] = [0,0,1]  (Down unchanged)
  EXPECT_NEAR(R(0, 2), 0.0, kAccept);
  EXPECT_NEAR(R(1, 2), 0.0, kAccept);
  EXPECT_NEAR(R(2, 2), 1.0, kAccept);
}

// ─────────────────────────────────────────────────────────────────────────────
// GOLDEN TEST 5 — Quaternion round-trip: q → R → q, error < 1e-6
// ─────────────────────────────────────────────────────────────────────────────
TEST(QuatRotmatRoundtrip, QToRToQ)
{
  const Vec4 q_in = quatYaw(M_PI / 4.0);   // 45° yaw
  const Mat3 R = quatToRotmat(q_in);
  const Vec4 q_out = rotmatToQuat(R);

  // Allow sign equivalence: q ≡ −q
  const double err_pos = (q_out - q_in).norm();
  const double err_neg = (q_out + q_in).norm();
  const double err = std::min(err_pos, err_neg);
  EXPECT_LT(err, kAccept);
}

// ─────────────────────────────────────────────────────────────────────────────
// GOLDEN TEST 6 — Rotation matrix round-trip: R → q → R, error < 1e-6
// ─────────────────────────────────────────────────────────────────────────────
TEST(QuatRotmatRoundtrip, RToQToR)
{
  // Construct a non-trivial rotation: 30° pitch
  const Mat3 R_in = quatToRotmat(quatPitch(M_PI / 6.0));
  const Vec4 q = rotmatToQuat(R_in);
  const Mat3 R_out = quatToRotmat(q);

  const double err = (R_out - R_in).norm();
  EXPECT_LT(err, kAccept);
}

// ─────────────────────────────────────────────────────────────────────────────
// Identity transform tests
// ─────────────────────────────────────────────────────────────────────────────
TEST(MakeTransform, IdentityPositionAndQuatGivesIdentityMatrix)
{
  const Mat4 T = makeTransform(Vec3::Zero(), identityQuat());
  const double err = (T - Mat4::Identity()).norm();
  EXPECT_LT(err, kTol);
}

TEST(InvertTransform, IdentityInvertedIsIdentity)
{
  const Mat4 T_inv = invertTransform(Mat4::Identity());
  const double err = (T_inv - Mat4::Identity()).norm();
  EXPECT_LT(err, kTol);
}

TEST(InvertTransform, TTimesItsInverseIsIdentity)
{
  const Vec3 p{3.0, -2.5, 7.0};
  const Vec4 q = quatRoll(M_PI / 3.0);  // 60° roll
  const Mat4 T = makeTransform(p, q);
  const Mat4 T_inv = invertTransform(T);

  const double err = (T * T_inv - Mat4::Identity()).norm();
  EXPECT_LT(err, kAccept);
}

TEST(InvertTransform, InvTimesTIsIdentity)
{
  const Vec3 p{-1.0, 4.0, -9.0};
  const Vec4 q = quatYaw(M_PI / 5.0);
  const Mat4 T = makeTransform(p, q);
  const Mat4 T_inv = invertTransform(T);

  const double err = (T_inv * T - Mat4::Identity()).norm();
  EXPECT_LT(err, kAccept);
}

// ─────────────────────────────────────────────────────────────────────────────
// Composition tests
// ─────────────────────────────────────────────────────────────────────────────
TEST(ComposeTransforms, IdentityComposedWithTIsT)
{
  const Vec3 p{10.0, -5.0, 3.0};
  const Vec4 q = quatPitch(M_PI / 4.0);
  const Mat4 T = makeTransform(p, q);

  const Mat4 result = composeTransforms(Mat4::Identity(), T);
  const double err = (result - T).norm();
  EXPECT_LT(err, kTol);
}

TEST(ComposeTransforms, TComposedWithItsInverseIsIdentity)
{
  const Vec3 p{0.0, 15.0, -7620.0};
  const Vec4 q = quatYaw(0.7854);  // ~45° yaw
  const Mat4 T = makeTransform(p, q);
  const Mat4 T_inv = invertTransform(T);

  const Mat4 result = composeTransforms(T, T_inv);
  const double err = (result - Mat4::Identity()).norm();
  EXPECT_LT(err, kAccept);
}

TEST(ComposeTransforms, KnownTranslationChain)
{
  // T_AB: translate +5 in X, no rotation
  const Mat4 T_AB = makeTransform(Vec3{5.0, 0.0, 0.0}, identityQuat());
  // T_BC: translate +3 in X, no rotation
  const Mat4 T_BC = makeTransform(Vec3{3.0, 0.0, 0.0}, identityQuat());
  // Expected: T_AC translates +8 in X
  const Mat4 T_AC = composeTransforms(T_AB, T_BC);

  EXPECT_NEAR(T_AC(0, 3), 8.0, kTol);
  EXPECT_NEAR(T_AC(1, 3), 0.0, kTol);
  EXPECT_NEAR(T_AC(2, 3), 0.0, kTol);
  // Rotation block should still be identity
  const double R_err = (T_AC.topLeftCorner<3, 3>() - Mat3::Identity()).norm();
  EXPECT_LT(R_err, kTol);
}

TEST(ComposeTransforms, Associativity)
{
  const Mat4 T1 = makeTransform(Vec3{1.0, 2.0, 0.0}, quatRoll(0.2));
  const Mat4 T2 = makeTransform(Vec3{0.0, 0.0, -5.0}, quatPitch(0.3));
  const Mat4 T3 = makeTransform(Vec3{3.0, -1.0, 0.0}, quatYaw(0.5));

  const Mat4 left  = composeTransforms(composeTransforms(T1, T2), T3);
  const Mat4 right = composeTransforms(T1, composeTransforms(T2, T3));

  const double err = (left - right).norm();
  EXPECT_LT(err, kTol);
}

// ─────────────────────────────────────────────────────────────────────────────
// QUATERNION ORDER TRAP — must FAIL if [x,y,z,w] is passed as [w,x,y,z]
// This test passes when the library correctly detects the convention mismatch.
// ─────────────────────────────────────────────────────────────────────────────
TEST(QuatConvention, ScalarLastOrderProducesDifferentMatrix)
{
  // Correct BAM convention: [w, x, y, z] for 90° yaw
  const Vec4 q_correct = quatYaw(M_PI / 2.0);  // [~0.707, 0, 0, ~0.707]

  // Wrong order: treat scipy/ROS2 [x, y, z, w] as if it were [w, x, y, z]
  // This is the scalar-last value mistakenly fed as scalar-first.
  Vec4 q_wrong;
  q_wrong << q_correct[1], q_correct[2], q_correct[3], q_correct[0];
  // q_wrong = [0, 0, ~0.707, ~0.707]  — completely different quaternion

  const Mat3 R_correct = quatToRotmat(q_correct);
  const Mat3 R_wrong   = quatToRotmat(q_wrong);

  // The two rotation matrices must be detectably different.
  // If this assertion fails, the convention trap is broken.
  const double diff = (R_correct - R_wrong).norm();
  EXPECT_GT(diff, 0.5);  // should be >> 0 for any non-identity rotation
}

TEST(QuatConvention, ScalarLastRollOrderProducesDifferentMatrix)
{
  const Vec4 q_correct = quatRoll(M_PI / 2.0);  // [~0.707, ~0.707, 0, 0]

  Vec4 q_wrong;
  q_wrong << q_correct[1], q_correct[2], q_correct[3], q_correct[0];

  const Mat3 R_correct = quatToRotmat(q_correct);
  const Mat3 R_wrong   = quatToRotmat(q_wrong);

  const double diff = (R_correct - R_wrong).norm();
  EXPECT_GT(diff, 0.5);
}

// ─────────────────────────────────────────────────────────────────────────────
// applyTransform / applyRotation helpers
// ─────────────────────────────────────────────────────────────────────────────
TEST(ApplyTransform, PureTranslation)
{
  const Mat4 T = makeTransform(Vec3{1.0, 2.0, 3.0}, identityQuat());
  const Vec3 p_in{0.0, 0.0, 0.0};
  const Vec3 p_out = applyTransform(T, p_in);
  EXPECT_NEAR(p_out[0], 1.0, kTol);
  EXPECT_NEAR(p_out[1], 2.0, kTol);
  EXPECT_NEAR(p_out[2], 3.0, kTol);
}

TEST(ApplyRotation, NinetyDegYawRotatesNorthToEast)
{
  const Mat4 T = makeTransform(Vec3::Zero(), quatYaw(M_PI / 2.0));
  const Vec3 north{1.0, 0.0, 0.0};
  const Vec3 rotated = applyRotation(T, north);
  EXPECT_NEAR(rotated[0], 0.0, kAccept);
  EXPECT_NEAR(rotated[1], 1.0, kAccept);  // → East
  EXPECT_NEAR(rotated[2], 0.0, kAccept);
}

// ─────────────────────────────────────────────────────────────────────────────
// T_WC composition — mirrors data_contract.md §11.1 label computation
// ─────────────────────────────────────────────────────────────────────────────
TEST(PipelineComposeTransforms, TWCFromTWBAndTBoC)
{
  // Aircraft at NED [0, 0, -7620] facing North (identity rotation)
  const Vec3 p_lead{0.0, 0.0, -7620.0};
  const Vec4 q_lead = identityQuat();
  const Mat4 T_WB_lead = makeTransform(p_lead, q_lead);

  // Camera offset from aircraft CG: 2 m forward (X), 0.5 m up (Z = -0.5 NED)
  const Vec3 t_BoC{2.0, 0.0, -0.5};
  const Vec4 q_BoC = identityQuat();  // camera aligned with body frame
  const Mat4 T_BoC = makeTransform(t_BoC, q_BoC);

  const Mat4 T_WC = composeTransforms(T_WB_lead, T_BoC);

  // Camera should be at NED [2, 0, -7620.5]
  EXPECT_NEAR(T_WC(0, 3), 2.0,     kTol);
  EXPECT_NEAR(T_WC(1, 3), 0.0,     kTol);
  EXPECT_NEAR(T_WC(2, 3), -7620.5, kTol);
}

// ─────────────────────────────────────────────────────────────────────────────
// WP2 stubs — must throw std::logic_error
// ─────────────────────────────────────────────────────────────────────────────
TEST(CameraStubs, PixelToRayThrowsLogicError)
{
  const Vec2 uv{320.0, 240.0};
  const Mat3K K = Mat3K::Identity();
  EXPECT_THROW(pixelToRay(uv, K), std::logic_error);
}

TEST(CameraStubs, RayToPixelThrowsLogicError)
{
  const Vec3 ray{0.0, 0.0, 1.0};
  const Mat3K K = Mat3K::Identity();
  EXPECT_THROW(rayToPixel(ray, K), std::logic_error);
}

// ─────────────────────────────────────────────────────────────────────────────
// Zero-norm quaternion rejection
// ─────────────────────────────────────────────────────────────────────────────
TEST(QuatToRotmat, ZeroNormQuatThrows)
{
  const Vec4 q_zero = Vec4::Zero();
  EXPECT_THROW(quatToRotmat(q_zero), std::invalid_argument);
}

TEST(MakeTransform, ZeroNormQuatThrows)
{
  EXPECT_THROW(makeTransform(Vec3::Zero(), Vec4::Zero()), std::invalid_argument);
}

int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
