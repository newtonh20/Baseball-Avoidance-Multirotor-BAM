// Copyright 2026 Newton Campbell
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0

/// @file camera_stubs.hpp
/// @brief Pixel ↔ ray projection helpers (WP2 stubs).
///
/// These functions implement the standard pinhole camera projection model.
/// They are intentionally left as stubs for Work Package 2 pending:
///   - Camera intrinsic matrix K  (data_contract.md OI-04: image resolution
///     and focal length [TBD])
///   - Camera extrinsic offset T_BoC  (data_contract.md OI-01)
///
/// Both functions throw std::logic_error until WP2 is complete.
/// Tests assert this throw; they will be updated when implemented.

#pragma once

#include <stdexcept>
#include <string_view>

#include "bam_transform_math/types.hpp"

namespace bam_transform_math
{

/// @brief Convert pixel coordinates [u, v] to a unit ray in the camera frame.
///
/// Pinhole model (to be implemented in WP2):
///   ray = normalize( K⁻¹ * [u, v, 1]ᵀ )
///
/// @param uv  Pixel coordinates [u, v].
/// @param K   3×3 camera intrinsic matrix.
/// @returns   Unit ray direction in the camera frame (not yet implemented).
/// @throws    std::logic_error always — WP2 stub.
[[nodiscard]] inline Vec3 pixelToRay(
  [[maybe_unused]] const Vec2 & uv,
  [[maybe_unused]] const Mat3K & K)
{
  throw std::logic_error(
    "bam_transform_math::pixelToRay: stub — to be completed in WP2. "
    "Blocked on: camera intrinsic matrix K (OI-04) and T_BoC (OI-01).");
}

/// @brief Project a unit ray in the camera frame to pixel coordinates [u, v].
///
/// Pinhole model (to be implemented in WP2):
///   [u, v] = ( K * ray / ray.z() ).head<2>()
///
/// @param ray  Unit ray direction in the camera frame.
/// @param K    3×3 camera intrinsic matrix.
/// @returns    Pixel coordinates [u, v] (not yet implemented).
/// @throws     std::logic_error always — WP2 stub.
[[nodiscard]] inline Vec2 rayToPixel(
  [[maybe_unused]] const Vec3 & ray,
  [[maybe_unused]] const Mat3K & K)
{
  throw std::logic_error(
    "bam_transform_math::rayToPixel: stub — to be completed in WP2. "
    "Blocked on: camera intrinsic matrix K (OI-04) and T_BoC (OI-01).");
}

}  // namespace bam_transform_math
