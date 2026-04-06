# `bam_transform_math` — C++ ROS 2 Jazzy Transform Library

**Epic:** CM-44  
**Part of:** FormationFlight / Baseball Avoidance Multirotor (BAM)  
**Language:** C++20  
**Build system:** `ament_cmake` (ROS 2 Jazzy)  
**Dependencies:** Eigen3 only (header-only; ships with ROS 2 Jazzy)  
**Testing:** GTest via `ament_cmake_gtest` + GitHub Actions CI  

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Conventions — Read First](#2-conventions--read-first)
   - 2.1 [Quaternion Order](#21-quaternion-order)
   - 2.2 [Transform Direction](#22-transform-direction)
   - 2.3 [Coordinate Frame (NED)](#23-coordinate-frame-ned)
   - 2.4 [Homogeneous Matrix Layout](#24-homogeneous-matrix-layout)
3. [Package Structure](#3-package-structure)
4. [API Reference](#4-api-reference)
5. [Usage Examples](#5-usage-examples)
   - 5.1 [Building T_WB from a Pose CSV Row](#51-building-t_wb-from-a-pose-csv-row)
   - 5.2 [Composing T_WC = T_WB * T_BoC](#52-composing-t_wc--t_wb--t_boc)
   - 5.3 [Relative Position in Lead Body Frame](#53-relative-position-in-lead-body-frame)
   - 5.4 [Relative Quaternion Between Two Aircraft](#54-relative-quaternion-between-two-aircraft)
   - 5.5 [Using This Library in Another ROS 2 Package](#55-using-this-library-in-another-ros-2-package)
6. [⚠️ Common Pitfalls](#6-️-common-pitfalls)
7. [Building and Testing](#7-building-and-testing)
8. [WP2 Stubs](#8-wp2-stubs)
9. [Design Notes](#9-design-notes)

---

## 1. Purpose

This package provides a minimal, header-only C++ library for SE(3) rigid-body
transform math used throughout the BAM ML relative pose estimation pipeline.
It is the authoritative C++ implementation of the coordinate frame, quaternion,
and homogeneous transform conventions defined in
[`/docs/data_contract.md`](../docs/data_contract.md).

Core capabilities:

- Build 4×4 SE(3) homogeneous transforms `T_WB` from position + quaternion
- Compose transforms: `T_WC = T_WB * T_BoC`
- Analytically invert transforms using `Rᵀ` (numerically superior to a
  general matrix inverse)
- Convert quaternion ↔ rotation matrix using Shepperd's method
- Stub helpers for pixel ↔ ray projection (WP2; throw `std::logic_error`)
- `applyTransform` and `applyRotation` convenience helpers

The library is **intentionally header-only and depends only on Eigen3**,
which ships as part of every ROS 2 Jazzy install.  There is no `rclcpp`
dependency; this library compiles and tests successfully with or without a
running ROS 2 environment.

---

## 2. Conventions — Read First

> ⚠️ **The single most dangerous mistake in this pipeline is silently
> using the wrong quaternion component order.**

### 2.1 Quaternion Order

All functions in this library use **scalar-first `[w, x, y, z]`**
(Hamilton convention), matching `data_contract.md` §4.1 and the MATLAB
generator `Gen_ML_Rel_Data.m`.

```
[w, x, y, z]   ←  THIS library · data contract · MATLAB output
[x, y, z, w]   ←  scipy · PyTorch3D · ROS 2 TF2 / geometry_msgs
```

Within the library, quaternions are passed and returned as `Vec4`
(`Eigen::Matrix<double,4,1>`) with the layout `[w, x, y, z]`.  Eigen's
`Eigen::Quaternion<double>` is used only internally (inside
`quat_utils.hpp`) and never exposed in the public API, specifically to
avoid confusion with Eigen's internal `[x, y, z, w]` storage order.

**Converting to ROS 2 `geometry_msgs::msg::Quaternion`:**

```cpp
#include "bam_transform_math/bam_transform_math.hpp"
#include <geometry_msgs/msg/quaternion.hpp>

bam_transform_math::Vec4 q_wxyz = bam_transform_math::quatYaw(0.5);

geometry_msgs::msg::Quaternion ros_q;
ros_q.w = q_wxyz[0];  // BAM [0]=w
ros_q.x = q_wxyz[1];  // BAM [1]=x
ros_q.y = q_wxyz[2];  // BAM [2]=y
ros_q.z = q_wxyz[3];  // BAM [3]=z
```

**Converting from ROS 2 / TF2 to BAM:**

```cpp
bam_transform_math::Vec4 q_wxyz;
q_wxyz << ros_q.w, ros_q.x, ros_q.y, ros_q.z;
```

### 2.2 Transform Direction

The pose CSV files store the quaternion as `q_i2b` — it rotates vectors
**from the World (NED) frame into the aircraft body frame** (§4.2):

```
q_i2b :  v_body  = R(q_i2b) · v_world
         v_world = R(q_i2b)ᵀ · v_body
```

`makeTransform(position, q_i2b)` builds `T_WB`:

```
T_WB = | R(q_i2b)   p_W |
       | 0  0  0     1  |
```

To map a point from world to body, use `invertTransform(T_WB)`.

### 2.3 Coordinate Frame (NED)

| Axis | Direction | Notes |
|------|-----------|-------|
| `X`  | North     | Forward |
| `Y`  | East      | Right |
| `Z`  | Down      | **Altitude stored negative-up** |

`pos_z_m = -7620.0` → aircraft is 7620 m above datum.  
Consumers must negate `pos_z_m` to get altitude for display; do NOT
negate before passing to `makeTransform`.

### 2.4 Homogeneous Matrix Layout

```
T = | R₃ₓ₃   t₃ₓ₁ |    R ∈ SO(3),  t = translation in parent frame
    | 0  0  0   1  |
```

All matrices are `Eigen::Matrix<double,4,4>` (column-major, float64).

---

## 3. Package Structure

```
bam_transform_math/
├── CMakeLists.txt
├── package.xml
├── README.md
├── include/
│   └── bam_transform_math/
│       ├── bam_transform_math.hpp   ← umbrella header (include this)
│       ├── types.hpp                ← Eigen type aliases
│       ├── quat_utils.hpp           ← quatToRotmat / rotmatToQuat
│       ├── transform_ops.hpp        ← makeTransform / compose / invert
│       └── camera_stubs.hpp         ← pixelToRay / rayToPixel (WP2 stubs)
└── test/
    └── test_transform_math.cpp      ← GTest suite (24 tests)
```

The library is **header-only** — there is no compiled `.so` or `.a`.
Downstream packages link against the `INTERFACE` CMake target.

---

## 4. API Reference

All public symbols live in the `bam_transform_math` namespace.

### Type Aliases (`types.hpp`)

| Alias | Underlying type | Description |
|-------|-----------------|-------------|
| `Scalar` | `double` | All computation is float64 |
| `Vec2` | `Eigen::Matrix<double,2,1>` | Pixel / 2D coordinates |
| `Vec3` | `Eigen::Matrix<double,3,1>` | 3D position / direction |
| `Vec4` | `Eigen::Matrix<double,4,1>` | Quaternion `[w,x,y,z]` |
| `Mat3` | `Eigen::Matrix<double,3,3>` | Rotation matrix |
| `Mat4` | `Eigen::Matrix<double,4,4>` | SE(3) homogeneous transform |
| `Mat3K` | `Eigen::Matrix<double,3,3>` | Camera intrinsic matrix |

### `quatToRotmat(q_wxyz) → Mat3`  (`quat_utils.hpp`)

Convert scalar-first `[w,x,y,z]` quaternion to a 3×3 rotation matrix.
Normalises the input; throws `std::invalid_argument` on zero-norm.

### `rotmatToQuat(R) → Vec4`  (`quat_utils.hpp`)

Convert 3×3 rotation matrix to `[w,x,y,z]` quaternion via **Shepperd's
method** (numerically stable at all orientations including 180° rotations).
Returns canonical form with `w ≥ 0`.

### `makeTransform(position, q_wxyz) → Mat4`  (`transform_ops.hpp`)

Build 4×4 SE(3) homogeneous transform from position + quaternion.
Throws `std::invalid_argument` on zero-norm quaternion.

### `composeTransforms(T_AB, T_BC) → Mat4`  (`transform_ops.hpp`)

Compose `T_AC = T_AB * T_BC`.  Follow subscript chaining: inner frame
label must match between the two arguments.

### `invertTransform(T) → Mat4`  (`transform_ops.hpp`)

Analytically invert SE(3) transform: `T⁻¹ = | Rᵀ  −Rᵀt |`.
                                              `| 0     1  |`

### `applyTransform(T, p) → Vec3`  (`transform_ops.hpp`)

Apply transform `T` to 3D point `p` (handles homogeneous lifting internally).

### `applyRotation(T, v) → Vec3`  (`transform_ops.hpp`)

Apply only the rotation block of `T` to direction vector `v` (translation
is NOT applied — use this for vectors, not points).

### `pixelToRay(uv, K) → Vec3`  (`camera_stubs.hpp`) *(WP2 stub)*

Always throws `std::logic_error`.  Planned: `ray = normalize(K⁻¹·[u,v,1]ᵀ)`.

### `rayToPixel(ray, K) → Vec2`  (`camera_stubs.hpp`) *(WP2 stub)*

Always throws `std::logic_error`.  Planned: `[u,v] = (K·ray/ray.z()).head<2>()`.

---

## 5. Usage Examples

### 5.1 Building T_WB from a Pose CSV Row

```cpp
#include "bam_transform_math/bam_transform_math.hpp"
#include <fstream>
#include <sstream>

// CSV layout: time_s, pos_x_m, pos_y_m, pos_z_m, q_w, q_x, q_y, q_z
std::ifstream f("lead_poses.csv");
std::string line;
std::getline(f, line);  // skip header
std::getline(f, line);  // first data row (nominal pose)

std::istringstream ss(line);
std::vector<double> row;
for (double v; ss >> v; ) { row.push_back(v); if (ss.peek() == ',') ss.ignore(); }

bam_transform_math::Vec3 position{row[1], row[2], row[3]};
bam_transform_math::Vec4 q_wxyz{row[4], row[5], row[6], row[7]};

auto T_WB_lead = bam_transform_math::makeTransform(position, q_wxyz);
```

### 5.2 Composing T_WC = T_WB * T_BoC

```cpp
// Camera extrinsic (pending OI-01; use placeholder for now)
bam_transform_math::Vec3 t_BoC{2.0, 0.0, -0.5};   // metres, in B_lead frame
bam_transform_math::Vec4 q_BoC{1.0, 0.0, 0.0, 0.0}; // identity rotation

auto T_BoC = bam_transform_math::makeTransform(t_BoC, q_BoC);
auto T_WC  = bam_transform_math::composeTransforms(T_WB_lead, T_BoC);
```

### 5.3 Relative Position in Lead Body Frame

```cpp
// Absolute NED positions
bam_transform_math::Vec3 pos_lead {row_lead[1],  row_lead[2],  row_lead[3]};
bam_transform_math::Vec3 pos_trail{row_trail[1], row_trail[2], row_trail[3]};

// rel_pos in World (NED) frame
bam_transform_math::Vec3 rel_pos_ned = pos_trail - pos_lead;

// Rotate into lead body frame using only the rotation block
auto T_WB_lead = bam_transform_math::makeTransform(
  pos_lead, bam_transform_math::Vec4{row_lead[4], row_lead[5], row_lead[6], row_lead[7]});
auto T_BW_lead = bam_transform_math::invertTransform(T_WB_lead);

bam_transform_math::Vec3 rel_pos_body =
  bam_transform_math::applyRotation(T_BW_lead, rel_pos_ned);
```

### 5.4 Relative Quaternion Between Two Aircraft

```cpp
// data_contract.md §11.1: q_rel = q_i2b_trail * conj(q_i2b_lead)
// Equivalently: R_rel = R_trail * R_lead^T
bam_transform_math::Vec4 q_lead {row_lead[4],  row_lead[5],  row_lead[6],  row_lead[7]};
bam_transform_math::Vec4 q_trail{row_trail[4], row_trail[5], row_trail[6], row_trail[7]};

auto R_lead  = bam_transform_math::quatToRotmat(q_lead);
auto R_trail = bam_transform_math::quatToRotmat(q_trail);
auto R_rel   = R_trail * R_lead.transpose();
auto q_rel   = bam_transform_math::rotmatToQuat(R_rel);  // [w, x, y, z]
```

### 5.5 Using This Library in Another ROS 2 Package

In your downstream `package.xml`:

```xml
<depend>bam_transform_math</depend>
```

In your `CMakeLists.txt`:

```cmake
find_package(bam_transform_math REQUIRED)
target_link_libraries(my_node bam_transform_math::bam_transform_math)
```

In C++:

```cpp
#include "bam_transform_math/bam_transform_math.hpp"
```

---

## 6. ⚠️ Common Pitfalls

### Pitfall 1 — Wrong quaternion order from ROS 2 / scipy

ROS 2 `geometry_msgs::msg::Quaternion`, TF2, and scipy all use scalar-last
`[x, y, z, w]`.  Passing these directly to any function in this library
will produce a silently wrong rotation.  The test `ScalarLastOrderProducesDifferentMatrix`
asserts this causes a detectable error (diff > 0.5 in rotation matrix norm).

**Fix:**
```cpp
// From ROS 2 msg → BAM:
bam_transform_math::Vec4 q;
q << ros_q.w, ros_q.x, ros_q.y, ros_q.z;  // explicit scalar-first
```

### Pitfall 2 — Altitude sign (NED Z is negative-up)

`pos_z_m = -7620.0` means 7620 m above datum.  Do **not** negate before
passing to `makeTransform`.  Only negate for human-readable display.

### Pitfall 3 — Differentiating adjacent pose columns

Each column is a statistically independent random draw (§5.3).  No kinematic
continuity exists between samples.  Never compute velocity by finite differences.

### Pitfall 4 — T_BoC values are [TBD]

Camera extrinsic `T_BoC` is pending Blender model extraction (OI-01).
Check `metadata.json:camera_extrinsic` for `null` before composing `T_WC`.

### Pitfall 5 — Non-standard foot-to-metre conversion

The MATLAB generator uses `ft_2_mtr = 0.3028` (not SI `0.3048`).  All
position values carry this ~0.066% offset.  Do not silently substitute the
SI value (OI-11).

---

## 7. Building and Testing

### Prerequisites

- ROS 2 Jazzy installed and sourced
- `colcon` build tool

### Build

```bash
# From your ROS 2 workspace root
cd ~/ros2_ws
colcon build --packages-select bam_transform_math
source install/setup.bash
```

### Run Tests

```bash
colcon test --packages-select bam_transform_math
colcon test-result --verbose
```

Or run the GTest binary directly after building:

```bash
./build/bam_transform_math/test_transform_math
```

### CI

Tests run automatically on every push and PR via
[`.github/workflows/test_transform_math.yml`](../.github/workflows/test_transform_math.yml).

### Test Coverage

| Category | Test name | Golden? |
|----------|-----------|---------|
| Identity quat → identity matrix | `IdentityQuatGivesIdentityMatrix` | ✅ Golden 1 |
| 90° roll exact matrix | `NinetyDegRollGolden` | ✅ Golden 2 |
| 90° pitch exact matrix | `NinetyDegPitchGolden` | ✅ Golden 3 |
| 90° yaw exact matrix | `NinetyDegYawGolden` | ✅ Golden 4 |
| q → R → q roundtrip | `QToRToQ` | ✅ Golden 5 |
| R → q → R roundtrip | `RToQToR` | ✅ Golden 6 |
| Identity transform | `IdentityPositionAndQuatGivesIdentityMatrix` | — |
| Identity inversion | `IdentityInvertedIsIdentity` | — |
| T × T⁻¹ = I | `TTimesItsInverseIsIdentity` | — |
| T⁻¹ × T = I | `InvTimesTIsIdentity` | — |
| I ∘ T = T | `IdentityComposedWithTIsT` | — |
| T ∘ T⁻¹ = I (compose) | `TComposedWithItsInverseIsIdentity` | — |
| Known translation chain | `KnownTranslationChain` | — |
| Associativity | `Associativity` | — |
| Quat order trap (yaw) | `ScalarLastOrderProducesDifferentMatrix` | — |
| Quat order trap (roll) | `ScalarLastRollOrderProducesDifferentMatrix` | — |
| Pure translation apply | `PureTranslation` | — |
| Yaw rotates North→East | `NinetyDegYawRotatesNorthToEast` | — |
| T_WC pipeline compose | `TWCFromTWBAndTBoC` | — |
| pixelToRay stub throws | `PixelToRayThrowsLogicError` | — |
| rayToPixel stub throws | `RayToPixelThrowsLogicError` | — |
| Zero-norm quat rejection | `ZeroNormQuatThrows` (×2) | — |

All golden tests have numerical error < 1e-6 (acceptance criterion).

---

## 8. WP2 Stubs

| Function | Blocked on |
|----------|------------|
| `pixelToRay(uv, K)` | Camera intrinsic matrix `K` (OI-04: image resolution and focal length `[TBD]`) |
| `rayToPixel(ray, K)` | Same — also requires `T_BoC` (OI-01) |

Both functions throw `std::logic_error` with an explanatory message that
cites the open items.  Tests assert `std::logic_error` is thrown; they
will be updated to assert known pixel ↔ ray correspondences when WP2 begins.

**WP2 implementation formulas:**

```cpp
// pixelToRay:
Eigen::Vector3d ray_h = K.inverse() * Eigen::Vector3d(uv[0], uv[1], 1.0);
return ray_h.normalized();

// rayToPixel:
Eigen::Vector3d proj = K * ray;
return proj.head<2>() / proj[2];
```

---

## 9. Design Notes

**Why header-only?**  
Keeps the package usable as a plain CMake `INTERFACE` target.  Downstream
packages link with zero compiled objects to manage, and the build system
need not produce a shared library just for math utilities.

**Why Eigen3 only?**  
Eigen is already a transitive dependency of every ROS 2 navigation, perception,
and geometry package.  Adding it introduces no new install footprint.  Using
`tf2_eigen` or `geometry_msgs` would pull in `rclcpp` and the full ROS 2
message infrastructure, which is inappropriate for a pure math library.

**Why Shepperd's method?**  
The naive `arccos((trace(R)−1)/2)` formula becomes numerically unstable near
180° rotations.  Shepperd selects the largest quaternion component as the
pivot, guaranteeing well-conditioned arithmetic across all SO(3).

**Why `[[nodiscard]]` everywhere?**  
All transform functions return a new matrix; silently ignoring the return
value is almost always a bug.  `[[nodiscard]]` promotes this to a compiler
warning with `-Wall`.

**Float64 everywhere**  
All computation uses `double`.  The acceptance-criteria tolerance of 1e-6
is comfortably achievable in float64 for composed and inverted transforms
but is tight for float32 at realistic pose scales (metres to kilometres).
