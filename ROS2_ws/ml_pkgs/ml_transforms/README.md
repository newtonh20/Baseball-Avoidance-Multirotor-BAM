# ml_transforms

`ml_transforms` is a dependency-light ROS 2 Jazzy C++ library package for rigid transform math used by the Baseball Avoidance Multirotor (BAM) / FormationFlight synthetic ML relative pose pipeline.

It is intentionally a **library package** (`ament_cmake`) with no required runtime node executable. Other ROS 2 packages and plain C++ modules can link against it.

## Purpose

This package provides deterministic, tested primitives for:

- Building rigid transforms from position + quaternion
- Composing transforms (`T_AC = T_AB * T_BC`)
- Inverting transforms analytically (without generic 4x4 inverse)
- Converting quaternion `[w,x,y,z]` to/from rotation matrices
- Converting transform objects to/from 4x4 homogeneous matrices
- Placeholder camera pixel/ray helpers for WP2 integration

## Data-contract conventions (critical)

These conventions are mandatory and reflected in the API and tests:

1. **Quaternion storage order is scalar-first**: `[w, x, y, z]`.
2. **Stored quaternion direction is `q_i2b`**: rotates vectors from inertial world NED into body frame.
3. **World frame is right-handed NED**:
   - `X = North`
   - `Y = East`
   - `Z = Down`
4. **Altitude sign in stored `pos_z_m` is negative-up** due to NED `+Z Down`.
5. **Camera extrinsic `T_BoC` is supported in composition APIs**, while camera projection details are deferred to WP2.

## ROS quaternion warning

ROS messages (`geometry_msgs::msg::Quaternion`) store quaternion fields as `x, y, z, w`, while this library uses `w, x, y, z`.

When bridging between ROS messages and this library, always reorder explicitly.

## Package structure

```text
ROS2_ws/ml_pkgs/ml_transforms
├── CMakeLists.txt
├── package.xml
├── README.md
├── include/ml_transforms/
│   ├── transform_math.hpp
│   ├── quaternion.hpp
│   ├── rotation_matrix.hpp
│   ├── transform.hpp
│   └── camera_geometry.hpp
├── src/
├── test/
└── .github/workflows/test_ml_transforms.yml
```

## Build and test

From your ROS 2 workspace root:

```bash
colcon build --packages-select ml_transforms
colcon test --packages-select ml_transforms
colcon test-result --verbose
```

## API overview

Namespace: `ml_transforms`

- `struct QuaternionWXYZ { double w, x, y, z; }`
- `struct Transform { Eigen::Matrix3d R; Eigen::Vector3d t; }`
- `Transform makeTransform(position, q_wxyz)`
- `Transform composeTransforms(T_AB, T_BC)`
- `Transform invertTransform(T)`
- `Eigen::Matrix3d quatToRotmat(q_wxyz)`
- `QuaternionWXYZ rotmatToQuat(R)`
- `Eigen::Matrix4d toHomogeneousMatrix(T)`
- `Transform fromHomogeneousMatrix(H)`
- `pixelToRay(...)`, `rayToPixel(...)` (WP2 stubs that currently throw)

## Usage examples

### 1) Build `T_WB` from position + scalar-first quaternion

```cpp
#include "ml_transforms/transform_math.hpp"

using ml_transforms::QuaternionWXYZ;
using ml_transforms::Transform;

Eigen::Vector3d p_WB(12.0, -4.0, -150.0);  // NED: z is down
QuaternionWXYZ q_i2b{0.9238795, 0.0, 0.0, 0.3826834};
Transform T_WB = ml_transforms::makeTransform(p_WB, q_i2b);
```

### 2) Compose with camera extrinsic `T_BoC`

```cpp
Transform T_BoC = ml_transforms::makeTransform(
  Eigen::Vector3d(0.2, 0.0, 0.05),
  QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});

Transform T_WC = ml_transforms::composeTransforms(T_WB, T_BoC);
```

### 3) Relative position in body frame (`relative_position_body` style)

```cpp
Eigen::Vector3d relative_position_ned = target_pos_ned - ownship_pos_ned;
Eigen::Vector3d relative_position_body = T_WB.R * relative_position_ned;  // q_i2b convention
```

### 4) Convert to homogeneous matrix

```cpp
Eigen::Matrix4d H_WB = ml_transforms::toHomogeneousMatrix(T_WB);
```

## Testing overview

Tests include:

- Identity and inversion sanity checks
- Golden 90° roll/pitch/yaw matrix tests
- Composition associativity and known compositions
- Quaternion order mismatch detection (`wxyz` vs accidental `xyzw`)
- Quaternion and rotation round-trip checks
- Homogeneous matrix validation and parsing checks
- Contract-aligned `q_i2b`/`relative_position_body` example tests
- Camera WP2 stubs throwing `std::logic_error`

## WP2 camera note

`pixelToRay(...)` and `rayToPixel(...)` are intentionally stubs in this package version and throw:

- `pixelToRay: to be completed in WP2`
- `rayToPixel: to be completed in WP2`

## Common pitfalls

1. **Wrong quaternion order** (`xyzw` passed where `wxyz` is expected).
2. **Wrong rotation direction** (`q_b2i` mistaken for stored `q_i2b`).
3. **NED sign confusion** (`pos_z_m` is down-positive; altitude up is negative).
4. **Silent ROS bridge errors** if `geometry_msgs::msg::Quaternion` fields are copied without reordering.
5. **Treating camera pixel/ray stubs as implemented geometry** before WP2.
