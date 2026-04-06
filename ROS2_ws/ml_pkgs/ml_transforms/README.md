# ml_transforms

`ml_transforms` is a ROS 2 Jazzy C++ library package that provides robust, dependency-light rigid-body transform and quaternion math utilities for the Baseball Avoidance Multirotor / FormationFlight synthetic ML relative pose pipeline.

This package is a reusable **library** (`ament_cmake` + Eigen) and is intentionally not a ROS node package.

---

## Purpose

The library provides:

- deterministic SE(3) transform construction from position + quaternion
- robust transform composition and analytic inversion
- quaternion `[w, x, y, z]` (scalar-first) ⇄ rotation matrix conversion
- homogeneous 4x4 conversion helpers
- WP2 camera pixel↔ray API stubs (currently throw `logic_error` intentionally)

This supports workflows involving lead/trail aircraft relative pose labels such as:

- `relative_position_body`
- `relative_quaternion_wxyz`

---

## Non-negotiable project conventions

### 1) Quaternion storage order is scalar-first: `[w, x, y, z]`

This package assumes scalar-first quaternions in every API.

> ⚠️ Important: this differs from `geometry_msgs::msg::Quaternion` field ordering (`x, y, z, w`) and many libraries (e.g., SciPy default conventions).

### 2) Stored quaternion direction is `q_i2b`

`q_i2b` rotates vectors from inertial/world NED into body frame:

\[
\mathbf{v}_b = R_{i2b}\,\mathbf{v}_i
\]

### 3) World frame is NED, right-handed

- `X = North`
- `Y = East`
- `Z = Down`

Altitude therefore appears as negative-up when stored in `pos_z_m`.

### 4) Camera extrinsic `T_BoC`

The library supports composition with camera/body optical transforms (e.g. `T_WC = T_WB * T_BoC`) even though production camera numbers may still be TBD.

### 5) Synthetic ML relative pose pipeline alignment

The API is intentionally minimal and explicit for downstream dataset assembly and ROS2 consumers.

---

## Package structure

```text
ROS2_ws/ml_pkgs/ml_transforms/
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

---

## Build and test

From `ROS2_ws`:

```bash
colcon build --packages-select ml_transforms
colcon test --packages-select ml_transforms
colcon test-result --verbose
```

---

## Public API overview

Namespace: `ml_transforms`

Core types:

- `QuaternionWXYZ {w, x, y, z}`
- `Transform {R, t}`

Core functions:

- `makeTransform(position, q_wxyz)`
- `composeTransforms(T_AB, T_BC)`
- `invertTransform(T)`
- `quatToRotmat(q_wxyz)`
- `rotmatToQuat(R)`
- `toHomogeneousMatrix(T)` / `fromHomogeneousMatrix(H)`
- `applyTransform(T_AB, p_B)`
- `applyRotation(R_AB, v_B)`
- `pixelToRay(...)` / `rayToPixel(...)` (WP2 stubs)

---

## Usage examples

### Build `T_WB` from NED position + scalar-first quaternion

```cpp
#include "ml_transforms/transform_math.hpp"

Eigen::Vector3d p_WB_ned(120.0, 15.0, -45.0);  // Z-down convention
ml_transforms::QuaternionWXYZ q_i2b{0.9238795, 0.0, 0.0, 0.3826834};
ml_transforms::Transform T_WB = ml_transforms::makeTransform(p_WB_ned, q_i2b);
```

### Compose camera pose: `T_WC = T_WB * T_BoC`

```cpp
ml_transforms::Transform T_BoC = ml_transforms::makeTransform(
  Eigen::Vector3d(0.12, 0.0, 0.03),
  ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});

ml_transforms::Transform T_WC = ml_transforms::composeTransforms(T_WB, T_BoC);
```

### Compute body-frame relative position from NED

```cpp
Eigen::Vector3d rel_ned = target_pos_ned - ownship_pos_ned;
Eigen::Vector3d rel_body = ml_transforms::applyRotation(T_WB.R, rel_ned);
```

### Convert to homogeneous 4x4

```cpp
Eigen::Matrix4d H_WB = ml_transforms::toHomogeneousMatrix(T_WB);
```

---

## Testing overview

The package includes gtests covering:

- identity, composition, inversion, and homogeneous round-trips
- 90° roll/pitch/yaw golden quaternion-to-rotation tests
- quaternion/rotation round-trip numerical checks
- explicit wrong-order (`xyzw` treated as `wxyz`) detection
- malformed input exception paths
- WP2 stub exceptions for camera geometry
- contract-aligned examples (`q_i2b`, NED sign conventions)

Golden checks are validated to `< 1e-6`.

---

## WP2 camera note

`pixelToRay` and `rayToPixel` are intentionally unimplemented and throw:

- `pixelToRay: to be completed in WP2`
- `rayToPixel: to be completed in WP2`

This preserves API stability without introducing premature camera-model assumptions.

---

## Common pitfalls

1. **Quaternion order confusion (`xyzw` vs `wxyz`)**
   - Always reorder when crossing ROS message boundaries.

2. **Direction confusion (`q_i2b`)**
   - This package expects inertial→body rotation semantics.

3. **NED sign errors**
   - NED `z` increases downward; altitude-up is typically negative in stored `pos_z_m`.

4. **Silent normalization assumptions**
   - Quaternion APIs normalize internally and reject near-zero norms.

5. **Assuming camera projection exists already**
   - Pixel/ray helpers are stubs until WP2.
