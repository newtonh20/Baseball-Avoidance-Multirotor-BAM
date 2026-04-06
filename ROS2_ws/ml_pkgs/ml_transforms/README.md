# ml_transforms

`ml_transforms` is a ROS 2 Jazzy C++ library package (`ament_cmake`) that provides
minimal, dependency-light rigid-body transform and quaternion utilities for the
Baseball Avoidance Multirotor / FormationFlight synthetic ML pipeline.

It is designed as a reusable library (not a node package), so other ROS 2
packages and standalone C++ tools can link against it.

---

## 1) Purpose

This package centralizes convention-sensitive math used by dataset generation and
relative pose labeling workflows, including:

- Constructing `T_WB` from position + quaternion
- SE(3) composition and inversion
- Quaternion <-> rotation matrix conversion
- 4x4 homogeneous conversion
- WP2 placeholder camera projection helpers (`pixelToRay`, `rayToPixel`)

The goal is predictable, testable math with explicit frame and quaternion
conventions.

---

## 2) Project data-contract conventions (critical)

These conventions are intentionally strict in both code and tests:

1. **Quaternion storage order is scalar-first `[w, x, y, z]`.**
   - This is **not** ROS message storage order.
2. **Stored quaternion direction is `q_i2b`.**
   - Rotates vectors from inertial world (NED) into body frame.
3. **World frame is right-handed NED**:
   - `X = North`, `Y = East`, `Z = Down`
4. **Altitude sign follows NED**:
   - Higher altitude is more negative `pos_z_m`.
5. **Camera extrinsic `T_BoC` is conceptually supported now**:
   - Numeric calibration values can be filled in later.

### ROS bridge warning

`geometry_msgs::msg::Quaternion` fields are `x, y, z, w`, while this package
requires `[w, x, y, z]`. Reorder explicitly when bridging to/from ROS messages.

---

## 3) Package structure

```text
ml_transforms/
├── CMakeLists.txt
├── package.xml
├── README.md
├── include/ml_transforms/
│   ├── camera_geometry.hpp
│   ├── quaternion.hpp
│   ├── rotation_matrix.hpp
│   ├── transform.hpp
│   └── transform_math.hpp
├── src/
│   ├── camera_geometry.cpp
│   ├── quaternion.cpp
│   ├── rotation_matrix.cpp
│   ├── transform.cpp
│   └── transform_math.cpp
├── test/
│   ├── test_camera_geometry.cpp
│   ├── test_quaternion.cpp
│   └── test_transform_math.cpp
└── .github/workflows/test_ml_transforms.yml
```

---

## 4) Build and test

From the `ROS2_ws` root:

```bash
colcon build --packages-select ml_transforms
colcon test --packages-select ml_transforms
colcon test-result --verbose
```

---

## 5) API overview

Namespace: `ml_transforms`

### Core types

- `QuaternionWXYZ {w,x,y,z}` scalar-first quaternion
- `Transform {R, t}` with mapping `p_A = R_AB * p_B + t_AB`

### Core functions

- `makeTransform(position, q_wxyz)`
- `composeTransforms(T_AB, T_BC)`
- `invertTransform(T)`
- `quatToRotmat(q_wxyz)`
- `rotmatToQuat(R)`
- `toHomogeneousMatrix(T)`
- `fromHomogeneousMatrix(H)`
- `applyTransform(T_AB, p_B)`
- `applyRotation(R_AB, v_B)`
- `normalizeQuaternion(q)`
- `conjugateQuaternion(q)`
- `pixelToRay(uv, K)` (WP2 stub)
- `rayToPixel(ray, K)` (WP2 stub)

---

## 6) Usage examples

### Build `T_WB`

```cpp
#include "ml_transforms/transform_math.hpp"

Eigen::Vector3d p_WB(100.0, 5.0, -120.0);          // NED: z is Down
ml_transforms::QuaternionWXYZ q_i2b{1.0, 0.0, 0.0, 0.0};
auto T_WB = ml_transforms::makeTransform(p_WB, q_i2b);
```

### Compose camera pose `T_WC = T_WB * T_BoC`

```cpp
auto T_BoC = ml_transforms::makeTransform(Eigen::Vector3d(0.2, 0.0, 0.1),
                                          ml_transforms::QuaternionWXYZ{1.0, 0.0, 0.0, 0.0});
auto T_WC = ml_transforms::composeTransforms(T_WB, T_BoC);
```

### Relative position in body frame

```cpp
Eigen::Vector3d rel_ned = p_target_ned - p_ego_ned;
Eigen::Matrix3d R_i2b = T_WB.R;
Eigen::Vector3d relative_position_body = R_i2b * rel_ned;
```

### Convert to homogeneous matrix

```cpp
Eigen::Matrix4d H_WB = ml_transforms::toHomogeneousMatrix(T_WB);
auto T_WB_again = ml_transforms::fromHomogeneousMatrix(H_WB);
```

---

## 7) Testing overview

The unit tests cover:

- Identity/inverse consistency
- Multiple golden 90-degree rotations (roll, pitch, yaw)
- Composition associativity and known composition cases
- Homogeneous matrix conversions and malformed input errors
- Quaternion roundtrips with sign ambiguity handling
- Explicit wrong-order quaternion misuse detection (`xyzw` fed as `wxyz`)
- Contract-aligned examples (`q_i2b`, NED sign handling)
- WP2 camera helper stubs throwing `std::logic_error`

Numerical tolerances are explicit and tight (typically `1e-6` or tighter).

---

## 8) WP2 note: camera geometry stubs

`pixelToRay(...)` and `rayToPixel(...)` intentionally throw:

- `pixelToRay: to be completed in WP2`
- `rayToPixel: to be completed in WP2`

This keeps the API stable while preventing accidental partial projection logic.

---

## 9) Common pitfalls

1. **Mixing quaternion order**
   - If you pass ROS message fields directly (`x,y,z,w`) into this library,
     your rotations will be wrong.
2. **Frame-direction confusion**
   - `q_i2b` maps inertial/world vectors into body frame.
3. **NED sign misunderstanding**
   - Positive down means altitude-up is negative `z`.
4. **Using generic matrix inverse for SE(3)**
   - Use `invertTransform` for analytic rigid transform inversion.

---

## 10) Intended use in BAM ML workflows

This library supports synthetic lead/trail aircraft relative pose pipelines where
labels such as `relative_position_body` and `relative_quaternion_wxyz` are
produced and consumed consistently across dataset assembly and ROS 2 tooling.
