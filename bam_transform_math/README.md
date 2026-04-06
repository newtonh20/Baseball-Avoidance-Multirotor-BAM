# `bam_transform_math` — Rigid-Body Transform Library

**Part of:** FormationFlight / Baseball Avoidance Multirotor (BAM)  
**Work Package:** WP1 (transform utilities) — WP2 stubs included  
**Dependencies:** `numpy` only  
**Python:** ≥ 3.9  

---

## Table of Contents

1. [Purpose](#1-purpose)
2. [Conventions — Read First](#2-conventions--read-first)
   - 2.1 [Quaternion Order](#21-quaternion-order)
   - 2.2 [Transform Direction](#22-transform-direction)
   - 2.3 [Coordinate Frame (NED)](#23-coordinate-frame-ned)
   - 2.4 [Homogeneous Matrix Layout](#24-homogeneous-matrix-layout)
3. [API Reference](#3-api-reference)
4. [Usage Examples](#4-usage-examples)
   - 4.1 [Building a Transform from a Pose Row](#41-building-a-transform-from-a-pose-row)
   - 4.2 [Composing T\_WC from T\_WB and T\_BoC](#42-composing-t_wc-from-t_wb-and-t_boc)
   - 4.3 [Relative Position in Lead Body Frame](#43-relative-position-in-lead-body-frame)
   - 4.4 [Relative Quaternion Between Two Aircraft](#44-relative-quaternion-between-two-aircraft)
   - 4.5 [Loading from a CSV Pose File](#45-loading-from-a-csv-pose-file)
5. [⚠️ Common Pitfalls](#5-️-common-pitfalls)
6. [Running the Tests](#6-running-the-tests)
7. [WP2 Stubs](#7-wp2-stubs)
8. [Design Notes](#8-design-notes)

---

## 1. Purpose

This library provides a small, dependency-light set of SE(3) transform utilities
for the BAM ML relative pose estimation pipeline. It is the authoritative
implementation of the coordinate frame, quaternion, and homogeneous transform
conventions defined in [`/docs/data_contract.md`](../docs/data_contract.md).

Core capabilities:

- Build 4×4 SE(3) homogeneous transforms `T_WB` from position + quaternion
- Compose transforms: `T_WC = T_WB @ T_BoC`
- Analytically invert transforms using `R⁻¹ = Rᵀ` (numerically superior to `np.linalg.inv`)
- Convert quaternion ↔ rotation matrix using Shepperd's method
- Stub helpers for pixel ↔ ray (camera projection, to be completed in WP2)

The library is intentionally minimal. It contains **no** import of `scipy`,
`transforms3d`, `pyquaternion`, ROS 2 message types, or any other package beyond
`numpy`. This keeps it usable in any environment including embedded CI runners,
MATLAB-adjacent workflows, and restricted Python installs.

---

## 2. Conventions — Read First

> ⚠️ **The single most dangerous mistake in this pipeline is silently using
> the wrong quaternion component order.** Read this section before writing
> any code that touches quaternions.

### 2.1 Quaternion Order

All functions in this library use **scalar-first `[w, x, y, z]`** (Hamilton
convention), matching the data contract (Section 4.1) and the MATLAB generator
`Gen_ML_Rel_Data.m`.

```
[w, x, y, z]   ←  THIS library, the data contract, MATLAB output
[x, y, z, w]   ←  scipy.spatial.transform, PyTorch3D, ROS 2 / TF2
```

**When passing quaternions from this library to scipy or ROS 2**, reorder:

```python
import numpy as np
from scipy.spatial.transform import Rotation

q_wxyz = np.array([w, x, y, z])       # from this library or a pose CSV
q_xyzw = np.roll(q_wxyz, shift=-1)    # [x, y, z, w]  →  scipy / ROS 2
R_scipy = Rotation.from_quat(q_xyzw)
```

**When receiving a quaternion from scipy or ROS 2**, reverse the reorder:

```python
q_xyzw = rotation.as_quat()           # scipy output
q_wxyz = np.roll(q_xyzw, shift=1)     # [w, x, y, z]  →  this library
```

The unit tests include a "quaternion order trap" test that verifies `[x, y, z, w]`
input produces a **detectably different** rotation matrix from `[w, x, y, z]` —
ensuring silent convention errors are caught during CI.

### 2.2 Transform Direction

The pose CSV files and `.mat` arrays store the quaternion as `q_i2b` — it rotates
a vector **from the World (NED inertial) frame into the aircraft body frame**
(data contract Section 4.2):

```
q_i2b :  v_body    = R(q_i2b) · v_world
         v_world   = R(q_i2b)ᵀ · v_body   =  R(q_b2i) · v_body
```

The `make_transform` function builds `T_WB` (a transform *from* body *to* world,
i.e., the transform that maps body-frame vectors into world-frame coordinates)
by using the rotation matrix derived from `q_i2b` directly in the upper-left 3×3
block, and the world-frame position as the translation column:

```
T_WB = | R(q_i2b)   p_W |
       | 0  0  0     1  |
```

To rotate a vector **from world to body** using a transform, use `invert_transform`:

```python
T_WB   = make_transform(position, q_i2b)
T_BW   = invert_transform(T_WB)         # maps world → body

v_world_h = np.array([x, y, z, 1.0])
v_body_h  = T_BW @ v_world_h            # result in body frame
```

### 2.3 Coordinate Frame (NED)

The World frame is **North-East-Down (NED)**, right-handed (data contract
Section 3.2):

| Axis | Direction | Notes |
|------|-----------|-------|
| `X`  | North     | Forward |
| `Y`  | East      | Right |
| `Z`  | Down      | **Altitude is stored negative-up** |

> `pos_z_m = -7620.0` means the aircraft is **7620 m above** the datum.  
> Consumers must negate `pos_z_m` to get altitude above datum.

Body frames (`B_lead`, `B_trail`) share the same axis orientation relative to
each aircraft's nose/wing/belly:

| Axis | Body direction |
|------|---------------|
| `X`  | Nose (forward) |
| `Y`  | Right wing |
| `Z`  | Belly (down) |

### 2.4 Homogeneous Matrix Layout

All transforms in this library are 4×4 `float64` NumPy arrays:

```
T = | R₃ₓ₃   t₃ₓ₁ |
    | 0  0  0   1  |

R  — 3×3 rotation matrix (orthogonal, det = +1)
t  — 3-vector translation in the *source* frame's coordinates
```

To apply a transform to a point `p`:

```python
p_h = np.array([p[0], p[1], p[2], 1.0])   # homogeneous
q_h = T @ p_h
p_out = q_h[:3]                             # de-homogenize
```

To apply only the rotational part:

```python
R = T[:3, :3]
v_rotated = R @ v
```

---

## 3. API Reference

All public symbols are importable from the top-level package:

```python
from bam_transform_math import (
    make_transform,
    compose_transforms,
    invert_transform,
    quat_to_rotmat,
    rotmat_to_quat,
    pixel_to_ray,    # WP2 stub — raises NotImplementedError
    ray_to_pixel,    # WP2 stub — raises NotImplementedError
)
```

### `make_transform(position, quaternion) → np.ndarray (4, 4)`

Build a 4×4 SE(3) homogeneous transform from a position vector and quaternion.

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `position` | array-like | `(3,)` | Translation `[x, y, z]` in meters |
| `quaternion` | array-like | `(4,)` | Rotation `[w, x, y, z]`, **scalar-first** |

The quaternion is normalized before use. Raises `ValueError` on wrong shape.

### `compose_transforms(T_AB, T_BC) → np.ndarray (4, 4)`

Compose two SE(3) transforms: `T_AC = T_AB @ T_BC`.

The naming convention follows the subscript chaining rule: the inner
frame label (`B`) must match between the two arguments.

### `invert_transform(T) → np.ndarray (4, 4)`

Analytically invert an SE(3) transform using `R⁻¹ = Rᵀ`:

```
T⁻¹ = | Rᵀ   −Rᵀt |
      | 0       1  |
```

This exploits rotation matrix orthogonality and is more numerically stable
than `np.linalg.inv` for valid SE(3) matrices.

### `quat_to_rotmat(q) → np.ndarray (3, 3)`

Convert a `[w, x, y, z]` unit quaternion to a 3×3 rotation matrix using the
standard SU(2) → SO(3) homomorphism formula. The quaternion is normalized
before use.

### `rotmat_to_quat(R) → np.ndarray (4,)`

Convert a 3×3 rotation matrix to a `[w, x, y, z]` unit quaternion using
**Shepperd's method** with stable branch selection. This avoids the
numerical instabilities near `trace(R) ≈ −1` that affect naive implementations.
Returns the quaternion with `w ≥ 0` (canonical form).

### `pixel_to_ray(uv, K)` *(WP2 stub)*

Convert pixel coordinates `[u, v]` to a unit ray in the camera frame using
intrinsic matrix `K`. **Always raises `NotImplementedError`** until WP2.

### `ray_to_pixel(ray, K)` *(WP2 stub)*

Project a unit ray in the camera frame to pixel coordinates `[u, v]`. **Always
raises `NotImplementedError`** until WP2.

---

## 4. Usage Examples

### 4.1 Building a Transform from a Pose Row

The pose CSV files (`lead_poses.csv`, `trail_poses.csv`) have the column layout
defined in the data contract Section 8.2:

```
time_s, pos_x_m, pos_y_m, pos_z_m, q_w, q_x, q_y, q_z
```

```python
import numpy as np
from bam_transform_math import make_transform

lead_poses = np.loadtxt("lead_poses.csv", delimiter=",", skiprows=1)

# Extract frame k (k=0 is the nominal/mean pose)
k = 0
row = lead_poses[k]
position   = row[1:4]   # [pos_x_m, pos_y_m, pos_z_m]  (NED, meters)
quaternion = row[4:8]   # [q_w, q_x, q_y, q_z]  (scalar-first)

T_WB_lead = make_transform(position, quaternion)
print(T_WB_lead)
```

> **Altitude reminder:** `position[2]` (i.e., `pos_z_m`) is negative for
> above-datum flight. Do **not** negate it before passing to `make_transform` —
> the transform uses the raw NED value. Only negate if you need to display
> altitude to a human.

### 4.2 Composing T\_WC from T\_WB and T\_BoC

The camera world pose is obtained by composing the lead aircraft's world pose
with the fixed camera extrinsic offset `T_BoC` (data contract Section 7):

```
T_WC  =  T_WB_lead  @  T_BoC
```

```python
from bam_transform_math import make_transform, compose_transforms

# Lead aircraft world pose (built from pose CSV row as above)
T_WB_lead = make_transform(position_lead, q_lead)

# Camera extrinsic offset: position and orientation of camera C
# relative to lead body frame B_lead (extracted from Blender model).
# Values are [TBD] pending OI-01 in the data contract.
t_BoC = np.array([x_cam, y_cam, z_cam])     # meters, in B_lead coords
q_BoC = np.array([w, x, y, z])              # rotation B_lead → C, scalar-first

T_BoC = make_transform(t_BoC, q_BoC)

# Camera pose in world frame:
T_WC = compose_transforms(T_WB_lead, T_BoC)
```

### 4.3 Relative Position in Lead Body Frame

The `labels.json` field `relative_position_body` requires the trail-minus-lead
position vector expressed in the lead body frame (data contract Section 11.1):

```python
from bam_transform_math import make_transform, invert_transform
import numpy as np

# Absolute NED positions from pose arrays
pos_lead  = lead_poses[k, 1:4]
pos_trail = trail_poses[k, 1:4]

# Relative position in World (NED) frame
rel_pos_ned = pos_trail - pos_lead           # shape (3,)

# Rotation: world → lead body
q_lead    = lead_poses[k, 4:8]              # [w, x, y, z]
T_WB_lead = make_transform(pos_lead, q_lead)
T_BW_lead = invert_transform(T_WB_lead)     # body ← world

# Apply only the rotation block (no translation needed here)
R_BW = T_BW_lead[:3, :3]
rel_pos_body = R_BW @ rel_pos_ned            # shape (3,)

print("rel_pos_body [x_m, y_m, z_m]:", rel_pos_body)
```

### 4.4 Relative Quaternion Between Two Aircraft

The `labels.json` field `relative_quaternion_wxyz` is defined as the rotation
from the lead body frame to the trail body frame (data contract Section 11.1):

```
q_rel  =  q_i2b_trail  *  conj(q_i2b_lead)
```

```python
from bam_transform_math import quat_to_rotmat, rotmat_to_quat
import numpy as np

q_lead  = lead_poses[k, 4:8]     # [w, x, y, z]
q_trail = trail_poses[k, 4:8]    # [w, x, y, z]

R_lead  = quat_to_rotmat(q_lead)
R_trail = quat_to_rotmat(q_trail)

# R_rel = R_trail @ R_lead^T  (i.e., R_trail @ R_lead_inv)
R_rel   = R_trail @ R_lead.T
q_rel   = rotmat_to_quat(R_rel)   # [w, x, y, z]
```

### 4.5 Loading from a CSV Pose File

```python
import numpy as np
from bam_transform_math import make_transform

def load_poses(csv_path):
    """Return (N, 8) array: [time_s, x, y, z, qw, qx, qy, qz] per row."""
    return np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.float64)

def pose_row_to_transform(row):
    """Convert one pose CSV row to a 4x4 SE(3) transform."""
    position   = row[1:4]   # NED meters
    quaternion = row[4:8]   # [w, x, y, z] scalar-first
    return make_transform(position, quaternion)

lead  = load_poses("run_20260309_143022_0001/lead_poses.csv")
trail = load_poses("run_20260309_143022_0001/trail_poses.csv")

T_WB_lead  = pose_row_to_transform(lead[0])
T_WB_trail = pose_row_to_transform(trail[0])
```

---

## 5. ⚠️ Common Pitfalls

### Pitfall 1 — Wrong quaternion order from scipy/ROS 2

Scipy, ROS 2, and PyTorch3D all use `[x, y, z, w]` (scalar-last). Passing
`[x, y, z, w]` to any function in this library without reordering will produce
a silently wrong rotation. The unit tests assert this produces a measurably
different result (`test_quaternion_order_trap`).

**Fix:** `q_wxyz = np.roll(q_xyzw, shift=1)` before calling any function here.

### Pitfall 2 — Altitude sign (NED Z is negative-up)

The `pos_z_m` column in all pose files is the NED Z coordinate. Because NED Z
points **down**, altitude above datum is stored as a **negative number**. Do
not negate it before passing to `make_transform`; the transform is correct
with the raw NED value.

**Fix:** Only negate when displaying altitude to a human: `altitude_m = -pos_z_m`.

### Pitfall 3 — Computing derivatives between pose columns

Each pose column is a statistically independent random draw (data contract
Section 5.3). There is no kinematic continuity. Do not differentiate adjacent
columns to estimate velocity.

### Pitfall 4 — Using `q_i2b` directly as a body-to-world rotation

`q_i2b` rotates vectors *from* world *into* body. If you want to express a
body-frame vector in world coordinates, you need the conjugate (`R.T` / `invert_transform`).

### Pitfall 5 — T\_BoC values are [TBD]

The camera extrinsic offset `T_BoC` is pending extraction from the aircraft
Blender model (data contract OI-01). Do not hardcode placeholder values in
production code paths. Use `metadata.json:camera_extrinsic` and check for
`null` values before composing `T_WC`.

### Pitfall 6 — Non-standard foot-to-meter conversion

The MATLAB generator uses `ft_2_mtr = 0.3028` (not the SI standard `0.3048`).
All position values in the dataset carry this ~0.066% offset. Do not silently
substitute the SI value (data contract OI-11).

---

## 6. Running the Tests

Tests live in `bam_transform_math/tests/test_transform_math.py` and use only
`pytest` + `numpy`. No other test dependencies.

```bash
# From the repo root
pip install pytest numpy
pytest bam_transform_math/tests/ -v
```

CI runs automatically on push and PR via
[`.github/workflows/test_transform_math.yml`](../.github/workflows/test_transform_math.yml).

### Test Coverage

| Category | Tests |
|----------|-------|
| Identity transform | Build, invert, and compose identity; verify `T @ T⁻¹ = I` |
| Golden rotations | 90° roll, pitch, yaw — analytically known rotation matrices |
| Inversion | Analytical vs `np.linalg.inv` on a general transform; error `< 1e-6` |
| Composition | Associativity; known translation chain; pose concatenation |
| Quat ↔ rotmat roundtrip | `R → q → R` and `q → R → q`; error `< 1e-6` |
| Quaternion order trap | Verifies `[x,y,z,w]` input produces a **different** result from `[w,x,y,z]` |
| WP2 stubs | Both `pixel_to_ray` and `ray_to_pixel` raise `NotImplementedError` |

All golden tests have numerical error `< 1e-6` (acceptance criterion).

---

## 7. WP2 Stubs

Two functions are implemented as stubs pending camera intrinsic data (Work
Package 2):

| Function | Blocked on |
|----------|------------|
| `pixel_to_ray(uv, K)` | Camera intrinsic matrix `K` (data contract OI-04: image resolution and focal length [TBD]) |
| `ray_to_pixel(ray, K)` | Same — also requires `T_BoC` (data contract OI-01) |

Both functions raise `NotImplementedError` with an explanatory message. Tests
for both stubs are already written and passing (they assert `NotImplementedError`
is raised).

When WP2 begins, implement the standard pinhole projection formulas:

```
pixel_to_ray(uv, K):   ray = K⁻¹ · [u, v, 1]ᵀ,  normalize
ray_to_pixel(ray, K):  uv  = (K · ray / ray_z)[0:2]
```

Update the corresponding tests to assert known pixel ↔ ray correspondences
rather than `NotImplementedError`.

---

## 8. Design Notes

**Why no scipy/transforms3d?**  
The pipeline runs across MATLAB-adjacent environments, minimal CI containers,
and eventually embedded hardware. Keeping the single dependency as `numpy`
ensures the library is portable everywhere numpy runs without version conflicts.

**Why Shepperd's method for `rotmat_to_quat`?**  
The naive `arccos((trace(R) - 1) / 2)` formula is numerically unstable near
180° rotations. Shepperd's method selects the largest quaternion component as
the pivot, guaranteeing numerical stability across all orientations.

**Why analytical `invert_transform` instead of `np.linalg.inv`?**  
For a valid SE(3) matrix, `Rᵀ` is the exact inverse of `R` up to floating-point
rounding. `np.linalg.inv` performs a general LU decomposition, which is both
slower and accumulates more rounding error. The analytical form is `O(n)` matrix
transpose + one matrix-vector product.

**Float64 everywhere**  
All internal computation uses `float64`. The golden tests assert numerical error
`< 1e-6`, which is comfortably achievable in float64 but not in float32 for
composed and inverted transforms.
