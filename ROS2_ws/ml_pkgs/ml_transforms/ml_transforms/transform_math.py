"""
transform_math.py
=================
Minimal SE3 homogeneous-transform utilities.

Conventions
-----------
* Quaternion order: **[w, x, y, z]** (scalar first, ROS2 / Hamilton convention).
  This matches geometry_msgs/msg/Quaternion when you extract fields as
  (w, x, y, z).  Do NOT pass [x, y, z, w] – tests explicitly guard against
  this mistake.
* Homogeneous transform T is a 4×4 float64 numpy array::

      T = | R  t |
          | 0  1 |

  where R ∈ SO(3) (3×3) and t ∈ ℝ³ (column).
* All functions accept plain numpy arrays; no ROS message types required.

Dependencies
------------
numpy only.  No scipy, no transforms3d, no tf2.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Build a transform
# ---------------------------------------------------------------------------

def make_transform(
    position: np.ndarray | list,
    quaternion: np.ndarray | list,
) -> np.ndarray:
    """Build a 4×4 SE3 homogeneous transform from position + quaternion.

    Parameters
    ----------
    position:
        3-vector [x, y, z] in metres.
    quaternion:
        4-vector **[w, x, y, z]** (scalar first).  Will be normalised
        internally; raises ValueError if near-zero.

    Returns
    -------
    T : np.ndarray, shape (4, 4), dtype float64
        Homogeneous transform  T = [[R, t], [0, 1]].
    """
    p = np.asarray(position, dtype=np.float64).ravel()
    q = np.asarray(quaternion, dtype=np.float64).ravel()

    if p.shape != (3,):
        raise ValueError(f'position must be a 3-vector, got shape {p.shape}')
    if q.shape != (4,):
        raise ValueError(f'quaternion must be a 4-vector [w,x,y,z], got shape {q.shape}')

    R = quat_to_rotmat(q)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


# ---------------------------------------------------------------------------
# Compose / invert
# ---------------------------------------------------------------------------

def compose_transforms(
    T_AB: np.ndarray,
    T_BC: np.ndarray,
) -> np.ndarray:
    """Compose two SE3 transforms:  T_AC = T_AB · T_BC.

    Parameters
    ----------
    T_AB, T_BC : np.ndarray, shape (4, 4)
        Homogeneous transforms.

    Returns
    -------
    T_AC : np.ndarray, shape (4, 4), dtype float64
    """
    T_AB = np.asarray(T_AB, dtype=np.float64)
    T_BC = np.asarray(T_BC, dtype=np.float64)
    if T_AB.shape != (4, 4):
        raise ValueError(f'T_AB must be (4,4), got {T_AB.shape}')
    if T_BC.shape != (4, 4):
        raise ValueError(f'T_BC must be (4,4), got {T_BC.shape}')
    return T_AB @ T_BC


def invert_transform(T: np.ndarray) -> np.ndarray:
    """Analytically invert an SE3 homogeneous transform.

    Exploits the block structure::

        T_inv = | R^T   -R^T t |
                |  0       1   |

    This is exact for valid SE3 matrices and numerically superior to
    ``np.linalg.inv`` because it avoids a full LU factorisation.

    Parameters
    ----------
    T : np.ndarray, shape (4, 4)
        A valid SE3 homogeneous transform.

    Returns
    -------
    T_inv : np.ndarray, shape (4, 4), dtype float64
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f'T must be (4,4), got {T.shape}')
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv


# ---------------------------------------------------------------------------
# Quaternion ↔ rotation matrix
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: np.ndarray | list) -> np.ndarray:
    """Convert a quaternion **[w, x, y, z]** to a 3×3 rotation matrix.

    Uses the standard closed-form formula::

        R = (w²-‖v‖²)I + 2vvᵀ + 2w [v]×

    where [v]× is the skew-symmetric cross-product matrix of v = (x,y,z).

    Parameters
    ----------
    q : array-like, shape (4,)
        Quaternion **[w, x, y, z]** (scalar first).  Need not be
        unit-length; normalisation is applied internally.

    Returns
    -------
    R : np.ndarray, shape (3, 3), dtype float64

    Raises
    ------
    ValueError
        If ``q`` is the zero vector (degenerate).
    """
    q = np.asarray(q, dtype=np.float64).ravel()
    if q.shape != (4,):
        raise ValueError(f'quaternion must be shape (4,), got {q.shape}')
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError('Quaternion has near-zero norm; cannot normalise.')
    w, x, y, z = q / norm

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3×3 rotation matrix to a quaternion **[w, x, y, z]**.

    Uses Shepperd's method with stable branch selection to avoid
    division by near-zero values when certain diagonal elements are small.

    Parameters
    ----------
    R : np.ndarray, shape (3, 3)
        A valid SO(3) rotation matrix.

    Returns
    -------
    q : np.ndarray, shape (4,), dtype float64
        Unit quaternion **[w, x, y, z]**, w ≥ 0 (canonical form).
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f'R must be (3,3), got {R.shape}')

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    # Canonical form: w >= 0
    if q[0] < 0.0:
        q = -q
    return q


# ---------------------------------------------------------------------------
# Pixel ↔ ray stubs  (WP2)
# ---------------------------------------------------------------------------

def pixel_to_ray(
    uv: np.ndarray | list,
    K: np.ndarray,
) -> np.ndarray:
    """[STUB – WP2] Back-project a pixel to a unit ray in the camera frame.

    Parameters
    ----------
    uv : array-like, shape (2,)
        Pixel coordinates (u, v) in pixels.
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    ray : np.ndarray, shape (3,)
        Unit direction vector in the camera frame.

    Raises
    ------
    NotImplementedError
        Always – implementation deferred to WP2.
    """
    raise NotImplementedError(
        'pixel_to_ray: back-projection not yet implemented – to be completed in WP2.'
    )


def ray_to_pixel(
    ray: np.ndarray | list,
    K: np.ndarray,
) -> np.ndarray:
    """[STUB – WP2] Project a unit ray onto the image plane.

    Parameters
    ----------
    ray : array-like, shape (3,)
        Unit direction vector in the camera frame.
    K : np.ndarray, shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    uv : np.ndarray, shape (2,)
        Pixel coordinates (u, v).

    Raises
    ------
    NotImplementedError
        Always – implementation deferred to WP2.
    """
    raise NotImplementedError(
        'ray_to_pixel: projection not yet implemented – to be completed in WP2.'
    )
