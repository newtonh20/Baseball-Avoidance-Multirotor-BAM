"""
bam_transform_math.transform_math
==================================
Minimal, dependency-light Python library for rigid-body (SE3) transform math
used by the Baseball Avoidance Multirotor (BAM) project.

Conventions
-----------
* Quaternions are stored as ``[w, x, y, z]`` (scalar-first / Hamilton convention).
* Homogeneous transform matrices are 4×4 float64 numpy arrays in the form::

      T = | R  t |
          | 0  1 |

  where ``R`` is a 3×3 rotation matrix and ``t`` is a 3-vector translation.
* Only numpy is used; no scipy, transforms3d, or other external packages.
"""

import numpy as np


def make_transform(position, quaternion):
    """Build a 4×4 SE3 homogeneous transform matrix.

    Parameters
    ----------
    position : array_like, shape (3,)
        Translation vector [x, y, z].
    quaternion : array_like, shape (4,)
        Rotation quaternion ``[w, x, y, z]`` (scalar first).
        The quaternion is normalized before use.

    Returns
    -------
    T : np.ndarray, shape (4, 4), dtype float64
        Homogeneous transform matrix.
    """
    position = np.asarray(position, dtype=np.float64)
    quaternion = np.asarray(quaternion, dtype=np.float64)

    if position.shape != (3,):
        raise ValueError(f"position must be shape (3,), got {position.shape}")
    if quaternion.shape != (4,):
        raise ValueError(f"quaternion must be shape (4,), got {quaternion.shape}")

    R = quat_to_rotmat(quaternion)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def compose_transforms(T_AB, T_BC):
    """Compose two SE3 transforms: T_AC = T_AB @ T_BC.

    Parameters
    ----------
    T_AB : array_like, shape (4, 4)
        Transform from frame A to frame B.
    T_BC : array_like, shape (4, 4)
        Transform from frame B to frame C.

    Returns
    -------
    T_AC : np.ndarray, shape (4, 4), dtype float64
        Composed transform from frame A to frame C.
    """
    T_AB = np.asarray(T_AB, dtype=np.float64)
    T_BC = np.asarray(T_BC, dtype=np.float64)

    if T_AB.shape != (4, 4):
        raise ValueError(f"T_AB must be shape (4, 4), got {T_AB.shape}")
    if T_BC.shape != (4, 4):
        raise ValueError(f"T_BC must be shape (4, 4), got {T_BC.shape}")

    return T_AB @ T_BC


def invert_transform(T):
    """Analytically invert an SE3 homogeneous transform.

    For a transform ``T = | R  t |``, the inverse is ``T^-1 = | R^T  -R^T t |``.
                           ``    | 0  1 |``                    ``              | 0     1    |``

    This is numerically superior to ``np.linalg.inv`` for valid rotation matrices
    because it exploits the orthogonality of R (R^{-1} = R^T).

    Parameters
    ----------
    T : array_like, shape (4, 4)
        SE3 homogeneous transform matrix.

    Returns
    -------
    T_inv : np.ndarray, shape (4, 4), dtype float64
        Inverse transform.
    """
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"T must be shape (4, 4), got {T.shape}")

    R = T[:3, :3]
    t = T[:3, 3]

    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def quat_to_rotmat(q):
    """Convert a quaternion to a 3×3 rotation matrix.

    Uses the standard formula derived from the double-cover homomorphism
    SU(2) → SO(3)::

        R = (w²+x²-y²-z²)  2(xy-wz)       2(xz+wy)
            2(xy+wz)        (w²-x²+y²-z²)  2(yz-wx)
            2(xz-wy)        2(yz+wx)        (w²-x²-y²+z²)

    Parameters
    ----------
    q : array_like, shape (4,)
        Quaternion ``[w, x, y, z]``. Normalized before use.

    Returns
    -------
    R : np.ndarray, shape (3, 3), dtype float64
        Corresponding rotation matrix.
    """
    q = np.asarray(q, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError(f"q must be shape (4,), got {q.shape}")

    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError("Quaternion has zero norm.")
    q = q / norm

    w, x, y, z = q

    R = np.array([
        [w*w + x*x - y*y - z*z,  2*(x*y - w*z),           2*(x*z + w*y)],
        [2*(x*y + w*z),           w*w - x*x + y*y - z*z,  2*(y*z - w*x)],
        [2*(x*z - w*y),           2*(y*z + w*x),           w*w - x*x - y*y + z*z],
    ], dtype=np.float64)

    return R


def rotmat_to_quat(R):
    """Convert a 3×3 rotation matrix to a quaternion [w, x, y, z].

    Uses Shepperd's method with stable branch selection to avoid numerical
    issues near degenerate configurations (trace near −1).

    Parameters
    ----------
    R : array_like, shape (3, 3)
        Valid rotation matrix (orthogonal, det = +1).

    Returns
    -------
    q : np.ndarray, shape (4,), dtype float64
        Unit quaternion ``[w, x, y, z]``.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must be shape (3, 3), got {R.shape}")

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
        s = 2.0 * np.sqrt(1.0 - R[0, 0] + R[1, 1] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 - R[0, 0] - R[1, 1] + R[2, 2])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= np.linalg.norm(q)
    return q


def pixel_to_ray(uv, K):
    """Convert pixel coordinates to a unit ray in the camera frame.

    .. note::
        This function is a stub and will be implemented in Work Package 2.

    Parameters
    ----------
    uv : array_like, shape (2,)
        Pixel coordinates ``[u, v]``.
    K : array_like, shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    ray : np.ndarray, shape (3,)
        Unit ray direction in the camera frame.

    Raises
    ------
    NotImplementedError
        Always – implementation deferred to WP2.
    """
    raise NotImplementedError("pixel_to_ray: to be completed in WP2")


def ray_to_pixel(ray, K):
    """Project a unit ray in the camera frame to pixel coordinates.

    .. note::
        This function is a stub and will be implemented in Work Package 2.

    Parameters
    ----------
    ray : array_like, shape (3,)
        Unit ray direction in the camera frame.
    K : array_like, shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    uv : np.ndarray, shape (2,)
        Pixel coordinates ``[u, v]``.

    Raises
    ------
    NotImplementedError
        Always – implementation deferred to WP2.
    """
    raise NotImplementedError("ray_to_pixel: to be completed in WP2")
