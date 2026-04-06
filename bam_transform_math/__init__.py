"""
bam_transform_math
==================
Minimal numpy-only library for SE3 rigid-body transform mathematics,
supporting the Baseball Avoidance Multirotor (BAM) project.

Exported functions
------------------
make_transform      -- Build 4×4 homogeneous transform from position + quaternion.
compose_transforms  -- Compose two SE3 transforms (matrix multiply).
invert_transform    -- Analytically invert an SE3 transform.
quat_to_rotmat      -- Convert quaternion [w,x,y,z] → 3×3 rotation matrix.
rotmat_to_quat      -- Convert 3×3 rotation matrix → quaternion [w,x,y,z].
pixel_to_ray        -- (stub, WP2) Pixel coordinates → unit ray in camera frame.
ray_to_pixel        -- (stub, WP2) Unit ray in camera frame → pixel coordinates.
"""

from .transform_math import (
    make_transform,
    compose_transforms,
    invert_transform,
    quat_to_rotmat,
    rotmat_to_quat,
    pixel_to_ray,
    ray_to_pixel,
)

__all__ = [
    "make_transform",
    "compose_transforms",
    "invert_transform",
    "quat_to_rotmat",
    "rotmat_to_quat",
    "pixel_to_ray",
    "ray_to_pixel",
]
