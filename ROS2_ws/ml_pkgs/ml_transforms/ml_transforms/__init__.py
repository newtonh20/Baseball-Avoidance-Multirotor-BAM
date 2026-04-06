"""ml_transforms – public API.

ROS2 Jazzy ament_python package providing minimal SE3 transform utilities
for the Baseball Avoidance Multirotor (BAM) project.

All functions use only numpy; no ROS message types are imported here so
the library can be used in both ROS2 nodes and plain Python scripts / CI.
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
    'make_transform',
    'compose_transforms',
    'invert_transform',
    'quat_to_rotmat',
    'rotmat_to_quat',
    'pixel_to_ray',
    'ray_to_pixel',
]
