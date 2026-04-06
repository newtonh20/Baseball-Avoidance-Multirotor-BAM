#pragma once

#include <Eigen/Core>

namespace ml_transforms {

/**
 * Lightweight rigid transform T_AB mapping vectors from frame B into frame A:
 * p_A = R_AB * p_B + t_AB.
 */
struct Transform {
  Eigen::Matrix3d R{Eigen::Matrix3d::Identity()};
  Eigen::Vector3d t{Eigen::Vector3d::Zero()};
};

/**
 * Convert Transform to homogeneous 4x4 matrix.
 */
Eigen::Matrix4d toHomogeneousMatrix(const Transform& T);

/**
 * Construct Transform from homogeneous 4x4 matrix.
 *
 * Throws std::invalid_argument when the matrix does not represent SE(3).
 */
Transform fromHomogeneousMatrix(const Eigen::Matrix4d& H);

/**
 * Apply transform T_AB to point p_B.
 */
Eigen::Vector3d applyTransform(const Transform& T_AB, const Eigen::Vector3d& p_B);

/**
 * Apply rotation R_AB to vector v_B.
 */
Eigen::Vector3d applyRotation(const Eigen::Matrix3d& R_AB,
                              const Eigen::Vector3d& v_B);

}  // namespace ml_transforms
