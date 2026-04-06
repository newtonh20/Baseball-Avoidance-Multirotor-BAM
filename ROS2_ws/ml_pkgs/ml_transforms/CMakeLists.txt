cmake_minimum_required(VERSION 3.8)
project(ml_transforms)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME}
  src/camera_geometry.cpp
  src/quaternion.cpp
  src/rotation_matrix.cpp
  src/transform.cpp
  src/transform_math.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  PUBLIC
    Eigen3
)

target_link_libraries(${PROJECT_NAME}
  PUBLIC
    Eigen3::Eigen
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_include_directories(include)
ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(eigen3_cmake_module Eigen3)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_quaternion test/test_quaternion.cpp)
  if(TARGET test_quaternion)
    target_link_libraries(test_quaternion ${PROJECT_NAME})
  endif()

  ament_add_gtest(test_transform_math test/test_transform_math.cpp)
  if(TARGET test_transform_math)
    target_link_libraries(test_transform_math ${PROJECT_NAME})
  endif()

  ament_add_gtest(test_camera_geometry test/test_camera_geometry.cpp)
  if(TARGET test_camera_geometry)
    target_link_libraries(test_camera_geometry ${PROJECT_NAME})
  endif()
endif()

ament_package()
