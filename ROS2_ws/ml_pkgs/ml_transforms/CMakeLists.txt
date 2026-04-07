cmake_minimum_required(VERSION 3.8)
project(ml_transforms)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

add_library(${PROJECT_NAME}
  src/transform_math.cpp
  src/quaternion.cpp
  src/rotation_matrix.cpp
  src/transform.cpp
  src/camera_geometry.cpp
)

target_include_directories(${PROJECT_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} PUBLIC Eigen3::Eigen)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(Eigen3 eigen3_cmake_module)
ament_export_include_directories(include)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_quaternion test/test_quaternion.cpp)
  target_link_libraries(test_quaternion ${PROJECT_NAME})

  ament_add_gtest(test_transform_math test/test_transform_math.cpp)
  target_link_libraries(test_transform_math ${PROJECT_NAME})

  ament_add_gtest(test_camera_geometry test/test_camera_geometry.cpp)
  target_link_libraries(test_camera_geometry ${PROJECT_NAME})
endif()

ament_package()
