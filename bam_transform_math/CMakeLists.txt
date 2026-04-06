cmake_minimum_required(VERSION 3.16)
project(bam_transform_math VERSION 0.1.0 LANGUAGES CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler warnings ─────────────────────────────────────────────────────────
add_compile_options(
  -Wall
  -Wextra
  -Wpedantic
  -Wshadow
  -Wnon-virtual-dtor
  -Wold-style-cast
  -Woverloaded-virtual
  $<$<CONFIG:Release>:-O3>
  $<$<CONFIG:Debug>:-g -O0>
)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# ── Header-only interface library ─────────────────────────────────────────────
# The library is intentionally header-only: all implementation lives in .hpp
# files inside include/bam_transform_math/.  This keeps the package
# dependency-light and usable as a plain CMake INTERFACE target by downstream
# ROS 2 packages without needing to link a shared object.
add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(${PROJECT_NAME} INTERFACE Eigen3::Eigen)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_20)

# ── Install ───────────────────────────────────────────────────────────────────
install(
  DIRECTORY include/
  DESTINATION include
)

install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  INCLUDES DESTINATION include
)

ament_export_targets(${PROJECT_NAME}Targets HAS_LIBRARY_TARGET)
ament_export_dependencies(eigen3_cmake_module Eigen3)

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)

  # Suppress copyright / cpplint lint (keep CI clean without distraction)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_transform_math
    test/test_transform_math.cpp
  )
  target_include_directories(test_transform_math PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  )
  target_link_libraries(test_transform_math
    ${PROJECT_NAME}::${PROJECT_NAME}
  )
  target_compile_features(test_transform_math PRIVATE cxx_std_20)
endif()

# ── ament_package ─────────────────────────────────────────────────────────────
ament_package()
