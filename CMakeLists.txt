cmake_minimum_required(VERSION 3.22)

cmake_policy(SET CMP0048 NEW)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    set(IS_MAIN_FILE ON)
else()
    set(IS_MAIN_FILE OFF)
endif()

option(EIGEN_EXPM_MULTIPLY_BUILD_TESTS "Build tests" ${IS_MAIN_FILE})

project(eigen_expm_multiply LANGUAGES CXX VERSION 0.0.1)

add_subdirectory(libs)

add_library(eigen_expm_multiply INTERFACE)
add_library(jakkes::eigen_expm_multiply ALIAS eigen_expm_multiply)
target_include_directories(eigen_expm_multiply INTERFACE include)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(eigen_expm_multiply INTERFACE Eigen3::Eigen)

if (EIGEN_EXPM_MULTIPLY_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
