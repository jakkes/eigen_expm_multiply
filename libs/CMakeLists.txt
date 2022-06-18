if (NOT TARGET gtest AND EIGEN_EXPM_MULTIPLY_BUILD_TESTS)
    add_subdirectory(googletest)
endif()
