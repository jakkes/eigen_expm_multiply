add_executable(test_eigen_expm_multiply main.cc)
add_test(test_eigen_expm_multiply test_eigen_expm_multiply)

target_sources(
    test_eigen_expm_multiply
    PRIVATE
        big_sparse_test.cc
        big_sparse_test_2.cc
        big_sparse_test_3.cc
        big_sparse_test_4.cc
)

target_link_libraries(test_eigen_expm_multiply PRIVATE gtest eigen_expm_multiply)
