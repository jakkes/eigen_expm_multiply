#include <iostream>
#include <cmath>

#include <gtest/gtest.h>
#include <Eigen/SparseCore>
#include <Eigen/Dense>

#include <Eigen/expm_multiply.h>


TEST(dummy, dummy) {
    ASSERT_EQ(1, 1);
}

TEST(expm_multiply, scalar) {
    Eigen::SparseMatrix<double> A{1,1};
    A.coeffRef(0, 0) = 5.0;

    Eigen::VectorXd b{1};
    b(0) = 2.0;

    auto result = Eigen::expm_multiply(A, b);
    ASSERT_NEAR((*result)(0), std::exp(5.0) * 2.0, 1e-3);
}

TEST(expm_multiply, diagonal) {
    Eigen::SparseMatrix<double> A{4,4};
    Eigen::VectorXd b{4};
    for (int i = 0; i < 4; i++) {
        A.coeffRef(i, i) = i;
        b(i) = 4 - i;
    }

    auto result = Eigen::expm_multiply(A, b);
    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR((*result)(i), std::exp(i) * (4 - i), 1e-3);
    }
}


// Expected result obtained from scipy
TEST(expm_multiply, dense) {
    Eigen::SparseMatrix<double> A{4,4};
    std::vector<Eigen::Triplet<double>> triplets{};
    triplets.reserve(16);
    triplets.emplace_back(0, 0, -2.58967563);
    triplets.emplace_back(0, 1, 0.17246303);
    triplets.emplace_back(0, 2, 0.64872998);
    triplets.emplace_back(0, 3, 0.66021286);
    triplets.emplace_back(1, 0, 0.11138618);
    triplets.emplace_back(1, 1, -0.15902188);
    triplets.emplace_back(1, 2, -1.2542628);
    triplets.emplace_back(1, 3, -0.3140965 );
    triplets.emplace_back(2, 0, -0.27349205);
    triplets.emplace_back(2, 1, 0.28710904);
    triplets.emplace_back(2, 2, -1.81543034);
    triplets.emplace_back(2, 3, 1.48233082);
    triplets.emplace_back(3, 0, 1.18596378);
    triplets.emplace_back(3, 1, -0.32232263);
    triplets.emplace_back(3, 2, 1.07553302);
    triplets.emplace_back(3, 3, 0.82501958);
    A.setFromTriplets(triplets.begin(), triplets.end());

    Eigen::VectorXd b{4};
    b(0) = 1.06447902;
    b(1) = -0.27261665;
    b(2) = 0.52023932;
    b(3) = -0.07119798;

    Eigen::VectorXd F{4};
    F(0) = 0.39751863;
    F(1) = -0.85562135;
    F(2) = 0.62733198;
    F(3) = 1.72522841;

    auto result = Eigen::expm_multiply(A, b);

    for (int i = 0; i < 4; i++) {
        ASSERT_NEAR((*result)(i), F(i), 1e-3);
    }
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
