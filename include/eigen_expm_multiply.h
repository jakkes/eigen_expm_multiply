#ifndef JAKKES_EIGEN_EXPM_MULTIPLY_H_
#define JAKKES_EIGEN_EXPM_MULTIPLY_H_

#include <memory>

#include <Eigen/Dense>
#include <Eigen/SparseCore>


namespace Eigen
{

    /**
     * @brief Computes the matrix-vector operation exp(A)b.
     * 
     * @param A Matrix of shape (n, n)
     * @param b Vector of shape (n, 1)
     * @return VectorXd Vector of shape (n, 1)
     */
    std::unique_ptr<VectorXd> expm_multiply(const SparseMatrix<double> &A, const VectorXd &b);
}

#endif /* JAKKES_EIGEN_EXPM_MULTIPLY_H_ */
