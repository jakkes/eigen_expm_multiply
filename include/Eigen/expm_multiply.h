#ifndef JAKKES_EIGEN_EXPM_MULTIPLY_H_
#define JAKKES_EIGEN_EXPM_MULTIPLY_H_

#include <memory>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "detail/expm_multiply.h"


namespace Eigen
{

    /**
     * @brief Computes the matrix-vector operation exp(A)b.
     * 
     * @param A Matrix of shape (n, n)
     * @param b Vector of shape (n, 1)
     * @return VectorXd Vector of shape (n, 1)
     */
    template<typename Scalar, int Rows>
    std::unique_ptr<Matrix<Scalar, Rows, 1>> expm_multiply(const SparseMatrix<Scalar> &A, const Matrix<Scalar, Rows, 1> &b)
    {
        assert(A.rows() == A.cols());
        assert(A.rows() == b.rows());
        assert(b.cols() == 1);

        auto mu = detail::expm_multiply::trace(A) / A.rows();
        auto a = detail::expm_multiply::subtractIdentity(A, mu);

        detail::expm_multiply::BestMS best_ms;
        detail::expm_multiply::LazyOperatorNormInfo<Scalar> norm_info{*a};
        if (norm_info.norm(1) == 0) {
            best_ms.m = 0;
            best_ms.s = 1;
        } else {
            best_ms = detail::expm_multiply::frag(norm_info);
        }

        auto out = std::make_unique<Matrix<Scalar, Rows, 1>>();
        detail::expm_multiply::core(*a, b, mu, best_ms.m, best_ms.s, out.get());
        return out;
    }
}

#endif /* JAKKES_EIGEN_EXPM_MULTIPLY_H_ */
