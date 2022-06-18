#ifndef JAKKES_EIGEN_EXPM_MULTIPLY_H_
#define JAKKES_EIGEN_EXPM_MULTIPLY_H_

#include <memory>

#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include "__eigen_expm_multiply_impl.h"


namespace jakkes
{

    /**
     * @brief Computes the matrix-vector operation exp(A)b.
     * 
     * @param A Matrix of shape (n, n)
     * @param b Vector of shape (n, 1)
     * @return Eigen::VectorXd Vector of shape (n, 1)
     */
    std::unique_ptr<Eigen::VectorXd> expm_multiply(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b)
    {
        assert(A.rows() == A.cols());
        assert(A.rows() == b.rows());
        assert(b.cols() == 1);

        double trace = __expm_multiply_impl::trace(A);
        auto mu = trace / A.rows();
        auto a = __expm_multiply_impl::subtractIdentity(A, mu);

        __expm_multiply_impl::BestMS best_ms;
        __expm_multiply_impl::LazyOperatorNormInfo norm_info{*a};
        if (norm_info.norm(1) == 0) {
            best_ms.m = 0;
            best_ms.s = 1;
        } else {
            best_ms = __expm_multiply_impl::frag(norm_info);
        }

        auto out = std::make_unique<Eigen::VectorXd>();
        __expm_multiply_impl::core(*a, b, mu, best_ms.m, best_ms.s, out.get());
        return out;
    }
}

#endif /* JAKKES_EIGEN_EXPM_MULTIPLY_H_ */
