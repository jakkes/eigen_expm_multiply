#include "eigen_expm_multiply.h"
#include "impl.h"


namespace Eigen
{
    std::unique_ptr<Eigen::VectorXd> expm_multiply(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &b)
    {
        assert(A.rows() == A.cols());
        assert(A.rows() == b.rows());
        assert(b.cols() == 1);

        auto mu = trace(A) / A.rows();
        auto a = subtractIdentity(A, mu);

        BestMS best_ms;
        LazyOperatorNormInfo norm_info{*a};
        if (norm_info.norm(1) == 0) {
            best_ms.m = 0;
            best_ms.s = 1;
        } else {
            best_ms = frag(norm_info);
        }

        auto out = std::make_unique<Eigen::VectorXd>();
        core(*a, b, mu, best_ms.m, best_ms.s, out.get());
        return out;
    }
}
