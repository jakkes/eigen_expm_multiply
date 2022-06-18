#ifndef JAKKES_EIGEN_EXPM_MULTIPLY_IMPL_H_
#define JAKKES_EIGEN_EXPM_MULTIPLY_IMPL_H_

#include <unordered_map>
#include <cmath>
#include <memory>

#include <Eigen/SparseCore>


namespace jakkes::__expm_multiply_impl
{
    static constexpr int m_max = 55;
    static constexpr int p_max = 8;

    double onenorm(const Eigen::SparseMatrix<double> &A)
    {
        double out{0.0};
        for (int i = 0; i < A.outerSize(); i++) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                out += std::abs(it.value());
            }
        }
        return out;
    }

    double trace(const Eigen::SparseMatrix<double> &A)
    {
        double out{0.0};
        for (int i = 0; i < A.outerSize(); i++) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                if (it.row() == it.col()) {
                    out += it.value();
                }
            }
        }
        return out;
    }

    std::unique_ptr<Eigen::SparseMatrix<double>> subtractIdentity(const Eigen::SparseMatrix<double> &A, double scale=1.0)
    {
        auto out = std::make_unique<Eigen::SparseMatrix<double>>(A.rows(), A.cols());
        std::vector<Eigen::Triplet<double>> triplets{};
        triplets.reserve(A.size() + A.rows());

        for (int i = 0; i < A.outerSize(); i++) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                triplets.emplace_back(it.row(), it.col(), it.value());
            }
            triplets.emplace_back(i, i, -scale);
        }

        out->setFromTriplets(triplets.begin(), triplets.end());
        return out;
    }

    // https://github.com/scipy/scipy/blob/4cf21e753cf937d1c6c2d2a0e372fbc1dbbeea81/scipy/sparse/linalg/_expm_multiply.py#L228
    std::unordered_map<int, double> _theta{
        {1, 2.29e-16},
        {2, 2.58e-8},
        {3, 1.39e-5},
        {4, 3.40e-4},
        {5, 2.40e-3},
        {6, 9.07e-3},
        {7, 2.38e-2},
        {8, 5.00e-2},
        {9, 8.96e-2},
        {10, 1.44e-1},
        {11, 2.14e-1},
        {12, 3.00e-1},
        {13, 4.00e-1},
        {14, 5.14e-1},
        {15, 6.41e-1},
        {16, 7.81e-1},
        {17, 9.31e-1},
        {18, 1.09},
        {19, 1.26},
        {20, 1.44},
        {21, 1.62},
        {22, 1.82},
        {23, 2.01},
        {24, 2.22},
        {25, 2.43},
        {26, 2.64},
        {27, 2.86},
        {28, 3.08},
        {29, 3.31},
        {30, 3.54},
        {35, 4.7},
        {40, 6.0},
        {45, 7.2},
        {50, 8.5},
        {55, 9.9}
    };

    class LazyOperatorNormInfo
    {
        private:
            const Eigen::SparseMatrix<double> &A;
            std::unordered_map<int, double> norms{};
            std::unordered_map<int, Eigen::SparseMatrix<double>> powers{};
        
        public:
            LazyOperatorNormInfo(const Eigen::SparseMatrix<double> &A)
            : A{A}
            {}

            inline double norm(int p) {
                if (norms.find(p) == norms.end()) {
                    norms[p] = std::pow(onenorm(power(p)), 1.0 / p);
                }
                return norms[p];
            }

            inline double alpha(int p) {
                return std::max( norm(p), norm(p+1) );
            }
        
        private:
            inline const Eigen::SparseMatrix<double> &power(int p) {
                assert(p > 0);
                if (p == 1) return A;

                if (powers.find(p) == powers.end()) {
                    powers[p] = power(p-1) * A;
                }

                return powers[p];
            }
    };

    struct BestMS {
        int m;
        size_t s;
    };

    BestMS frag(LazyOperatorNormInfo &norm_info)
    {
        int best_m;
        size_t best_s;
        bool first{true};
        if (norm_info.norm(1) <= 63.36) {
            for (const auto &m_theta : _theta) {
                size_t s = std::ceil(norm_info.norm(1) / m_theta.second);
                if (first || m_theta.first * s < best_m * best_s) {
                    best_m = m_theta.first;
                    best_s = s;
                    first = false;
                }
            }
        }
        else {
            for (int p = 2; p < p_max + 1; p++) {
                for (int m = p * (p-1) - 1; m < m_max + 1; m++) {
                    if (_theta.find(m) == _theta.end()) continue;
                    size_t s = std::ceil(norm_info.alpha(p) / _theta[m]);
                    if (first || m * s < best_m * best_s) {
                        first = false;
                        best_m = m;
                        best_s = s;
                    }
                }
            }
            best_s = std::max(best_s, 1UL);
        }

        return BestMS{best_m, best_s};
    }

    void core(
        const Eigen::SparseMatrix<double> &A,
        const Eigen::VectorXd &b,
        double mu,
        int m,
        size_t s,
        Eigen::VectorXd *F
    )
    {
        static double tol = std::pow(2, -53);
        *F = b;
        auto B = b;
        double eta = std::exp(mu / s);

        for (int i = 0; i < s; i++) {
            double c1 = B.lpNorm<Eigen::Infinity>();
            for (int j = 0; j < m; j++) {
                double coeff = 1.0 / s / (j+1);
                B = coeff * (A * B);
                double c2 = B.lpNorm<Eigen::Infinity>();
                *F += B;
                if (c1+c2 <= tol * F->lpNorm<Eigen::Infinity>()) {
                    break;
                }
                c1 = c2;
            }
            *F = eta * (*F);
            B = *F;
        }
    }
}

#endif /* JAKKES_EIGEN_EXPM_MULTIPLY_IMPL_H_ */
