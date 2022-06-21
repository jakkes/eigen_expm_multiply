#ifndef EIGEN_DETAIL_EXPM_MULTIPLY_H_
#define EIGEN_DETAIL_EXPM_MULTIPLY_H_


#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <memory>
#include <numeric>
#include <algorithm>

#include <Eigen/Dense>
#include <Eigen/SparseCore>


namespace Eigen::detail::expm_multiply
{
    static constexpr int m_max = 55;
    static constexpr int p_max = 8;

    template<typename d>
    inline void checkNotParallelToOneVector(const Eigen::MatrixBase<d> &v) {
        for (int i = 1; i < v.size(); i++) {
            if (v(i-1) != v(i)) return;
        }
        throw std::runtime_error{"Invalid sample"};
    }

    template<typename Scalar>
    inline Scalar onenorm(const Eigen::SparseMatrix<Scalar> &A) {
        assert(!A.IsRowMajor);
        assert(A.cols() > 0);
        Scalar max{-1.0};
        for (int i = 0; i < A.outerSize(); i++) {
            Scalar sum{0.0};
            for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(A, i); it; ++it) {
                sum += std::abs(it.value());
            }
            if (sum > max) max = sum;
        }
        return max;
    }

    template<typename Scalar>
    inline Scalar sign(Scalar x) {
        if (x > 0) {
            return 1.0;
        } else if (x < 0) {
            return -1.0;
        } else {
            return 0.0;
        }
    }

    template<typename Scalar>
    struct max_abssum_result {
        Scalar value;
        int col;
    };

    template<typename Scalar, int Rows, int Cols>
    inline max_abssum_result<Scalar> max_abssum_per_col(const Eigen::Matrix<Scalar, Rows, Cols>  &A)
    {
        assert(A.cols() > 0);

        max_abssum_result<Scalar> re{0, 0};
        for (int i = 0; i < A.cols(); i++) {
            Scalar sum{0.0};
            for (int j = 0; j < A.rows(); j++) {
                sum += std::abs(A(j, i));
            }
            if (sum > re.value) {
                re.value = sum;
                re.col = i;
            }
        }
        return re;
    }

    template<typename Scalar>
    inline
    std::unique_ptr<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>> unit_norm_random(int size)
    {
        auto random_vector = std::make_unique<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>();
        random_vector->setRandom(size);
        *random_vector /= random_vector->squaredNorm();
        return random_vector;
    }

    template<typename d1, typename d2>
    inline bool are_parallel(
        const Eigen::MatrixBase<d1> &a, const Eigen::MatrixBase<d2> &b
    )
    {
        bool positive_product = a(0) * b(0) > 0;
        for (int k = 1; k < a.rows(); k++) {
            if (a(k) * b(k) > 0 ^ positive_product) return false;
        }
        return true;
    }

    template<typename Scalar, int Rows, int Cols>
    inline bool all_parallel(const Eigen::Matrix<Scalar, Rows, Cols> &A, const Eigen::Matrix<Scalar, Rows, Cols> &B)
    {
        for (int i = 0; i < A.cols(); i++) {
            for (int j = 0; j < B.cols(); j++) {
                if (!are_parallel(A.col(i), B.col(j))) {
                    return false;
                }
            }
        }
        return true;
    }

    template<typename Scalar, int Rows, int Cols>
    inline void update_S(Eigen::Matrix<Scalar, Rows, Cols> &S, const Eigen::Matrix<Scalar, Rows, Cols> &S_old)
    {
        for (int k = 0; k < 1000; k++)
        {
            if (are_parallel(S.col(0), S.col(1))) {
                S.col(1) = *unit_norm_random<Scalar>(S.rows());
                continue;
            }

            bool replaced{false};
            for (int j = 0; j < 2; j++) {
                for (int i = 0; i < 2; i++) {
                    if (are_parallel(S.col(i), S_old.col(j))) {
                        S.col(i) = *unit_norm_random<Scalar>(S.rows());
                        replaced = true;
                    }
                }
            }
            if (!replaced) return;
        }

        throw std::runtime_error{"Reached iteration limit in update_S()"};
    }

    template<typename Scalar, int Rows>
    inline std::vector<int> argsort(const Eigen::Matrix<Scalar, Rows, 1> &data)
    {
        std::vector<int> indices{};
        indices.resize(data.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(
            indices.begin(), indices.end(),
            [&data] (int i, int j) { return data(i) < data(j); }
        );

        return indices;
    }

    template<typename Scalar>
    inline Scalar onenormest(const Eigen::SparseMatrix<Scalar> &A, int max_iterations=5)
    {
        Eigen::SparseMatrix<Scalar> AT = A.transpose();
        std::unordered_set<int> ind_hist{};
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ind{A.rows()};
        Eigen::Matrix<Scalar, Eigen::Dynamic, 2> S{A.rows(), 2};
        Eigen::Matrix<Scalar, Eigen::Dynamic, 2> S_old{A.rows(), 2};
        Eigen::Matrix<Scalar, Eigen::Dynamic, 2> X{A.rows(), 2};

        X << Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Ones(A.rows()) / A.rows(), *unit_norm_random<Scalar>(A.rows());
        checkNotParallelToOneVector(X.col(1));

        max_abssum_result<Scalar> est_old{0, 0};
        max_abssum_result<Scalar> est{};

        int ind_best = 0;

        for (int k = 1; ; k++)
        {
            Eigen::Matrix<Scalar, Eigen::Dynamic, 2> Y = A * X;
            est = max_abssum_per_col(Y);

            if (est.value > est_old.value || k == 2) {
                ind_best = est.col;
            }

            if (k >= 2 && est.value <= est_old.value) {
                est = est_old;
                break;
            }

            est_old = est;
            S_old = S;

            if (k > max_iterations) {
                break;
            }

            S  = Y.cwiseSign();
            if (all_parallel(S, S_old)) break;

            update_S(S, S_old);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 2> Z = AT * S;
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> h = Z.rowwise().template lpNorm<Eigen::Infinity>();

            if (k >= 2 && h.maxCoeff() == h.coeff(ind_best)) {
                break;
            }

            auto ind = argsort(h);
            if ( ind_hist.find(ind[0]) != ind_hist.end() && ind_hist.find(ind[1]) != ind_hist.end() )
            {
                break;
            }

            int j = 0;
            for (int i = 0; i < 2; i++) {
                for (;; j++) {
                    if (ind_hist.find(ind[j]) == ind_hist.end()) {
                        ind[i] = ind[j];
                        break;
                    }
                }
            }
            assert(j < ind.size());

            for (int i = 0; i < 2; i++) {
                X.col(i).setZero();
                X(ind[i], i) = 1.0;
            }

            ind_hist.insert(ind[0]);
            ind_hist.insert(ind[1]);
        }

        return est.value;
    }

    template<typename Scalar>
    inline Scalar trace(const Eigen::SparseMatrix<Scalar> &A)
    {
        Scalar out{0};
        for (int i = 0; i < A.outerSize(); i++) {
            for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(A, i); it; ++it) {
                if (it.row() == it.col()) {
                    out += it.value();
                }
            }
        }
        return out;
    }

    template<typename Scalar>
    inline std::unique_ptr<Eigen::SparseMatrix<Scalar>> subtractIdentity(const Eigen::SparseMatrix<Scalar> &A, Scalar scale=1)
    {
        auto out = std::make_unique<Eigen::SparseMatrix<Scalar>>(A.rows(), A.cols());
        std::vector<Eigen::Triplet<Scalar>> triplets{};
        triplets.reserve(A.size() + A.rows());

        for (int i = 0; i < A.outerSize(); i++) {
            for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(A, i); it; ++it) {
                triplets.emplace_back(it.row(), it.col(), it.value());
            }
            triplets.emplace_back(i, i, -scale);
        }

        out->setFromTriplets(triplets.begin(), triplets.end());
        return out;
    }

    template<typename Scalar>
    class LazyOperatorNormInfo
    {
        private:
            const Eigen::SparseMatrix<Scalar> &A;
            std::unordered_map<int, Scalar> norms{};
            std::unordered_map<int, Eigen::SparseMatrix<Scalar>> powers{};
        
        public:
            LazyOperatorNormInfo(const Eigen::SparseMatrix<Scalar> &A)
            : A{A}
            {
                norms[1] = onenorm(A);
            }

            inline Scalar norm(int p) {
                if (norms.find(p) == norms.end()) {
                    norms[p] = std::pow(onenormest(power(p)), 1.0 / p);
                }
                return norms[p];
            }

            inline Scalar alpha(int p) {
                return std::max( norm(p), norm(p+1) );
            }
        
        private:
            inline const Eigen::SparseMatrix<Scalar> &power(int p) {
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

    template<typename Scalar>
    inline BestMS frag(LazyOperatorNormInfo<Scalar> &norm_info)
    {
        // https://github.com/scipy/scipy/blob/4cf21e753cf937d1c6c2d2a0e372fbc1dbbeea81/scipy/sparse/linalg/_expm_multiply.py#L228
        static std::unordered_map<int, double> _theta{
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

    template<typename Scalar, int Rows>
    inline void core(
        const Eigen::SparseMatrix<Scalar> &A,
        const Eigen::Matrix<Scalar, Rows, 1> &b,
        Scalar mu,
        int m,
        size_t s,
        Eigen::Matrix<Scalar, Rows, 1> *F
    )
    {
        static double tol = std::pow(2, -53);
        *F = b;
        Eigen::Matrix<Scalar, Rows, 1> B = b;
        double eta = std::exp(mu / s);

        for (int i = 0; i < s; i++) {
            double c1 = B.template lpNorm<Eigen::Infinity>();
            for (int j = 0; j < m; j++) {
                double coeff = 1.0 / s / (j+1);
                B = coeff * (A * B);
                double c2 = B.template lpNorm<Eigen::Infinity>();
                *F += B;
                if (c1+c2 <= tol * F->template lpNorm<Eigen::Infinity>()) {
                    break;
                }
                c1 = c2;
            }
            *F = eta * (*F);
            B = *F;
        }
    }
}

#endif /* EIGEN_DETAIL_EXPM_MULTIPLY_H_ */
