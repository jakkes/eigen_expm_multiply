#ifndef JAKKES_EIGEN_EXPM_MULTIPLY_IMPL_H_
#define JAKKES_EIGEN_EXPM_MULTIPLY_IMPL_H_

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <memory>
#include <numeric>
#include <algorithm>

#include <Eigen/SparseCore>


namespace Eigen
{
    static constexpr int m_max = 55;
    static constexpr int p_max = 8;

    void checkNotParallelToOneVector(const Eigen::VectorXd &v);

    double onenorm(const Eigen::SparseMatrix<double> &A);

    inline double sign(double x) {
        if (x > 0) {
            return 1.0;
        } else if (x < 0) {
            return -1.0;
        } else {
            return 0.0;
        }
    }

    struct max_abssum_result {
        double value;
        int col;
    };

    max_abssum_result max_abssum_per_col(const Eigen::MatrixXd &A);

    std::unique_ptr<Eigen::VectorXd> unit_norm_random(int size);

    inline bool are_parallel(const Eigen::VectorXd &a, const Eigen::VectorXd &b)
    {
        bool positive_product = a(0) * b(0) > 0;
        for (int k = 1; k < a.rows(); k++) {
            if (a(k) * b(k) > 0 ^ positive_product) return false;
        }
        return true;
    }

    inline bool all_parallel(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
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

    void update_S(Eigen::MatrixXd &S, const Eigen::MatrixXd &S_old);

    std::vector<int> argsort(const Eigen::VectorXd &data);

    double onenormest(const Eigen::SparseMatrix<double> &A, int max_iterations=5);

    inline double trace(const Eigen::SparseMatrix<double> &A)
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

    std::unique_ptr<Eigen::SparseMatrix<double>> subtractIdentity(const Eigen::SparseMatrix<double> &A, double scale=1.0);

    class LazyOperatorNormInfo
    {
        private:
            const Eigen::SparseMatrix<double> &A;
            std::unordered_map<int, double> norms{};
            std::unordered_map<int, Eigen::SparseMatrix<double>> powers{};
        
        public:
            LazyOperatorNormInfo(const Eigen::SparseMatrix<double> &A)
            : A{A}
            {
                norms[1] = onenorm(A);
            }

            inline double norm(int p) {
                if (norms.find(p) == norms.end()) {
                    norms[p] = std::pow(onenormest(power(p)), 1.0 / p);
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

    BestMS frag(LazyOperatorNormInfo &norm_info);

    void core(
        const Eigen::SparseMatrix<double> &A,
        const Eigen::VectorXd &b,
        double mu,
        int m,
        size_t s,
        Eigen::VectorXd *F
    );
}

#endif /* JAKKES_EIGEN_EXPM_MULTIPLY_IMPL_H_ */
