#ifndef FUNCTIONGENERATOR_H
#define FUNCTIONGENERATOR_H
#include <eigen3/Eigen/Eigen>
#include <eigen3/Eigen/LU>
#include <functional>
#include <vector>

extern "C" {
typedef double (*cfunc_double)(double);
typedef std::complex<double> (*cfunc_complex)(double);
};

//! Namespace for error calculation helper functions.
namespace FGError {
//! Model used to calculate error in approximation the input function.
enum ErrorModel : int8_t { standard = 0, relative = 1 };
using Eigen::Dynamic;
using Eigen::Matrix;

template <typename T>
inline double standard_error(Matrix<T, Dynamic, 1> &coeffs) {
    double maxcoeff = 0.0;
    int n = coeffs.size();
    for (auto i = n - 2; i < n; ++i)
        maxcoeff = std::max(std::abs(coeffs[i]), maxcoeff);

    return maxcoeff / std::max(1.0, std::abs(coeffs[0]));
}

template <typename T>
inline double relative_error(Matrix<T, Dynamic, 1> &coeffs) {
    double maxcoeff = 0.0;
    int n = coeffs.size();
    for (auto i = n - 2; i < n; ++i)
        maxcoeff = std::max(std::abs(coeffs[i]), maxcoeff);

    return maxcoeff / std::abs(coeffs[0]);
}

template <typename T>
inline double calc_error(Matrix<T, Dynamic, 1> &coeffs,
                         ErrorModel error_model) {
    switch (error_model) {
    case relative:
        return FGError::relative_error(coeffs);
    case standard:
        return FGError::standard_error(coeffs);
    default:
        return standard_error(coeffs);
    }
}

} // namespace FGError

//! Class to approximate functions on a given range by sub-divisions of
//! Chebyshev polynomial expansions.
/*!

This is a template "header-only" class which aims to be virtually
indistinguishable, but often orders of magnitude faster, than special functions
on a given range. This goal is accomplished by recursively dividing the input
function into sub-regions which are fit to Chebyshev polynomials of a given
order until the error is within some given tolerance.

To aid in optimization for C++, this class is provided as a template library,
which can result in a 50% performance increase vs. a shared library
implementation. See README for more information.

@tparam n_ Order of expansion to use for Chebyshev subunits
@tparam table_size_ Number of elements in lookup table which is used to assist
in finding the appropriate Chebyshev subunit.
*/
template <uint16_t n_, uint16_t table_size_, typename T> class FunctionGenerator {
    typedef Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> FGmatrix;
    typedef Eigen::Matrix<T, Eigen::Dynamic, 1> FGvec;

  public:
    //!  Default constructor for FunctionGenerator.
    //!
    /*!

    @param f One dimensional function to approximate.
    @param a Lower bound of function domain to approximate.
    @param b Upper bound of function domain to approximate.
    @param tol Desired accuracy of approximation according to error_model.
    @param mw Minimum width of a sub-division.
    @param error_model Which model to use to calculate error for 'tol'
    parameter.
    */
    // FunctionGenerator(T(*fin)(double), double a, double b, double tol = 1E-12,
    //     double mw = 1E-15,
    //     FGError::ErrorModel error_model = FGError::ErrorModel::standard)
    //     : a_(a), b_(b), tol_(tol), mw_(mw),
    //       scale_factor_(table_size_ / (b_ - a_)), bounds_table_(table_size_),
    //       error_model_(error_model) {
    //     init(fin);
    // }

    FunctionGenerator(
        cfunc_double fin, double a, double b, double tol = 1E-12, double mw = 1E-15,
        FGError::ErrorModel error_model = FGError::ErrorModel::standard)
        : a_(a), b_(b), tol_(tol), mw_(mw),
          scale_factor_(table_size_ / (b_ - a_)), bounds_table_(table_size_),
          error_model_(error_model) {
        init(fin);
    }

    T operator()(double x) {
        const int index = bisect_lookup(x);
        const double a = lbs_[index];
        const double b = lbs_[index + 1];
        const double xinterp = 2 * (x - a) / (b - a) - 1.0;

        return chbevl(xinterp, &coeffs_[index * n_]);
    }

  private:
    const double a_;
    const double b_;
    const double tol_;
    const double mw_;
    const double scale_factor_;

    std::vector<double> lbs_;
    std::vector<T> coeffs_;
    std::vector<std::pair<uint16_t, uint16_t>> bounds_table_;

    const FGError::ErrorModel error_model_;

    T chbevl(const double x, const T *c) {
        const double x2 = 2 * x;

        T c0 = c[0];
        T c1 = c[1];
        for (int i = 2; i < n_; ++i) {
            T tmp = c1;
            c1 = c[i] - c0;
            c0 = tmp + c0 * x2;
        }

        return c1 + c0 * x;
    }

    void init(cfunc_double f) {
        FGmatrix V = calc_vandermonde();
        Eigen::PartialPivLU<FGmatrix> VLU = Eigen::PartialPivLU<FGmatrix>(V);

        fit(f, a_, b_, VLU);
        init_lookup();

        // Add one element to bounds_table to handle call on largest upper bound
        bounds_table_.push_back(std::make_pair(lbs_.size() - 1, lbs_.size()));

        // Fixup lbs_ to include last upper bound in case we are in last
        // subdivision for bisect_lookup
        lbs_.push_back(b_);
    }

    FGmatrix calc_vandermonde() {
        FGmatrix V(n_, n_);

        auto x = get_chebyshev_nodes(-1, 1, n_);
        for (int j = 0; j < n_; ++j) {
            V(0, j) = 1;
            V(1, j) = x(j);
        }

        for (int i = 2; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) {
                V(i, j) = T(2) * V(i - 1, j) * x(j) - V(i - 2, j);
            }
        }
        V = V.transpose().eval();
        return V;
    }

    Eigen::VectorXd get_chebyshev_nodes(double lb, double ub, int order) {
        Eigen::VectorXd res(order);
        for (int k = 0; k < order; ++k)
            res[order - k - 1] =
                0.5 * (lb + ub) +
                0.5 * (ub - lb) * cos(M_PI * (k + 0.5) / order);

        return res;
    }

    void fit(cfunc_double f, double a, double b, Eigen::PartialPivLU<FGmatrix> &VLU) {
        double m = 0.5 * (a + b);
        auto xvec = get_chebyshev_nodes(a, b, n_);
        FGvec yvec(n_);

        for (int i = 0; i < n_; ++i)
            yvec[i] = f(xvec[i]);

        FGvec coeffs = VLU.solve(yvec);
        double tail_energy = FGError::calc_error<T>(coeffs, error_model_);

        if (tail_energy < tol_ || (b - a) < mw_) {
            lbs_.push_back(a);
            std::vector<T> coeffstmp(n_);
            // Reverse list for cache performance reasons
            for (int i = 0; i < n_; ++i)
                coeffstmp[i] = coeffs[n_ - i - 1];

            // append to single coeffs_ vector, for cache reasons.
            for (int i = 0; i < n_; ++i)
                coeffs_.push_back(coeffstmp[i]);

            // Require that number of subdivisions is less than maximum possible value in
            // lookup table. If you are failing here, then use a domain of your function that
            // is more well behaved.
            assert((coeffs_.size() / n_ < UINT16_MAX) &&
                   "Too many subdivisions. See comment here for details.");
        } else {
            fit(f, a, m, VLU);
            fit(f, m, b, VLU);
        }
    }

    void init_lookup() {
        for (int i = 0; i < table_size_; ++i) {
            double x0 = a_ + i * (b_ - a_) / table_size_;
            double x1 = a_ + (i + 1) * (b_ - a_) / table_size_;
            bounds_table_[i].first = bisect(x0);
            bounds_table_[i].second = bisect(x1);
        }
    }

    uint16_t bisect_bracketed(double x, int n1, int n2) {
        while (n2 - n1 > 1) {
            const uint16_t m = n1 + (n2 - n1) / 2;
            if (x < lbs_[m])
                n2 = m;
            else
                n1 = m;
        }

        return n1;
    }

    uint16_t bisect(double x) { return bisect_bracketed(x, 0, lbs_.size()); }

    uint16_t bisect_lookup(double x) {
        int table_index = (x - a_) * scale_factor_;
        auto bisect_bounds = bounds_table_[table_index];
        return bisect_bracketed(x, bisect_bounds.first, bisect_bounds.second);
    }
};

#endif
