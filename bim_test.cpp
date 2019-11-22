#include <cmath>
#include <iostream>

#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

#include "function_generator.hpp"

typedef struct {
    double etaR;
    double eta;
    double eta0;
    double G;
    double delta;
    double lam;
    double gam;
    size_t N;
} param_t;

typedef std::vector<std::vector<double>> dmat;
typedef std::vector<double> dvec;

using std::cout;
using std::endl;

Eigen::VectorXd inteqnsolve(const param_t &params, const dmat &positions,
                            const dmat &tangents, const dmat &normals,
                            const double L, const double tol) {
    auto f_bk0 = [](double x) { return gsl_sf_bessel_Kn(0, x); };
    auto f_bk1 = [](double x) { return gsl_sf_bessel_Kn(1, x); };
    auto f_bk2 = [](double x) { return gsl_sf_bessel_Kn(2, x); };
    auto f_bk3 = [](double x) { return gsl_sf_bessel_Kn(3, x); };
    static FunctionGenerator<8, 4096, double> bessel_0(f_bk0, 1e-10, 100);
    static FunctionGenerator<8, 4096, double> bessel_1(f_bk1, 1e-10, 100);
    static FunctionGenerator<8, 4096, double> bessel_2(f_bk2, 1e-10, 100);
    static FunctionGenerator<8, 4096, double> bessel_3(f_bk3, 1e-10, 100);

    Eigen::MatrixXd sys(2 * params.N, 2 * params.N);

    const double etaR = params.etaR;
    const double eta = params.eta;
    const double eta0 = params.eta0;
    const double lam = params.lam;
    const double delta = params.delta;
    const double gam = params.gam;
    const size_t N = params.N;

    Eigen::VectorXd RHS(2 * params.N);
    for (size_t m = 0; m < params.N; ++m) {
        double temp[2] = {0.0, 0.0};
        for (size_t n = 0; n < params.N; ++n) {
            if (m != n) {
                const double d[2] = {positions[n][0] - positions[m][0],
                                     positions[n][1] - positions[m][1]};
                const double r = sqrt(d[0] * d[0] + d[1] * d[1]);
                const double rbar = r * params.lam;
                const double rbar2 = rbar * rbar;
                const double rbar3 = rbar2 * rbar;
                const double r2 = r * r;
                const double r4 = r2 * r2;
                const double r6 = r4 * r2;

                const double bk[4] = {bessel_0(rbar), bessel_1(rbar),
                                      bessel_2(rbar), bessel_3(rbar)};

                // FS_TRAC
                double T[2][2][2];
                {
                    double coeff1 =
                        (4 - rbar * rbar - 2 * rbar * rbar * bk[2]) /
                        (2 * M_PI * lam * lam * r4);
                    double coeff2 =
                        (4 - 2 * rbar * rbar * bk[2] - rbar3 * bk[1]) /
                        (2 * M_PI * lam * lam * r4);
                    double coeff3 =
                        (-8 + rbar3 * bk[3]) / (M_PI * lam * lam * r6);

                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            for (int k = 0; k < 2; ++k) {
                                T[i][j][k] =
                                    (i == j) * d[k] * coeff1 +
                                    ((i == k) * d[j] + (j == k) * d[i]) *
                                        coeff2 +
                                    d[i] * d[j] * d[k] * coeff3;
                            }
                        }
                    }
                }
                double M1[2][2] = {
                    {normals[n][0] * T[0][0][0], normals[n][0] * T[1][0][0]},
                    {normals[n][0] * T[0][0][1], normals[n][0] * T[1][0][1]}};
                double M2[2][2] = {
                    {normals[n][1] * T[0][1][0], normals[n][1] * T[1][1][0]},
                    {normals[n][1] * T[0][1][1], normals[n][1] * T[1][1][1]}};

                // fs_vel_p
                double G_prime[2][2] = {{0}};
                {
                    double drds =
                        (tangents[n][0] * d[0] + tangents[n][1] * d[1]) / r;

                    double coeff1 =
                        (2 * r * (-1 + rbar * bk[1] + rbar2 * bk[0]) +
                         r * rbar2 * (bk[0] - rbar * bk[1])) *
                        drds;
                    double coeff2 = (2 - rbar2 * bk[2]);
                    double coeff3 = (rbar2 * bk[1] * drds * lam);
                    double coeff4 = coeff2 / (r2);

                    for (int i = 0; i < 2; ++i)
                        G_prime[i][i] =
                            coeff1 -
                            4 * r * drds * (-1 + rbar * bk[1] + rbar2 * bk[0]);

                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            G_prime[i][j] +=
                                coeff2 * (tangents[n][i] * d[j] +
                                          tangents[n][j] * d[i]) +
                                (coeff3 - 4 * r * drds * coeff4) * d[i] * d[j];
                            G_prime[i][j] /=
                                2 * M_PI * (eta + etaR) * r4 / pow(delta, 2);
                        }
                    }
                }

                double term1[2][2] = {
                    {-eta0 * G_prime[0][0], -eta0 * G_prime[0][1]},
                    {-eta0 * G_prime[1][0], -eta0 * G_prime[1][1]}};
                double term2[2][2] = {
                    {-etaR * G_prime[0][1], etaR * G_prime[0][0]},
                    {-etaR * G_prime[1][1], etaR * G_prime[1][0]}};

                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j) {
                        sys(2 * m + i, 2 * n + j) =
                            -(L / N) * (M1[i][j] + M2[i][j] + 2 * term1[i][j] +
                                        2 * term2[i][j]);
                    }
                }

                for (int i = 0; i < 2; ++i)
                    for (int j = 0; j < 2; ++j)
                        temp[i] -= G_prime[i][j] * (etaR * positions[n][j] +
                                                    gam * tangents[n][j]);
            } else {
                for (int i = 0; i < 2; ++i) {
                    for (int j = 0; j < 2; ++j)
                        sys(2 * m + i, 2 * n + j) = (i == j) * 0.5;
                }
            }
        }
        RHS(2 * m) = (L / N) * temp[0];
        RHS(2 * m + 1) = (L / N) * temp[1];
    }

    Eigen::GMRES<Eigen::MatrixXd> solver(sys);
    return solver.solve(RHS);
}

double trapzp(dvec &a) {
    double area = 0.0;
    for (auto &y : a)
        area += y;
    return area * 2 * M_PI / a.size();
}

dvec D(dvec &numer, dvec &denom) {
    const int N = numer.size();
    dvec res(N);

    gsl_fft_complex_wavetable *wt;
    gsl_fft_complex_workspace *work;

    work = gsl_fft_complex_workspace_alloc(N);
    wt = gsl_fft_complex_wavetable_alloc(N);

    double data[2 * N];
    for (int i = 0; i < N; ++i) {
        data[i * 2] = numer[i];
        data[i * 2 + 1] = 0.0;
    }

    gsl_fft_complex_forward(data, 1, N, wt, work);

    double kdelta = 2 * M_PI / N / (denom[1] - denom[0]);
    for (int i = 0; i < N / 2 + 1; ++i) {
        double k = kdelta * i;
        if (fabs(data[2 * i]) < 1E-5)
            data[2 * i] = 0.0;
        if (fabs(data[2 * i + 1]) < 1E-5)
            data[2 * i + 1] = 0.0;
        std::swap(data[2 * i], data[2 * i + 1]);
        data[2 * i] *= -k;
        data[2 * i + 1] *= k;
    }
    for (int i = N / 2 + 1; i < N; ++i) {
        double k = -0.5 * kdelta * N + kdelta * (i - N / 2 - 0.5);
        if (fabs(data[2 * i]) < 1E-5)
            data[2 * i] = 0.0;
        if (fabs(data[2 * i + 1]) < 1E-5)
            data[2 * i + 1] = 0.0;
        std::swap(data[2 * i], data[2 * i + 1]);
        data[2 * i] *= -k;
        data[2 * i + 1] *= k;
    }

    gsl_fft_complex_inverse(data, 1, N, wt, work);

    for (int i = 0; i < N; ++i)
        res[i] = data[2 * i];

    gsl_fft_complex_wavetable_free(wt);
    gsl_fft_complex_workspace_free(work);

    return res;
}

dvec D2(dvec &numer, dvec &denom) {
    const int N = numer.size();
    dvec res(N);

    gsl_fft_complex_wavetable *wt;
    gsl_fft_complex_workspace *work;

    work = gsl_fft_complex_workspace_alloc(N);
    wt = gsl_fft_complex_wavetable_alloc(N);

    double data[2 * N];
    for (int i = 0; i < N; ++i) {
        data[i * 2] = numer[i];
        data[i * 2 + 1] = 0.0;
    }

    gsl_fft_complex_forward(data, 1, N, wt, work);

    double kdelta = 2 * M_PI / N / (denom[1] - denom[0]);
    for (int i = 0; i < N / 2 + 1; ++i) {
        double k = kdelta * i;
        data[2 * i] *= -k * k;
        data[2 * i + 1] *= -k * k;
    }
    for (int i = N / 2 + 1; i < N; ++i) {
        double k = -0.5 * kdelta * N + kdelta * (i - N / 2 - 0.5);
        data[2 * i] *= -k * k;
        data[2 * i + 1] *= -k * k;
    }

    gsl_fft_complex_inverse(data, 1, N, wt, work);

    for (int i = 0; i < N; ++i)
        res[i] = data[2 * i];

    gsl_fft_complex_wavetable_free(wt);
    gsl_fft_complex_workspace_free(work);

    return res;
}

dvec linspace(double a, double b, size_t N) {
    dvec res(N);
    for (size_t i = 0; i < N; ++i)
        res[i] = a + i * b / (N - 1);

    return res;
}

double integrand(double ap, void *params) {
    double *p = (double *)params;
    double mp = p[0];
    double eps = p[1];

    return sqrt(pow(mp * eps * cos(ap * mp), 2) +
                pow(1 + eps * sin(ap * mp), 2));
}

double integrate(double a, void *params) {
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    double result, error;
    gsl_function F;
    F.function = integrand;
    F.params = params;

    gsl_integration_qags(&F, 0, a, 0, 1e-7, 1000, w, &result, &error);

    gsl_integration_workspace_free(w);

    return result;
}

double func_to_zero(double x, void *params) {
    return ((double *)params)[2] - integrate(x, params);
}

dvec cumtrapz(dvec &X, dvec &Y) {
    dvec res(X.size());
    for (size_t i = 1; i < X.size(); ++i) {
        double dx = 0.5 * (X[i] - X[i - 1]);
        res[i] = dx * (X[i - 1] + X[i]);
    }
    return res;
}

int main(int argc, char *argv[]) {
    param_t params;

    //// --------- physical parameters (let Omega = 1) --------
    params.etaR = 1; // rotational viscosity
    params.eta = 1;  // shear viscosity
    params.eta0 = 1; // odd viscosity
    params.G = 10;   // substrate drag (big Gamma)
    params.delta =
        sqrt((params.eta + params.etaR) / params.G); // BL length scale
    params.lam = 1.0 / params.delta;
    params.gam = 0.01; // line tension (little gamma)

    //// -------- numerical parameters --------
    params.N = pow(2, 7) - 1; // number of points on curve
    double dt = 0.001;
    double t = 0; // time
    double t_max = 0.1;
    double soltol = 1e-12;

    //// -------- initialize boundary (periodic BCs) ---------
    dvec alpha = linspace(0, 2 * M_PI, params.N + 1);
    alpha.resize(params.N);

    double eps = 0.1; // perturbation amplitude
    int mp = 4;       // perturbation mode
    dvec x(alpha.size());
    dvec y(alpha.size());
    for (size_t i = 0; i < alpha.size(); ++i) {
        x[i] = cos(alpha[i]) + eps * sin(mp * alpha[i]) * cos(alpha[i]);
        y[i] = sin(alpha[i]) + eps * sin(mp * alpha[i]) * sin(alpha[i]);
    }

    //// -------- geometric quantities --------
    dvec dxda = D(x, alpha); // dx/d(alpha)
    dvec dyda = D(y, alpha); // dy/d(alpha)

    dvec tmp(alpha.size());
    for (size_t i = 0; i < dxda.size(); ++i) {
        tmp[i] = sqrt(dxda[i] * dxda[i] + dyda[i] * dyda[i]);
    }
    double L_n = trapzp(tmp);
    dvec s_i = linspace(0, L_n, params.N + 1);
    s_i.resize(params.N);

    dvec a_i(params.N);

    for (size_t i = 0; i < a_i.size(); ++i) {
        double params[3] = {(double)mp, eps, s_i[i]};

        const gsl_root_fsolver_type *T;
        gsl_root_fsolver *s;
        gsl_function F;
        F.function = func_to_zero;
        F.params = params;

        T = gsl_root_fsolver_brent;
        s = gsl_root_fsolver_alloc(T);
        gsl_root_fsolver_set(s, &F, 0, 10);

        double x0;
        double result = 0;
        int status;
        do {
            status = gsl_root_fsolver_iterate(s);
            x0 = result;
            result = gsl_root_fsolver_root(s);
            status = gsl_root_test_delta(result, x0, 0, 1e-5);
        } while (status == GSL_CONTINUE);

        gsl_root_fsolver_free(s);

        a_i[i] = result;
    }

    dvec x_i(params.N);
    dvec y_i(params.N);
    for (size_t i = 0; i < params.N; ++i) {
        x_i[i] = cos(a_i[i]) + eps * sin(mp * a_i[i]) * cos(a_i[i]);
        y_i[i] = sin(a_i[i]) + eps * sin(mp * a_i[i]) * sin(a_i[i]);
    }

    dmat positions_n(params.N, dvec(2));
    for (size_t i = 0; i < params.N; ++i) {
        positions_n[i][0] = x_i[i];
        positions_n[i][1] = y_i[i];
    }

    // -------- (x,y) -> (theta,L) --------
    dvec x_ip = D(x_i, alpha);
    dvec x_ipp = D2(x_i, alpha);
    dvec y_ip = D(y_i, alpha);
    dvec y_ipp = D2(y_i, alpha);
    dmat tangents_n(params.N, dvec(2));
    for (size_t i = 0; i < params.N; ++i) {
        tangents_n[i][0] = x_ip[i] * 2 * M_PI / L_n;
        tangents_n[i][1] = y_ip[i] * 2 * M_PI / L_n;
    }

    dmat normals_n(params.N, dvec(2));
    for (size_t i = 0; i < params.N; ++i) {
        normals_n[i][0] = -y_ip[i] * 2 * M_PI / L_n;
        normals_n[i][1] = x_ip[i] * 2 * M_PI / L_n;
    }

    dvec kappa_n(params.N);
    double norm_factor = 0.0;
    for (size_t i = 0; i < params.N; ++i)
        norm_factor += pow(x_ip[i], 2) + pow(y_ip[i], 2);
    norm_factor = pow(norm_factor, -1.5);
    for (size_t i = 0; i < params.N; ++i)
        kappa_n[i] = norm_factor * (x_ip[i] * y_ipp[i] - y_ip[i] * x_ipp[i]);

    dvec theta_n = cumtrapz(alpha, kappa_n);
    double shift =
        2 * M_PI - L_n / (2 * M_PI) * trapzp(kappa_n) + atan2(y_ip[0], x_ip[0]);

    for (size_t i = 0; i < params.N; ++i) {
        theta_n[i] = shift + L_n / (2 * M_PI) * theta_n[i];
    }

    dvec dthda_n(params.N);
    for (size_t i = 0; i < params.N; ++i)
        dthda_n[i] = L_n / (2 * M_PI) * kappa_n[i];

    tmp.resize(params.N);
    for (size_t i = 0; i < params.N; ++i)
        tmp[i] = pow(x_i[i], 2) + pow(y_i[i], 2);
    double area_n = 0.5 * trapzp(tmp);

    // -------- given curve, solve linear system for flow --------
    for (size_t i = 0; i < 1000; ++i)
        inteqnsolve(params, positions_n, tangents_n, normals_n, L_n, soltol);
}
