#include <cmath>
#include <iostream>
#include <vector>

// FFT routines for solving derivatives
// TODO?: could replace this with FFTW. Might be faster. Just one more
// dependency...
#include <gsl/gsl_fft_complex.h>

// Includes for root finding in initialization
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>

// For Bessel K functions
#include <gsl/gsl_sf.h>

// For arrays, matrices, and GMRES
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>

// option parser
#include <getopt.h>

// Output container for later processing
#include <hdf5.h>

// Accelerates bessel function calls
#include "function_generator.hpp"

// Runtime simulation parameters
typedef struct {
    double etaR;   // rotational viscosity
    double eta;    // shear viscosity
    double eta0;   // odd viscosity
    double G;      // substrate drag (big Gamma)
    double delta;  // BL length scale
    double lam;    // delta^-1
    double gam;    // line tension (little gamma)
    double dt;     // timestep
    double soltol; // desired GMRes accuracy
    double t_max;  // Time to simulate to
    int N;         // Number of points on RING
    int n_record;  // number of timesteps between output
    std::string output_file;
} param_t;

// Some convenience types
typedef Eigen::ArrayXd dvec;
typedef Eigen::ArrayXXd dvecvec;

// Creates and solves BIM matrix
Eigen::VectorXd inteqnsolve(const param_t &params, const dvecvec &positions,
                            const dvecvec &tangents, const dvecvec &normals,
                            const double L, const double soltol) {
    auto f_bk0 = [](double x) { return gsl_sf_bessel_Kn(0, x); };
    auto f_bk1 = [](double x) { return gsl_sf_bessel_Kn(1, x); };
    auto f_bk2 = [](double x) { return gsl_sf_bessel_Kn(2, x); };
    auto f_bk3 = [](double x) { return gsl_sf_bessel_Kn(3, x); };
    static FunctionGenerator<8, 4096, double> bessel_0(f_bk0, 1e-10, 100);
    static FunctionGenerator<8, 4096, double> bessel_1(f_bk1, 1e-10, 100);
    static FunctionGenerator<8, 4096, double> bessel_2(f_bk2, 1e-10, 100);
    static FunctionGenerator<8, 4096, double> bessel_3(f_bk3, 1e-10, 100);

    const double etaR = params.etaR;
    const double eta = params.eta;
    const double eta0 = params.eta0;
    const double lam = params.lam;
    const double delta = params.delta;
    const double gam = params.gam;
    const int N = params.N;

    Eigen::MatrixXd sys(2 * N, 2 * N);
    Eigen::VectorXd RHS(2 * N);
    const double fs_coeff = 0.5 / (M_PI * lam * lam);
    for (int m = 0; m < N; ++m) {
        double temp[2] = {0.0, 0.0};
        for (int n = 0; n < N; ++n) {
            if (m != n) {
                const double d[2] = {positions(n, 0) - positions(m, 0),
                                     positions(n, 1) - positions(m, 1)};
                const double r = sqrt(d[0] * d[0] + d[1] * d[1]);
                const double rbar = r * lam;
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
                    const double coeff1 =
                        fs_coeff * (4 - rbar2 - 2 * rbar2 * bk[2]) / r4;
                    const double coeff2 =
                        fs_coeff * (4 - 2 * rbar2 * bk[2] - rbar3 * bk[1]) / r4;
                    const double coeff3 =
                        2 * fs_coeff * (-8 + rbar3 * bk[3]) / r6;

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
                    {normals(n, 0) * T[0][0][0], normals(n, 0) * T[1][0][0]},
                    {normals(n, 0) * T[0][0][1], normals(n, 0) * T[1][0][1]}};
                double M2[2][2] = {
                    {normals(n, 1) * T[0][1][0], normals(n, 1) * T[1][1][0]},
                    {normals(n, 1) * T[0][1][1], normals(n, 1) * T[1][1][1]}};

                // fs_vel_p
                double G_prime[2][2] = {{0}};
                {
                    double drds =
                        (tangents(n, 0) * d[0] + tangents(n, 1) * d[1]) / r;

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
                                coeff2 * (tangents(n, i) * d[j] +
                                          tangents(n, j) * d[i]) +
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
                        temp[i] -= G_prime[i][j] * (etaR * positions(n, j) +
                                                    gam * tangents(n, j));
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
    solver.setTolerance(soltol);

    return solver.solve(RHS);
}

// Trapezoidal integration with periodic boundary
double trapzp(const dvec &a) { return a.sum() * 2 * M_PI / a.size(); }

void writeH5(hid_t fid, std::string path, const std::vector<hsize_t> &dims,
             const dvec &arr) {
    hid_t dataspace_id = H5Screate_simple(dims.size(), dims.data(), NULL);
    hid_t dataset_id =
        H5Dcreate2(fid, path.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    // Dump output to file
    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             arr.data());

    // End access to the dataset and release resources used by it.
    H5Dclose(dataset_id);

    // Terminate access to the data space.
    H5Sclose(dataspace_id);
}

void writeH5(hid_t fid, std::string path, double val) {
    hsize_t dims[1] = {1};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id =
        H5Dcreate2(fid, path.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             &val);

    // End access to the dataset and release resources used by it.
    H5Dclose(dataset_id);

    // Terminate access to the data space.
    H5Sclose(dataspace_id);
}

void writeH5(hid_t fid, std::string path, int val) {
    hsize_t dims[1] = {1};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id =
        H5Dcreate2(fid, path.c_str(), H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT,
                   H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);

    // End access to the dataset and release resources used by it.
    H5Dclose(dataset_id);

    // Terminate access to the data space.
    H5Sclose(dataspace_id);
}

dvec D(const dvec &numer, double delta_a) {
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

    double kdelta = 2 * M_PI / N / delta_a;
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

std::pair<dvec, dvec> D1D2(const dvec &numer, double delta_a) {
    const int N = numer.size();
    dvec dx(N);
    dvec ddx(N);

    gsl_fft_complex_wavetable *wt;
    gsl_fft_complex_workspace *work;

    work = gsl_fft_complex_workspace_alloc(N);
    wt = gsl_fft_complex_wavetable_alloc(N);

    double data1[2 * N];
    for (int i = 0; i < N; ++i) {
        data1[i * 2] = numer[i];
        data1[i * 2 + 1] = 0.0;
    }

    gsl_fft_complex_forward(data1, 1, N, wt, work);

    double data2[2 * N];
    for (int i = 0; i < 2 * N; ++i)
        data2[i] = data1[i];

    double kdelta = 2 * M_PI / N / delta_a;
    for (int i = 0; i < N / 2 + 1; ++i) {
        double k = kdelta * i;
        // For first derivative (i*k)
        std::swap(data1[2 * i], data1[2 * i + 1]);
        data1[2 * i] *= -k;
        data1[2 * i + 1] *= k;

        // For second derivative (-k^2)
        data2[2 * i] *= -k * k;
        data2[2 * i + 1] *= -k * k;
    }
    for (int i = N / 2 + 1; i < N; ++i) {
        double k = -0.5 * kdelta * N + kdelta * (i - N / 2 - 0.5);

        // For first derivative (i*k)
        std::swap(data1[2 * i], data1[2 * i + 1]);
        data1[2 * i] *= -k;
        data1[2 * i + 1] *= k;

        // For second derivative (-k^2)
        data2[2 * i] *= -k * k;
        data2[2 * i + 1] *= -k * k;
    }

    gsl_fft_complex_inverse(data1, 1, N, wt, work);
    for (int i = 0; i < N; ++i)
        dx[i] = data1[2 * i];

    gsl_fft_complex_inverse(data2, 1, N, wt, work);
    for (int i = 0; i < N; ++i)
        ddx[i] = data2[2 * i];

    gsl_fft_complex_wavetable_free(wt);
    gsl_fft_complex_workspace_free(work);

    return std::make_pair(dx, ddx);
}

dvec linspace_trunced(double a, double b, int N) {
    dvec res(N - 1);
    for (int i = 0; i < N - 1; ++i)
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

    gsl_integration_qags(&F, 0, a, 0, 1e-11, 1000, w, &result, &error);

    gsl_integration_workspace_free(w);

    return result;
}

double func_to_zero(double x, void *params) {
    return ((double *)params)[2] - integrate(x, params);
}

dvec cumtrapz(const dvec &X, const dvec &Y) {
    dvec res(X.size());
    res[0] = 0.0;
    for (int i = 1; i < X.size(); ++i) {
        double dx = 0.5 * (X[i] - X[i - 1]);
        res[i] = res[i - 1] + dx * (Y[i - 1] + Y[i]);
    }
    return res;
}

void printVec(const dvec &x) {
    std::cout << "[";
    for (int i = 0; i < x.size(); ++i)
        printf("%.10f, ", x[i]);
    std::cout << "]\n";
}

void printMat(const dvecvec &x) {
    std::cout << "[";
    for (int i = 0; i < x.rows(); ++i)
        printf("%.10f, %.10f; ", x(i, 0), x(i, 1));
    std::cout << "]\n";
}

dvec find_zeros(double mp, double eps, const dvec &s_i) {
    dvec a_i(s_i.size());

    for (int i = 0; i < s_i.size(); ++i) {
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
            status = gsl_root_test_delta(result, x0, 0, 1e-10);
        } while (status == GSL_CONTINUE);

        gsl_root_fsolver_free(s);

        a_i[i] = result;
    }
    return a_i;
}

void print_params(param_t &params) {
    std::cout << "etaR: " << params.etaR << std::endl;
    std::cout << "eta: " << params.eta << std::endl;
    std::cout << "eta0: " << params.eta0 << std::endl;
    std::cout << "G: " << params.G << std::endl;
    std::cout << "delta: " << params.delta << std::endl;
    std::cout << "lam: " << params.lam << std::endl;
    std::cout << "gam: " << params.gam << std::endl;
    std::cout << "dt: " << params.dt << std::endl;
    std::cout << "soltol: " << params.soltol << std::endl;
    std::cout << "t_max: " << params.t_max << std::endl;
    std::cout << "N: " << params.N << std::endl;
    std::cout << "n_record: " << params.n_record << std::endl;
}

param_t parse_args(int argc, char *argv[]) {
    param_t params;

    // Set default parameters.
    params.etaR = 1;
    params.eta = 1;
    params.eta0 = 1;
    params.G = 10;
    params.gam = 0.01;
    params.n_record = 100;
    params.N = pow(2, 7) - 1;
    params.dt = 0.001;
    params.t_max = 10.0;
    params.soltol = 1e-12;

    while (true) {
        static struct option long_options[] = {
            {"etaR", required_argument, 0, 'r'},
            {"eta", required_argument, 0, 'e'},
            {"eta0", required_argument, 0, 'n'},
            {"G", required_argument, 0, 'G'},
            {"gam", required_argument, 0, 'g'},
            {"dt", required_argument, 0, 'd'},
            {"t_max", required_argument, 0, 't'},
            {"soltol", required_argument, 0, 's'},
            {"n_record", required_argument, 0, 'f'},
            {"N", required_argument, 0, 'N'},
            {0, 0, 0, 0}};

        int option_index = 0;
        int c = getopt_long(argc, argv, "r:e:n:G:g:d:t:s:f:", long_options,
                            &option_index);

        if (c == -1)
            break;

        switch (c) {
        case 0:
            // If this option set a flag, do nothing else now.
            if (long_options[option_index].flag != 0)
                break;
            printf("option %s", long_options[option_index].name);
            if (optarg)
                printf(" with arg %s", optarg);
            printf("\n");
            break;

        case 'r':
            printf("option -r (--etaR) with value `%s'\n", optarg);
            params.etaR = atof(optarg);
            break;

        case 'e':
            printf("option -e (--eta) with value `%s'\n", optarg);
            params.eta = atof(optarg);
            break;

        case 'n':
            printf("option -n (--eta0) with value `%s'\n", optarg);
            params.eta0 = atof(optarg);
            break;

        case 'G':
            printf("option -G (--G) with value `%s'\n", optarg);
            params.G = atof(optarg);
            break;

        case 'g':
            printf("option -g (--gam) with value `%s'\n", optarg);
            params.gam = atof(optarg);
            break;

        case 'd':
            printf("option -d (--dt) with value `%s'\n", optarg);
            params.dt = atof(optarg);
            break;

        case 't':
            printf("option -t (--t_max) with value `%s'\n", optarg);
            params.t_max = atof(optarg);
            break;

        case 's':
            printf("option -s (--soltol) with value `%s'\n", optarg);
            params.soltol = atof(optarg);
            break;

        case 'f':
            printf("option -f (--n_record) with value `%s'\n", optarg);
            params.n_record = atoi(optarg);
            break;

        case 'N':
            printf("option -N (--N) with value `%s'\n", optarg);
            params.N = atoi(optarg);
            break;

        case '?':
            // getopt_long already printed an error message.
            break;

        default:
            abort();
        }
    }

    // Derived parameters
    params.delta = sqrt((params.eta + params.etaR) / params.G);
    params.lam = 1.0 / params.delta;

    return params;
}

int main(int argc, char *argv[]) {
    param_t params = parse_args(argc, argv);
    print_params(params);

    //// -------- initialize boundary (periodic BCs) ---------
    dvec alpha = linspace_trunced(0, 2 * M_PI, params.N + 1);
    double delta_alpha = alpha(1) - alpha(0);

    double eps = 0.1; // perturbation amplitude
    int mp = 4;       // perturbation mode
    dvec x = alpha.cos() + eps * (mp * alpha).sin() * alpha.cos();
    dvec y = alpha.sin() + eps * (mp * alpha).sin() * alpha.sin();

    //// -------- geometric quantities --------
    dvec dxda = D(x, delta_alpha); // dx/d(alpha)
    dvec dyda = D(y, delta_alpha); // dy/d(alpha)

    double L_n = trapzp((dxda * dxda + dyda * dyda).sqrt());
    dvec s_i = linspace_trunced(0, L_n, params.N + 1);

    dvec a_i = find_zeros(mp, eps, s_i);

    dvec x_i = a_i.cos() + eps * (mp * a_i).sin() * a_i.cos();
    dvec y_i = a_i.sin() + eps * (mp * a_i).sin() * a_i.sin();

    // -------- (x,y) -> (theta,L) --------
    dvecvec positions_n(params.N, 2);
    positions_n.col(0) = x_i;
    positions_n.col(1) = y_i;

    auto[x_ip, x_ipp] = D1D2(x_i, delta_alpha);
    auto[y_ip, y_ipp] = D1D2(y_i, delta_alpha);

    dvecvec tangents_n(params.N, 2);
    tangents_n.col(0) = 2 * M_PI / L_n * x_ip;
    tangents_n.col(1) = 2 * M_PI / L_n * y_ip;

    dvecvec normals_n(params.N, 2);
    normals_n.col(0) = -2 * M_PI / L_n * y_ip;
    normals_n.col(1) = 2 * M_PI / L_n * x_ip;

    // -------- given curve, solve linear system for flow --------
    dvec uv_np1 = inteqnsolve(params, positions_n, tangents_n, normals_n, L_n,
                              params.soltol);

    dvec U_np1 = dvec(params.N).setZero();
    for (int i = 0; i < params.N; ++i)
        for (int j = 0; j < 2; ++j)
            U_np1[i] += normals_n(i, j) * uv_np1[2 * i + j];

    dvec kappa_n =
        (x_ip * y_ipp - y_ip * x_ipp) / (x_ip * x_ip + y_ip * y_ip).pow(1.5);
    dvec theta_n = (L_n / (2 * M_PI)) * cumtrapz(alpha, kappa_n) +
                   atan2(y_ip[0], x_ip[0]) + 2 * M_PI -
                   L_n / (2 * M_PI) * trapzp(kappa_n);
    dvec dthda_n = L_n / (2 * M_PI) * kappa_n;
    double area_n = 0.5 * trapzp(x_i * x_i + y_i * y_i);
    dvec T_np1 = cumtrapz(alpha, dthda_n * U_np1) -
                 trapzp(dthda_n * U_np1) * alpha / (2 * M_PI);

    // update theta and L (Euler forward for 1 step)
    double L_np1 = L_n - params.dt * trapzp(dthda_n * U_np1);
    dvec theta_np1 = theta_n + params.dt * (2 * M_PI / L_n) *
                                   (D(U_np1, delta_alpha) + dthda_n * T_np1);

    dvec costheta = theta_np1.cos();
    dvec sintheta = theta_np1.sin();
    dvecvec tangents_np1(params.N, 2);
    tangents_np1.col(0) = costheta;
    tangents_np1.col(1) = sintheta;

    dvecvec normals_np1(params.N, 2);
    normals_np1.col(0) = -sintheta;
    normals_np1.col(1) = costheta;

    // update 1 point, then use (x,y) = integral of tangent
    double X_np1 = positions_n(0, 0) + params.dt * U_np1[0] * normals_n(0, 0) +
                   T_np1[0] * tangents_n(0, 0);
    double Y_np1 = positions_n(0, 1) + params.dt * U_np1[0] * normals_n(0, 1) +
                   T_np1[0] * tangents_n(0, 1);

    dvecvec positions_np1(params.N, 2);
    positions_np1.col(0) =
        X_np1 +
        L_np1 / (2 * M_PI) * (cumtrapz(alpha, costheta) - trapzp(costheta));

    positions_np1.col(1) =
        Y_np1 +
        L_np1 / (2 * M_PI) * (cumtrapz(alpha, sintheta) - trapzp(sintheta));

    // using new positions, compute new curvature and therefore
    std::tie(x_ip, x_ipp) = D1D2(positions_np1.col(0), delta_alpha);
    std::tie(y_ip, y_ipp) = D1D2(positions_np1.col(1), delta_alpha);

    dvec dthda_np1 = L_np1 / (2 * M_PI) * (x_ip * y_ipp - y_ip * x_ipp) /
                     (x_ip.square() + y_ip.square()).pow(1.5);
    dvec D_U_np1 = D(U_np1, delta_alpha);

    size_t n_steps = (int)(params.t_max / params.dt);
    size_t n_meas =
        n_steps / params.n_record + (n_steps % params.n_record != 0);
    dvec theta_t(n_meas * params.N);
    dvec U_t(n_meas * params.N);
    dvec positions_t(n_meas * params.N * 2);
    for (size_t i_step = 0; i_step < n_steps; ++i_step) {
        // compute U and T
        dvec uv_np2 = inteqnsolve(params, positions_np1, tangents_np1,
                                  normals_np1, L_np1, params.soltol);
        dvec U_np2 = dvec(params.N);
        for (int i = 0; i < params.N; ++i) {
            U_np2[i] = normals_np1(i, 0) * uv_np2[2 * i] +
                       normals_np1(i, 1) * uv_np2[2 * i + 1];
        }

        dvec T_np2 = cumtrapz(alpha, dthda_np1 * U_np2) -
                     alpha / (2 * M_PI) * trapzp(dthda_np1 * U_np2);

        // update theta and L
        double L_np2 = L_np1 - 0.5 * params.dt *
                                   (3 * trapzp(dthda_np1 * U_np2) -
                                    trapzp(dthda_n * U_np1)); // AB2

        dvec D_U_np2 = D(U_np2, delta_alpha);
        dvec theta_np2 =
            theta_np1 +
            0.5 * params.dt *
                (3 * (2 * M_PI / L_np2) * (D_U_np2 + dthda_np1 * T_np2) -
                 (2 * M_PI / L_np1) * (D_U_np1 + dthda_n * T_np1));

        dvec costheta = theta_np2.cos();
        dvec sintheta = theta_np2.sin();

        dvecvec tangents_np2(params.N, 2);
        tangents_np2.col(0) = costheta;
        tangents_np2.col(1) = sintheta;

        dvecvec normals_np2(params.N, 2);
        normals_np2.col(0) = -sintheta;
        normals_np2.col(1) = costheta;

        // integrate tangent to get X(alpha)
        double X_np2 =
            positions_np1(0, 0) + 0.5 * params.dt *
                                      (3 * U_np2[0] * normals_np2(0, 0) -
                                       U_np1[0] * normals_np1(0, 0));
        double Y_np2 =
            positions_np1(0, 1) + 0.5 * params.dt *
                                      (3 * U_np2[0] * normals_np2(0, 1) -
                                       U_np1[0] * normals_np1(0, 1));

        dvecvec positions_np2(params.N, 2);
        positions_np2.col(0) =
            X_np2 +
            L_np2 / (2 * M_PI) * (cumtrapz(alpha, costheta) - trapzp(costheta));
        positions_np2.col(1) =
            Y_np2 +
            L_np2 / (2 * M_PI) * (cumtrapz(alpha, sintheta) - trapzp(sintheta));

        // calculate new curvature
        std::tie(x_ip, x_ipp) = D1D2(positions_np2.col(0), delta_alpha);
        std::tie(y_ip, y_ipp) = D1D2(positions_np2.col(1), delta_alpha);
        dvec dthda_np2 = L_np2 / (2 * M_PI) * (x_ip * y_ipp - y_ip * x_ipp) /
                         (x_ip.square() + y_ip.square()).pow(1.5);

        // change n and n-1 timestep info
        dthda_n = dthda_np1;
        dthda_np1 = dthda_np2;
        U_np1 = U_np2;
        T_np1 = T_np2;
        positions_np1 = positions_np2;
        normals_np1 = normals_np2;
        tangents_np1 = tangents_np2;
        L_np1 = L_np2;
        theta_np1 = theta_np2;
        D_U_np1 = D_U_np2;

        if (i_step % params.n_record == 0) {
            int i_record = i_step / params.n_record;

            for (int i = 0; i < params.N; ++i)
                for (int j = 0; j < 2; ++j)
                    positions_t[i_record * 2 * params.N + i * 2 + j] =
                        positions_np2(i, j);
            for (int i = 0; i < params.N; ++i)
                theta_t[i_record * params.N + i] = theta_np2[i];
            for (int i = 0; i < params.N; ++i)
                U_t[i_record * params.N + i] = U_np2[i];
        }
    }

    hid_t file_id =
        H5Fcreate("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    writeH5(file_id, "/alpha", {(hsize_t)params.N}, alpha);
    writeH5(file_id, "/theta_t", {(hsize_t)n_meas, (hsize_t)params.N}, theta_t);
    writeH5(file_id, "/U_t", {(hsize_t)n_meas, (hsize_t)params.N}, U_t);
    writeH5(file_id, "/positions_t",
            {(hsize_t)n_meas, (hsize_t)params.N, (hsize_t)2}, positions_t);
    writeH5(file_id, "/area_n", area_n);
    writeH5(file_id, "/etaR", params.etaR);
    writeH5(file_id, "/eta", params.eta);
    writeH5(file_id, "/eta0", params.eta0);
    writeH5(file_id, "/G", params.G);
    writeH5(file_id, "/dt", params.dt);
    writeH5(file_id, "/n_record", params.n_record);

    H5Fclose(file_id);

    return 0;
}
