#include <cmath>
#include <iostream>

#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/IterativeSolvers>
//#include <Eigen/IterativeLinearSolvers>

#include <getopt.h>

#include "function_generator.hpp"

#include <hdf5.h>

typedef struct {
    double etaR;
    double eta;
    double eta0;
    double G;
    double delta;
    double lam;
    double gam;
    double dt;
    double soltol;
    double t_max;
    int N;
    int n_record;
} param_t;

// typedef std::vector<double> dvec;
typedef Eigen::ArrayXd dvec;
typedef Eigen::ArrayXXd vecvec;

using std::cout;
using std::endl;

Eigen::VectorXd inteqnsolve(const param_t &params, const vecvec &positions,
                            const vecvec &tangents, const vecvec &normals,
                            const double L, const double soltol) {
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
    const int N = params.N;

    Eigen::VectorXd RHS(2 * params.N);
    const double fs_coeff = 0.5 / (M_PI * lam * lam);

    for (int m = 0; m < params.N; ++m) {
        double temp[2] = {0.0, 0.0};
        for (int n = 0; n < params.N; ++n) {
            if (m != n) {
                const double d[2] = {positions(n, 0) - positions(m, 0),
                                     positions(n, 1) - positions(m, 1)};
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

double trapzp(dvec a) {
    double area = a.sum();
    return area * 2 * M_PI / a.size();
}

void writeH5(hid_t fid, std::string path, std::vector<vecvec> time_data) {
    hsize_t dims[3] = {(hsize_t)time_data.size(), (hsize_t)time_data[0].rows(),
                       (hsize_t)time_data[0].cols()};
    hid_t dataspace_id = H5Screate_simple(3, dims, NULL);
    hid_t dataset_id =
        H5Dcreate2(fid, path.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<double> flattened(time_data.size() * time_data[0].rows() *
                                  time_data[0].cols());

    long int offset = 0;
    for (auto &arr : time_data) {
        for (int i = 0; i < arr.rows(); ++i) {
            for (int j = 0; j < arr.cols(); ++j)
                flattened[arr.cols() * i + j + offset] = arr(i, j);
        }
        offset += arr.rows() * arr.cols();
    }

    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             flattened.data());

    /* End access to the dataset and release resources used by it. */
    H5Dclose(dataset_id);

    /* Terminate access to the data space. */
    H5Sclose(dataspace_id);
}

void writeH5(hid_t fid, std::string path, std::vector<dvec> time_data) {
    hsize_t dims[2] = {(hsize_t)time_data.size(), (hsize_t)time_data[0].size()};
    hid_t dataspace_id = H5Screate_simple(2, dims, NULL);
    hid_t dataset_id =
        H5Dcreate2(fid, path.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<double> flattened(time_data.size() * time_data[0].size());

    long int offset = 0;
    for (auto &arr : time_data) {
        for (int i = 0; i < arr.size(); ++i)
            flattened[i + offset] = arr[i];
        offset += arr.size();
    }

    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             flattened.data());

    /* End access to the dataset and release resources used by it. */
    H5Dclose(dataset_id);

    /* Terminate access to the data space. */
    H5Sclose(dataspace_id);
}

void writeH5(hid_t fid, std::string path, dvec arr) {
    hsize_t dims[1] = {(hsize_t)arr.size()};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id =
        H5Dcreate2(fid, path.c_str(), H5T_NATIVE_DOUBLE, dataspace_id,
                   H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
             arr.data());

    /* End access to the dataset and release resources used by it. */
    H5Dclose(dataset_id);

    /* Terminate access to the data space. */
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

    /* End access to the dataset and release resources used by it. */
    H5Dclose(dataset_id);

    /* Terminate access to the data space. */
    H5Sclose(dataspace_id);
}

void writeH5(hid_t fid, std::string path, int val) {
    hsize_t dims[1] = {1};
    hid_t dataspace_id = H5Screate_simple(1, dims, NULL);
    hid_t dataset_id =
        H5Dcreate2(fid, path.c_str(), H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT,
                   H5P_DEFAULT, H5P_DEFAULT);

    H5Dwrite(dataset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &val);

    /* End access to the dataset and release resources used by it. */
    H5Dclose(dataset_id);

    /* Terminate access to the data space. */
    H5Sclose(dataspace_id);
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

dvec linspace(double a, double b, int N) {
    dvec res(N);
    for (int i = 0; i < N; ++i)
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

dvec cumtrapz(dvec X, dvec Y) {
    dvec res(X.size());
    for (int i = 1; i < X.size(); ++i) {
        double dx = 0.5 * (X[i] - X[i - 1]);
        res[i] = res[i - 1] + dx * (Y[i - 1] + Y[i]);
    }
    return res;
}

void printVec(dvec &x) {
    for (int i = 0; i < x.size(); ++i)
        cout << x[i] << endl;
}

dvec find_zeros(double mp, double eps, dvec &s_i) {
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
            status = gsl_root_test_delta(result, x0, 0, 1e-5);
        } while (status == GSL_CONTINUE);

        gsl_root_fsolver_free(s);

        a_i[i] = result;
    }
    return a_i;
}

void print_params(param_t &params) {
    cout << "etaR: " << params.etaR << endl;
    cout << "eta: " << params.eta << endl;
    cout << "eta0: " << params.eta0 << endl;
    cout << "G: " << params.G << endl;
    cout << "delta: " << params.delta << endl;
    cout << "lam: " << params.lam << endl;
    ;
    cout << "gam: " << params.gam << endl;
    cout << "dt: " << params.dt << endl;
    cout << "soltol: " << params.soltol << endl;
    cout << "t_max: " << params.t_max << endl;
    cout << "N: " << params.N << endl;
    cout << "n_record: " << params.n_record << endl;
}

param_t parse_args(int argc, char *argv[]) {
    param_t params;

    //// Set default parameters.

    //// --------- physical parameters (let Omega = 1) --------
    params.etaR = 1;       // rotational viscosity
    params.eta = 1;        // shear viscosity
    params.eta0 = 1;       // odd viscosity
    params.G = 10;         // substrate drag (big Gamma)
    params.gam = 0.01;     // line tension (little gamma)
    params.n_record = 100; // Number of timesteps between output

    //// -------- numerical parameters --------
    params.N = pow(2, 7) - 1; // number of points on curve
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
            /* If this option set a flag, do nothing else now. */
            if (long_options[option_index].flag != 0)
                break;
            printf("option %s", long_options[option_index].name);
            if (optarg)
                printf(" with arg %s", optarg);
            printf("\n");
            break;

        case 'r':
            printf("option -r (etaR) with value `%s'\n", optarg);
            params.etaR = atof(optarg);
            break;

        case 'e':
            printf("option -e (eta) with value `%s'\n", optarg);
            params.eta = atof(optarg);
            break;

        case 'n':
            printf("option -n (eta0) with value `%s'\n", optarg);
            params.eta0 = atof(optarg);
            break;

        case 'G':
            printf("option -G (G) with value `%s'\n", optarg);
            params.G = atof(optarg);
            break;

        case 'g':
            printf("option -g (gam) with value `%s'\n", optarg);
            params.gam = atof(optarg);
            break;

        case 'd':
            printf("option -d (dt) with value `%s'\n", optarg);
            params.dt = atof(optarg);
            break;

        case 't':
            printf("option -t (t_max) with value `%s'\n", optarg);
            params.t_max = atof(optarg);
            break;

        case 's':
            printf("option -s (soltol) with value `%s'\n", optarg);
            params.soltol = atof(optarg);
            break;

        case 'f':
            printf("option -f (n_record) with value `%s'\n", optarg);
            params.n_record = atoi(optarg);
            break;

        case 'N':
            printf("option -N (N) with value `%s'\n", optarg);
            params.N = atoi(optarg);
            break;

        case '?':
            /* getopt_long already printed an error message. */
            break;

        default:
            abort();
        }
    }

    // Derived parameters
    params.delta =
        sqrt((params.eta + params.etaR) / params.G); // BL length scale
    params.lam = 1.0 / params.delta;

    return params;
}

int main(int argc, char *argv[]) {
    param_t params = parse_args(argc, argv);
    print_params(params);

    //// -------- initialize boundary (periodic BCs) ---------
    dvec alpha = linspace(0, 2 * M_PI, params.N + 1);
    alpha.resize(params.N);

    double eps = 0.1; // perturbation amplitude
    int mp = 4;       // perturbation mode
    dvec x = alpha.cos() + eps * (mp * alpha).sin() * alpha.cos();
    dvec y = alpha.sin() + eps * (mp * alpha).sin() * alpha.sin();

    //// -------- geometric quantities --------
    dvec dxda = D(x, alpha); // dx/d(alpha)
    dvec dyda = D(y, alpha); // dy/d(alpha)

    double L_n = trapzp((dxda * dxda + dyda * dyda).sqrt());
    dvec s_i = linspace(0, L_n, params.N + 1);
    s_i.resize(params.N);

    dvec a_i = find_zeros(mp, eps, s_i);

    dvec x_i = a_i.cos() + eps * (mp * a_i).sin() * a_i.cos();
    dvec y_i = a_i.sin() + eps * (mp * a_i).sin() * a_i.sin();

    vecvec positions_n(params.N, 2);
    positions_n.col(0) = x_i;
    positions_n.col(1) = y_i;

    // -------- (x,y) -> (theta,L) --------
    dvec x_ip = D(x_i, alpha);
    dvec x_ipp = D2(x_i, alpha);
    dvec y_ip = D(y_i, alpha);
    dvec y_ipp = D2(y_i, alpha);

    vecvec tangents_n(params.N, 2);
    tangents_n.col(0) = x_ip;
    tangents_n.col(1) = y_ip;

    vecvec normals_n(params.N, 2);
    normals_n.col(0) = -2 * M_PI / L_n * y_ip;
    normals_n.col(1) = 2 * M_PI / L_n * x_ip;

    dvec kappa_n =
        (x_ip * y_ipp - y_ip * x_ipp) / (x_ip * x_ip + y_ip * y_ip).pow(1.5);

    dvec theta_n = (L_n / (2 * M_PI)) * cumtrapz(alpha, kappa_n) +
                   atan2(y_ip[0], x_ip[0]) + 2 * M_PI -
                   L_n / (2 * M_PI) * trapzp(kappa_n);

    dvec dthda_n = L_n / (2 * M_PI) * kappa_n;

    double area_n = 0.5 * trapzp(x_i * x_i + y_i * y_i);

    // -------- given curve, solve linear system for flow --------
    dvec uv_np1 = inteqnsolve(params, positions_n, tangents_n, normals_n, L_n,
                              params.soltol);

    dvec U_n = dvec(params.N).setZero();
    for (int i = 0; i < params.N; ++i)
        for (int j = 0; j < 2; ++j)
            U_n[i] += normals_n(i, j) * uv_np1[2 * i + j];

    dvec T_n = cumtrapz(alpha, dthda_n * U_n) -
               trapzp(dthda_n * U_n) * alpha / (2 * M_PI);

    // update theta and L (Euler forward for 1 step)
    double L_np1 = L_n - params.dt * trapzp(dthda_n * U_n);
    dvec theta_np1 = theta_n + params.dt * (2 * M_PI / L_n) *
                                   (D(U_n, alpha) + dthda_n * T_n);

    vecvec tangents_np1(params.N, 2);
    tangents_np1.col(0) = theta_np1.cos();
    tangents_np1.col(1) = theta_np1.sin();

    vecvec normals_np1(params.N, 2);
    normals_np1.col(0) = -theta_np1.sin();
    normals_np1.col(1) = theta_np1.cos();

    // // update 1 point, then use (x,y) = integral of tangent
    double X_np1[2] = {
        positions_n(0, 0) + params.dt * U_n[0] * normals_n(0, 0) +
            T_n[0] * tangents_n(0, 0),
        positions_n(0, 1) + params.dt * U_n[0] * normals_n(0, 1) +
            T_n[0] * tangents_n(0, 1)};
    dvec x_np1 = X_np1[0] +
                 L_np1 / (2 * M_PI) * cumtrapz(alpha, theta_np1.cos()) -
                 L_np1 / (2 * M_PI) * trapzp(theta_np1.cos());
    dvec y_np1 = X_np1[1] +
                 L_np1 / (2 * M_PI) * cumtrapz(alpha, theta_np1.sin()) -
                 L_np1 / (2 * M_PI) * trapzp(theta_np1.sin());

    vecvec positions_np1(params.N, 2);
    positions_np1.col(0) = x_np1;
    positions_np1.col(1) = y_np1;

    // using new positions, compute new curvature and therefore
    x_ip = D(x_i, alpha);
    x_ipp = D2(x_i, alpha);
    y_ip = D(y_i, alpha);
    y_ipp = D2(y_i, alpha);
    dvec dthda_np1 = L_np1 / (2 * M_PI) * (x_ip * y_ipp - y_ip * x_ipp) /
                     (x_ip.square() + y_ip.square()).pow(1.5);

    double t = 0; // time
    t += params.dt;
    dvec uv_n = uv_np1;
    int i_step = 0;
    dvec D_U_last = D(U_n, alpha);
    std::vector<dvec> theta_t;
    std::vector<dvec> U_t;
    std::vector<vecvec> positions_t;
    while (t < params.t_max) {
        // compute U and T
        dvec uv_np2 = inteqnsolve(params, positions_np1, tangents_np1,
                                  normals_np1, L_np1, params.soltol);
        dvec U_np1 = dvec(params.N).setZero();
        for (int i = 0; i < params.N; ++i)
            for (int j = 0; j < 2; ++j)
                U_np1[i] += normals_np1(i, j) * uv_np2[2 * i + j];

        dvec T_np1 = cumtrapz(alpha, dthda_np1 * U_np1) -
                     alpha / (2 * M_PI) * trapzp(dthda_np1 * U_np1);

        // update theta and L
        double L_np2 = L_np1 - 0.5 * params.dt *
                                   (3 * trapzp(dthda_np1 * U_np1) -
                                    trapzp(dthda_n * U_n)); // AB2

        // FIXME: Can cache the call to D(U_n, alpha)
        dvec D_U_np1 = D(U_np1, alpha);
        dvec theta_np2 =
            theta_np1 +
            0.5 * params.dt *
                (3 * (2 * M_PI / L_np2) * (D_U_np1 + dthda_np1 * T_np1) -
                 (2 * M_PI / L_np1) * (D_U_last + dthda_n * T_n));

        vecvec tangents_np2(params.N, 2);
        tangents_np2.col(0) = theta_np2.cos();
        tangents_np2.col(1) = theta_np2.sin();

        vecvec normals_np2(params.N, 2);
        normals_np2.col(0) = -theta_np2.sin();
        normals_np2.col(1) = theta_np2.cos();

        // integrate tangent to get X(alpha)
        double X_np2[2] = {
            positions_np1(0, 0) + 0.5 * params.dt *
                                      (3 * U_np1[0] * normals_np2(0, 0) -
                                       U_n[0] * normals_np1(0, 0)),
            positions_np1(0, 1) + 0.5 * params.dt *
                                      (3 * U_np1[0] * normals_np2(0, 1) -
                                       U_n[0] * normals_np1(0, 1))};
        dvec x_np2 = X_np2[0] +
                     L_np2 / (2 * M_PI) * cumtrapz(alpha, theta_np2.cos()) -
                     L_np2 / (2 * M_PI) * trapzp(theta_np2.cos());
        dvec y_np2 = X_np2[1] +
                     L_np2 / (2 * M_PI) * cumtrapz(alpha, theta_np2.sin()) -
                     L_np2 / (2 * M_PI) * trapzp(theta_np2.sin());

        vecvec positions_np2(params.N, 2);
        positions_np2.col(0) = x_np2;
        positions_np2.col(1) = y_np2;

        // calculate new curvature
        x_ip = D(x_np2, alpha);
        x_ipp = D2(x_np2, alpha);
        y_ip = D(y_np2, alpha);
        y_ipp = D2(y_np2, alpha);
        dvec dthda_np2 = L_np2 / (2 * M_PI) * (x_ip * y_ipp - y_ip * x_ipp) /
                         (x_ip.square() + y_ip.square()).pow(1.5);

        // change n and n-1 timestep info
        dthda_n = dthda_np1;
        U_n = U_np1;
        T_n = T_np1;
        uv_n = uv_np2;
        positions_np1 = positions_np2;
        normals_np1 = normals_np2;
        tangents_np1 = tangents_np2;
        L_np1 = L_np2;
        dthda_np1 = dthda_np2;
        theta_np1 = theta_np2;
        uv_np1 = uv_np2;
        D_U_last = D_U_np1;

        t += params.dt;

        i_step++;
        if (i_step % params.n_record == 0) {
            positions_t.push_back(positions_np1);
            theta_t.push_back(theta_np1);
            U_t.push_back(U_np1);
        }
    }

    hid_t file_id =
        H5Fcreate("test.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    writeH5(file_id, "/alpha", alpha);
    writeH5(file_id, "/theta_t", theta_t);
    writeH5(file_id, "/U_t", U_t);
    writeH5(file_id, "/positions_t", positions_t);
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
