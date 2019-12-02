/*==========================================================
 * mexcpp.cpp - example in MATLAB External Interfaces
 *
 * Illustrates how to use some C++ language features in a MEX-file.
 *
 * The routine simply defines a class, constructs a simple object,
 * and displays the initial values of the internal variables. It
 * then sets the data members of the object based on the input given
 * to the MEX-file and displays the changed values.
 *
 * This file uses the extension .cpp. Other common C++ extensions such
 * as .C, .cc, and .cxx are also supported.
 *
 * The calling syntax is:
 *
 *              mexcpp( num1, num2 )
 *
 * This is a MEX-file for MATLAB.
 * Copyright 1984-2018 The MathWorks, Inc.
 *
 *========================================================*/

#include "function_generator.hpp"
#include "mex.hpp"
#include "mexAdapter.hpp"
#include <gsl/gsl_sf.h>

#include <iostream>

namespace md = matlab::data;
namespace mm = matlab::mex;

using std::cout;
using std::endl;

class MexFunction : public mm::Function {
  public:
    void operator()(mm::ArgumentList outputs, mm::ArgumentList inputs) {
        checkArguments(outputs, inputs);
        const md::StructArray params(inputs[0]);
        loadStruct(params[0]);

        const md::TypedArray<double> positions(inputs[1]);
        const md::TypedArray<double> tangents(inputs[2]);
        const md::TypedArray<double> normals(inputs[3]);
        const double L = inputs[4][0];
        const double toler = inputs[5][0];
        const md::TypedArray<double> vel_prev(inputs[6]);

        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();

        // matlabPtr->feval(u"besselk", 0, 1.5);

        // auto f_bk0 = [](double x) { return gsl_sf_bessel_Kn(0, x); };
        // auto f_bk1 = [](double x) { return gsl_sf_bessel_Kn(1, x); };
        // auto f_bk2 = [](double x) { return gsl_sf_bessel_Kn(2, x); };
        // auto f_bk3 = [](double x) { return gsl_sf_bessel_Kn(3, x); };
        // FunctionGenerator<8, 4096, double> bessel_0(f_bk0, 1e-10, 100);
        // FunctionGenerator<8, 4096, double> bessel_1(f_bk1, 1e-10, 100);
        // FunctionGenerator<8, 4096, double> bessel_2(f_bk2, 1e-10, 100);
        // FunctionGenerator<8, 4096, double> bessel_3(f_bk3, 1e-10, 100);

        md::ArrayFactory factory;
        // double order[4] = {0, 1, 2, 4};
        // double val[4] = {1.0, 1.0, 1.0, 1.0};
        // mm::ArgumentList args = std::vector<md::TypedArray<double>>(
        //     {factory.createArray({1, 4}, order, order + 4),
        //      factory.createArray({1, 4}, val, val + 4)});
        // matlab::data::TypedArray<double> result =
        //     std::move(matlabPtr->feval(u"besselk", args));
        // for (auto el : result)
        //     cout << el << endl;

        md::TypedArray<double> sys =
            factory.createArray<double>({2 * N_, 2 * N_});
        md::TypedArray<double> RHS = factory.createArray<double>({N_, 2});

        // std::vector<std::vector<double>> sys;
        // sys.resize(2*N_, std::vector<double>(2*N_));

        double rbar_min = 1e300;
        double rbar_max = 0;
        for (size_t m = 0; m < N_; ++m) {
            double temp[2] = {0.0, 0.0};
            for (size_t n = 0; n < N_; ++n) {
                if (m != n) {
                    double d[2] = {positions[0][n] - positions[0][m],
                                   positions[1][n] - positions[1][m]};
                    double r = sqrt(d[0] * d[0] + d[1] * d[1]);
                    double rbar = r * lam_;
                    rbar_min = std::min(rbar_min, rbar);
                    rbar_max = std::max(rbar_max, rbar);
                    double rbar2 = rbar * rbar;
                    // double bk[4] = {bessel_0(rbar), bessel_1(rbar),
                    //                 bessel_2(rbar), bessel_3(rbar)};

                    double bk[4] = {0.0};
                    for (int i = 0; i < 4; ++i) bk[i] =
                        gsl_sf_bessel_Kn(i, rbar);

                    // FS_TRAC
                    double T[2][2][2];
                    {
                        double coeff1 =
                            (4 - rbar * rbar - 2 * rbar * rbar * bk[2]) /
                            (2 * M_PI * lam_ * lam_ * pow(r, 4));
                        double coeff2 = (4 - 2 * rbar * rbar * bk[2] -
                                         pow(rbar, 3) * bk[1]) /
                                        (2 * M_PI * lam_ * lam_ * pow(r, 4));
                        double coeff3 = (-8 + pow(rbar, 3) * bk[3]) /
                                        (M_PI * lam_ * lam_ * pow(r, 6));

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
                    double M1[2][2] = {{normals[0][n] * T[0][0][0],
                                        normals[0][n] * T[1][0][0]},
                                       {normals[0][n] * T[0][0][1],
                                        normals[0][n] * T[1][0][1]}};
                    double M2[2][2] = {{normals[1][n] * T[0][1][0],
                                        normals[1][n] * T[1][1][0]},
                                       {normals[1][n] * T[0][1][1],
                                        normals[1][n] * T[1][1][1]}};

                    // fs_vel_p
                    double G_prime[2][2] = {{0}};
                    {
                        double drds =
                            (tangents[0][n] * d[0] + tangents[1][n] * d[1]) / r;

                        double coeff1 =
                            (2 * r * (-1 + rbar * bk[1] + rbar2 * bk[0]) +
                             r * rbar2 * (bk[0] - rbar * bk[1])) *
                            drds;
                        double coeff2 = (2 - rbar2 * bk[2]);
                        double coeff3 = (rbar2 * bk[1] * drds * lam_);
                        double coeff4 = coeff2 / (r * r);

                        for (int i = 0; i < 2; ++i)
                            G_prime[i][i] = coeff1 - 4 * r * drds *
                                                         (-1 + rbar * bk[1] +
                                                          rbar2 * bk[0]);

                        for (int i = 0; i < 2; ++i) {
                            for (int j = 0; j < 2; ++j) {
                                G_prime[i][j] +=
                                    coeff2 * (tangents[i][n] * d[j] +
                                              tangents[j][n] * d[i]) +
                                    (coeff3 - 4 * r * drds * coeff4) * d[i] *
                                        d[j];
                                G_prime[i][j] /= 2 * M_PI * (eta_ + etaR_) *
                                                 pow(r, 4) / pow(delta_, 2);
                            }
                        }
                    }

                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            sys[2 * m + i][2 * n + j] =
                                -(L / N_) * (M1[i][j] + M2[i][j]) -
                                (L / N_) * G_prime[i][j] * 2 *
                                    ((1 - eta0_) * (i == j) +
                                     etaR_ * (i != j) * (j > i ? 1 : -1));
                        }
                    }

                    for (int i = 0; i < 2; ++i)
                        for (int j = 0; j < 2; ++j)
                            temp[i] -=
                                G_prime[i][j] * (etaR_ * positions[j][n] +
                                                 gam_ * tangents[j][n]);
                } else {
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j)
                            sys[2 * m + i][2 * n + j] = (i == j) * 0.5;
                    }
                }
            }
        }

        std::cout << rbar_min << " " << rbar_max << endl;
    }

    void loadStruct(md::Struct input) {
        etaR_ = input["etaR"][0];
        eta_ = input["eta"][0];
        eta0_ = input["eta0"][0];
        G_ = input["G"][0];
        delta_ = input["delta"][0];
        lam_ = input["lam"][0];
        gam_ = input["gam"][0];
        N_ = input["N"][0];
    }

    void arrayProduct(md::TypedArray<double> &inMatrix, double multiplier) {

        for (auto &elem : inMatrix) {
            elem *= multiplier;
        }
    }

    void checkArguments(mm::ArgumentList outputs, mm::ArgumentList inputs) {
        std::shared_ptr<matlab::engine::MATLABEngine> matlabPtr = getEngine();
        md::ArrayFactory factory;

        if (inputs.size() != 7) {
            matlabPtr->feval(u"error", 0,
                             std::vector<md::Array>({factory.createScalar(
                                 "Two inputs required")}));
        }

        if (inputs[0].getType() != md::ArrayType::STRUCT) {
            matlabPtr->feval(u"error", 0,
                             std::vector<md::Array>({factory.createScalar(
                                 "Input 1 must be a struct")}));
        }

        if (inputs[1].getType() != md::ArrayType::DOUBLE) {
            matlabPtr->feval(u"error", 0,
                             std::vector<md::Array>({factory.createScalar(
                                 "Input 2 must be a 2D array")}));
        }
    }

  private:
    double etaR_;
    double eta_;
    double eta0_;
    double G_;
    double delta_;
    double lam_;
    double gam_;
    unsigned long N_;
};
