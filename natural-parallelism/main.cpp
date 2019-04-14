#include <array>
#include <cmath>
#include <iostream>
#include <vector>

#include "common/test_functions.h"

using double2 = std::array<double, 2>;
using double3 = std::array<double, 3>;

using tensor3d = std::vector< std::vector <std::vector<double> > >;
using tensor2d = std::vector< std::vector<double> >;
using tensor1d = std::vector<double>;

void start_initialization(tensor3d& y, double h, int N) {
    for (int i = 0 ; i <= N ; i++) {
        for (int j = 0; j <= N; j++) {
            for (int k = 0 ; k <= N ; k++) {
                y[i][j][k] = func::u0({i*h, j*h, k*h});
            }
        }
    }
}

int main(int argc, char * argv[]) {
    auto start = std::chrono::steady_clock::now();
    int N = 100; // number of iterations (start conditions intializing not included)
    if (argc == 2 ) {
        N = std::atoi(argv[1]);
    }

    tensor3d y(N+1, tensor2d(N+1, tensor1d(N+1)));

    double T = 1;
    int j0 = 100;
    double t = T / j0;

    double l = 1;
    double h = l / N;

    start_initialization(y, h, N);

    double epsilon = 2 * h * h / t;

    for (int j = 0; j < j0; j++) {
        for (int i = 0; i <= N ; i++) {
            for (int k = 0; k <= N ; k++) {
                y[0][i][k] = func::a0({i*h, k*h}, (j)*t);
                y[N][i][k] = func::a1({i*h, k*h}, (j)*t);
                y[i][0][k] = func::b0({i*h, k*h}, (j)*t);
                y[i][N][k] = func::b1({i*h, k*h}, (j)*t);
                y[i][k][0] = func::c0({i*h, k*h}, (j)*t);
                y[i][k][N] = func::c1({i*h, k*h}, (j)*t);
            }
        }

        for (int i2 = 0; i2 <= N; i2++) {
            for (int i3 = 0; i3 <= N; i3++) {
                double ai[N+1];
                double bi[N+1];

                ai[0] = 0;
                bi[0] = func::a0({i2*h, i3*h}, (j+(double)1/3)*t);
                for (int i1 = 1; i1 < N ; ++i1) {
                    ai[i1] = 1 / (2 + epsilon - ai[i1 - 1]);
                    bi[i1] =
                            ((y[i1 + 1][i2][i3] + y[i1 - 1][i2][i3] + bi[i1 - 1]) +
                             (epsilon - 2) * y[i1][i2][i3]) /
                            (2 + epsilon - ai[i1 - 1]);
                }
                y[N][i2][i3] = func::a1({i2*h, i3*h}, (j+(double)1/3)*t);
                for (int i1 = N - 1; i1 >= 0; --i1) {
                    y[i1][i2][i3] = ai[i1] * y[i1 + 1][i2][i3] + bi[i1];
                }
            }
        }
/*
        --------------------------------------------
*/
        for (int i1 = 0; i1 <= N; i1++) {
            for (int i3 = 0; i3 <= N; i3++) {
                double ai[N+1];
                double bi[N+1];

                ai[0] = 0;
                bi[0] = func::b0({i1*h, i3*h}, (j+(double)2/3)*t);
                for (int i2 = 1; i2 < N; ++i2) {
                    ai[i2] = 1 / (2 + epsilon - ai[i2 - 1]);
                    bi[i2] =
                            ((y[i1][i2+1][i3] + y[i1][i2-1][i3] + bi[i2 - 1]) +
                             (epsilon - 2) * y[i1][i2][i3]) /
                            (2 + epsilon - ai[i2 - 1]);
                }
                y[i1][N][i3] = func::b1({i1*h, i3*h}, (j+(double)2/3)*t);
                for (int i2 = N - 1; i2 >= 0; --i2) {
                    y[i1][i2][i3] = ai[i2] * y[i1][i2+1][i3] + bi[i2];
                }
            }
        }
/*
        --------------------------------------------
*/
        for (int i1 = 0; i1 <= N; i1++) {
            for (int i2 = 0; i2 <= N; i2++) {
                double ai[N+1];
                double bi[N+1];

                ai[0] = 0;
                bi[0] = func::c0({i1*h, i2*h}, (j+(double)3/3)*t);
                for (int i3 = 1; i3 < N; ++i3) {
                    ai[i3] = 1 / (2 + epsilon - ai[i3 - 1]);
                    bi[i3] =
                            ((y[i1][i2][i3+1] + y[i1][i2][i3-1] + bi[i3 - 1]) +
                             (epsilon - 2) * y[i1][i2][i3]) /
                            (2 + epsilon - ai[i3 - 1]);
                }
                y[i1][i2][N] = func::b1({i1*h, i2*h}, (j+(double)3/3)*t);
                for (int i3 = N - 1; i3 >= 0; --i3) {
                    y[i1][i2][i3] = ai[i3] * y[i1][i2][i3+1] + bi[i3];
                }
            }
        }

        double error = 0;
        for (int i1 = 0; i1 <= N; ++i1) {
            for (int i2 = 0; i2 <= N; ++i2) {
                for (int i3 = 0; i3 <= N; ++i3) {
                    if (std::abs(func::u({i1*h, i2*h, i3*h}, (j+1)*t) - y[i1][i2][i3]) > error) {
                        error = std::abs(func::u({i1*h, i2*h, i3*h}, (j+1)*t) - y[i1][i2][i3]);
                    }
                }
            }
        }
        std::cout << error << '\n';
    }

    auto finish = std::chrono::steady_clock::now();
    auto time_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    std::cout << "Execution time (in milliseconds): " << static_cast<float>(time_in_milliseconds.count())  << '\n';

    return 0;
}
