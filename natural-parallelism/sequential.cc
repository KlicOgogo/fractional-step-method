#include <cmath>
#include <iostream>

#include "common/constants.h"
#include "common/test_functions.h"

void start_initialization(tensor3d& y, double h, int N) {
    double x1_curr, x2_curr;
    for (int i = 0 ; i <= N ; ++i) {
        x1_curr = i * h;
        for (int j = 0; j <= N; ++j) {
            x2_curr = j * h;
            for (int k = 0; k <= N; ++k) {
                y[i][j][k] = func::u0({x1_curr, x2_curr, k*h});
            }
        }
    }
}

int main(int argc, char* argv[]) {
    auto start = std::chrono::steady_clock::now();
    int N = 100; // grid size
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }

    tensor3d y(N+1, tensor2d(N+1, tensor1d(N+1)));
    const double h = consts::l / N; // grid step
    start_initialization(y, h, N);

    double eps = 2 * h * h / consts::t;
    double error = 0;
    double x1_curr, x2_curr, x3_curr, t_curr;
    for (int j = 0; j < consts::j0; ++j) {
        t_curr = consts::t * j;
        double row_curr, col_curr;
        for (int i = 0; i <= N ; ++i) {
            row_curr = i * h;
            for (int k = 0; k <= N ; ++k) {
                col_curr = k * h;
                y[0][i][k] = func::a0({row_curr, col_curr}, t_curr);
                y[N][i][k] = func::a1({row_curr, col_curr}, t_curr);
                y[i][0][k] = func::b0({row_curr, col_curr}, t_curr);
                y[i][N][k] = func::b1({row_curr, col_curr}, t_curr);
                y[i][k][0] = func::c0({row_curr, col_curr}, t_curr);
                y[i][k][N] = func::c1({row_curr, col_curr}, t_curr);
            }
        }

        t_curr = (1.0 / 3 + j) * consts::t;
        for (int i2 = 0; i2 <= N; ++i2) {
            x2_curr = i2 * h;
            for (int i3 = 0; i3 <= N; ++i3) {
                x3_curr = i3 * h;
                double ai[N+1], bi[N+1];
                ai[0] = 0;
                bi[0] = func::a0({x2_curr, x3_curr}, t_curr);
                for (int i1 = 1; i1 < N; ++i1) {
                    ai[i1] = 1.0 / (2.0 + eps - ai[i1-1]);
                    bi[i1] = (y[i1+1][i2][i3] + y[i1-1][i2][i3] + bi[i1-1] + (eps - 2.0) * y[i1][i2][i3]) * ai[i1];
                }
                y[N][i2][i3] = func::a1({x2_curr, x3_curr}, t_curr);
                for (int i1 = N - 1; i1 >= 0; --i1) {
                    y[i1][i2][i3] = ai[i1] * y[i1+1][i2][i3] + bi[i1];
                }
            }
        }
/*
        --------------------------------------------
*/
        t_curr = (2.0 / 3 + j) * consts::t;
        for (int i1 = 0; i1 <= N; ++i1) {
            x1_curr = i1 * h;
            for (int i3 = 0; i3 <= N; ++i3) {
                x3_curr = i3 * h;
                double ai[N+1], bi[N+1];
                ai[0] = 0;
                bi[0] = func::b0({x1_curr, x3_curr}, t_curr);
                for (int i2 = 1; i2 < N; ++i2) {
                    ai[i2] = 1.0 / (2.0 + eps - ai[i2-1]);
                    bi[i2] = (y[i1][i2+1][i3] + y[i1][i2-1][i3] + bi[i2-1] + (eps - 2.0) * y[i1][i2][i3]) * ai[i2];
                }
                y[i1][N][i3] = func::b1({x1_curr, x3_curr}, t_curr);
                for (int i2 = N - 1; i2 >= 0; --i2) {
                    y[i1][i2][i3] = ai[i2] * y[i1][i2+1][i3] + bi[i2];
                }
            }
        }
/*
        --------------------------------------------
*/
        t_curr = (1.0 + j) * consts::t;
        for (int i1 = 0; i1 <= N; ++i1) {
            x1_curr = i1 * h;
            for (int i2 = 0; i2 <= N; ++i2) {
                x2_curr = i2 * h;
                double ai[N+1], bi[N+1];
                ai[0] = 0;
                bi[0] = func::c0({x1_curr, x2_curr}, t_curr);
                for (int i3 = 1; i3 < N; ++i3) {
                    ai[i3] = 1.0 / (2.0 + eps - ai[i3 - 1]);
                    bi[i3] = (y[i1][i2][i3+1] + y[i1][i2][i3-1] + bi[i3-1] + (eps - 2) * y[i1][i2][i3]) * ai[i3];
                }
                y[i1][i2][N] = func::b1({x1_curr, x2_curr}, t_curr);
                for (int i3 = N - 1; i3 >= 0; --i3) {
                    y[i1][i2][i3] = ai[i3] * y[i1][i2][i3+1] + bi[i3];
                }
            }
        }

        t_curr = (j + 1) * consts::t;
        error = 0;
        for (int i1 = 0; i1 <= N; ++i1) {
            x1_curr = i1 * h;
            for (int i2 = 0; i2 <= N; ++i2) {
                x2_curr = i2 * h;
                for (int i3 = 0; i3 <= N; ++i3) {
                    x3_curr = i3 * h;
                    if (std::abs(func::u({x1_curr, x2_curr, x3_curr}, t_curr) - y[i1][i2][i3]) > error) {
                        error = std::abs(func::u({x1_curr, x2_curr, x3_curr}, t_curr) - y[i1][i2][i3]);
                    }
                }
            }
        }
        std::cout << error << '\n';
    } // consts::j0

    auto finish = std::chrono::steady_clock::now();
    auto time_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    std::cout << "Execution time (in milliseconds): " << static_cast<float>(time_in_milliseconds.count()) << '\n';
    return 0;
}
