#include <iostream>
#include "startConditions.h"
#include <math.h>
#include <ctime>

using namespace std;

int main() {
    clock_t begin = clock();


    double T = 1;
    int j0 = 100;
    double t = T / j0;

    double l = 1;
    int N = 100;
    double h = l / N;

    auto *** y = new double**[N+1];
    for (int i = 0 ; i <= N ; i++) {
        y[i] = new double*[N+1];
        for (int j = 0; j <= N; j++) {
            y[i][j] = new double[N+1];
            for (int k = 0 ; k <= N ; k++) {
                y[i][j][k] = u0(i*h, j*h, k*h);
            }
        }
    }

    double epsilon = 2 * h * h / t;

    for (int j = 0 ; j < j0; j++) {
        for (int i = 0 ; i <= N ; i++) {
            for (int k = 0 ; k <= N ; k++) {
                y[0][i][k] = a0(i*h, k*h, (j)*t);
                y[N][i][k] = a1(l, i*h, k*h, (j)*t);

                y[i][0][k] = b0(i*h, k*h, (j)*t);
                y[i][N][k] = b1(l, i*h, k*h, (j)*t);

                y[i][k][0] = c0(i*h, k*h, (j)*t);
                y[i][k][N] = c1(l, i*h, k*h, (j)*t);
            }
        }

        for (int i2 = 0 ; i2 <= N ; i2++) {
            for (int i3 = 0 ; i3 <= N ; i3++) {
                auto * ai = new double[N+1];
                auto * bi = new double[N+1];

                ai[0] = 0;
                bi[0] = a0(i2*h, i3*h, (j+(double)1/3)*t);
                for (int i1 = 1; i1 < N ; ++i1)
                {
                    ai[i1] = 1 / (2 + epsilon - ai[i1 - 1]);
                    bi[i1] =
                            ((y[i1 + 1][i2][i3] + y[i1 - 1][i2][i3] + bi[i1 - 1]) +
                             (epsilon - 2) * y[i1][i2][i3]) /
                            (2 + epsilon - ai[i1 - 1]);
                }
                y[N][i2][i3] = a1(l, i2*h, i3*h, (j+(double)1/3)*t);
                for (int i1 = N - 1; i1 >= 0; --i1)
                {
                    y[i1][i2][i3] =
                            ai[i1] * y[i1 + 1][i2][i3] + bi[i1];
                }

                delete[] ai;
                delete[] bi;

            }
        }
/*
        --------------------------------------------
*/
        for (int i1 = 0 ; i1 <= N ; i1++) {
            for (int i3 = 0 ; i3 <= N ; i3++) {

                auto * ai = new double[N+1];
                auto * bi = new double[N+1];

                ai[0] = 0;
                bi[0] = b0(i1*h, i3*h, (j+(double)2/3)*t);
                for (int i2 = 1 ; i2 < N ; ++i2)
                {
                    ai[i2] = 1 / (2 + epsilon - ai[i2 - 1]);
                    bi[i2] =
                            ((y[i1][i2+1][i3] + y[i1][i2-1][i3] + bi[i2 - 1]) +
                             (epsilon - 2) * y[i1][i2][i3]) /
                            (2 + epsilon - ai[i2 - 1]);
                }
                y[i1][N][i3] = b1(l, i1*h, i3*h, (j+(double)2/3)*t);
                for (int i2 = N - 1; i2 >= 0; --i2)
                {
                    y[i1][i2][i3] =
                            ai[i2] * y[i1][i2+1][i3] + bi[i2];
                }

                delete[] ai;
                delete[] bi;
            }
        }

/*
        --------------------------------------------
*/
        for (int i1 = 0 ; i1 <= N ; i1++) {
            for (int i2 = 0 ; i2 <= N ; i2++) {

                auto * ai = new double[N+1];
                auto * bi = new double[N+1];

                ai[0] = 0;
                bi[0] = c0(i1*h, i2*h, (j+(double)3/3)*t);
                for (int i3 = 1; i3 < N ; ++i3)
                {
                    ai[i3] = 1 / (2 + epsilon - ai[i3 - 1]);
                    bi[i3] =
                            ((y[i1][i2][i3+1] + y[i1][i2][i3-1] + bi[i3 - 1]) +
                             (epsilon - 2) * y[i1][i2][i3]) /
                            (2 + epsilon - ai[i3 - 1]);
                }
                y[i1][i2][N] = b1(l, i1*h, i2*h, (j+(double)3/3)*t);
                for (int i3 = N - 1; i3 >= 0; --i3)
                {
                    y[i1][i2][i3] =
                            ai[i3] * y[i1][i2][i3+1] + bi[i3];
                }

                delete[] ai;
                delete[] bi;
            }
        }



        double maxDifference = 0;
        for (int i1 = 0; i1 <= N; ++i1)
        {
            for (int i2 = 0; i2 <= N; ++i2)
            {
                for (int i3 = 0; i3 <= N; ++i3)
                {
                    if (fabs(exp(3*(j+1)*t + i1*h + i2*h + i3*h) - y[i1][i2][i3]) > maxDifference) {
                        maxDifference = fabs(exp(3*(j+1)*t + i1*h + i2*h + i3*h) - y[i1][i2][i3]);
                    }
                }
            }
        }

        std::cout << maxDifference << std::endl;
    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Time: " << elapsed_secs << endl;

    return 0;
}
