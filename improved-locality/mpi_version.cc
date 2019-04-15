/* TODO: change MPI_Send and MPI_Recv to non blocking calls; reduce memory usage
 *
 * ACHTUNG: assume N1 % pcnt =  0
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>

#include "common/constants.h"
#include "common/structures.h"
#include "common/test_functions.h"

#define N1 144
#define N2 100
#define N3 100
#define T 100
#define Q2 30
#define Q3 30
#define H1 0.01
#define H2 0.01
#define H3 0.01
#define TAU 0.01

#define MIN(x, y) (((x) > (y)) ? (y) : (x))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))


double y[T][N1][N2][N3] = {{{{0.}}}};

double a0[N2][N3][T];
double a1[N2][N3][T];

double b0[N1][N3][T];
double b1[N1][N3][T];

double c0[N1][N2][T];
double c1[N1][N2][T];

#define LAMBDA1 0.6
#define LAMBDA2 0.8
#define LAMBDA3 1.

int i, i1, i2, i3, j;

// u = t * e^(LAMBDA1*x1 + LAMBDA2*x2 + LAMBDA3*x3)

double calcExactSolution(double t, double x1, double x2, double x3) {
	return t * exp(LAMBDA1 * x1 + LAMBDA2 * x2 + LAMBDA3 * x3);
}

void setBorderConditions() {
	for (i2 = 0; i2 < N2; ++i2) {
		for (i3 = 0; i3 < N3; ++i3) {
			for (j = 0; j < T; ++j) {
				a0[i2][i3][j] = exp(LAMBDA2 * i2 * H2 + LAMBDA3 * i3 * H3) * (j + (1. / 3.)) * TAU;
				a1[i2][i3][j] = exp(LAMBDA1 * (N1 - 1) * H1 + LAMBDA2 * i2 * H2 + LAMBDA3 * i3 * H3) * (j + (1. / 3.)) * TAU;
			}
		}
	}
	for (i1 = 0; i1 < N1; ++i1) {
		for (i3 = 0; i3 < N3; ++i3) {
			for (j = 0; j < T; ++j) {
				b0[i1][i3][j] = exp(LAMBDA1 * i1 * H1 + LAMBDA3 * i3 * H3) * (j + (2. / 3.)) * TAU;
				b1[i1][i3][j] = exp(LAMBDA1 * i1 * H1 + LAMBDA2 * (N2 - 1) * H2 + LAMBDA3 * i3 * H3) * (j + (2. / 3.)) * TAU;
			}
		}
	}
	for (i1 = 0; i1 < N1; ++i1) {
		for (i2 = 0; i2 < N2; ++i2) {
			for (j = 0; j < T; ++j) {
				c0[i1][i2][j] = exp(LAMBDA1 * i1 * H1 + LAMBDA2 * i2 * H2) * (j + 1) * TAU;
				c1[i1][i2][j] = exp(LAMBDA1 * i1 * H1 + LAMBDA2 * i2 * H2 + LAMBDA3 * (N3 - 1) * H3) * (j + 1) * TAU;
			}
		}
	}
}

void setInitialApproximation() {
	for (i1 = 0; i1 < N1; ++i1) {
		for (i2 = 0; i2 < N2; ++i2) {
			for (i3 = 0; i3 < N3; ++i3) {
				y[0][i1][i2][i3] = 0.0;
			}
		}
	}
}

double tempY[2][N1][N2][N3];
double alphaArr[N1][N2][N3];
double betaArr[N1][N2][N3];

int main(int argc, char* argv[]) {
	std::ofstream output_file;
    output_file.open("result.time", std::ios_base::app);

    MPI_Init(&argc, &argv);
	int my_rank, pcnt;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &pcnt);

	auto start = std::chrono::steady_clock::now();

	const int r1 = N1 / pcnt; // should be N1 % pcnt = 0
	const int r2 = (N2 + Q2 - 1) / Q2;
	const int r3 = (N3 + Q3 - 1) / Q3;

	setBorderConditions();
	setInitialApproximation();

	double epsilon1 = 2 * H1 * H1 / TAU;
	double epsilon2 = 2 * H2 * H2 / TAU;
	double epsilon3 = 2 * H3 * H3 / TAU;

	size_t maxSize = (N1 > N2) ? N1 : N2;
	if (maxSize < N3) {
		maxSize = N3;
	}

	double *alpha = new double[sizeof(*alpha) * maxSize];
	double *beta = new double[sizeof(*beta) * maxSize];

	const size_t splitSize = r2 * r3;
	double *alphaLast = new double[sizeof(*alphaLast) * splitSize];
	double *betaLast = new double [sizeof(*betaLast) * splitSize];
	double *yLast = new double[sizeof(*yLast) * splitSize];

	for (int j = 0; j < T - 1; ++j) {
		for (int q2 = 0; q2 < Q2; q2++) {
			for (int q3 = 0; q3 < Q3; q3++) {
				if (my_rank) {
					// TODO: change to non blocking calls
					MPI_Recv(alphaLast, splitSize, MPI_DOUBLE, my_rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(betaLast, splitSize, MPI_DOUBLE, my_rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}

				for (int i2 = r2 * q2; i2 < MIN(r2 * (q2 + 1), N2); ++i2) {
					for (int i3 = r3 * q3; i3 < MIN(r3 * (q3 + 1), N3); ++i3) {
						int lastIdx = (i2 - r2 * q2) * r2 + i3 - r3 * q3;
						if (my_rank == 0) {
							alphaLast[lastIdx] = 0.;
							betaLast[lastIdx] = a0[i2][i3][j];
						}

						const int startIdx = my_rank * r1 + 1;
						const int stopIdx = MIN((my_rank + 1) * r1 + 1, N1 - 1);
						alphaArr[startIdx - 1][i2][i3] = alphaLast[lastIdx];
						betaArr[startIdx - 1][i2][i3] = betaLast[lastIdx];
						for (i = startIdx; i < stopIdx; ++i) {
							alphaArr[i][i2][i3] = 1 / (2 + epsilon1 - alphaArr[i - 1][i2][i3]);
							betaArr[i][i2][i3] =
								((y[j][i + 1][i2][i3] + y[j][i - 1][i2][i3] + betaArr[i - 1][i2][i3]) +
								(epsilon1 - 2) * y[j][i][i2][i3]) /
									(2 + epsilon1 - alphaArr[i - 1][i2][i3]);
						}
						alphaLast[lastIdx] = alphaArr[stopIdx - 1][i2][i3];
						betaLast[lastIdx] = betaArr[stopIdx - 1][i2][i3];
					}
				}
				if (my_rank != pcnt - 1) {
					// TODO: change to non blocking calls
					MPI_Send(alphaLast, splitSize, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
					MPI_Send(betaLast, splitSize, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
				}
			}
		}
		for (int q2 = 0; q2 < Q2; q2++) {
			for (int q3 = 0; q3 < Q3; q3++) {
				if (my_rank != pcnt - 1) {
					// TODO: change to non blocking calls
					MPI_Recv(yLast, splitSize, MPI_DOUBLE, my_rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				for (i2 = r2 * q2; i2 < MIN(r2 * (q2 + 1), N2); ++i2) {
					for (i3 = r3 * q3; i3 < MIN(r3 * (q3 + 1), N3); ++i3) {
						int lastIdx = (i2 - r2 * q2) * r2 + i3 - r3 * q3;
						int startIdx = (my_rank + 1) * r1 - 1;
						if (my_rank == pcnt - 1) {
							yLast[lastIdx] = a1[i2][i3][j];
							startIdx = N1 - 2;
						}
						const int stopIdx = my_rank * r1;
						tempY[0][startIdx + 1][i2][i3] = yLast[lastIdx];
						for (i = startIdx; i >= stopIdx; --i) {
							tempY[0][i][i2][i3] =
								alphaArr[i][i2][i3] * tempY[0][i + 1][i2][i3] + betaArr[i][i2][i3];
						}
						yLast[lastIdx] = tempY[0][stopIdx][i2][i3];
					}
				}
				if (my_rank) {
					// TODO: change to non blocking calls
					MPI_Send(yLast, splitSize, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
				}
			}
		}

		for (i1 = r1 * my_rank; i1 < r1 * (my_rank + 1); ++i1) {
			for (i3 = 0; i3 < N3; ++i3) {
				alpha[0] = 0;
				beta[0] = b0[i1][i3][j];
				for (i = 1; i < N2 - 1; ++i) {
					alpha[i] = 1 / (2 + epsilon2 - alpha[i - 1]);
					beta[i] =
						((tempY[0][i1][i + 1][i3] + tempY[0][i1][i - 1][i3] + beta[i - 1]) +
						(epsilon2 - 2) * tempY[0][i1][i][i3]) /
							(2 + epsilon2 - alpha[i - 1]);
				}
				tempY[1][i1][N2 - 1][i3] = b1[i1][i3][j];
				for (i = N2 - 2; i >= 0; --i) {
					tempY[1][i1][i][i3] = alpha[i] * tempY[1][i1][i + 1][i3] + beta[i];
				}
			}
		}

		for (i1 = r1 * my_rank; i1 < r1 * (my_rank + 1); ++i1) {
			for (i2 = 0; i2 < N2; ++i2) {
				alpha[0] = 0;
				beta[0] = c0[i1][i2][j];
				for (i = 1; i < N3 - 1; ++i) {
					alpha[i] = 1 / (2 + epsilon3 - alpha[i - 1]);
					beta[i] =
						((tempY[1][i1][i2][i + 1] + tempY[1][i1][i2][i - 1] + beta[i - 1]) +
						(epsilon3 - 2) * tempY[1][i1][i2][i]) /
							(2 + epsilon3 - alpha[i - 1]);
				}
				y[j + 1][i1][i2][N3 - 1] = c1[i1][i2][j];
				for (i = N3 - 2; i >= 0; --i) {
					y[j + 1][i1][i2][i] =
						alpha[i] * y[j + 1][i1][i2][i + 1] + beta[i];
				}
			}
		}

		// TODO: change to non blocking calls
		if (my_rank != pcnt - 1) {
			MPI_Recv(y[j + 1][(my_rank + 1) * r1 + 1], N2 * N3, MPI_DOUBLE, my_rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(y[j + 1][(my_rank + 1) * r1], N2 * N3, MPI_DOUBLE, my_rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (my_rank) {
			MPI_Send(y[j + 1][my_rank * r1 + 1], N2 * N3, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
			MPI_Send(y[j + 1][my_rank * r1], N2 * N3, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
		}

		double error = 0.;
		for (i1 = my_rank * r1; i1 < (my_rank + 1) * r1; ++i1) {
			for (i2 = 0; i2 < N2; ++i2) {
				for (i3 = 0; i3 < N3; ++i3) {
					double diff = std::abs(calcExactSolution(TAU * j, H1 * i1, H2 * i2, H3 * i3) - y[j][i1][i2][i3]);
					if (diff > error) {
						error = diff;
					}
				}
			}
		}
		std::cout << error << '\n';
	}

	auto finish = std::chrono::steady_clock::now();
    auto time_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    std::cout << "Execution time (in milliseconds): " << static_cast<float>(time_in_milliseconds.count()) << '\n';
    if (my_rank == 0) {
        output_file << N1+1 << " " << static_cast<float>(time_in_milliseconds.count()) << '\n';
    }
    output_file.close();

	free(alpha);
	free(beta);
	free(alphaLast);
	free(betaLast);
	free(yLast);
	MPI_Finalize();
	return EXIT_SUCCESS;
}
