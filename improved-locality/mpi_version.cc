/* TODO: change MPI_Send and MPI_Recv to non blocking calls; reduce memory usage
 *
 * ACHTUNG: assume N % pcnt =  0
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>

#include "common/constants.h"
#include "common/structures.h"
#include "common/test_functions.h"

constexpr int N = 100;
constexpr double H = 0.01;

constexpr int T = 100;
constexpr int Q2 = 30;
constexpr int Q3 = 30;
constexpr double TAU = 0.01;

double y[T][N][N][N] = {{{{0.}}}};
double a0[N][N][T];
double a1[N][N][T];
double b0[N][N][T];
double b1[N][N][T];
double c0[N][N][T];
double c1[N][N][T];

int i, i1, i2, i3, j;

void setBorderConditions() {
	for (i1 = 0; i1 < N; ++i1) {
		for (i2 = 0; i2 < N; ++i2) {
			for (j = 0; j < T; ++j) {
				a0[i1][i2][j] = func::a0({i1 * H, i2 * H}, (j + 1.0 / 3.0) * TAU);
				a1[i1][i2][j] = func::a1({i1 * H, i2 * H}, (j + 1.0 / 3.0) * TAU);
				b0[i1][i2][j] = func::b0({i1 * H, i2 * H}, (j + 2.0 / 3.0) * TAU);
				b1[i1][i2][j] = func::b1({i1 * H, i2 * H}, (j + 2.0 / 3.0) * TAU);
				c0[i1][i2][j] = func::c0({i1 * H, i2 * H}, (j + 1.0) * TAU);
				c1[i1][i2][j] = func::c1({i1 * H, i2 * H}, (j + 1.0) * TAU);
			}
		}
	}
}

void setInitialApproximation() {
	for (i1 = 0; i1 < N; ++i1) {
		for (i2 = 0; i2 < N; ++i2) {
			for (i3 = 0; i3 < N; ++i3) {
				y[0][i1][i2][i3] = func::u0({i1 * H, i2 * H, i3 * H});
			}
		}
	}
}

double tempY[2][N][N][N];
double alphaArr[N][N][N];
double betaArr[N][N][N];

int main(int argc, char* argv[]) {
	std::ofstream output_file;
    output_file.open("result.time", std::ios_base::app);

    MPI_Init(&argc, &argv);
	int my_rank, pcnt;
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &pcnt);

	auto start = std::chrono::steady_clock::now();

	const int r1 = N / pcnt; // should be N % pcnt = 0
	const int r2 = (N + Q2 - 1) / Q2;
	const int r3 = (N + Q3 - 1) / Q3;

	setBorderConditions();
	setInitialApproximation();

	double eps = 2 * H * H / TAU;

	constexpr int double_size = sizeof(eps);
	tensor1d alpha = tensor1d(N);
	tensor1d beta = tensor1d(N);

	const size_t splitSize = r2 * r3;
	tensor1d alphaLast = tensor1d(splitSize);
	tensor1d betaLast = tensor1d(splitSize);
	tensor1d yLast = tensor1d(splitSize);

	for (int j = 0; j < T - 1; ++j) {
		for (int q2 = 0; q2 < Q2; q2++) {
			for (int q3 = 0; q3 < Q3; q3++) {
				if (my_rank) {
					// TODO: change to non blocking calls
					MPI_Recv(alphaLast.data(), splitSize, MPI_DOUBLE, my_rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(betaLast.data(), splitSize, MPI_DOUBLE, my_rank - 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}

				for (int i2 = r2 * q2; i2 < std::min(r2 * (q2 + 1), N); ++i2) {
					for (int i3 = r3 * q3; i3 < std::min(r3 * (q3 + 1), N); ++i3) {
						int lastIdx = (i2 - r2 * q2) * r2 + i3 - r3 * q3;
						if (my_rank == 0) {
							alphaLast[lastIdx] = 0.;
							betaLast[lastIdx] = a0[i2][i3][j];
						}

						const int startIdx = my_rank * r1 + 1;
						const int stopIdx = std::min((my_rank + 1) * r1 + 1, N - 1);
						alphaArr[startIdx - 1][i2][i3] = alphaLast[lastIdx];
						betaArr[startIdx - 1][i2][i3] = betaLast[lastIdx];
						for (i = startIdx; i < stopIdx; ++i) {
							alphaArr[i][i2][i3] = 1 / (2 + eps - alphaArr[i - 1][i2][i3]);
							betaArr[i][i2][i3] =
								((y[j][i + 1][i2][i3] + y[j][i - 1][i2][i3] + betaArr[i - 1][i2][i3]) +
								(eps - 2) * y[j][i][i2][i3]) /
									(2 + eps - alphaArr[i - 1][i2][i3]);
						}
						alphaLast[lastIdx] = alphaArr[stopIdx - 1][i2][i3];
						betaLast[lastIdx] = betaArr[stopIdx - 1][i2][i3];
					}
				}
				if (my_rank != pcnt - 1) {
					// TODO: change to non blocking calls
					MPI_Send(alphaLast.data(), splitSize, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
					MPI_Send(betaLast.data(), splitSize, MPI_DOUBLE, my_rank + 1, 0, MPI_COMM_WORLD);
				}
			}
		}
		for (int q2 = 0; q2 < Q2; q2++) {
			for (int q3 = 0; q3 < Q3; q3++) {
				if (my_rank != pcnt - 1) {
					// TODO: change to non blocking calls
					MPI_Recv(yLast.data(), splitSize, MPI_DOUBLE, my_rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
				}
				for (i2 = r2 * q2; i2 < std::min(r2 * (q2 + 1), N); ++i2) {
					for (i3 = r3 * q3; i3 < std::min(r3 * (q3 + 1), N); ++i3) {
						int lastIdx = (i2 - r2 * q2) * r2 + i3 - r3 * q3;
						int startIdx = (my_rank + 1) * r1 - 1;
						if (my_rank == pcnt - 1) {
							yLast[lastIdx] = a1[i2][i3][j];
							startIdx = N - 2;
						}
						const int stopIdx = my_rank * r1;
						tempY[0][startIdx + 1][i2][i3] = yLast[lastIdx];
						for (i = startIdx; i >= stopIdx; --i) {
							tempY[0][i][i2][i3] = alphaArr[i][i2][i3] * tempY[0][i + 1][i2][i3] + betaArr[i][i2][i3];
						}
						yLast[lastIdx] = tempY[0][stopIdx][i2][i3];
					}
				}
				if (my_rank) {
					// TODO: change to non blocking calls
					MPI_Send(yLast.data(), splitSize, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
				}
			}
		}

		for (i1 = r1 * my_rank; i1 < r1 * (my_rank + 1); ++i1) {
			for (i3 = 0; i3 < N; ++i3) {
				alpha[0] = 0;
				beta[0] = b0[i1][i3][j];
				for (i = 1; i < N - 1; ++i) {
					alpha[i] = 1 / (2 + eps - alpha[i - 1]);
					beta[i] =
						((tempY[0][i1][i + 1][i3] + tempY[0][i1][i - 1][i3] + beta[i - 1]) +
						(eps - 2) * tempY[0][i1][i][i3]) /
							(2 + eps - alpha[i - 1]);
				}
				tempY[1][i1][N - 1][i3] = b1[i1][i3][j];
				for (i = N - 2; i >= 0; --i) {
					tempY[1][i1][i][i3] = alpha[i] * tempY[1][i1][i + 1][i3] + beta[i];
				}
			}
		}

		for (i1 = r1 * my_rank; i1 < r1 * (my_rank + 1); ++i1) {
			for (i2 = 0; i2 < N; ++i2) {
				alpha[0] = 0;
				beta[0] = c0[i1][i2][j];
				for (i = 1; i < N - 1; ++i) {
					alpha[i] = 1 / (2 + eps - alpha[i - 1]);
					beta[i] =
						((tempY[1][i1][i2][i + 1] + tempY[1][i1][i2][i - 1] + beta[i - 1]) +
						(eps - 2) * tempY[1][i1][i2][i]) /
							(2 + eps - alpha[i - 1]);
				}
				y[j + 1][i1][i2][N - 1] = c1[i1][i2][j];
				for (i = N - 2; i >= 0; --i) {
					y[j + 1][i1][i2][i] = alpha[i] * y[j + 1][i1][i2][i + 1] + beta[i];
				}
			}
		}

		// TODO: change to non blocking calls
		if (my_rank != pcnt - 1) {
			MPI_Recv(y[j + 1][(my_rank + 1) * r1 + 1], N * N, MPI_DOUBLE, my_rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			MPI_Recv(y[j + 1][(my_rank + 1) * r1], N * N, MPI_DOUBLE, my_rank + 1, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		}
		if (my_rank) {
			MPI_Send(y[j + 1][my_rank * r1 + 1], N * N, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
			MPI_Send(y[j + 1][my_rank * r1], N * N, MPI_DOUBLE, my_rank - 1, 0, MPI_COMM_WORLD);
		}

		double error = 0.0;
		for (i1 = my_rank * r1; i1 < (my_rank + 1) * r1; ++i1) {
			for (i2 = 0; i2 < N; ++i2) {
				for (i3 = 0; i3 < N; ++i3) {
					error = std::max(error, std::abs(func::u({H * i1, H * i2, H * i3}, TAU * j) - y[j][i1][i2][i3]));
				}
			}
		}
		std::cout << my_rank << ": " << error << '\n';
	}

	auto finish = std::chrono::steady_clock::now();
    auto time_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    std::cout << "Execution time (in milliseconds): " << static_cast<float>(time_in_milliseconds.count()) << '\n';
    if (my_rank == 0) {
        output_file << N+1 << " " << static_cast<float>(time_in_milliseconds.count()) << '\n';
    }
    output_file.close();
	MPI_Finalize();
	return 0;
}
