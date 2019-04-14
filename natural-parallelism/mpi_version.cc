#include <cmath>
#include <iostream>
#include <fstream>
#include "mpi.h"

#include "common/constants.h"
#include "common/test_functions.h"

/**
 * Allocate contiguous 3d array
 */
double ***alloc3d(int x, int y, int z) {
    double *data = new double [x*y*z];
    double ***array = new double **[x];
    for (int i=0; i<x; i++) {
        array[i] = new double *[y];
        for (int j=0; j<y; j++) {
            array[i][j] = &(data[(i*y+j)*z]);
        }
    }
    return array;
}

double ***alloc_memory_for_first_receive(int myRank, int processRank, int size, int r_last_process, int r, int N) {
    double*** buffer;
    if (myRank == size-1 && processRank == size-1) {
        buffer = alloc3d(r_last_process, r_last_process, N+1);
    } else if (myRank == size-1) {
        buffer = alloc3d(r_last_process, r, N+1);
    } else if (processRank == size-1) {
        buffer = alloc3d(r, r_last_process, N+1);
    } else {
        buffer = alloc3d(r, r, N+1);
    }
    return buffer;
}

void clear_memory_after_first_receive(double*** buffer, int myRank, int processRank, int size, int r_last_process, int r) {
    delete[] buffer[0][0];
    if (myRank == size-1 && processRank == size-1) {
        for (int i = 0; i < r_last_process; ++i) {
            delete [] buffer[i];
        }
    } else if (myRank == size-1) {
        for (int i = 0; i < r_last_process; ++i) {
            delete [] buffer[i];
        }
    } else if (processRank == size-1) {
        for (int i = 0; i < r; ++i) {
            delete [] buffer[i];
        }
    } else {
        for (int i = 0; i < r; ++i) {
            delete [] buffer[i];
        }
    }
    delete [] buffer;
}

double ***alloc_memory_for_second_receive(int myRank, int processRank, int size, int r_last_process, int r, int N) {
    double*** buffer;
    if (myRank == size-1 && processRank == size-1) {
        buffer = alloc3d(r_last_process, r_last_process, N+1);
    } else if (myRank == size-1) {
        buffer = alloc3d(r, r_last_process, N+1);
    } else if (processRank == size-1) {
        buffer = alloc3d(r_last_process, r, N+1);
    } else {
        buffer = alloc3d(r, r, N+1);
    }
    return buffer;
}

void clear_memory_after_second_receive(double*** buffer, int myRank, int processRank, int size, int r_last_process, int r) {
    delete[] buffer[0][0];
    if (myRank == size-1 && processRank == size-1) {
        for (int i = 0; i < r_last_process; ++i) {
            delete [] buffer[i];
        }
    } else if (myRank == size-1) {
        for (int i = 0; i < r; ++i) {
            delete [] buffer[i];
        }
    } else if (processRank == size-1) {
        for (int i = 0; i < r_last_process; ++i) {
            delete [] buffer[i];
        }
    } else {
        for (int i = 0; i < r; ++i) {
            delete [] buffer[i];
        }
    }
    delete [] buffer;
}


// так как сторона куба не всегда кратна кол-ву процессов, то надо посчитать куб какого размера должны прислать текущему процессу
int calculate_receive_count(int N, int myRank, int processRank, int size, int r, int r_last_process) {
    int receive_count = N+1;
    if (myRank == size-1) {
        receive_count *= r_last_process;
    } else {
        receive_count *= r;
    }
    if (processRank == size-1) {
        receive_count *= r_last_process;
    } else {
        receive_count *= r;
    }
    return receive_count;
}

int main(int argc, char* argv[]) {
    std::ofstream outputFile;
    outputFile.open("result.time", std::ios_base::app);
    int myRank, size;
    MPI_Init(&argc, &argv);                    /* starts MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);    /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &size);      /* get number of processes */

    auto start = std::chrono::steady_clock::now();

    int N = 100; // grid size
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }

    double h = consts::l / N;
    int r = (N+1 + size - 1) / size;   // деление с округлением вверх
    int r_last_process = (N+1) - r * (size-1);

    double *** y = alloc3d(N+1, N+1, N+1);
    for (int i = 0 ; i <= N ; i++) {
        for (int j = r * myRank ; j < (myRank == size -1 ? N+1 : (myRank+1) * r); j++) {
            for (int k = 0 ; k <= N ; k++) {
                y[i][j][k] = func::u0({i*h, j*h, k*h});
            }
        }
    }

    double eps = 2 * h * h / consts::t;

    MPI_Request recv_request = MPI_REQUEST_NULL;
    MPI_Status status;
    MPI_Request send_requests[size-1];
    MPI_Status statuses[size-1];

    for (int j = 0 ; j < consts::j0; j++) {
        for (int i = 0 ; i <= N ; i++) {
            for (int k = r * myRank ; k < (myRank == size -1 ? N+1 : (myRank+1) * r) ; k++) {
                y[0][k][i] = func::a0({i*h, k*h}, j*consts::t);
                y[N][k][i] = func::a1({i*h, k*h}, j*consts::t);

                y[i][k][0] = func::c0({i*h, k*h}, j*consts::t);
                y[i][k][N] = func::c1({i*h, k*h}, j*consts::t);
            }
        }

        for (int i = 0 ; i <= N ; i++) {
            for (int k = 0 ; k <= N ; k++) {
                if (myRank == 0) {
                    y[i][0][k] = func::b0({i*h, k*h}, j*consts::t);
                } else if (myRank == size -1) {
                    y[i][N][k] = func::b1({i*h, k*h}, j*consts::t);
                }
            }
        }

        for (int i2 = r * myRank ; i2 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i2++) {
            for (int i3 = 0 ; i3 <= N ; i3++) {
                double ai[N+1], bi[N+1];
                ai[0] = 0;
                bi[0] = func::a0({i2*h, i3*h}, (j+(double)1/3)*consts::t);
                for (int i1 = 1; i1 < N ; ++i1) {
                    ai[i1] = 1 / (2 + eps - ai[i1 - 1]);
                    bi[i1] =
                            ((y[i1 + 1][i2][i3] + y[i1 - 1][i2][i3] + bi[i1 - 1]) +
                             (eps - 2) * y[i1][i2][i3]) /
                            (2 + eps - ai[i1 - 1]);
                }
                y[N][i2][i3] = func::a1({i2*h, i3*h}, (j+(double)1/3)*consts::t);
                for (int i1 = N - 1; i1 >= 0; --i1) {
                    y[i1][i2][i3] = ai[i1] * y[i1 + 1][i2][i3] + bi[i1];
                }
            }
        }

        for (int i = 0 ; i < size ; i++) {
            if (i != myRank) {  // i тот кому будем посылать
                MPI_Datatype subarray_3d;
                int starts[3] = {r*i, myRank*r, 0};
                int subsizes[3] = {
                        i == size-1 ? r_last_process : r,      // если посылаем последнму
                        myRank == size-1 ? r_last_process : r, // если посылает последний
                        N+1};
                int bigsizes[3] = {N+1, N+1, N+1};
                MPI_Type_create_subarray(3, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray_3d);
                MPI_Type_commit(&subarray_3d);
                int send_request_index = i < myRank ? i : i-1;
                MPI_Isend(&(y[0][0][0]), 1, subarray_3d, i, 100500, MPI_COMM_WORLD, &send_requests[send_request_index]);
                MPI_Type_free(&subarray_3d);
            }
        }


        for (int i = 0 ; i < size ; i++) {
            double*** buffer;
            int index = i;
            if (i != myRank) {
                int receive_count = calculate_receive_count(N, myRank, i, size, r, r_last_process);
                double *** buffer = alloc_memory_for_first_receive(myRank, i, size, r_last_process, r, N);

                MPI_Irecv(&(buffer[0][0][0]), receive_count, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
                MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

                for (int i1 = myRank*r ; i1 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i1++) {
                    for (int i2 = index*r ; i2 < (index == size -1 ? N+1 : (index+1) * r) ; i2++) {
                        for (int i3 = 0 ; i3 <= N ; i3++) {
                            y[i1][i2][i3] = buffer[i1-myRank*r][i2-index*r][i3];
                        }
                    }
                }
                clear_memory_after_first_receive(buffer, myRank, i, size, r_last_process, r);
            }
        }
        MPI_Waitall(size-1, send_requests, MPI_STATUS_IGNORE);
/*
        --------------------------------------------
*/
        for (int i1 = r * myRank ; i1 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i1++) {
            for (int i3 = 0 ; i3 <= N ; i3++) {
                double ai[N+1], bi[N+1];
                ai[0] = 0;
                bi[0] = func::b0({i1*h, i3*h}, (j+(double)2/3)*consts::t);
                for (int i2 = 1 ; i2 < N ; ++i2) {
                    ai[i2] = 1 / (2 + eps - ai[i2 - 1]);
                    bi[i2] =
                            ((y[i1][i2+1][i3] + y[i1][i2-1][i3] + bi[i2 - 1]) +
                             (eps - 2) * y[i1][i2][i3]) /
                            (2 + eps - ai[i2 - 1]);
                }
                y[i1][N][i3] = func::b1({i1*h, i3*h}, (j+(double)2/3)*consts::t);
                for (int i2 = N - 1; i2 >= 0; --i2) {
                    y[i1][i2][i3] = ai[i2] * y[i1][i2+1][i3] + bi[i2];
                }
            }
        }
/*
        --------------------------------------------
*/

        for (int i1 = r * myRank ; i1 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i1++) {
            for (int i2 = 0 ; i2 <= N ; i2++) {
                double ai[N+1], bi[N+1];
                ai[0] = 0;
                bi[0] = func::c0({i1*h, i2*h}, (j+(double)3/3)*consts::t);
                for (int i3 = 1; i3 < N ; ++i3) {
                    ai[i3] = 1 / (2 + eps - ai[i3 - 1]);
                    bi[i3] =
                            ((y[i1][i2][i3+1] + y[i1][i2][i3-1] + bi[i3 - 1]) +
                             (eps - 2) * y[i1][i2][i3]) /
                            (2 + eps - ai[i3 - 1]);
                }
                y[i1][i2][N] = func::b1({i1*h, i2*h}, (j+(double)3/3)*consts::t);
                for (int i3 = N - 1; i3 >= 0; --i3) {
                    y[i1][i2][i3] = ai[i3] * y[i1][i2][i3+1] + bi[i3];
                }
            }
        }

        double error = 0;
        for (int i1 = r * myRank ; i1 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i1++) {
            for (int i2 = 0; i2 <= N; ++i2) {
                for (int i3 = 0; i3 <= N; ++i3) {
                    if (abs(func::u({i1*h, i2*h, i3*h}, (j+1)*consts::t) - y[i1][i2][i3]) > error) {
                        error = abs(func::u({i1*h, i2*h, i3*h}, (j+1)*consts::t) - y[i1][i2][i3]);
                    }
                }
            }
        }

        std::cout << '\n' << myRank << ": " << error << '\n';
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0 ; i < size ; i++) {
            if (i != myRank) {
                MPI_Datatype subarray_3d;
                int starts[3] = {myRank*r, r*i, 0};
                int subsizes[3] = {myRank == size-1 ? r_last_process : r, // если посылает последний
                                   i == size-1 ? r_last_process : r,      // если посылаем последнму
                                   N+1};
                int bigsizes[3] = {N+1, N+1, N+1};
                MPI_Type_create_subarray(3, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray_3d);
                MPI_Type_commit(&subarray_3d);
                int send_request_index = i < myRank ? i : i-1;
                MPI_Isend(&(y[0][0][0]), 1, subarray_3d, i, 100500, MPI_COMM_WORLD, &send_requests[send_request_index]);
                MPI_Type_free(&subarray_3d);
            }
        }

        for (int i = 0 ; i < size ; i++) {
            int index = i;
            if (i != myRank) {
                int receive_count = calculate_receive_count(N, myRank, i, size, r, r_last_process);
                double*** buffer = alloc_memory_for_second_receive(myRank, i, size, r_last_process, r ,N);

                MPI_Irecv(&(buffer[0][0][0]), receive_count, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
                MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

                for (int i1 = index*r ; i1 < (index == size -1 ? N+1 : (index+1) * r) ; i1++) {
                    for (int i2 = myRank*r ; i2 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i2++) {
                        for (int i3 = 0 ; i3 <= N ; i3++) {
                            y[i1][i2][i3] = buffer[i1-index*r][i2-myRank*r][i3];
                        }
                    }
                }
                clear_memory_after_second_receive(buffer, myRank, i, size, r_last_process, r);
            }
        }
    }

    auto finish = std::chrono::steady_clock::now();
    auto time_in_milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start);
    std::cout << "Execution time (in milliseconds): " << static_cast<float>(time_in_milliseconds.count()) << '\n';
    if (myRank == 0) {
        outputFile << N+1 << " " << static_cast<float>(time_in_milliseconds.count()) << '\n';
    }
    outputFile.close();
    MPI_Finalize();
    return 0;
}
