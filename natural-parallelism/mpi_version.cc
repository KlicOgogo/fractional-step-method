#include <cmath>
#include <iostream>
#include <fstream>
#include "mpi.h"

#include "common/constants.h"
#include "common/structures.h"
#include "common/test_functions.h"

/**
 * Allocate contiguous 3d array
 */
buffer3D alloc3d(int x, int y, int z) {
    return buffer3D({x, y, z});
}

buffer3D first_recv_alloc(int my_rank, int process_rank, int size, int r_last, int r, int N) {
    int size_x = (my_rank == size-1) ? r_last : r;
    int size_y = (process_rank == size-1) ? r_last : r;
    return alloc3d(size_x, size_y, N+1);
}

buffer3D second_recv_alloc(int my_rank, int process_rank, int size, int r_last, int r, int N) {
    int size_x = (process_rank == size-1) ? r_last : r;
    int size_y = (my_rank == size-1) ? r_last : r;
    return alloc3d(size_x, size_y, N+1);
}

// calculating size of memory to receive
int calculate_receive_count(int N, int my_rank, int process_rank, int size, int r, int r_last) {
    int first_multiplier = (my_rank == size-1) ? r_last : r;
    int second_multiplier = (process_rank == size-1) ? r_last : r;
    return first_multiplier * second_multiplier * (N+1);
}

int main(int argc, char* argv[]) {
    std::ofstream output_file;
    output_file.open("result.time", std::ios_base::app);
    int my_rank, size;
    MPI_Init(&argc, &argv);                    /* starts MPI */
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);    /* get current process id */
    MPI_Comm_size(MPI_COMM_WORLD, &size);      /* get number of processes */

    auto start = std::chrono::steady_clock::now();

    int N = 100; // grid size
    if (argc == 2) {
        N = std::atoi(argv[1]);
    }

    const double h = consts::l / N; // grid step
    int r = (N + 1 + size - 1) / size;
    int r_last = (N + 1) - r * (size - 1);

    buffer3D y = alloc3d(N+1, N+1, N+1);
    double x1_curr, x2_curr, x3_curr;
    for (int i = 0; i <= N; i++) {
        x1_curr = i * h;
        for (int j = r * my_rank; j < (my_rank == size -1 ? N+1 : (my_rank+1) * r); j++) {
            x2_curr = j * h;
            for (int k = 0; k <= N; k++) {
                x3_curr = k * h;
                y.get(i, j, k) = func::u0({x1_curr, x2_curr, x3_curr});
            }
        }
    }

    double eps = 2 * h * h / consts::t;

    MPI_Request recv_request = MPI_REQUEST_NULL;
    MPI_Status status;
    MPI_Request send_requests[size-1];
    MPI_Status statuses[size-1];

    double error = 0;
    double t_curr;
    for (int j = 0; j < consts::j0; ++j) {
        t_curr = consts::t * j;
        double row_curr, col_curr;
        for (int i = 0; i <= N; ++i) {
            row_curr = i * h;
            for (int k = r * my_rank; k < (my_rank == size -1 ? N+1 : (my_rank+1) * r); ++k) {
                col_curr = k * h;
                y.get(0, k, i) = func::a0({row_curr, col_curr}, t_curr);
                y.get(N, k, i) = func::a1({row_curr, col_curr}, t_curr);
                y.get(i, k, 0) = func::c0({row_curr, col_curr}, t_curr);
                y.get(i, k, N) = func::c1({row_curr, col_curr}, t_curr);
            }
        }

        for (int i = 0; i <= N; ++i) {
            row_curr = i * h;
            for (int k = 0; k <= N; ++k) {
                col_curr = k * h;
                if (my_rank == 0) {
                    y.get(i, 0, k) = func::b0({row_curr, col_curr}, t_curr);
                } else if (my_rank == size -1) {
                    y.get(i, N, k) = func::b1({row_curr, col_curr}, t_curr);
                }
            }
        }

        t_curr = (1.0 / 3 + j) * consts::t;
        for (int i2 = r * my_rank; i2 < (my_rank == size -1 ? N+1 : (my_rank+1) * r); ++i2) {
            x2_curr = i2 * h;
            for (int i3 = 0; i3 <= N; ++i3) {
                x3_curr = i3 * h;
                double ai[N+1], bi[N+1];
                ai[0] = 0;
                bi[0] = func::a0({x2_curr, x3_curr}, t_curr);
                for (int i1 = 1; i1 < N; ++i1) {
                    ai[i1] = 1.0 / (2.0 + eps - ai[i1-1]);
                    bi[i1] = (y.get(i1+1, i2, i3) + y.get(i1-1, i2 ,i3) + bi[i1-1] + (eps - 2.0) * y.get(i1, i2, i3)) * ai[i1];
                }
                y.get(N, i2, i3) = func::a1({x2_curr, x3_curr}, t_curr);
                for (int i1 = N - 1; i1 >= 0; --i1) {
                    y.get(i1, i2, i3) = ai[i1] * y.get(i1+1, i2, i3) + bi[i1];
                }
            }
        }

        for (int i = 0; i < size; ++i) {
            if (i != my_rank) {  // i тот кому будем посылать
                MPI_Datatype subarray_3d;
                int starts[3] = {r*i, my_rank*r, 0};
                int subsizes[3] = {
                        i == size-1 ? r_last : r,      // если посылаем последнму
                        my_rank == size-1 ? r_last : r, // если посылает последний
                        N+1};
                int bigsizes[3] = {N+1, N+1, N+1};
                MPI_Type_create_subarray(3, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray_3d);
                MPI_Type_commit(&subarray_3d);
                int send_request_index = i < my_rank ? i : i-1;
                MPI_Isend(&(y.get(0, 0, 0)), 1, subarray_3d, i, 100500, MPI_COMM_WORLD, &send_requests[send_request_index]);
                MPI_Type_free(&subarray_3d);
            }
        }


        for (int i = 0; i < size; ++i) {
            int index = i;
            if (i != my_rank) {
                int receive_count = calculate_receive_count(N, my_rank, i, size, r, r_last);
                buffer3D buffer = first_recv_alloc(my_rank, i, size, r_last, r, N);

                MPI_Irecv(&(buffer.get(0, 0, 0)), receive_count, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
                MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

                for (int i1 = my_rank*r; i1 < (my_rank == size -1 ? N+1 : (my_rank+1) * r); ++i1) {
                    for (int i2 = index*r; i2 < (index == size -1 ? N+1 : (index+1) * r); ++i2) {
                        for (int i3 = 0; i3 <= N; ++i3) {
                            y.get(i1, i2, i3) = buffer.get(i1-my_rank*r, i2-index*r, i3);
                        }
                    }
                }
            }
        }
        MPI_Waitall(size-1, send_requests, MPI_STATUS_IGNORE);
/*
        --------------------------------------------
*/
        t_curr = (2.0 / 3 + j) * consts::t;
        for (int i1 = r * my_rank; i1 < (my_rank == size -1 ? N+1 : (my_rank+1) * r); ++i1) {
            x1_curr = i1 * h;
            for (int i3 = 0; i3 <= N; ++i3) {
                x3_curr = i3 * h;
                double ai[N+1], bi[N+1];
                ai[0] = 0;
                bi[0] = func::b0({x1_curr, x3_curr}, t_curr);
                for (int i2 = 1; i2 < N; ++i2) {
                    ai[i2] = 1.0 / (2.0 + eps - ai[i2-1]);
                    bi[i2] = (y.get(i1, i2+1, i3) + y.get(i1, i2-1, i3) + bi[i2-1] + (eps - 2.0) * y.get(i1, i2, i3)) * ai[i2];
                }
                y.get(i1, N, i3) = func::b1({x1_curr, x3_curr}, t_curr);
                for (int i2 = N - 1; i2 >= 0; --i2) {
                    y.get(i1, i2, i3) = ai[i2] * y.get(i1, i2+1, i3) + bi[i2];
                }
            }
        }
/*
        --------------------------------------------
*/
        t_curr = (1.0 + j) * consts::t;
        for (int i1 = r * my_rank; i1 < (my_rank == size -1 ? N+1 : (my_rank+1) * r); ++i1) {
            x1_curr = i1 * h;
            for (int i2 = 0; i2 <= N; ++i2) {
                x2_curr = i2 * h;
                double ai[N+1], bi[N+1];
                ai[0] = 0;
                bi[0] = func::c0({x1_curr, x2_curr}, t_curr);
                for (int i3 = 1; i3 < N; ++i3) {
                    ai[i3] = 1.0 / (2.0 + eps - ai[i3 - 1]);
                    bi[i3] = (y.get(i1, i2, i3+1) + y.get(i1, i2, i3-1) + bi[i3-1] + (eps - 2) * y.get(i1, i2, i3)) * ai[i3];
                }
                y.get(i1, i2, N) = func::b1({x1_curr, x2_curr}, t_curr);
                for (int i3 = N - 1; i3 >= 0; --i3) {
                    y.get(i1, i2, i3) = ai[i3] * y.get(i1, i2, i3+1) + bi[i3];
                }
            }
        }

        error = 0;
        t_curr = (j + 1) * consts::t;
        for (int i1 = r * my_rank; i1 < (my_rank == size -1 ? N+1 : (my_rank+1) * r); ++i1) {
            x1_curr = i1 * h;
            for (int i2 = 0; i2 <= N; ++i2) {
                x2_curr = i2 * h;
                for (int i3 = 0; i3 <= N; ++i3) {
                    x3_curr = i3 * h;
                    if (std::abs(func::u({x1_curr, x2_curr, x3_curr}, t_curr) - y.get(i1, i2, i3)) > error) {
                        error = std::abs(func::u({x1_curr, x2_curr, x3_curr}, t_curr) - y.get(i1, i2, i3));
                    }
                }
            }
        }

        std::cout << my_rank << ": " << error << '\n';
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < size; ++i) {
            if (i != my_rank) {
                MPI_Datatype subarray_3d;
                int starts[3] = {my_rank*r, r*i, 0};
                int subsizes[3] = {my_rank == size-1 ? r_last : r, // если посылает последний
                                   i == size-1 ? r_last : r,      // если посылаем последнму
                                   N+1};
                int bigsizes[3] = {N+1, N+1, N+1};
                MPI_Type_create_subarray(3, bigsizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &subarray_3d);
                MPI_Type_commit(&subarray_3d);
                int send_request_index = i < my_rank ? i : i-1;
                MPI_Isend(&(y.get(0, 0, 0)), 1, subarray_3d, i, 100500, MPI_COMM_WORLD, &send_requests[send_request_index]);
                MPI_Type_free(&subarray_3d);
            }
        }

        for (int i = 0; i < size; ++i) {
            int index = i;
            if (i != my_rank) {
                int receive_count = calculate_receive_count(N, my_rank, i, size, r, r_last);
                buffer3D buffer = second_recv_alloc(my_rank, i, size, r_last, r, N);

                MPI_Irecv(&(buffer.get(0, 0, 0)), receive_count, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
                MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

                for (int i1 = index*r; i1 < (index == size -1 ? N+1 : (index+1) * r); ++i1) {
                    for (int i2 = my_rank*r; i2 < (my_rank == size -1 ? N+1 : (my_rank+1) * r); ++i2) {
                        for (int i3 = 0; i3 <= N; ++i3) {
                            y.get(i1, i2, i3) = buffer.get(i1-index*r, i2-my_rank*r, i3);
                        }
                    }
                }
            }
        }
    } // consts::j0

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
