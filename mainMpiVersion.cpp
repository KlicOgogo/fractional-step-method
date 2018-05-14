#include <iostream>
#include "startConditions.h"
#include <math.h>
#include <ctime>
#include "/home/alexandr/magistratura/mpi/installationDir/include/mpi.h"

using namespace std;

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

int main(int argc, char **argv) {
    int myRank, size;

    MPI_Init (&argc, &argv);                    /* starts MPI */
    MPI_Comm_rank (MPI_COMM_WORLD, &myRank);    /* get current process id */
    MPI_Comm_size (MPI_COMM_WORLD, &size);      /* get number of processes */

    double begin = MPI_Wtime();
    double T = 1;
    int j0 = 100;
    double t = T / j0;

    double l = 1;
    int N = 599;//фактичски N будет на 1 больше. Так сделано для удобства индексации, чтобы можно было обращаться по индексу N (y[N][i2][i3])
    double h = l / N;
    int r = (N+1 + size - 1) / size;   // деление с округлением вверх
    int r_last_process = (N+1) - r * (size-1);

    double *** y = alloc3d(N+1, N+1, N+1);
    for (int i = 0 ; i <= N ; i++) {
        for (int j = r * myRank ; j < (myRank == size -1 ? N+1 : (myRank+1) * r); j++) {
            for (int k = 0 ; k <= N ; k++) {
                y[i][j][k] = u0(i*h, j*h, k*h);
            }
        }
    }

    double epsilon = 2 * h * h / t;

    MPI_Request recv_request = MPI_REQUEST_NULL;
    MPI_Status status;
    MPI_Request send_requests[size-1];
    MPI_Status statuses[size-1];

    for (int j = 0 ; j < j0; j++) {
        for (int i = 0 ; i <= N ; i++) {
            for (int k = r * myRank ; k < (myRank == size -1 ? N+1 : (myRank+1) * r) ; k++) {
                y[0][k][i] = a0(i*h, k*h, j*t);
                y[N][k][i] = a1(l, i*h, k*h, j*t);

                y[i][k][0] = c0(i*h, k*h, j*t);
                y[i][k][N] = c1(l, i*h, k*h, j*t);
            }
        }

        for (int i = 0 ; i <= N ; i++) {
            for (int k = 0 ; k <= N ; k++) {
                if (myRank == 0) {
                    y[i][0][k] = b0(i*h, k*h, j*t);
                } else if (myRank == size -1) {
                    y[i][N][k] = b1(l, i*h, k*h, j*t);
                }
            }
        }

        for (int i2 = r * myRank ; i2 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i2++) {
            for (int i3 = 0 ; i3 <= N ; i3++) {
                double * ai = new double[N+1];
                double * bi = new double[N+1];

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
            double*** buffer1;
            int index = i;
            if (i != myRank) {
                int recieve_count = N+1;
                if (myRank == size-1) {
                    recieve_count *= r_last_process;
                } else {
                    recieve_count *= r;
                }
                if (i == size-1) {
                    recieve_count *= r_last_process;
                } else {
                    recieve_count *= r;
                }
                if (myRank == size-1 && i == size-1) {
                    buffer1 = alloc3d(r_last_process, r_last_process, N+1);
                } else if (myRank == size-1) {
                    buffer1 = alloc3d(r_last_process, r, N+1);
                } else if (i == size-1) {
                    buffer1 = alloc3d(r, r_last_process, N+1);
                } else {
                    buffer1 = alloc3d(r, r, N+1);
                }


                //MPI_Irecv(&(buffer1[0][0][0]), r * r * (N + 1), MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
                MPI_Irecv(&(buffer1[0][0][0]), recieve_count, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
                //MPI_Wait(&send_requests[0], MPI_STATUS_IGNORE);
                MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
//                int send_request_index = i < myRank ? i : i-1;
//                MPI_Wait(&send_requests[send_request_index], MPI_STATUS_IGNORE);

                for (int i1 = myRank*r ; i1 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i1++) {
                    for (int i2 = index*r ; i2 < (index == size -1 ? N+1 : (index+1) * r) ; i2++) {
                        for (int i3 = 0 ; i3 <= N ; i3++) {
                            y[i1][i2][i3] = buffer1[i1-myRank*r][i2-index*r][i3];
                        }
                    }
                }
            }
        }
        MPI_Waitall(size-1, send_requests, MPI_STATUS_IGNORE);
/*
        --------------------------------------------
*/
        for (int i1 = r * myRank ; i1 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i1++) {
            for (int i3 = 0 ; i3 <= N ; i3++) {
                double * ai = new double[N+1];
                double * bi = new double[N+1];

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

        for (int i1 = r * myRank ; i1 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i1++) {
            for (int i2 = 0 ; i2 <= N ; i2++) {

                double * ai = new double[N+1];
                double * bi = new double[N+1];

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
/*
      Проверка точности приближённого решения
*/
        double maxDifference = 0;
        for (int i1 = r * myRank ; i1 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i1++)
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

        cout << endl << myRank << ": " << maxDifference << endl;
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
                //MPI_Type_free(&subarray_3d);
            }
        }

        for (int i = 0 ; i < size ; i++) {
            int index = i;
            if (i != myRank) {
                double*** buffer1;
                int recieve_count = N+1;
                if (myRank == size-1) {
                    recieve_count *= r_last_process;
                } else {
                    recieve_count *= r;
                }
                if (i == size-1) {
                    recieve_count *= r_last_process;
                } else {
                    recieve_count *= r;
                }
                if (myRank == size-1 && i == size-1) {
                    buffer1 = alloc3d(r_last_process, r_last_process, N+1);
                } else if (myRank == size-1) {
                    buffer1 = alloc3d(r, r_last_process, N+1);
                } else if (i == size-1) {
                    buffer1 = alloc3d(r_last_process, r, N+1);
                } else {
                    buffer1 = alloc3d(r, r, N+1);
                }

                MPI_Irecv(&(buffer1[0][0][0]), recieve_count, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
                MPI_Wait(&recv_request, MPI_STATUS_IGNORE);

                for (int i1 = index*r ; i1 < (index == size -1 ? N+1 : (index+1) * r) ; i1++) {
                    for (int i2 = myRank*r ; i2 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i2++) {
                        for (int i3 = 0 ; i3 <= N ; i3++) {
                            y[i1][i2][i3] = buffer1[i1-index*r][i2-myRank*r][i3];
                        }
                    }
                }

//                for (int i1 = index*r ; i1 < (index == size -1 ? N+1 : (index+1) * r) ; i1++) {
//                    for (int i2 = myRank*r ; i2 < (myRank == size -1 ? N+1 : (myRank+1) * r) ; i2++) {
//                        delete [] buffer1[i][j];
//                    }
//                    delete [] buffer1[i];
//                }
//                delete [] buffer1;
            }
        }
    }

    double end = MPI_Wtime();
    double elapsed_secs = double(end - begin);
    cout << "Time: " << elapsed_secs << endl;

    MPI_Finalize();
    return 0;
}