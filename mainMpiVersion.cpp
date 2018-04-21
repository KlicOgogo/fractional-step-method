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

    int previous_process_rank;
    int next_process_rank;
    for (int i = 0 ; i < size ; i++) {
        previous_process_rank = myRank == 0 ? MPI_PROC_NULL : myRank-1;
        next_process_rank = myRank == size-1 ? MPI_PROC_NULL : myRank+1;
    }

    clock_t begin = clock();
    double T = 1;
    int j0 = 100;
    double t = T / j0;

    double l = 1;
    int N = 99;       //фактичски N будет 100. Так сделано для удобства индексирования
    double h = l / N;
    int r = (N+1) / size;

    double *** y = alloc3d(N+1, N+1, N+1);
    for (int i = 0 ; i <= N ; i++) {
        for (int j = 0 + r * myRank ; j < (myRank+1) * r; j++) {
            for (int k = 0 ; k <= N ; k++) {
                y[i][j][k] = u0(i*h, j*h, k*h);
            }
        }
    }

    double epsilon = 2 * h * h / t;

    MPI_Request recv_request = MPI_REQUEST_NULL;
    MPI_Request send_request = MPI_REQUEST_NULL;


    for (int j = 0 ; j < j0; j++) {
        for (int i = 0 ; i <= N ; i++) {
            for (int k = 0 + r * myRank ; k < (myRank+1) * r ; k++) {
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

        for (int i2 = 0 + r * myRank ; i2 < (myRank+1) * r ; i2++) {
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

        if (myRank == 0) {
            MPI_Datatype mysubarray1;
            int starts1[3] = {r, 0, 0};
            int subsizes1[3] = {r, r, N+1};
            int bigsizes1[3] = {N+1, N+1, N+1};
            MPI_Type_create_subarray(3, bigsizes1, subsizes1, starts1, MPI_ORDER_C, MPI_DOUBLE, &mysubarray1);
            MPI_Type_commit(&mysubarray1);
            MPI_Isend(&(y[0][0][0]), 1, mysubarray1, 1, 100500, MPI_COMM_WORLD, &send_request);

        } else {
            MPI_Datatype mysubarray1;
            int starts1[3] = {0, r, 0};
            int subsizes1[3] = {r, r, N+1};
            int bigsizes1[3] = {N+1, N+1, N+1};
            MPI_Type_create_subarray(3, bigsizes1, subsizes1, starts1, MPI_ORDER_C, MPI_DOUBLE, &mysubarray1);
            MPI_Type_commit(&mysubarray1);
            MPI_Isend(&(y[0][0][0]), 1, mysubarray1, 0, 100500, MPI_COMM_WORLD, &send_request);
        }

        double*** buffer1 = alloc3d(r, r, N+1);
        if (myRank == 0) {

            MPI_Irecv(&(buffer1[0][0][0]), r*r*(N+1), MPI_DOUBLE, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            for (int i1 = 0 ; i1 < r ; i1++) {
                for (int i2 = r ; i2 <= N ; i2++) {
                    for (int i3 = 0 ; i3 <= N ; i3++) {
                        y[i1][i2][i3] = buffer1[i1][i2-r][i3];
                    }
                }
            }
        } else {
            MPI_Irecv(&(buffer1[0][0][0]), r*r*(N+1), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            for (int i1 = r ; i1 <= N ; i1++) {
                for (int i2 = 0 ; i2 < r; i2++) {
                    for (int i3 = 0 ; i3 <= N ; i3++) {
                        y[i1][i2][i3] = buffer1[i1-r][i2][i3];
                    }
                }
            }
        }


/*
        --------------------------------------------
*/
        for (int i1 = 0 + r * myRank ; i1 < (myRank+1) * r ; i1++) {
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

        for (int i1 = 0 + r * myRank ; i1 < (myRank+1) * r ; i1++) {
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
//
//
//
        double maxDifference = 0;
        for (int i1 = 0 + r * myRank ; i1 < (myRank+1) * r ; i1++)
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

        if (myRank == 0) {
            MPI_Datatype mysubarray1;
            int starts1[3] = {0, r, 0};
            int subsizes1[3] = {r, r, N+1};
            int bigsizes1[3] = {N+1, N+1, N+1};
            MPI_Type_create_subarray(3, bigsizes1, subsizes1, starts1, MPI_ORDER_C, MPI_DOUBLE, &mysubarray1);
            MPI_Type_commit(&mysubarray1);
            MPI_Isend(&(y[0][0][0]), 1, mysubarray1, 1, 100500, MPI_COMM_WORLD, &send_request);
        } else {
            MPI_Datatype mysubarray1;
            int starts1[3] = {r, 0, 0};
            int subsizes1[3] = {r, r, N+1};
            int bigsizes1[3] = {N+1, N+1, N+1};
            MPI_Type_create_subarray(3, bigsizes1, subsizes1, starts1, MPI_ORDER_C, MPI_DOUBLE, &mysubarray1);
            MPI_Type_commit(&mysubarray1);
            MPI_Isend(&(y[0][0][0]), 1, mysubarray1, 0, 100500, MPI_COMM_WORLD, &send_request);
        }

        if (myRank == 0) {
            MPI_Irecv(&(buffer1[0][0][0]), r*r*(N+1), MPI_DOUBLE, 1, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            for (int i1 = r ; i1 <= N ; i1++) {
                for (int i2 = 0 ; i2 < r; i2++) {
                    for (int i3 = 0 ; i3 <= N ; i3++) {
                        y[i1][i2][i3] = buffer1[i1-r][i2][i3];
                    }
                }
            }
        } else {
            MPI_Irecv(&(buffer1[0][0][0]), r*r*(N+1), MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &recv_request);
            MPI_Wait(&send_request, MPI_STATUS_IGNORE);
            MPI_Wait(&recv_request, MPI_STATUS_IGNORE);
            for (int i1 = 0 ; i1 < r ; i1++) {
                for (int i2 = r ; i2 <= N ; i2++) {
                    for (int i3 = 0 ; i3 <= N ; i3++) {
                        y[i1][i2][i3] = buffer1[i1][i2-r][i3];
                    }
                }
            }
        }
    }

    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    cout << "Time: " << elapsed_secs << endl;

    MPI_Finalize();
    return 0;
}