#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

#define N 4
#define BLOCKSIZE 2
#define MAXVAL 10
#define DEBUG 1

void printMatrix(int a[N][N]);

int main(int argc, char* argv[]) {
    int numprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int nprocs = (N*N)/(BLOCKSIZE*BLOCKSIZE);
    int sr = sqrt(nprocs);
    int n = N;
    int blocksize = BLOCKSIZE;
    int maxval = MAXVAL;
    int debug = DEBUG;
    int indxC, indxR;
    // Create matrices:
    int inA[n][n], inB[n][n], outSerial[n][n], outParallel[n][n];
    // Initialize with random data
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inA[i][j] = rand() % maxval;
            srand(rand()); // reseed to get new random number for inB
            inB[i][j] = rand() % maxval;
        }
    }

    // MPI INIT:
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Status status;

    // Let the below code run only once, do this by designating the processor with rank 0 to run this:
    if (rank == 0) {
        if (debug) {
            printf("A: \n");
            printMatrix(inA);
            printf("B: \n");
            printMatrix(inB);
        }

        // Serial Matrix multiply:
        double starttimeSerial, endtimeSerial;
        starttimeSerial = MPI_Wtime();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                outSerial[i][j] = 0;
                for (int k = 0; k < n; k++) {
                    outSerial[i][j] = outSerial[i][j] + inA[i][k] * inB[k][j];
                }
            }
        }
        endtimeSerial = MPI_Wtime();
        if (debug) {
            printf("Serial output: \n");
            printMatrix(outSerial);
        }
        printf("SERIAL TIME: %f\n", endtimeSerial - starttimeSerial);
    }
    // END SERIAL SECTION

    // Begin parallel:
    double starttimeParallel, endtimeParallel;
    starttimeParallel = MPI_Wtime();
    // Broadcast blocks of matrix A and B:
    MPI_Bcast(inA, n*n/numprocs, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(inB, n*n/numprocs, MPI_INT, 0, MPI_COMM_WORLD);

    if (debug) {
        if (1) {
            printf("RANK: %d\n", rank);
            printf("block A:\n");

            for (int i = (rank/sr)*blocksize; i < (rank/sr)*blocksize + blocksize; i++) {
                for (int j = (rank*blocksize) % n; j < (rank*blocksize) % n + blocksize; j++) {
                    printf("%d, ", inA[i][j]);
                }
                printf("\n");
            }
            printf("block B:\n");
            for (int i = (rank/sr)*blocksize; i < (rank/sr)*blocksize + blocksize; i++) {
                for (int j = (rank*blocksize) % n; j < (rank*blocksize) % n + blocksize; j++) {
                    printf("%d, ", inB[i][j]);
                }
                printf("\n");
            }
        }
    }

    // Perform matrix mult:
    for (int i = rank*n/numprocs; i < n; i++) {
        for (int j = 0; j < n; j++) {
            outParallel[i][j] = 0;
            for (int k = 0; k < n; k++) {
                outParallel[i][j] += inA[i][k] * inB[k][j];
            }
        }
    }

    // Gather results to single proc:
    MPI_Gather(outParallel[rank*n/numprocs], n*n/numprocs, MPI_INT, outParallel, n*n/numprocs, MPI_INT, 0, MPI_COMM_WORLD);
    endtimeParallel = MPI_Wtime();
    if (rank == 0) {
        if (debug) {
            printf("Parallel matrix: \n");
            printMatrix(outParallel);
        }
        printf("Parallel time: %f\n", endtimeParallel - starttimeParallel);
    }

    MPI_Finalize();
}

void printMatrix(int a[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d, ", a[i][j]);
        }
        printf("\n");
    }
}