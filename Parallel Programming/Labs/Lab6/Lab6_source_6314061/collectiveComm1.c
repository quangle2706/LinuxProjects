/*
 * Collective communication
 * 1.1 MPI process 0 initializes a variable to given value, then modifies the variable
 * (for example, by calculating the square of its value) and finally broadcasts it to all the others
 * MPI processes
 *
*/

// 6314061

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

    int rank, size;
    int value;

    double start_time, end_time, elapsed_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime(); // start the timer

    if (rank == 0) {
        // Process 0 initializes the variable to a given value
        value = 10;

        // Calculate the square of the variable
        value = value * value;
    }

    // Broadcast the value to all other processes
    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Process %d received value %d\n", rank, value);

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime(); // end the timer

    elapsed_time = end_time - start_time;

    double maxtime, mintime, avgtime;
    MPI_Reduce(&elapsed_time, &maxtime, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time, &mintime, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
    MPI_Reduce(&elapsed_time, &avgtime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);

    if (rank == 0) {
        avgtime /= size;
        printf("Min: %lf Max: %lf Avg: %lf\n", mintime, maxtime, avgtime);
    }

    MPI_Finalize();
    return 0;
}