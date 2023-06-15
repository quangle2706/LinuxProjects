// 6314061

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 16

int main(int argc, char** argv) {
    int rank, size;
    int arr[ARRAY_SIZE];
    int* chunk = NULL;
    int chunk_size;
    int* updated_chunk = NULL;
    int i;

    double start_time, end_time, elapsed_time;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Barrier(MPI_COMM_WORLD);
    start_time = MPI_Wtime(); // start the timer

    if (rank == 0) {
        // Process 0 initializes the array with the index of each element
        for (i = 0; i < ARRAY_SIZE; i++) {
            arr[i] = i;
        }
    }

    // Divide the array into equal-sized chunks and scatter them to other processes
    chunk_size = ARRAY_SIZE / size;
    chunk = malloc(sizeof(int) * chunk_size);

    // Scatter array chunks to every processes
    MPI_Scatter(arr, chunk_size, MPI_INT, chunk, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Update the chunk with the rank of the receiving process
    updated_chunk = malloc(sizeof(int) * chunk_size);
    for (i = 0; i < chunk_size; i++) {
        updated_chunk[i] = chunk[i] + rank; // add its rank value to value of array relatively
    }

    // Gather the updated chunks back to process 0
    MPI_Gather(updated_chunk, chunk_size, MPI_INT, arr, chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) { // The root after receiving all updated value => print value of new array
        // Print the updated array
        printf("The updated array is:\n");
        for (i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    }

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