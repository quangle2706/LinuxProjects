#include <stdio.h>
#include <mpi.h>
#include <sys/time.h>

int main(int argc, char** argv) {

    double time_start, time_end, elapsed_time;
    struct timeval tv;

    // Initialize the MPI environment
    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Measure time
    gettimeofday(&tv, NULL);
    time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d"
           " out of %d processors\n",
           processor_name, world_rank, world_size);

    int token;
    if (world_rank != 0) {
        MPI_Recv(&token, 1, MPI_INT, world_rank - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token %d from process %d\n",
               world_rank, token, world_rank - 1);
    } else {
        // Set the token's value if you are process 0
        token = 112;
    }

    MPI_Send(&token, 1, MPI_INT, (world_rank + 1) % world_size,
             0, MPI_COMM_WORLD);
    // Now process 0 can receive from the last process.
    if (world_rank == 0) {
        MPI_Recv(&token, 1, MPI_INT, world_size - 1, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d received token %d from process %d\n",
               world_rank, token, world_size - 1);
    }

    gettimeofday(&tv, NULL);
    time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
    elapsed_time = time_end - time_start;

    double total_time;
    MPI_Reduce(&elapsed_time, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD); // All to one to get sum

    if (world_rank == 0) {
        printf("Elapsed time = %f seconds\n", total_time/world_size);
    }

    // Finalize the MPI environment
    MPI_Finalize();

    return 0;
}