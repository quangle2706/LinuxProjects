// ~/TA/openmpi-4.1.4/bin/mpicc MPI_simple.c -o MPI_simple
// ./MPI_simple
// ~/TA/openmpi-4.1.4/bin/mpirun -np 4 MPI_simple
// ... it will run with np = 4

#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int numprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &namelen);

    printf("Process %d on %s out of %d\n", rank, processor_name, numprocs);
    MPI_Finalize();
}