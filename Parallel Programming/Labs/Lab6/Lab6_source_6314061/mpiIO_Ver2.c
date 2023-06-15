#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <sys/time.h>

#define numInt 10

int main(int argc, char *argv[]) {
    int numprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    MPI_File file;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &namelen);
    MPI_Status stat;
    int offset = sizeof (int) * numInt * rank;
    int writebuffer[numInt] = {0};
    int readbuffer[numInt] = {0};
    for (int i = 0; i < numInt; i++) {
        writebuffer[i] = i + (rank * numInt);
    }

    MPI_File_open(MPI_COMM_WORLD, "io.bin", MPI_MODE_RDWR, MPI_INFO_NULL, &file);
    MPI_File_write_at(file, offset, writebuffer, numInt, MPI_INT, &stat);
    MPI_File_close(&file);

    MPI_File_open(MPI_COMM_WORLD, "io.bin", MPI_MODE_RDWR, MPI_INFO_NULL, &file);
    MPI_File_read_at(file, offset, readbuffer, numInt, MPI_INT, &stat);
    MPI_File_close(&file);

    printf("Check if read/write was OK\n");
    for (int i = 0; i < numInt; i++) {
        printf("I am: %d, I read: %d and wrote: %d\n", rank, readbuffer[i], writebuffer[i]);
    }

    MPI_File_open(MPI_COMM_WORLD, "io.bin", MPI_MODE_RDWR, MPI_INFO_NULL, &file);
    for (int i = 0; i < numInt / 2; i++) {
        int *ToWrite = readbuffer + 2 * i;
        offset = sizeof (int) * (2 * rank + (numprocs * i * 2));
        MPI_File_write_at(file, offset, ToWrite, 2, MPI_INT, &stat);
    }

    MPI_File_close(&file);
    MPI_Finalize();
    return 0;
}