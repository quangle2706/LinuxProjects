// Task 3: MPI I/O

// 6314061

/**
* Hints: Use MPI_File_Open, MPI_File_write, MPI_File_seek, MPI_File_get_size,
 * MPI_File_read_at, MPI_File_set_view, MPI_File_write_all
*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define FILENAME "io.bin"
#define NUM_ELEMENTS 10

int main(int argc, char** argv) {
    int rank, size, i;
    int* data;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Allocate memory for data
    data = (int*)malloc(NUM_ELEMENTS * sizeof(int));

    // Create the file with the appropriate size
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    MPI_File_close(&file);

    // Write data to file
    MPI_Offset offset = rank * NUM_ELEMENTS * sizeof(int);
    MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    MPI_File_seek(file, offset, MPI_SEEK_SET);
    for (i = 0; i < NUM_ELEMENTS; i++) {
        data[i] = i + rank * NUM_ELEMENTS;
    }

    MPI_File_write(file, data, NUM_ELEMENTS, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&file);

    // Reopen the file to read data
    MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_RDONLY, MPI_INFO_NULL, &file);
    MPI_File_seek(file, offset, MPI_SEEK_SET);
    MPI_File_read_at(file, offset, data, NUM_ELEMENTS, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_close(&file);

    // Check if the read data is correct
    for (i = 0; i < NUM_ELEMENTS; i++) {
        if (data[i] != rank * NUM_ELEMENTS + i) {
            printf("Rank %d: Error reading data at index %d\n", rank, i);
            break;
        }
    }

    // Write the read data
    MPI_Status status;
    MPI_Datatype ftype; // use data type replication

    // Write 2 ints of each processor -> N/2 blocks
    MPI_Type_vector(NUM_ELEMENTS / 2, 2, size * 2, MPI_INT, &ftype);
    MPI_Type_commit(&ftype);

    MPI_File_open(MPI_COMM_WORLD, FILENAME, MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    MPI_File_set_view(file, rank * 2 * sizeof(int), MPI_INT, ftype, "native", MPI_INFO_NULL); // offset -> 0

    MPI_File_write_all(file, data, NUM_ELEMENTS, MPI_INT, &status); // MPI_STATUS_IGNORE -> &status
    MPI_File_close(&file);

    // Free memory and finalize MPI
    free(data);
    MPI_Finalize();
    return 0;
}

// Hint from Lab instruction
//MPI_File_open(MPI_COMM_WORLD, "io.bin", MPI_MODE_RDWR, MPI_INFO_NULL, &file);
//for (int i = 0; i < numInt / 2; i++) {
//    int *ToWrite = readbuffer * 2 * i;
//    offset = sizeof(int) * (2 * rank * (numprocs * i * 2));
//    MPI_File_write_at(file, offset, ToWrite, 2, MPI_INT, &stat);
//}
//
//MPI_File_close(&file);