// 6314061

// Task 2: Cartesian topology

#include <stdio.h>
#include <mpi.h>

int main(int argc, char** argv) {
    int size, rank;
    MPI_Comm comm_cart;
    int ndims = 2;
    int dims[2] = {4, 4}; // 2d cartesian topology 4x4
    int periods[2] = {0, 0}; // no periodic for 2-dimensional, 0 - false, 1 - true
    int reorder = 0; // default 0 or 1
    int coords[2];
    int my_cart_rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ierr = 0;
    ierr = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);
    if (ierr != 0) {
        printf("ERROR[%d] creating CART\n", ierr);
    }

    MPI_Cart_coords(comm_cart, rank, ndims, coords);
    MPI_Cart_rank(comm_cart, coords, &my_cart_rank);
    printf("Rank %d in MPI_COMM_WORLD has local rank %d in the Cartesian communicator (%d, %d)\n",
           rank, my_cart_rank, coords[0], coords[1]);

    MPI_Comm_free(&comm_cart);

    MPI_Finalize();
    return 0;
}


