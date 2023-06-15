// 6314061

// Task 2: Cartesian topology

#include <stdio.h>
#include <mpi.h>
#include <math.h>

#define SHIFT_ROW 0
#define SHIFT_COL 1
#define DISP 1

int main(int argc, char** argv) {
    int size, rank;
    MPI_Comm comm_cart;
    int ndims = 2;
    int dims[2] = {4, 4};
    int periods[2] = {1, 1}; // set true periodic for 2-dimensional, 0 - false, 1 - true
    int reorder = 0; // default 0 or 1 (arbitrary rank if necessary)
    int coords[2];
    int my_cart_rank;

    // Declare our neighbors
    enum DIRECTIONS {DOWN, UP, LEFT, RIGHT};
    char* neighbors_names[4] = {"down", "up", "left", "right"};
    int neighbors_ranks[4];

    /* start up initial MPI environment */
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Create cartesian topology for processes
    // create cartesian mapping
    int ierr = 0;
    ierr = MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &comm_cart);
    if (ierr != 0) {
        printf("ERROR[%d] creating CART\n", ierr);
    }

    // find my coordinates in the cartesian communicator group
    MPI_Cart_coords(comm_cart, rank, ndims, coords);

    // use my cartesian coordinates to find my rank in cartesian group
    MPI_Cart_rank(comm_cart, coords, &my_cart_rank);

    // get my neighbors; axis is coordinate dimension of shift
    // axis=0 ~ row shifting
    // axis=1 ~ col shifting
    MPI_Cart_shift(comm_cart, SHIFT_ROW, DISP, &neighbors_ranks[LEFT], &neighbors_ranks[RIGHT]);
    MPI_Cart_shift(comm_cart, SHIFT_COL, DISP, &neighbors_ranks[DOWN], &neighbors_ranks[UP]);
    printf("PW[%d] Coord(%d,%d) with local rank: %d has NEIGHBORS including %d, %d, %d, %d \n",
           rank, coords[0], coords[1], my_cart_rank, neighbors_ranks[LEFT], neighbors_ranks[RIGHT],
           neighbors_ranks[DOWN], neighbors_ranks[UP]);

    // calculate the average
    float sum_ranks = my_cart_rank + neighbors_ranks[LEFT] + neighbors_ranks[RIGHT] +
            neighbors_ranks[DOWN] + neighbors_ranks[UP];
    printf("With the average between its local rank and the local rank from each of its neighbors is %.2f\n", sum_ranks / 5.0);


    fflush(stdout);

    MPI_Comm_free(&comm_cart);

    MPI_Finalize();
    return 0;
}