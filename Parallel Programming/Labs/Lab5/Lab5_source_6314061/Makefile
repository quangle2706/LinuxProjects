all: hello MPI_ring

hello: MPI_simple.c
	mpicc MPI_simple.c -o hello

MPI_ring: MPI_ring.c
	mpicc MPI_ring.c -o MPI_ring

clean:
	rm -f *.o