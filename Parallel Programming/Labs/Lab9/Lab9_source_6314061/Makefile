all: dm

dm: denseMatrix.c
	~openmpi-4.1.4/bin/mpicc denseMatrix.c -o dm -lm

clean:
	rm -f *.o