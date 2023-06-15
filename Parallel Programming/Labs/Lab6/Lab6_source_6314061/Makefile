all: collectiveComm1 collectiveComm2 cartesianTopology1 cartesianTopology2 mpiIO

collectiveComm1: collectiveComm1.c
	mpicc collectiveComm1.c -o collectiveComm1

collectiveComm2: collectiveComm2.c
	mpicc collectiveComm2.c -o collectiveComm2

cartesianTopology1: cartesianTopology1.c
	mpicc cartesianTopology1.c -o cartesianTopology1

cartesianTopology2: cartesianTopology2.c
	mpicc cartesianTopology2.c -o cartesianTopology2

mpiIO: mpiIO.c
	mpicc mpiIO.c -o mpiIO

clean:
	rm -f *.o