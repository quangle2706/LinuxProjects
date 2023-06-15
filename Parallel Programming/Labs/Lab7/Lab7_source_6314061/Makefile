all: findMax

findMax: findMax.cu
	/usr/local/cuda-11.8/bin/nvcc findMax.cu -o findMax

clean:
	rm -f *.o