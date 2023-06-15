#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <cstdlib>
#include <stdlib.h>
#include <ctime>
#include <time.h>
#include <iostream>

#define N 100
#define BLOCK_SIZE 116
#define NUMBER_BLOCKS 5

__global__ void GPUfindMax(int* a, int* max) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N) {
        for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                if (a[threadIdx.x] < a[threadIdx.x + offset]) {
                    a[threadIdx.x] = a[threadIdx.x + offset];
                }
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        max[blockIdx.x] = a[0];
    }
}

int CPUFindMax(int* a) {
    int max = 0;
    for (int i = 0; i < N; i++) {
        if (max < a[i]) {
            max = a[i];
        }
    }
    return max;
}

int main() {
    cudaEvent_t start, stop;
    float total_time;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int* a = 0;
    a = (int*)malloc(N * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        a[i] = rand() % N;
    }

    int* GPUMaxes = (int*)malloc(NUMBER_BLOCKS * sizeof(int));
    int* d_a, * d_max;
    cudaMalloc((void**)&d_a, N * sizeof(int));
    cudaMalloc((void**)&d_max, NUMBER_BLOCKS * sizeof(int));

    cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(start, 0);
    GPUfindMax<<<NUMBER_BLOCKS, BLOCK_SIZE>>>(d_a, d_max);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&total_time, start, stop);
    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_max);

    int final_max = 0;
    for (int i = 0; i < NUMBER_BLOCKS; i++) {
        if (final_max < GPUMaxes[i]) {
            final_max = GPUMaxes[i];
        }
    }
    cudaEvent_t cpuStart, cpuStop;
    float CPUtotal_time;
    cudaEventCreate(&cpuStart);
    cudaEventCreate(&cpuStop);
    cudaEventRecord(cpuStart, 0);
    int cpuMax = CPUFindMax(a);
    cudaEventRecord(cpuStop, 0);
    cudaEventSynchronize(cpuStop);
    cudaEventElapsedTime(&CPUtotal_time, cpuStart, cpuStop);
    final_max = cpuMax;
    printf("GPU max: %d | GPU time: %f\n", final_max, CPUtotal_time);
    printf("CPU max: %d | CPU time: %f\n", cpuMax, total_time);

    free(a);
    free(GPUMaxes);
}

// /usr/local/cuda-11.8/bin/nvcc findMax.cu -o findMax
// ./findMax