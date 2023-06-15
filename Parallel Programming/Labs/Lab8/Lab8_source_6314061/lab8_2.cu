#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <cstdlib>
#include <time.h>
#include <chrono>
#include <sys/time.h>
//#include <windows.h>

using namespace std;

#define ROW 20
#define COL 20
#define LOOPS 2
#define VERBOSE 1
#define NUM_THREAD 100

void PrintGrid(int* grid) {
    for (int i = 0; i < ROW; i++) {
        for (int j = 0; j < COL; j++) {
            printf("%d ", grid[i * COL + j]);
        }
        printf("\n");
    }
}

void SerialGameOfLife(int* grid) {
    // make copy of grid for output:
    int* result = (int*)malloc(ROW * COL * sizeof(int));
    // copy grid to result:
    for (long long i = 0; i < (ROW * COL); i++) {
        result[i] = grid[i];
    }

    for (long long i = 0; i < ROW; i++) {
        for (long long j = 0; j < COL; j++) {
            int cell = grid[i + j * ROW];
            // check if cell is at edge:
            if (i == 0) { // UPPER EDGE
                // check if cell is at corner:
                if (i == 0 && j == 0) { // UPPER LEFT
                    int numFriends = grid[i * COL + (j + 1)] + grid[(i + 1) * COL + (j + 1)] + grid[(i + 1) * COL + j];
                    if (cell == 1) {
                        if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                            result[i * COL + j] = 0;
                        }
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                } else if (i == 0 && j == COL - 1) { // UPPER RIGHT
                    int numFriends = grid[i * COL + (j - 1)] + grid[(i + 1) * COL + (j - 1)] + grid[(i + 1) * COL + j];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                } else {
                    int numFriends = grid[i * COL + (j - 1)] + grid[(i + 1) * COL + (j - 1)] + grid[(i + 1) * COL + j] + grid[(i + 1) * COL + (j + 1)] + grid[i * COL + (j + 1)];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                }
            } else if (i == ROW - 1) { // LOWER EDGE
                // check if cell is at corner:
                if (i == ROW - 1 && j == 0) { // LOWER LEFT
                    int numFriends = grid[i * COL + (j + 1)] + grid[(i - 1) * COL + (j + 1)] + grid[(i - 1) * COL + j];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                } else if (i == ROW - 1 && j == COL - 1) { // LOWER RIGHT
                    int numFriends = grid[i * COL + (j - 1)] + grid[(i - 1) * COL + (j - 1)] + grid[(i - 1) * COL + j];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                } else {
                    int numFriends = grid[i * COL + (j - 1)] + grid[(i - 1) * COL + (j - 1)] + grid[(i - 1) * COL + j] + grid[(i - 1) * COL + (j + 1)] + grid[i * COL + (j + 1)];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                }
            } else if (j == 0) { // LEFT EDGE
                // check if cell is at corner:
                if (i == 0 && j == 0) { // UPPER LEFT
                    int numFriends = grid[i * COL + (j + 1)] + grid[(i + 1) * COL + (j + 1)] + grid[(i + 1) * COL + j];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                } else if (i == ROW - 1 && j == 0) { // LOWER LEFT
                    int numFriends = grid[i * COL + (j + 1)] + grid[(i - 1) * COL + (j + 1)] + grid[(i - 1) * COL + j];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                } else {
                    int numFriends = grid[i * COL + (j + 1)] + grid[(i - 1) * COL + (j + 1)] + grid[(i - 1) * COL + j] + grid[(i + 1) * COL + (j + 1)] + grid[(i + 1) * COL + j];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                }
            } else if (j == COL - 1) { // RIGHT EDGE
                // check if cell is at corner:
                if (i == 0 && j == COL - 1) { // UPPER RIGHT
                    int numFriends = grid[i * COL + (j - 1)] + grid[(i + 1) * COL + (j - 1)] + grid[(i + 1) * COL + j];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                } else if (i == ROW - 1 && j == COL - 1) { // LOWER RIGHT
                    int numFriends = grid[i * COL + (j - 1)] + grid[(i - 1) * ROW + (j - 1)] + grid[(i - 1) * ROW + j];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                } else {
                    int numFriends = grid[i * COL + (j - 1)] + grid[(i - 1) * COL + (j - 1)] + grid[(i - 1) * COL + j] + grid[(i + 1) * COL + (j - 1)] + grid[(i + 1) * COL + j];
                    if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                        result[i * COL + j] = 0;
                    } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                        result[i * COL + j] = 1;
                    }
                }
            } else { // Generic cell
                int numFriends = grid[(i * COL) + j - 1] + grid[((i - 1) * COL) + j - 1] + grid[((i - 1) * COL) + j] + grid[((i - 1) * COL) + j + 1] + grid[((i) * COL) + j + 1] + grid[((i + 1) * COL) + j + 1] + grid[((i + 1) * COL) + j] + grid[((i + 1) * COL) + j - 1];
                if (numFriends < 2 || numFriends >= 4) { // cell dies from loneliness or overpopulation
                    result[i * COL + j] = 0;
                } else if (numFriends == 2 || numFriends == 3) { // cell becomes alive due to having 3/2 neighbors
                    result[i * COL + j] = 1;
                }
            }
        }
    }
    // copy result back into grid
    for (int i = 0; i < (ROW * COL); i++) {
        grid[i] = result[i];
    }
}

__global__ void GPUGameOfLife(int* grid) {
    __shared__ int s_arr[100];
    int neighbor = 0;

    int r = threadIdx.x % ROW;
    int c = threadIdx.x / ROW;
    int cell = grid[r + c * COL];
    int row = r - 1;
    int col = c - 1;

    s_arr[r + c * ROW] = grid[r + c * ROW];
    __syncthreads();
    for (int i = row; i < row + 3; i++) {
        for (int j = col; j < col + 3; j++) {
            if (i >= 0 && j >= 0) {
                if (i < ROW && j < COL) {
                    if (i != r || j != c) {
                        if (s_arr[i + j * ROW] == 1) {
                            neighbor += 1;
                        }
                    }
                }
            }
        }
    }
    if (cell == 1 && neighbor <= 1) {
        cell = 0;
    } else if (cell == 0 && neighbor == 2) {
        cell = 1;
    } else if (cell == 0 && neighbor == 3) {
        cell = 1;
    } else if (cell == 1 && neighbor >= 4) {
        cell = 0;
    }
    __syncthreads();
    grid[r + c * ROW] = cell;
}


//int main(int argc, char** argv) {

//    if (argc != 4) {
//        printf("Your command line is not valid. Please input like this.\n");
//        printf("Usage: %s numRow numCol numLoops\n", argv[0]);
//        return 1;
//    }
//
//    int numRow = atoi(argv[1]);
//    int numCol = atoi(argv[2]);
//    int numLoops = atoi(argv[3]);
int main() {
    //struct timeval serial_start, serial_end;
    //struct timeval gpu_start, gpu_end;
    srand(time(NULL));
    int *grid;
    int *d_grid;
    int *gpu_output;
    grid = (int *) malloc(ROW * COL * sizeof(int));
    gpu_output = (int *) malloc(ROW * COL * sizeof(int));
    cudaMalloc((void **) &d_grid, ROW * COL * sizeof(int));

    // populate grid with random data
    for (long long i = 0; i < (ROW * COL); i++) {
        grid[i] = rand() % 2;
    }
    if (VERBOSE == 1) {
        printf("INITIAL STATE: \n");
        PrintGrid(grid);
    }

    // copy to gpu:
    cudaMemcpy(d_grid, grid, ROW * COL * sizeof(int), cudaMemcpyHostToDevice);

    //gettimeofday(&gpu_start, NULL);
    auto gpu_start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < LOOPS; i++) {
        GPUGameOfLife <<<1, NUM_THREAD>>>(d_grid);
    }

    cudaMemcpy(gpu_output, d_grid, ROW * COL * sizeof(int), cudaMemcpyDeviceToHost);
    //gettimeofday(&gpu_end, NULL);
    auto gpu_end_time = chrono::high_resolution_clock::now();
    if (VERBOSE) {
        printf("GPU: \n");
        printf("-------------\n");
        PrintGrid(gpu_output);
    }
    //gettimeofday(&serial_start, NULL);
    auto serial_start_time = chrono::high_resolution_clock::now();
    for (long long i = 0; i < LOOPS; i++) {
        SerialGameOfLife(grid);
        if (VERBOSE) {
            printf("CPU: \n");
            printf("-------------\n");
            PrintGrid(grid);
        }
    }
    //gettimeofday(&serial_end, NULL);
    auto serial_end_time = chrono::high_resolution_clock::now();

    if (VERBOSE == 1) {
        printf("FINAL STATE AFTER %d ITERATIONS: \n", LOOPS);
        PrintGrid(grid);
    }
    printf("ROWS: %d, COLS: %d, LOOPS: %d, NUM_THREADS: %d\n", ROW, COL, LOOPS, NUM_THREAD);
//    if (1 == 1) {
//        printf("endtime: %d, starttime: %d\n", serial_end.tv_sec, serial_start.tv_sec);
//    }
//    printf("SERIAL TIME (ms): %d\n", (serial_end.tv_sec - serial_start.tv_sec) * 1e6 + (serial_end.tv_usec - serial_start.tv_usec));
//    if (1 == 1) {
//        printf("endtime: %d, starttime: %d\n", gpu_end.tv_sec, gpu_start.tv_sec);
//    }
//    printf("GPU TIME (ms): %d\n", (gpu_end.tv_sec - gpu_start.tv_sec) * 1e6 + (gpu_end.tv_usec - gpu_start.tv_usec));

    auto gpu_time_diff = chrono::duration_cast<chrono::microseconds>(gpu_end_time - gpu_start_time);
    printf("GPU execution time: %d (ms)\n", gpu_time_diff.count());

    auto serial_time_diff = chrono::duration_cast<chrono::microseconds>(serial_end_time - serial_start_time);
    printf("Serial execution time: %d (ms)\n", serial_time_diff.count());

    cudaFree(d_grid);
    free(gpu_output);
    free(grid);
    return 0;
}