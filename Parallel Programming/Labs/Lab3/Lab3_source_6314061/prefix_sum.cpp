/**
 * Quang Le - 6314061
 */


#include <iostream>
#include <omp.h>
#include <sys/time.h>

void printArray(int* arr, int n) {
    for (int i = 0; i < n; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main() {

    int n = 4096;
    int* arr = (int*)malloc(sizeof (int) * n);
    int* X = (int*) malloc(sizeof (int) * n);

    // generate random n elements of arr
    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 10;
        X[i] = arr[i];
    }

    //printf("The init array: \n");
    //printArray(arr, n);

    int num_threads = 4; //omp_get_max_threads();
    int chunk_size = n / num_threads;


    struct timeval start_time, end_time;
    int ret = gettimeofday(&start_time, NULL);
    if (ret != -1) {
        printf("%d\n", start_time.tv_usec);
    }

    // Step 1: split arr among threads, every thread computes its own (partial) prefix sum
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        int thread_id = omp_get_thread_num();
        //printf("Thread id: %d and index i: %d\n", thread_id, i);
        int chunk_start = thread_id * chunk_size;
        int chunk_end = (thread_id + 1) * chunk_size;
        if (thread_id == num_threads - 1) {
            chunk_end = n;
        }

        if (i > chunk_start && i < chunk_end)
            X[i] += X[i - 1];
    }

    ret = gettimeofday(&end_time, NULL);
    if (ret != -1) {
        printf("%d\n", end_time.tv_usec);
    }

    printf("Step 1 - Time elapsed: %d\n", (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec));

    //printArray(arr, n);

    ret = gettimeofday(&start_time, NULL);
    if (ret != -1) {
        printf("%d\n", start_time.tv_usec);
    }

    // Step 2: create array T, perform simple prefix sum on T
    int* T = (int*) malloc(sizeof (int)*chunk_size);
    T[0] = 0;
    for (int i = 1; i < num_threads; i++) {
        T[i] = T[i - 1] + X[i * chunk_size - 1];
    }

    ret = gettimeofday(&end_time, NULL);
    if (ret != -1) {
        printf("%d\n", end_time.tv_usec);
    }

    printf("Step 2 - Time elapsed: %d\n", (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec));

    //printArray(T, num_threads);

    ret = gettimeofday(&start_time, NULL);
    if (ret != -1) {
        printf("%d\n", start_time.tv_usec);
    }

    // Step 3: every thread adds T[threadid] to all its element
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++) {
        int thread_id = omp_get_thread_num();
        X[i] += T[thread_id];
    }

    free(T);

    ret = gettimeofday(&end_time, NULL);
    if (ret != -1) {
        printf("%d\n", end_time.tv_usec);
    }

    printf("Step 3 - Time elapsed: %d\n", (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec));

    ret = gettimeofday(&start_time, NULL);
    if (ret != -1) {
        printf("%d\n", start_time.tv_usec);
    }

    // Step 4: finished, we rewrote prefix sum by removing dependencies
#pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i++)
        arr[i] = X[i];
    free(X);

    ret = gettimeofday(&end_time, NULL);
    if (ret != -1) {
        printf("%d\n", end_time.tv_usec);
    }

    printf("Step 4 - Time elapsed: %d\n", (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec));

//    ret = gettimeofday(&end_time, NULL);
//    if (ret != -1) {
//        printf("%d\n", end_time.tv_usec);
//    }
//
//    printf("Time elapsed: %d\n", (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec));

    return 0;
}