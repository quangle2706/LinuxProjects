#include <iostream>
#include "omp.h"
int main() {
    int numThreads, tid;
    #pragma omp parallel private(tid)
    {
        tid = omp_get_thread_num();
        printf("Hello World from thread = %d\n", tid);

        if (tid == 0) {
            numThreads = omp_get_num_threads();
            printf("Number of threads = %d\n", numThreads);
        }
    }
    return 0;
}
