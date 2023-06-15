// Matrix vector multiplication in Parallel
// Busher Bridi University of California, Merced 2021

#include <iostream>
#include <sys/time.h>
#include <omp.h>

int main() {
    // Get input
    int rows, cols, max_range;
    std::cout << "Enter rows: \n";
    std::cin >> rows;
    std::cout << "Enter columns: \n";
    std::cin >> cols;
    std::cout << "Enter max range for number generation: \n";
    std::cin >> max_range;

    // Generate matrix with input dimensions and random numbers:
    int matrix[rows][cols];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % max_range;
        }
    }

    // Generate vector with compatible dimensions and random numbers:
    int vec[cols];
    for (int i = 0; i < cols; i++) {
        vec[i] = rand() % max_range;
    }

    std::cout << "Starting Computation: \n";
    // Generate resultant matrix to hold result:
    int result[cols];
    struct timeval start, end;
    gettimeofday(&start, NULL);
    //#pragma omp parallel for
    #pragma omp parallel for num_threads(4)
    //#pragma omp for schedule(static, 4)
    //#pragma omp for schedule(guided, 4)
    //#pragma omp for schedule(dynamic, 4)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[i] += vec[j] * matrix[i][j];
        }
    }
    gettimeofday(&end, NULL);
    std::cout << "time: " << (end.tv_usec - start.tv_usec) << " microseconds" << std::endl;
    // DEBUG PRINT:
    // Print matrix:
//    for (int i = 0; i < rows; i++) {
//        for (int j = 0; j < cols; j++) {
//            std::cout << matrix[i][j];
//            std::cout << " | ";
//        }
//        std::cout << std::endl;
//    }
    return 0;
}
