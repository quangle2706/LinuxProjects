#include <iostream>
#include "omp.h"
int main() {
    #pragma omp parallel num_threads(4)
    {
        std::cout<<"Hello World\n";
    }
    return 0;
}
