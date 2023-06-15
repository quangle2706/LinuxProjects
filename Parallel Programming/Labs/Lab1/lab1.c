#include <stdio.h>
#include <sys/time.h>

/**
 * Panther's ID:        6314061
 * Semester:            Spring 2023
 * Professor's Name:    Wenqian Dong
 * This is the code for Lab 1 assignment - COP 4520
 */

/**
 * The function to test the gettimeofday function of C
 * to measure performance of a code block.
 * @param COUNT
 */
void myLoop(int COUNT) {
    struct timeval start_time, end_time;
    int ret = gettimeofday(&start_time, NULL);
    if (ret != -1) {
        printf("%d\n", start_time.tv_usec);
    }

    int i, sum;
    for (i = 0; i < COUNT; i++) {
        sum += i;
        //printf("%d,", sum);
    }

    ret = gettimeofday(&end_time, NULL);
    if (ret != -1) {
        printf("%d\n", end_time.tv_usec);
    }

    // when input (COUNT) too large the tv_usec time may out of range and it gives negative elapsed time
    // so this is another print for those cases.
    printf("Time elapsed: %d\n", (end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_usec - start_time.tv_usec));
    printf("Time elapsed Ver2: %d\n", (end_time.tv_usec - start_time.tv_usec));
}

int main() {
    int COUNT = 1000;
    //myLoop(COUNT);
    for (int i = 0; i < 10; i++) {
        myLoop(COUNT);
    }
}
