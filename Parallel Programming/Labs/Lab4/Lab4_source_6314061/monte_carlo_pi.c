#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#define MAX_THREADS 512
void *compute_pi(void *);

int total_hits, total_misses, hits[MAX_THREADS], sample_points, sample_points_per_thread, num_threads;

int main() {
   int i;
   pthread_t p_threads[MAX_THREADS];
   pthread_attr_t attr;
   double computed_pi;
   double time_start, time_end;
   struct timeval tv;
   // struct timezone tz;

   pthread_attr_init(&attr);
   pthread_attr_setscope(&attr, PTHREAD_SCOPE_SYSTEM);
   printf("Enter number of sample points: ");
   scanf("%d", &sample_points);
   printf("Enter number of pthreads: ");
   scanf("%d", &num_threads);

   gettimeofday(&tv, NULL);
   time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

   total_hits = 0;
   sample_points_per_thread = sample_points / num_threads;
   for (i = 0; i < num_threads; i++) {
       hits[i] = i;
       pthread_create(&p_threads[i], &attr, compute_pi, (void *)&hits[i]);
   }
   for (i = 0; i < num_threads; i++) {
       pthread_join(p_threads[i], NULL); // main thread wait subroutine thread finish
       total_hits += hits[i];
   }
   printf("total hits: %d\n", total_hits);
   computed_pi = 4 * (double)total_hits / ((double)(sample_points));
   gettimeofday(&tv, NULL);
   time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
   printf("Computed PI = %lf\n", computed_pi);
   printf(" %lf\n", time_end - time_start);

   return 0;
}

void *compute_pi(void *s) {
    int seed, i, *hit_pointer;
    double rand_no_x, rand_no_y;
    int local_hits;

    hit_pointer = (int *)s;
    seed = *hit_pointer;
    local_hits = 0;
    for (i = 0; i < sample_points_per_thread; i++) {
        rand_no_x = (double)(rand_r(&seed)) / (RAND_MAX) - 0.5; /// (double)((2 << 14) - 1);
        rand_no_y = (double)(rand_r(&seed)) / (RAND_MAX) - 0.5; /// (double)((2 << 14) - 1);
        if (((rand_no_x) * (rand_no_x) + (rand_no_y) * (rand_no_y)) < 0.25)
            local_hits++;
        seed *= i;
    }
    *hit_pointer = local_hits;
    pthread_exit(0);
}