/**
 * Panther's ID: 6314061
 *
 * Modify test03.c by replacing pthread_rwlock to Pthread mutex and
 * condition variables
 */

#define _MULTI_THREADED
#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include "check.h"

// pthread_mutex_lock(&mlock);
// pthread_mutex_unlock(&mlock);
pthread_rwlock_t rwlock;

// Using mutex
pthread_mutex_t mlock = PTHREAD_MUTEX_INITIALIZER; // a mutex read_write_lock

// Initialize necessary variables
int num_readers_active = 0; // a count of the number of active readers
int num_writers_waiting = 0; // a count pending_writers of pending writers
int writer_active = 0; // 0/1 integer specifying whether a writer is active
pthread_cond_t read_cond = PTHREAD_COND_INITIALIZER; // a condition variable readers_proceed - signaled when reader can proceed
pthread_cond_t write_cond = PTHREAD_COND_INITIALIZER; // a condition variable writer_proceed - signaled when one of the writers can proceed

void *rdlockThread(void *arg) {
    int rc;

    printf("Entered thread, getting read lock\n");
    //rc = pthread_rwlock_rdlock(&rwlock);
    // BEGIN READ:
    rc = pthread_mutex_lock(&mlock);
    while (num_writers_waiting > 0 || writer_active == 1) {
        pthread_cond_wait(&read_cond, &mlock);
    }
    num_readers_active++;
    compResults("pthread_rwlock_rdlock()\n", rc);
    printf("got the rwlock read lock\n");

    sleep(5);


    //rc = pthread_rwlock_unlock(&rwlock);
    rc = pthread_mutex_unlock(&mlock);
    printf("unlock the read lock\n");

    // END READ:
    rc = pthread_mutex_lock(&mlock);
    num_readers_active--;
    if (num_readers_active == 0) {
        pthread_cond_broadcast(&write_cond);
    }
    rc = pthread_mutex_unlock(&mlock);


    compResults("pthread_rwlock_unlock()\n", rc);
    //printf("Secondary thread unlocked\n");
    printf("Next thread unlocked\n");
    return NULL;
}

void *wrlockThread(void *arg) {
    int rc;

    printf("Entered thread, getting write lock\n");
    //rc = pthread_rwlock_wrlock(&rwlock);
    // BEGIN WRITE:
    rc = pthread_mutex_lock(&mlock);
    num_writers_waiting++;
    while (num_readers_active > 0 || writer_active == 1) {
        pthread_cond_wait(&write_cond, &mlock);
    }
    num_writers_waiting--;
    writer_active++; // set writer active to true
    compResults("pthread_rwlock_wrlock()\n", rc);


    //rc = pthread_rwlock_unlock(&rwlock);
    // END WRITE:
    rc = pthread_mutex_unlock(&mlock);
    printf("Got the rwlock write lock, now unlock\n");

    rc = pthread_mutex_lock(&mlock);
    writer_active = 0; // set writer active to false
    pthread_cond_broadcast(&read_cond);

    rc = pthread_mutex_unlock(&mlock);


    compResults("pthread_rwlock_unlock()\n", rc);
    printf("Next thread unlocked\n");
    return NULL;
}

int main(int argc, char **argv) {
    int rc = 0;
    pthread_t thread, thread1;
    pthread_t thread2, thread3;

    printf("Enter test case - %s\n", argv[0]);

    printf("Main, initialize the read write lock\n");
    rc = pthread_rwlock_init(&rwlock, NULL);
    compResults("pthread_rwlock_init()\n", rc);

//    printf("Main, grab a read lock\n");
//    rc = pthread_rwlock_rdlock(&rwlock);
//    compResults("pthread_rwlock_rdlock()\n", rc);
//
//    printf("Main, grab the same read lock again\n");
//    rc = pthread_rwlock_rdlock(&rwlock);
//    compResults("pthread_rwlock_rdlock() second\n", rc);

    printf("Main, create the read lock thread\n");
    rc = pthread_create(&thread, NULL, rdlockThread, NULL);
    compResults("pthread_create\n", rc);

    printf("Main, create the read lock thread\n");
    rc = pthread_create(&thread2, NULL, rdlockThread, NULL);
    compResults("pthread_create\n", rc);

//    printf("Main - unlock the first read lock\n");
//    rc = pthread_rwlock_unlock(&rwlock);
//    compResults("pthread_rwlock_unlock()\n", rc);

    printf("Main, create the write lock thread\n");
    rc = pthread_create(&thread1, NULL, wrlockThread, NULL);
    compResults("pthread_create\n", rc);

    printf("Main, create the write lock thread\n");
    rc = pthread_create(&thread3, NULL, wrlockThread, NULL);
    compResults("pthread_create\n", rc);

    sleep(5);
//    printf("Main - unlock the second read lock\n");
//    rc = pthread_rwlock_unlock(&rwlock);
//    compResults("pthread_rwlock_unlock()\n", rc);

    printf("Main, wait for the threads\n");
    rc = pthread_join(thread, NULL);
    compResults("pthread_join\n", rc);

    rc = pthread_join(thread1, NULL);
    compResults("pthread_join\n", rc);

    rc = pthread_join(thread2, NULL);
    compResults("pthread_join\n", rc);

    rc = pthread_join(thread3, NULL);
    compResults("pthread_join\n", rc);

    rc = pthread_rwlock_destroy(&rwlock);
    compResults("pthread_rwlock_destroy()\n", rc);

    printf("Main completed\n");
    return 0;

}