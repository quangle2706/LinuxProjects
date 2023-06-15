#include <stdio.h>
#include <stdbool.h>
#include <string.h>

/**
 * Panther ID:          6314061
 * Student's Name:      Quang Tuan Le
 * Professor's Name:    Prof. Caryl Rahn
 * Semester:            COP4338 RVC - Fall 2022
 *
 * I affirm that I wrote this program myself without any help from any other people
 * or sources from the internet.
 *
 * Description:
 * The program is for Assignment 1 which is called Assigning Seats - planeseats in an airplane.
 * This program asks user input if they would like to get a seat in first class or economy.
 */

/**
 * The main function operates based on the assignment 1 instruction.
 * Including asking and checking user choices, assigning seats, displaying information,
 * showing boarding pass.
 *
 * @return 0 exit code
 */
int main() {

    printf("WELCOME TO LE AIRLINE!\n");

    const int NUM_SEATS = 12;   // Number of seats
    int planeSeats[NUM_SEATS];  // Seat numbers

    for (int i = 0; i < NUM_SEATS; i++) {
        planeSeats[i] = 0; // initialize with 0 which means the seat is available
    }

    int firstClassIdx = 0;      // 4 (0-3) first seats are assigned for First Class (Start with index 0)
    int economyIdx = 4;         // 8 (4-11) last seats are assigned for Economy (Start with index 4)

    // Menu
    printf("Here is your choice for your flight!\n");
    printf("Please type 1 for \"first class\"\n");
    printf("PLease type 2 for \"economy\"\n");
    printf("Please type 0 to quit\n\n");

    char userType; // Hold user input
    while (true) {

        // Check if all the seats are full
        if (firstClassIdx == 4 && economyIdx == 12) {
            printf("Announcement: All the seats are full for this flight.\n");
            printf("Please come back tomorrow for the next flight!\n");
            printf("Thank you for your coming!\n");
            printf("Enter 0 to QUIT.\n\n");
        }

        printf("Please make your choice: \n");
        scanf(" %c%*c", &userType); // Get user choice, it has space before %c to ignore leading whitespace
        fflush(stdin); // clear the buffer in case user input string instead of char
        switch (userType) {
            case '1': // User picking the first class seat.
                if (firstClassIdx < 4) { // Check if we still have first class seats
                    printf("Thank you for choosing the first class seat.\n");
                    printf("Here is your boarding pass: \n");
                    printf("Your seat is No. %d of the first class.\n", firstClassIdx + 1);
                    printf("---------------------------------------\n");
                    planeSeats[firstClassIdx++] = 1; // assign seat to 1 and increment index by 1
                } else {
                    if (economyIdx < 12) { // In case any economy seat is available
                        printf("I'm sorry! The first class section is full.\n");
                        printf("Would you like a seat in economy section? (YES/NO)\n");
                        char answer[50]; // in case user input too many chars
                        scanf("%s", &answer);
                        fflush(stdin); // clear the buffer
                        if (strcmp(answer, "yes") == 0 || strcmp(answer, "YES") == 0) {
                            printf("Thank you for choosing the economy seat.\n");
                            printf("Here is your boarding pass: \n");
                            printf("Your seat is No. %d of the economy.\n", economyIdx + 1);
                            printf("---------------------------------------\n");
                            planeSeats[economyIdx++] = 1;
                        } else {
                            printf("The next flight will be available tomorrow.\n");
                        }
                    }
                }
                break;
            case '2': // User picking the economy seat.
                if (economyIdx < 12) { // Check if still have economy seats
                    printf("Thank you for choosing the economy seat.\n");
                    printf("Here is your boarding pass: \n");
                    printf("Your seat is No. %d of the economy.\n", economyIdx + 1);
                    printf("---------------------------------------\n");
                    planeSeats[economyIdx++] = 1;
                } else {
                    if (firstClassIdx < 4) { // In case any first class seat available
                        printf("I'm sorry! The economy section is full.\n");
                        printf("Would you like a seat in first class section? YES/NO \n");
                        char answer[50];
                        scanf("%s", &answer);
                        fflush(stdin); // clear the buffer
                        if (strcmp(answer, "yes") == 0 || strcmp(answer, "YES") == 0) {
                            printf("Thank you for choosing the first class seat.\n");
                            printf("Here is your boarding pass: \n");
                            printf("Your seat is No. %d of the first class.\n", firstClassIdx + 1);
                            printf("---------------------------------------\n");
                            planeSeats[firstClassIdx++] = 1;
                        } else {
                            printf("The next flight will be available tomorrow.\n");
                        }
                    }
                }
                break;
            case '0': // User would like to quit
                printf("Thank you for choosing our flight!\nHave a great day!\n");
                return 0; // Exit
            default: // User input is not valid
                printf("Your choice is not valid! Try again!\n");
                break;
        }
        printf("\n");
    }

    return 0; // Exit the program
}
