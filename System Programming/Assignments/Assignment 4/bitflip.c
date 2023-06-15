/**
 * Panther ID:          6314061
 * Student's Name:      Quang Tuan Le
 * Professor's Name:    Prof. Caryl Rahn
 * Semester:            COP4338 RVC - Fall 2022
 *
 * Assignment #4
 *
 * I affirm that I wrote this program myself without any help from any other people
 * or sources from the internet.
 *
 * Description:
 * This program is for the assignment 4 which is to task the user uses the command line:
 * usage: bitflip [-e] [-f] [-a] [-s] [-o outputfilename] intval
 * From the above command line arguments, this program reads the unsigned integer 16 bits
 * in range between 1 and 20000 inclusive on the command line as well as the desired options
 * to execute this number which are -e option is to flip all even bits, -f option is to flip
 * all odd bits, -a option is to flip all bits, -s option is to switch right and left bits,
 * -o option is to print the output to output file instead of to the screen.
 * This program is to help to work with bitwise operator, getopt function, and file execution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>

int debug = 0;

/**
 * This function flips all even bits of the 16-bit unsigned integer input value
 * and return the result
 * @param inputValue to be flipped all even bits
 * @return uint16_t from input value flipped all even bits
 */
uint16_t flipAllEvenBits(uint16_t inputValue) {
    // Iterate through even bit positions of the 16-bit number
    for (int i = 0; i < 16; i += 2) {
        // Using XOR to flip bit at specific position
        // 1 ^ 0 = 1 and 1 ^ 1 = 0
        // Based on this formula, we shift bit 1 of 1 to left
        // that corresponds to the position of the bit to be flipped
        inputValue = inputValue ^ (1 << i);
    }
    // Return the result
    return inputValue;
}

/**
 * This function flips all odd bits of the 16-bit unsigned integer input value
 * and return the result
 * @param inputValue to be flipped all odd bits
 * @return uint16_t from input value flipped all odd bits
 */
uint16_t flipAllOddBits(uint16_t inputValue) {
    // Iterate through odd bit positions of the 16-bit number
    for (int i = 1; i < 16; i += 2) {
        // Using XOR to flip bit at specific position
        // 1 ^ 0 = 1 and 1 ^ 1 = 0
        // Based on this formula, we shift bit 1 of 1 to left
        // that corresponds to the position of the bit to be flipped
        inputValue = inputValue ^ (1 << i);
    }
    // Return the result
    return inputValue;
}

/**
 * This function flips all bits of the 16-bit unsigned integer input value
 * and return the result
 * @param inputValue
 * @return uint16_t from input value flipped all bits
 */
uint16_t flipAllBits(uint16_t inputValue) {
    // Iterate all 16 bits
    for (int i = 0; i < 16; i++) {
        // Using XOR to flip bit at specific position
        // 1 ^ 0 = 1 and 1 ^ 1 = 0
        // Based on this formula, we shift bit 1 of 1 to left
        // that corresponds to the position of the bit to be flipped
        inputValue = inputValue ^ (1 << i);
    }
    // Return the result
    return inputValue;
}

/**
 * This function take the bits and switch them around
 * from right to left (the rightmost bit becomes the leftmost bit)
 * and return the result
 * @param inputValue
 * @return uint16_t from input value switched all right and left bits
 */
uint16_t switchAllBits(uint16_t inputValue) {

    // Iterate half of the 16 bits number
    for (int i = 0; i < 16/2; i++) {
        // switch bit i & (16 - i - 1)
        // get bit left & right
        uint16_t bitRight = (inputValue >> i) & 1;
        uint16_t bitLeft = (inputValue >> (16 - i - 1)) & 1;
        // set bit left & right to 0 temporarily
        inputValue = inputValue & (~(1 << i));
        inputValue = inputValue & (~(1 << (16 - i - 1)));
        // set bit left to right and right to left
        inputValue = inputValue | (bitLeft << i);
        inputValue = inputValue | (bitRight << (16 - i - 1));
    }

    // Return the result
    return inputValue;
}

/**
 * The main function operates based on the assignment #4 instruction.
 * This function receives and processes the command line passed by user and
 * uses getopt to parse the command line, check options, arguments, and do the assignment
 * which is read the unsigned 16-bit integer number and based on the option to execute
 * such as -e to flip all even bits, -f to flip all odd bits, -a to flip all bits
 * -s to switch all bits, -o to print to file.
 * @param argc the number of arguments passed
 * @param argv a pointer array which points to each arguments passed to the program
 * @return exit(0) on success and others on failure
 */
int main(int argc, char ** argv) {

    // usage: bitflip [-e] [-f] [-a] [-s] [-o outputfilename] intval

    extern char *optarg;
    extern int optind;

    int c, err = 0;
    int eFlag=0, fFlag=0, aFlag=0, sFlag=0, oFlag=0; // flag for each option
    uint16_t intValue; // integer number passed by an user
    char *outputFileName; // hold the output file name

    static char usage[] = "%s [-e] [-f] [-a] [-s] [-o outputfilename] intval\n";

    // use getopt to parse the command line
    while ((c = getopt(argc, argv, "efaso:")) != -1) {
        switch (c) {
            case 'e':
                eFlag = 1;
                break;
            case 'f':
                fFlag = 1;
                break;
            case 'a':
                aFlag = 1;
                break;
            case 's':
                sFlag = 1;
                break;
            case 'o':
                if (oFlag == 0) { // Make sure ignore the duplicates
                    oFlag = 1;
                    outputFileName = optarg;
                }
                break;
            case '?':
                err = 1;
                break;
        }
    }

    // Check command line
    if ((optind + 1) > argc) { /* need one argument for int value */
        fprintf(stderr, "optind=%d, argc=%d\n", optind, argc);
        fprintf(stderr, "%s: missing integer value\n", argv[0]);
        fprintf(stderr, usage, argv[0]);
        exit(100);
    } else if ((optind + 1) < argc) { /* should be only one int value in command line */
        fprintf(stderr, "optind=%d, argc=%d\n", optind, argc);
        fprintf(stderr, "There should only be one integer value\n");
        fprintf(stderr, usage, argv[0]);
        exit(100);
    } else if (err) {
        fprintf(stderr, usage, argv[0]);
        exit(100);
    }

    // Temporary integer to check range eligible
    int tempInt = atoi(argv[optind]);

    // Check if int value is in range 1 - 20000 inclusive
    if (tempInt < 1 || tempInt > 20000) {
        fprintf(stderr, "Your number is not valid!\nPlease input a number from 1 to 20000 inclusive\n");
        exit(100);
    }

    intValue = tempInt;

    // Print the input value
    fprintf(stdout, "Value: %d\n", intValue);

    // If output file is passed, print to file instead of screen
    if (oFlag) {
        // output file name
        FILE *pOutfile;
        pOutfile = fopen(outputFileName, "w");
        if (pOutfile == NULL) { // Check open file
            fprintf(stderr, "\aERROR opening %s\n", outputFileName);
            exit(100);
        }

        fprintf(pOutfile, "Value: %d\n", intValue);

        // Based on options are passed and call appropriate functions
        if (eFlag) {
            fprintf(pOutfile, "Even bits flipped: %d\n", flipAllEvenBits(intValue));
        }

        if (fFlag) {
            fprintf(pOutfile, "Odd bits flipped: %d\n", flipAllOddBits(intValue));
        }

        if (aFlag) {
            fprintf(pOutfile, "All bits flipped: %d\n", flipAllBits(intValue));
        }

        if (sFlag) {
            fprintf(pOutfile, "Switched all bits: %d\n", switchAllBits(intValue));
        }

        // close output file
        if (fclose(pOutfile) == EOF) {
            fprintf(stderr, "Output file close error.\n");
            exit(100);
        }

        // Inform to user if successful
        fprintf(stdout, "Successfully write to the output file name: %s.\n", outputFileName);

    } else { // print output to screen instead of output file

        // Based on options are passed and call appropriate functions
        if (eFlag) {
            fprintf(stdout, "Even bits flipped: %d\n", flipAllEvenBits(intValue));
        }

        if (fFlag) {
            fprintf(stdout, "Odd bits flipped: %d\n", flipAllOddBits(intValue));
        }

        if (aFlag) {
            fprintf(stdout, "All bits flipped: %d\n", flipAllBits(intValue));
        }

        if (sFlag) {
            fprintf(stdout, "Switched all bits: %d\n", switchAllBits(intValue));
        }
    }

    // Exit 0 on success
    exit(0);
}
