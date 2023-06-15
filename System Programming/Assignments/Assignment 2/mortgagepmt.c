/**
 * Panther ID:          6314061
 * Student's Name:      Quang Tuan Le
 * Professor's Name:    Prof. Caryl Rahn
 * Semester:            COP4338 RVC - Fall 2022
 *
 * Assignment #2
 *
 * I affirm that I wrote this program myself without any help from any other people
 * or sources from the internet.
 *
 * Description:
 * The program is to do a mortgage payment calculation.
 * All information needed for this will be passed to the program on the command line
 * as the usage below:
 *          usage: mortgagepmt [-s] -r rate [-d downpayment] price
 * -s is optional and means that the mortgage will be a 15 year mortgage (the default is 30 years)
 * -r is the yearly interest rate for the loan
 * -d is optional and is the down payment that will be made on the loan.
 * price is the price of the house
 *
 * To do that, the program will work with command line options and arguments by
 * using getopt function as well as working with Makefiles, Unix, Math functions with C.
 *
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <locale.h>

int debug = 0;

/**
 * The main function operates based on the assignment 2 instruction and the program needs.
 * This function receives and processes the command line passing by users and
 * uses getopt to parse the command line, check options, arguments, and calculate the monthly payment based on
 * the mortgage payment formula. Then, print the output of information.
 *
 * @param argc the number of arguments passed
 * @param argv a pointer array which points to each argument passed to the program
 *
 * @return 0 exit code on success or 1 on failure
 */
int main(int argc, char ** argv) {

    extern char *optarg;
    extern int optind;

    int c, err = 0;

    int sflag=0, rflag=0, dflag=0;

    double yearlyRate, downpayment = 0;
    int yearOfMortgage = 30; // set as default

    static char usage[] = "usage: %s [-s] -r rate [-d downpayment] price\n";

    // use getopt to parse the command line
    while ((c = getopt(argc, argv, "sr:d:")) != -1) {
        switch (c) {
            case 's':
                sflag = 1;
                yearOfMortgage = 15; // set the default year of mortgage to 15
                break;
            case 'r':
                rflag = 1;
                yearlyRate = atof(optarg);
                break;
            case 'd':
                dflag = 1;
                downpayment = atof(optarg);
                break;
            case '?':
                err = 1;
                break;
        }
    }

    // Check flags
    if (rflag == 0) { /* -r was mandatory */
        fprintf(stderr, "%s: missing -r option\n", argv[0]);
        fprintf(stderr, usage, argv[0]);
        exit(1);
    } else if ((optind + 1) > argc) { /* need one argument for price */
        printf("optind=%d, argc=%d\n", optind, argc);
        fprintf(stderr, "%s: missing price\n", argv[0]);
        fprintf(stderr, usage, argv[0]);
        exit(1);
    } else if ((optind + 1) < argc) { /* the case user inputs more than 1 price house */
        printf("optind=%d, argc=%d\n", optind, argc);
        printf("There should only be one house price.\n");
        fprintf(stderr, usage, argv[0]);
        exit(1);
    } else if (err) {
        fprintf(stderr, usage, argv[0]);
        exit(1);
    }

    // get the price of the house --- Using optind to read the price of the house
    double price = atof(argv[optind]);

    // check the interest rate if it is valid --- Validate the rate
    if (yearlyRate < 3 || yearlyRate > 10) {
        printf("The interest rate should be between 3%% and 10%%.\n");
        fprintf(stderr, usage, argv[0]);
        exit(1);
    }

    // check the down payment if it was passed and valid --- Validate the down payment
    if (dflag == 1 && downpayment > price) {
        printf("The down payment cannot be larger than the price of the home.\n");
        fprintf(stderr, usage, argv[0]);
        exit(1);
    }

    // The output monthly payment
    double result;

    // convert rate to exact value
    double exactYearlyRate = yearlyRate / 100.0;

    // Converting annual interest rate to a monthly interest rate
    double monthlyInterestRate = exactYearlyRate / 12;

    // Add 1 to the monthly rate
    double addingMonthlyRate = monthlyInterestRate + 1;

    // Multiply the number of years in the term of the mortgage by 12
    // to calculate the number of monthly payments you'll make
    int numMonthlyPayments = yearOfMortgage * 12;

    // Raise the result of 1 plus the month rate to the negative power of the number of monthly payments
    result = pow(addingMonthlyRate, -numMonthlyPayments);

    // Subtract that result from 1
    result = 1 - result;

    // Divide the monthly rate by the result
    result = monthlyInterestRate / result;

    // Multiply the result by the amount you want to borrow
    double borrowAmount = price - downpayment;
    result *= borrowAmount;

    // In addition, add on PMI of 1% of the amount borrowed / 12
    // if the down payment is less than 20% of the price of the home.
    // This will stay the same for the life of the loan.
    if (downpayment < (price * 0.2)) { /* Add in PMI when necessary */
        double PMI = 0.01 * (borrowAmount / 12);
        result += PMI;
    }

    // Support print payment format with commas
    setlocale(LC_NUMERIC, "");

    // Output should be
    printf("The payment on a loan of $%'.2f with an interest rate of %.3f%% for a term of %d years will be $%'.2f\n", (price - downpayment), yearlyRate, yearOfMortgage, result);

    // Exit with 0 on success
    exit(0);
}
