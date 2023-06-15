/**
 * Panther ID:          6314061
 * Student's Name:      Quang Tuan Le
 * Professor's Name:    Prof. Caryl Rahn
 * Semester:            COP4338 RVC - Fall 2022
 *
 * Assignment #3
 *
 * I affirm that I wrote this program myself without any help from any other people
 * or sources from the internet.
 *
 * Description:
 * The program is for the assignment 3 which is to ask the user uses the command line:
 * codesolve [-o output_file_name] input_file_name.
 * From the above command line arguments, this program will read the letters in an input file
 * and print the sorted letters to an output file (or standard output) which will be the solution.
 * To more specific, the program first reads the letters in from the input file and convert them
 * to upper case if they are not already in uppercase. If output_file_name is given with the -o
 * option, the program will output the sorted letters to the given output file; otherwise, the output
 * shall be to standard output.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>

/**
 * Node struct of Doubly Linked List
 * with key (letter), next (Node), and previous (Node)
 */
typedef struct Node_struct {
    char letter;
    struct Node_struct* next;
    struct Node_struct* previous;
} Node;

/**
 * This function creates new node and a passing character
 * is a value of the new node
 * @param letter
 * @return new Node with value as a letter passed
 */
Node* createNode(char letter) {
    Node* node = (Node*) malloc(sizeof(Node));
    node->letter = letter;
    node->next = NULL;
    node->previous = NULL;
    return node;
}

/**
 * This function creates and adds new node letter at first,
 * and the head must be changed after this function.
 * So the head has to be passed as an address not reference.
 * @param head Node should be pass as an address
 * @param letter char
 */
void addFirst(Node** head, char letter) {
    Node* newNode = createNode(letter);

    (*head)->previous = newNode;
    newNode->next = *head;
    *head = newNode;
}

/**
 * This function creates and adds new node letter at last (at the tail)
 * @param head Node as a reference
 * @param letter char
 */
void addLast(Node* head, char letter) {
    Node* newNode = createNode(letter);

    while (head->next != NULL) {
        head = head->next;
    }

    head->next = newNode;
    newNode->previous = head;
}

/**
 * This function creates and adds new Node after the
 * current node is passed in Doubly Linked List
 * @param currNode Node of Doubly Linked List
 * @param letter char
 */
void insertAfterNode(Node* currNode, char letter) {
    Node* newNode = createNode(letter);

    newNode->next = currNode->next;
    currNode->next = newNode;
    newNode->previous = currNode;

    if (newNode->next != NULL) {
        newNode->next->previous = newNode;
    }
}

/**
 * This function creates and adds new Node before the
 * current node is passed in Doubly Linked List
 * @param currNode Node of Doubly Linked List
 * @param letter char
 */
void insertBeforeNode(Node* currNode, char letter) {
    Node* newNode = createNode(letter);

    newNode->next = currNode;
    newNode->previous = currNode->previous;
    currNode->previous = newNode;

    if (newNode->previous != NULL) {
        newNode->previous->next = newNode;
    }
}

/**
 * This function returns the node contain a passing letter or
 * the node position where before the position need to add a passing letter
 * @param head Node reference
 * @param letter char
 * @return the Node at position may be add a new node before or after
 */
Node* findPosition(Node* head, char letter) {
    while (head != NULL && head->letter < letter) {
        head = head->next;
    }
    return head;
}

/**
 * This function checks if a passing letter is an alphabet or not
 * @param letter char
 * @return true/false
 */
bool isAlphabet(char letter) {
    if ((letter >= 'a' && letter <= 'z') || (letter >= 'A' && letter <= 'Z')) {
        return true;
    }
    return false;
}

/**
 * This function converts a passing letter to an upper case of it
 * @param letter char
 * @return an upper case of the passing letter
 */
char toUpperCase(char letter) {
    if (isAlphabet(letter) && letter >= 'a' && letter <= 'z') {
        letter -= 32;
    }
    return letter;
}

/**
 * The main function operates based on the assignment #3 instruction.
 * This function receives and processes the command line passed by user and
 * uses getopt to parse the command line, check options, arguments, and do the assignment
 * which is read the letters in the input file, and put them in alphabetical order in doubly linked list
 * , then based on the command options to be write to the output file or print to screen.
 * @param argc the number of arguments passed
 * @param argv a pointer array which points to each arguments passed to the program
 * @return exit(0) on success and others in failure
 */
int main(int argc, char ** argv) {

    // Command line: codesolve [-o output_file_name] input_file_name

    extern char *optarg;
    extern int optind;

    int c, err = 0;
    int oflag = 0;
    char *infile, *outfile; // hold the input file name, output file name

    static char usage[] = "usage: %s [-o output_file_name] input_file_name\n";

    // use getopt to parse the command line
    while ((c = getopt(argc, argv, "o:")) != -1) {
        switch (c) {
            case 'o':
                oflag = 1;
                outfile = optarg; // check if it changed after optarg change (because use the pointer)
                break;
            case '?':
                err = 1;
                break;
        }
    }

    if ((optind + 1) > argc) { /* need one argument for input file */
        fprintf(stderr, "optind=%d, argc=%d\n", optind, argc);
        fprintf(stderr, "%s: missing input file\n", argv[0]);
        fprintf(stderr, usage, argv[0]);
        exit(1);
    } else if ((optind + 1) < argc) { /* in case user put too many commands */
        fprintf(stderr, "optind=%d, argc=%d\n", optind, argc);
        fprintf(stderr, "Too many commands than needed.\n");
        fprintf(stderr, usage, argv[0]);
        exit(1);
    } else if (err) {
        fprintf(stderr, usage, argv[0]);
        exit(1);
    }

    // get the input file name
    infile = argv[optind];

    FILE *pInfile;
    pInfile = fopen(infile, "r");
    if (pInfile == NULL) {
        printf("\aERROR opening %s\n", infile);
        exit(100);
    }

    // Doubly Linked List
    Node* head = NULL;

    char currLetter;
    // Get the first letter to keep for the head
    // It should be a loop in case that a lot of spaces at top of in file not a letter
    while (head == NULL && (currLetter = fgetc(pInfile)) != EOF) {
        // Before create a node, the char which is read should be a letter
        if (isAlphabet(currLetter)) {
            head = createNode(toUpperCase(currLetter)); // upper case and create a node to head
        }
    }

    // when infile has no letter, then do nothing
    if (head == NULL) {
        printf("The input file has no letter to be read.\n");
        exit(100);
    }

    // when a head of Doubly Linked List is satisfied, then do the loop to read all remain letters in the input file
    Node* tempHead; // hold the temporary head, in case the head is not changed
    Node* currNode; // hold the current node when finding a position to add a next letter
    while ((currLetter = fgetc(pInfile)) != EOF) { // get until at the end of file
        if (isAlphabet(currLetter)) { // check if the current character is a letter

            // The first case is add to head when the current letter is before head letter in alphabetical order
            if (head->letter > toUpperCase(currLetter)) {
                // add first and change the head
                addFirst(&head, toUpperCase(currLetter));
            } else { // the rest of cases
                tempHead = head; // update the temp head every loop in case head changed
                currNode = findPosition(tempHead, toUpperCase(currLetter));

                if (currNode == NULL) { // add last node
                    // after find position and the curr node is NULL, it means at the tail
                    addLast(tempHead, toUpperCase(currLetter));
                } else if (currNode->letter < toUpperCase(currLetter)) {
                    // even though currNode should be higher than currLetter
                    // , still good to cover this case, so if currLetter higher order -> add after
                    insertAfterNode(currNode, toUpperCase(currLetter));
                } else if (currNode->letter > toUpperCase(currLetter)) {
                    // when currLetter lower order -> add before
                    insertBeforeNode(currNode, toUpperCase(currLetter));
                }
            }
        }
    }

    // If command line pass the output file
    // then write to the output file
    if (oflag == 1) {

        FILE *pOutfile;
        pOutfile = fopen(outfile, "w");

        currNode = head; // start at head
        while (currNode != NULL) { // until the end of doubly linked list
            fputc(currNode->letter, pOutfile); // put every letter at every node to the output file
            currNode = currNode->next; // move to the next node
        }

        // try to close the output file
        if (fclose(pOutfile) == EOF) {
            // if error, notify a message and exit
            printf("Output file close error.\n");
            exit(100);
        }

        printf("Successfully write to the output file.\n");
    }

    fclose(pInfile); // close the input file

    printf("Forward order: \n");
    currNode = head;
    while (currNode != NULL) {
        printf("%c", currNode->letter);
        currNode = currNode->next;
    }
    printf("\n");

    // Check print backward
    currNode = head;
    // after this loop the current node is at the tail
    while (currNode->next != NULL) {
        currNode = currNode->next;
    }

    printf("Backward order: \n");
    while (currNode != NULL) {
        printf("%c", currNode->letter);
        currNode = currNode->previous;
    }
    printf("\n");

    // free all nodes in Doubly Linked List
    currNode = head;
    while (currNode != NULL) {
        Node* temp = currNode;
        currNode = currNode->next;
        free(temp);
    }

    // Successfully operate the program
    exit(0);
}
