all: codesolve

codesolve: codesolve.o
	gcc -o codesolve codesolve.o

codesolve.o: codesolve.c
	gcc -c codesolve.c -o codesolve.o

clean:
	rm -f *.o