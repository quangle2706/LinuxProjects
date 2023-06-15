all: mortgagepmt

mortgagepmt: mortgagepmt.o
	gcc -o mortgagepmt mortgagepmt.o -lm

mortgagepmt.o: mortgagepmt.c
	gcc -c mortgagepmt.c -o mortgagepmt.o

clean:
	rm -f *.o