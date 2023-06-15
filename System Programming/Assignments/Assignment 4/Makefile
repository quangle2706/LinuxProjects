all: bitflip

bitflip: bitflip.o
	gcc -o bitflip bitflip.o

bitflip.o: bitflip.c
	gcc -c bitflip.c -o bitflip.o

clean:
	rm -f *.o