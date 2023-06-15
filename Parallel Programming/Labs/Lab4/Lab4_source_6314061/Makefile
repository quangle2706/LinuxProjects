all: monte_carlo_pi bankAccount test03

monte_carlo_pi: monte_carlo_pi.c
	gcc -lpthread monte_carlo_pi.c -o monte_carlo_pi

bankAccount: bankAccount.c
	gcc -lpthread bankAccount.c -o bankAccount

test03: test03.c
	gcc -lpthread test03.c -o test03

clean:
	rm -f *.o