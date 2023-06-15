all: prod_cons linked prefix_sum

prod_cons: prod_cons.c
	g++ -fopenmp prod_cons.c -o prod_cons

linked: linked.c
	gcc -fopenmp linked.c -o linked

prefix_sum: prefix_sum.cpp
	g++ -fopenmp prefix_sum.cpp -o prefix_sum

clean:
	rm -f *.o