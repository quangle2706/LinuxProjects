all: randomgraph serialPrims parallelPrims

parallelPrims: parallelPrims.c++
	g++ -fopenmp parallelPrims.c++ -o parallelPrims

serialPrims: serialPrims.c
	gcc serialPrims.c -o serialPrims

randomgraph: randomgraph.java
	javac randomgraph.java

clean:
	rm -f *.o