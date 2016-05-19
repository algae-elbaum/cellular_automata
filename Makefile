CC=g++

all: cpu_automata 

cpu_automata: cpu_automata.o
	$(CC) -std=c++11 -o $@ cpu_automata.o

cpu_automata.o: cpu/cpu_automata.cc
	$(CC) -std=c++11 -c -o $@ cpu/cpu_automata.cc

clean:
	rm cpu_automata.o cpu_automata
