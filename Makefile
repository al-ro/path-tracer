CFLAGS = -std=c++17 -O3

build:
	nvcc -dc $(CFLAGS) *.cpp *.cu && nvcc -rdc=true *.o -o PathTracer

.PHONY: run

run:
	./PathTracer

clean:
	rm -f PathTracer && rm -f *.o

all: clean build run