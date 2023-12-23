CFLAGS = -std=c++17 -O3

build:
	nvcc $(CFLAGS) -o PathTracer *.cpp *.cu

.PHONY: run

run:
	./PathTracer

clean:
	rm -f PathTracer

all: clean build run