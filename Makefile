CFLAGS = -std=c++17 -O3

build:
	nvcc $(CFLAGS) *.cpp *.cu -o PathTracer

.PHONY: run

run:
	./PathTracer

clean:
	rm -f PathTracer

all: clean build run