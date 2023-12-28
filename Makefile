CFLAGS = -std=c++17 -g

build:
	nvcc -arch=sm_80 $(CFLAGS) *.cpp *.cu -o PathTracer -diag-suppress 20012

.PHONY: run

run:
	./PathTracer

clean:
	rm -f PathTracer

all: clean build run