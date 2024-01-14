CFLAGS = -std=c++17 -O3 

build:
	nvcc -arch=sm_80 --use_fast_math $(CFLAGS) *.cpp *.cu -o PathTracer -diag-suppress 20012

.PHONY: run

run:
	./PathTracer

clean:
	rm -f PathTracer

all: clean build run