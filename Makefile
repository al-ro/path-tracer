CFLAGS = -std=c++17 -O3

build:
	g++ $(CFLAGS) -o PathTracer *.cpp

.PHONY: run

run:
	./PathTracer

clean:
	rm PathTracer

all: clean build run