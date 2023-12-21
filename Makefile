CFLAGS = -std=c++17 -g

build:
	g++ $(CFLAGS) -o PathTracer *.cpp

.PHONY: run

run:
	./PathTracer

clean:
	rm -f PathTracer

all: clean build run