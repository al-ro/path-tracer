CFLAGS = -std=c++17 -O3

PathTracer: main.cpp
	
	g++ $(CFLAGS) -o PathTracer *.cpp

.PHONY: run clean

run: PathTracer
	./PathTracer

clean:
	rm -f PathTracer