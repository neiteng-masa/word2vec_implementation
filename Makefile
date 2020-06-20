all:
	g++ src/train.cpp -o bin/train -mcmodel=large -std=c++1y -I./lib/ -pthread -Wall -O3 -mtune=native -march=native
