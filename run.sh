#!/bin/sh
echo $1 $2
g++ solution.cpp `pkg-config --libs opencv` -std=c++0x -DDEBUG -o solution && ./solution < $1
