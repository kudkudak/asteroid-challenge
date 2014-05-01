#!/bin/sh

g++ solution.cpp `pkg-config --libs opencv` -std=c++0x -O3 -o solution
