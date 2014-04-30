#!/bin/sh

g++ solution.cpp `pkg-config --libs opencv` -std=c++0x -o solution
