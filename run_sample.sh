#!/bin/sh
./compile.sh
javac src/AsteroidRejectTester.java && mv src/*.class . 
java AsteroidRejectTester -folder data/ -train example_train.txt -test example_test.txt -exec ./solution $1 $2
