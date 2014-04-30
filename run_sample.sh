#!/bin/sh
./compile.sh
javac src/AsteroidRejectTester.java && mv src/AsteroidRejectTester.class . 
java AsteroidRejectTester -folder data/ -train 1_train.txt -test 1_test.txt -exec ./solution $1 $2
