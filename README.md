NASA Asteroid Challenge solution
======================

AstroidChallenge consisted in predicting if asteroid detection is correct.
Technically submission had to be a self-contained 1-MB C++/Java files.

This repository contains code submitted to Asteroid Challenge held by topcoder.
It scored **29** place from 69 competitors, and 469 registrants. The final
solution consisted in a small ensemble of neural network trained using theano trained
on augumented set of asteroids (around 6 000 000, obtained using rotations, scaling, etc.). 
The neural network were ncoded as binary in C++ code.

**Disclaimer**: This repository contains not cleaned and documented code :) 

I have tested many things and it turned out to be a great learning experience.
Among other things I have tested:

* Convolutional neural networks 

* PCA and KMeans for feature detectors (very sensible features, but wasn't very
  helpful, didn't manage to investigate why)

* Autoencoder

* Random Forests (strong model, but unfortunately hard to port from python to
  C++)

* 3 and 4 layers feedforward networks with dropout/relu units

It turned out that simple feedforward networks joined with augumentation of the
dataset (and log transform!) were the best. My single biggest problem was
imposed code size limit, which was 1MB - it enforced me to use only small
neural networks.

I am leaving this code for reference, it is not documented, but might contain
interesting code snippets.

### Important files:

* solution.cpp - contains submitted solution
* encode_nn.cpp - code for encoding neural network
* trainer/model_scikitlearn.py - main tester for scikit-learn models (RandomForest were the strongest candidate, but hard to port to C++)
* trainer/model_theanonets.py - I trained tiny ensemble of networks using Theanonets
* trainer/model_theano_cnn_basic.py - Implementation of CNN in Theano
* trainer/realtime_aug.py - Inspired by Kaggle winner - realtime augumentation scripts for rotating, scaling, shearing
* trainer/model_kmeans.py - Feature detectors using KMeans - not used in the final submission 
