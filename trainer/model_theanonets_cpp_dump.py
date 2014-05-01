#!/usr/bin/env python
MODEL_NAME = "mt_128_64.32"
DEBUG = 1
import cPickle
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy
from theano.tensor.signal import downsample
import time
import sklearn

from th_logreg import LogisticRegression
from th_hiddenlayer import HiddenLayer
from th_cnn import LeNetConvPoolLayer
from data_api import *
from realtime_aug import *
import data_api



import matplotlib.pyplot as plt
import theanets
from theanonet_utils import load_mnist, plot_layers


e = theanets.Experiment(
    theanets.Regressor,
    activation='tanh',
    num_updates=100,
    layers=(train_set_x.shape[1],  128, 64, last_layer),
)
e.network.load(MODEL_NAME);
           
weights = [p.get_value().copy() for p in e.network.weights],
biases =[p.get_value().copy() for p in e.network.biases]
                
print weights
print biases

print "Predicting 0"
print e.network.predict(get_training_example_memory(0).reshape(1,-1))


