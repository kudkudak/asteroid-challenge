"""
general csv dataset wrapper for pylearn2, here with one hot and scaling/255 for digits
"""

import csv
import numpy as np
import os
import sys, os

sys.path.append(os.path.join(".", ".."))

from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess

import cPickle
import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy
from theano.tensor.signal import downsample
import time
import sklearn

from data_api import *
from realtime_aug import *
import data_api

import matplotlib.pyplot as plt
import theanets
from theanonet_utils import load_mnist, plot_layers


train_set_x, train_set_y, test_set_x, test_set_y, train_indices= \
    get_training_test_matrices_expanded(N=8000000, train_percentage=0.9, oversample_negative=True, generator=generator_fast, add_x_extra=True)



class AsteroidDataset( DenseDesignMatrix ):
    def __init__(self, N = 6000000, train_percentage=0.99):
        #TODO: split into training and validation

        print "normalizing"
        normalizer = sklearn.preprocessing.Scaler(copy=False)
        normalizer.fit_transform(train_set_x)
        normalizer.transform(test_set_x)

        self.one_hot = False		
        self.view_converter = None

        X = train_set_x
        y = train_set_y

        super( AsteroidDataset, self ).__init__( X=X, y=y )

class AsteroidDatasetValid( DenseDesignMatrix ):
    def __init__(self, N = 600000, train_percentage=0.99):
        #TODO: split into training and validation

        print "normalizing"
        normalizer = sklearn.preprocessing.Scaler(copy=False)
        normalizer.fit_transform(train_set_x)
        normalizer.transform(test_set_x)

        self.one_hot = False		
        self.view_converter = None

        X = test_set_x
        y = test_set_y

        super( AsteroidDatasetValid, self ).__init__( X=X, y=y )

