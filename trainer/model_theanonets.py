#!/usr/bin/env python
MODEL_NAME = "theanonets_feedforward.pkl"
DEBUG = 1

import matplotlib.pyplot as plt
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy
from theano.tensor.signal import downsample
import time

from th_logreg import LogisticRegression
from th_hiddenlayer import HiddenLayer
from th_cnn import LeNetConvPoolLayer
from data_api import *
from realtime_aug import *
import data_api



import matplotlib.pyplot as plt
import theanets
from theanonet_utils import load_mnist, plot_layers

if DEBUG >= 1:
    print "Theano configured for ",theano.config.floatX

N = 200000

train_set_x, train_set_y, test_set_x, test_set_y = \
    get_training_test_matrices_expanded(N=N, oversample_negative=True, generator=generator_fast, add_x_extra=True)

print train_set_x[0]

#
# train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
# train_set_x = train_set_x[:, 0:train_set_x.shape[1]-ExtraColumns]
# test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
# test_set_x = test_set_x[:, 0:test_set_x.shape[1]-ExtraColumns]


K = 16

e = theanets.Experiment(
    theanets.Classifier,
    activation = "tanh",
    # hidden_dropouts=0.1,
    # input_dropouts=0.1,
    # weight_l1 = 0.01,
    layers=(train_set_x.shape[1], train_set_x.shape[1]*2, 10, 2),
    train_batches=1000
)
e.run((train_set_x, train_set_y.astype("int32")), (test_set_x, test_set_y.astype("int32")))

print e.network.predict(test_set_x[0]), test_set_y[0]
print e.network.predict(test_set_x[1]), test_set_y[1]



plot_layers(e.network.weights)
plt.tight_layout()
plt.show()
