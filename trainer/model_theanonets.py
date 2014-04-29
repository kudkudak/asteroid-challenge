"""
 Any results true true true
e = theanets.Experiment(
    theanets.Regressor,
    layers=(train_set_x.shape[1], 128, 14,  last_layer),
)
"""


#!/usr/bin/env python
MODEL_NAME = "theanonets_feedforward.pkl"
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

if DEBUG >= 1:
    print "Theano configured for ",theano.config.floatX

onlyLast = True
UsePCAKmeans = True
PCAKmeansModel = "model_kmeans_pca_1.pkl"
N = 1800000
train_set_x, train_set_y, test_set_x, test_set_y = \
    get_training_test_matrices_expanded(N=N, oversample_negative=True, generator=generator_fast, add_x_extra=True)


if UsePCAKmeans: 
    ipixels = ImageChannels*ImageSideFinal*ImageSideFinal
    print "Loading PCA"
    pca, kmeans = cPickle.load(open(PCAKmeansModel, "r"))
    #TODO: add extra columns
    print "Transforming train"
    train_set_x_pca = kmeans.transform(pca.transform(train_set_x[:,0:ipixels]))
    print "Transforming test"
    test_set_x_pca = kmeans.transform(pca.transform(test_set_x[:,0:ipixels]))
    # Add pca variables
    train_set_x, train_set_y, test_set_x, test_set_y =np.hstack((train_set_x, train_set_x_pca)), train_set_y.astype("int32"), np.hstack((test_set_x, test_set_x_pca)), test_set_y.astype("int32")

if onlyLast:
    train_set_y = train_set_y[:,ImageChannels-1]
    test_set_y = test_set_y[:,ImageChannels-1]

last_layer = 1 if len(train_set_y.shape)==1 else train_set_y.shape[1]

print "LAST LAYER SIZE", last_layer

print "Normalizing"
normalizer = sklearn.preprocessing.Scaler(copy=False)
normalizer.fit_transform(train_set_x)
normalizer.transform(test_set_x)



e = theanets.Experiment(
    theanets.Regressor,
    num_updates=5,
    layers=(train_set_x.shape[1], 128, last_layer),
)
e.run((train_set_x, train_set_y.astype("int32").reshape(-1,last_layer)), (test_set_x, test_set_y.astype("int32").reshape(-1,last_layer)))

for xt, yt in zip(test_set_x, test_set_y):
    if yt == 0:
        print e.network.predict(xt), yt


plot_layers(e.network.weights)
plt.tight_layout()
plt.show()
