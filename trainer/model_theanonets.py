###TODO####
"""
 Any results true true true
e = theanets.Experiment(
    theanets.Regressor,
    layers=(train_set_x.shape[1], 128, 14,  last_layer),
)


=========16x16====================
1. 16x16 180k - 0.08 error i 81% accuracy - bardzo niskie negative precision
e = theanets.Experiment(
    theanets.Regressor,
    activation='tanh',
    num_updates=10,
    layers=(train_set_x.shape[1],  64, 10,  last_layer),
)
2. 128x64 - 0.06 error on validation more or less and 0.02 error


===========8x8====================



"""


#!/usr/bin/env python
MODEL_NAME = "mt_300_100_10_tanh.pkl"
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
UsePCAKmeans = False
PCAKmeansModel = "model_kmeans_pca_1_50_8x8.pkl"#"model_kmeans_pca_1.pkl"
N = 6000000

print N

train_set_x, train_set_y, test_set_x, test_set_y, train_indices= \
    get_training_test_matrices_expanded(N=N, train_percentage=0.99, oversample_negative=True, generator=generator_fast, add_x_extra=True)

print PCAKmeansModel

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
    train_set_x = train_set_x_pca
    test_set_x = test_set_x_pca
    # train_set_x, train_set_y, test_set_x, test_set_y =np.hstack((train_set_x, train_set_x_pca)), train_set_y.astype("int32"), np.hstack((test_set_x, test_set_x_pca)), test_set_y.astype("int32")

if onlyLast:
    train_set_y = train_set_y[:,ImageChannels-1]
    test_set_y = test_set_y[:,ImageChannels-1]

last_layer = 1 if len(train_set_y.shape)==1 else train_set_y.shape[1]

print "LAST LAYER SIZE", last_layer

print "Normalizing"
normalizer = sklearn.preprocessing.Scaler(copy=False)
normalizer.fit_transform(train_set_x)
normalizer.transform(test_set_x)
print "Normalizing pixels"
#dnormalizer = sklearn.preprocessing.Scaler(copy=False, axis=1)
#normalizer.fit_transform()
#normalizer.transform(test_set_x[:,0:ImageChannels*ImageSide**2])
#sklearn.preprocessing.scale(train_set_x[:,0:ImageChannels*ImageSide**2], axis=1, with_mean=True, with_std=True, copy=False)
#sklearn.preprocessing.scale(test_set_x[:,0:ImageChannels*ImageSide**2], axis=1, with_mean=True, with_std=True, copy=False)

e = theanets.Experiment(
    theanets.Regressor,
    activation='tanh',
    num_updates=20,
    decode_linear=False,
    batchsize=100,
    weight_l2=1e-4,
    hidden_l2=1e-4,
    layers=(train_set_x.shape[1],  300, 100, 10,  last_layer),
)
#e.add_trainer("hf")
e.run((train_set_x, train_set_y.astype("float32").reshape(-1,last_layer)), (test_set_x, test_set_y.astype("float32").reshape(-1,last_layer)))
e.network.save(MODEL_NAME)
import cPickle
cPickle.dump(normalizer, open(MODEL_NAME+".normalizer", "w"))

def _tp_tn_fp_fn(y_true, y_pred):
    tp, tn, fp, fn = 0., 0., 0., 0.
    for y_t, y_p in zip(y_true, y_pred):
        if y_t == y_p and y_t == 1:
            tp += 1
        if y_t != y_p and y_t == 1:
            fn += 1
        if y_t != y_p and y_t == 0:
            fp += 1
        if y_t == y_p and y_t == 0:
            tn += 1
    return tp, tn, fp, fn

score = 0.0

Y_pred = []
for xt, yt in zip(test_set_x, test_set_y):
    score = e.network.predict(xt.reshape(1,-1))
    if yt == 0:
        print score
    if score < 0.5:
        Y_pred.append(0)
    else:
        Y_pred.append(1)


tp, tn, fp, fn = _tp_tn_fp_fn(test_set_y, Y_pred)
print tp, tn, fp, fn
print "tp", "tn", "fp", "fn"

print "Accuracy ", (tp+tn)/(tp+tn+fp+fn), "Negative precision ", tn/(tn+fn+0.0001), "Precision ", tp/(tp+fp+0.00001)
print "True performance ", tn/(fp+tn)

Y_pred = []
for xt, yt in zip(test_set_x, test_set_y):
    score = e.network.predict(xt.reshape(1,-1))
    if yt == 1:
        print score
    if score < 0.5:
        Y_pred.append(0)
    else:
        Y_pred.append(1)



plot_layers(e.network.weights)
plt.tight_layout()
plt.show()
