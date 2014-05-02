#!/usr/bin/env python
MODEL_NAME = "mt_128_64_32_8x8_noPCA_noreg_0.9_800.pkl"
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
    layers=(get_training_example_memory(0)[0].shape[0],  128, 64, 32, 1),
)
e.network.load(MODEL_NAME);
normalizer = cPickle.load(open("mt_128_64_32_8x8_noPCA_noreg_0.9.pkl.normalizer"))#open(MODEL_NAME+".normalizer"))
           
weights = [p.get_value().copy() for p in e.network.weights],

biases =[p.get_value().copy() for p in e.network.biases]
"""
print "Weights"

print weights[0][0].shape
print weights[0][1].shape
print len(normalizer.mean_)
print len(normalizer.std_)
print len(weights[0][0].reshape(-1))
print len(weights[0][1].reshape(-1))
print len(weights[0][2].reshape(-1))
print len(biases[0].reshape(-1))
print len(biases[1].reshape(-1))
print len(biases[2].reshape(-1))
"""
for w in weights[0]:
    print "Weight shape ", w.shape

dumped = np.hstack( tuple( [np.hstack((w.T.reshape(-1),b.reshape(-1))) for w,b in zip(weights[0], biases) ] + [normalizer.mean_, normalizer.std_ ] ) )
print dumped
"""
print "Checking dumped values"
print dumped[0:70]
print weights[0][0].T[0, 0:70]
print dumped[(70*128+128): (70*128+128+128)]
print weights[0][1].T[0, 0:128]
print weights[0][0].T.shape
"""
open(MODEL_NAME+".cppdump", "w").write(" ".join((str(x) for x in dumped)))


print len(dumped)

tst_str="-1.49913 -1.24362 -1.41037 -1.11879 -1.41562 -1.32508 -1.3208 -1.17095 -1.1737 -1.03177 -1.31408 -1.01932 -1.41493 -1.43986 -1.4158 -1.24406 -1.74538 -1.31434 -1.37679 -1.22956 -1.36085 -1.33887 -1.64762 -1.24824 -1.33022 -1.41554 -1.22997 -1.255 -1.37009 -1.32633 -1.37353 -1.18314 -1.05987 -1.60996 -1.36104 -0.911397 -0.844543 -1.20042 -1.55688 -1.18325 -1.3751 -1.40414 -1.30923 -1.07137 -0.82004 -1.08152 -1.02208 -1.17163 -1.08533 -1.38033 -1.60088 -0.882443 -0.974504 -1.02202 -1.24592 -1.16724 -1.28597 -1.68702 -1.46696 -1.57608 -1.22446 -1.40168 -1.39717 -1.16398 93.0807 77.8635 81.4317 116.435 1.55162 101.044"
input_test = np.array([float(a) for a in tst_str.split(" ")])
print input_test
 
print e.network.feed_forward(input_test.reshape(1,-1))[-1]
z = e.network.feed_forward(input_test.reshape(1,-1))

exit(0)

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
Y_true = []

from realtime_aug import *
from data_api import *
for i in xrange(100000):
    w, label = get_training_example_memory(i, gen=lambda x:x)
    pixels = ImageChannels*aug_image_side**2

    score = 0.0    

    for k in xrange(10):
        aug = np.hstack((generator_fast(w[0:pixels].reshape(ImageChannels, aug_image_side, aug_image_side)).reshape(-1), w[pixels:]))
        normalizer.transform(aug, copy=False)
        score_part = e.network.predict(aug.reshape(1,-1))
        score += score_part
    
    score = score / 10.0    


    Y_true.append(label)
    if score<0.5:
        Y_pred.append(0)
    else:
        Y_pred.append(1)
    
    if label == 0:
        print score


tp, tn, fp, fn = _tp_tn_fp_fn(Y_true, Y_pred)
print tp, tn, fp, fn
print "tp", "tn", "fp", "fn"

print "Accuracy ", (tp+tn)/(tp+tn+fp+fn), "Negative precision ", tn/(tn+fn+0.0001), "Precision ", tp/(tp+fp+0.00001)
print "True performance ", tn/(fp+tn)
