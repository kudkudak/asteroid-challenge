#!/usr/bin/env python
MODEL_NAME = "mt_128_64_32_relu_8x8_noPCA.pkl"
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
normalizer = cPickle.load(open("500000.normalizer"))
           
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
"""
print len(dumped)

input_test = np.array([float(a) for a in "0.214742 0.214742 0.237616 0.214742 0.202117 0.226541 0.173861 0.226541 0.214742 0.188541 0.173861 0.226541 0.226541 0.226541 0.188541 0.188541 0.237616 0.157881 0.24805 0.140346 0.202117 0.202117 0.188541 0.188541 0.226541 0.202117 0.202117 0.202117 0.157881 0.157881 0.202117 0.173861 0.202117 0.226541 0.173861 0.173861 0.214742 0.214742 0.202117 0.173861 0.140346 0.140346 0.140346 0.202117 0.214742 0.157881 0.0743938 0.0743938 0.173861 0.0991536 0.202117 0.140346 0.0743938 0.157881 0.157881 0.157881 0.188541 0.157881 0.157881 0.0743938 0.120923 0.120923 0.140346 0.173861 18.47 6.48 1.893 0.5 1.54 0.66".split(" ")])
normalizer.transform(input_test, copy=False)

input_test = np.array([float(a) for a in "-0.975401 -0.979134 -0.885057 -0.987605 -1.04186 -0.932716 -1.15475 -0.924912 -0.979083 -1.10037 -1.18352 -0.972717 -0.972777 -0.956463 -1.10042 -1.09173 -0.884972 -1.2525 -0.917176 -1.44037 -1.17095 -1.11662 -1.12044 -1.09586 -0.937094 -1.07828 -1.17123 -1.30554 -1.50207 -1.36384 -1.07829 -1.16352 -1.04172 -0.972722 -1.29419 -1.43112 -1.24921 -1.11624 -1.07842 -1.16351 -1.3024 -1.3279 -1.38468 -1.17136 -1.11625 -1.30903 -1.61214 -1.58587 -1.15469 -1.48403 -1.06167 -1.34467 -1.62928 -1.25247 -1.23211 -1.22338 -1.08801 -1.22326 -1.22738 -1.59047 -1.39067 -1.38614 -1.29882 -1.15105 94.0955 104.432 91.8572 0.810043 6.28942 44.0421".split(" ")])
print input_test
 
print e.network.feed_forward(input_test.reshape(1,-1))

z1, z2, z3 = e.network.feed_forward(input_test.reshape(1,-1))


print weights[0][2]
print biases[2]
import math
print math.tanh(np.dot(z2, weights[0][2])+biases[2])
print "Without tanh"
print np.dot(z2, weights[0][2])+biases[2]
print z3
exit(0)
"""

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

               

