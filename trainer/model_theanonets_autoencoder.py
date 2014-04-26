"""
Parameters achieving anything
        e = theanets.Experiment(
            theanets.Autoencoder,
            layers=(train_set_x.shape[1], 64, 16, 64, train_set_x.shape[1]),
            num_updates=50,
            input_noise=0.2,
            train_batches=100,
            tied_weights=True,
        )

Fixed log scale:


"""

MODEL_NAME="learned_denoising_autoencoder_image.pkl"
AddExtra = False
Visualise = False
#!/usr/bin/env python

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


N = 200000

train_set_x, train_set_y, test_set_x, test_set_y = \
    get_training_test_matrices_expanded(N=N, oversample_negative=True, generator=generator_fast, add_x_extra=True)




print "Example image ", train_set_x[0]

if not AddExtra:
    train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
    train_set_x = train_set_x[:, 0:train_set_x.shape[1]-ExtraColumns]
    test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
    test_set_x = test_set_x[:, 0:test_set_x.shape[1]-ExtraColumns]



import matplotlib.pyplot as plt
import theanets

from theanonet_utils import load_mnist, plot_layers, plot_images

e=None

if not os.path.exists(MODEL_NAME):
    """
        e = theanets.Experiment(
            theanets.Autoencoder,
            layers=(train_set_x.shape[1], 64, 16, 64, train_set_x.shape[1]),
            num_updates=50,
            input_noise=0.2,
            train_batches=100,
            tied_weights=True,
        )
    """

    e = theanets.Experiment(
        theanets.Autoencoder,
        layers=(train_set_x.shape[1], 64, 16, 64, train_set_x.shape[1]),
        num_updates=100,
        input_noise=0.2,
        train_batches=100,
        tied_weights=True,
    )
    e.run(train_set_x, test_set_x)

    import cPickle

else:
    print "Loading file!"

    import pickle
    e = pickle.load(open(MODEL_NAME, "r"))

if Visualise:

    plot_layers(e.network.weights, tied_weights=True)
    plt.tight_layout()
    plt.show()

    test_set_x = test_set_x[:ImageSideFinal**2]
    plot_images(test_set_x, 121, 'Sample data')
    predicted = e.network.predict(test_set_x)
    plot_images(predicted, 122, 'Reconstructed data')
    plt.tight_layout()
    from visualize import *
    plt.show()
    import numpy as np


    for id, (im, im_pred) in enumerate(zip(test_set_x, predicted)):
        #plt.imshow(np.hstack((im.reshape(ImageChannels, ImageSideFinal, ImageSideFinal)[0], im_pred.reshape(ImageChannels, ImageSideFinal, ImageSideFinal)[0])), cmap='hot')
        im0=im.reshape(ImageChannels, ImageSideFinal, ImageSideFinal)[0]
        im1=im_pred.reshape(ImageChannels, ImageSideFinal, ImageSideFinal)[0]
        show_4_ex([im0,im1,im0,im1], title=str(test_set_y[id]))
        plt.show() 


if not os.path.exists(MODEL_NAME):
    try:
        cPickle.dump(e, open("learned_denoising_autoencoder_image.pkl", "w"))
    except Exception, e:
        print "Excpetion ",e


