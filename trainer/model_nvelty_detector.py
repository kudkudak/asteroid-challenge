# Novelty detector
from generate_aug_data import *
import matplotlib.pylab as plt   
from im_operators import * 
from visualize import *
import sklearn
import sklearn.covariance
from sklearn.covariance import EllipticEnvelope

novdet = sklearn.covariance.EllipticEnvelope()
train_set_x, train_set_y, test_set_x, test_set_y = get_training_test_matrices_expanded(N=200000, oversample_negative=True, generator=generator_fast, add_x_extra=True)
print "Fitting.."
novdet.fit(train_set_x)

for x in test_set_x:
    out = novdet.predict(x)
    show_4_ex(x.reshape(ImageChannels, ImageSideFinal, ImageSideFinal), title="Out="+str(out))




