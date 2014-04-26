"""
Showing examples
"""


from sklearn import svm
import cPickle
from data_api import *
import sklearn
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pylab as plt
from realtime_aug import generator_simple
print get_example_memory(0)

fromApi = False

if fromApi:
    trn, tst = get_cycled_training_test_generators_bare(generator=default_generator)
    from itertools import islice
    for ex, label in islice(trn, 1000):
        if label == 0:
            print ex.shape
            plt.title(str(label))
            plt.imshow(ex[0:4*(aug_image_side//CROP_FACTOR)**2].reshape(4,aug_image_side//CROP_FACTOR,aug_image_side//CROP_FACTOR)[0], cmap='hot')
            plt.show()
else:
    for i in xrange(10000):
        ex, label = get_example(i)
        print label
        if label[-1] == '0':
            print ex.shape
            plt.title(str(label))
            plt.imshow(ex[0:4*(aug_image_side//CROP_FACTOR)**2].reshape(4,aug_image_side//CROP_FACTOR,aug_image_side//CROP_FACTOR)[0], cmap='hot')
            plt.show() 
