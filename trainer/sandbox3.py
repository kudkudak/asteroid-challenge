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

trn, tst = get_cycled_training_test_generators_bare(generator=generator_simple)


from itertools import islice

for ex, label in islice(trn, 1):
    print ex.shape
    plt.title(str(label))
    plt.imshow(ex[0:4*(aug_image_side//CROP_FACTOR)**2].reshape(4,aug_image_side//CROP_FACTOR,aug_image_side//CROP_FACTOR)[0], cmap='hot')
    plt.show()


X_trn, Y_trn, X_tst, Y_tst = get_training_test_matrices_expanded(N=1000, generator=generator_simple)

print X_trn.shape
