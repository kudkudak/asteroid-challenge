MODEL_NAME="model_ica.pkl"
N=200000
Visualise=True
from sklearn import svm
import cPickle
from data_api import *
import sklearn
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import FastICA
from sklearn.cluster import KMeans

train_set_x, train_set_y, test_set_x, test_set_y = \
    get_training_test_matrices_expanded(N=N, oversample_negative=True, generator=generator_fast, add_x_extra=True)

train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
train_set_x = train_set_x[:, 0:train_set_x.shape[1]-ExtraColumns]
test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
test_set_x = test_set_x[:, 0:test_set_x.shape[1]-ExtraColumns]

X_tr, Y_tr, X_tst, Y_st = train_set_x, train_set_y, test_set_x, test_set_y

import os


ica = FastICA(n_components=49, whiten=True)
if not os.path.exists(MODEL_NAME):

    print "Training on ", train_set_x.shape
    print "Fitting ICA"

    ica.fit_transform(X_tr)
else:
    ica = cPickle.load(open(MODEL_NAME, "r"))

F = ica.unmixing_matrix_w 

import matplotlib.pylab as plt

if Visualise:
    for i, f in enumerate(F):
        plt.subplot(7, 7, i + 1)
        plt.imshow(f.reshape(ImageSideFinal, ImageSideFinal), cmap="gray")
        plt.axis("off")
        plt.show()


    N=min(1000, test_set_x.shape[0])
    x_plt, y_plt, clr_plt = [0]*int(N), [0]*int(N), [0]*int(N)
    for i in xrange(int(N)):
        act = pca.transform(test_set_x[i])
        print act
        x_plt[i] = act[0,0]
        y_plt[i] = act[0,1]
        clr_plt[i] = i % 2

    plt.scatter(x_plt, y_plt,s=90, c=clr_plt)
    plt.show()



if not os.path.exists(MODEL_NAME):
    import cPickle
    cPickle.dump((pca,kmeans), open(MODEL_NAME,"w"))
