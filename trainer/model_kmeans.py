MODEL_NAME="model_kmeans_pca_1_50_8x8.pkl"
N=400000
Visualise=True
from sklearn import svm
import cPickle
from data_api import *
import sklearn
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans
import os

train_set_x, train_set_y, test_set_x, test_set_y = \
    get_training_test_matrices_expanded(N=N, oversample_negative=False, generator=generator_fast, add_x_extra=True, train_percentage=0.99)

train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
train_set_x = train_set_x[:, 0:train_set_x.shape[1]-ExtraColumns]
test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
test_set_x = test_set_x[:, 0:test_set_x.shape[1]-ExtraColumns]

X_tr, Y_tr, X_tst, Y_st = train_set_x, train_set_y, test_set_x, test_set_y



print train_set_x.shape



pca = PCA(n_components=50, whiten=True)
kmeans = KMeans(n_clusters=50, n_init=1)
if not os.path.exists(MODEL_NAME):

    print "Training on ", train_set_x.shape
    print "Fitting PCA"

    pca.fit(X_tr)

    print "Fitted PCA"
    # Train KMeans on whitened data
    print "Transforming data"
    X_tr_white = pca.transform(X_tr)
    print "Fitting KMEANS"
    kmeans.fit(X_tr_white)
    import cPickle
    cPickle.dump((pca,kmeans), open(MODEL_NAME,"w"))
else:
    pca, kmeans = cPickle.load(open(MODEL_NAME, "r"))

F = pca.inverse_transform(kmeans.cluster_centers_)

print F[0].shape

import matplotlib.pylab as plt

if Visualise:
    for i, f in enumerate(F):
        if i > 46: break
        print f
        print f.shape
        plt.subplot(7, 7, i + 1)
        plt.imshow(np.hstack(f.reshape(ImageChannels,ImageSideFinal, ImageSideFinal)), cmap="hot")
        plt.axis("off")
    plt.show()


    N=min(1000, test_set_x.shape[0])
    x_plt, y_plt, clr_plt = [0]*int(N), [0]*int(N), [0]*int(N)
    for i in xrange(int(N)):
        act = kmeans.transform(pca.transform(test_set_x[i]))
        print act
        x_plt[i] = act[0,2]
        y_plt[i] = act[0,3]
        clr_plt[i] = test_set_y[i,3]

    plt.scatter(x_plt, y_plt,s=90, c=clr_plt)
    plt.show()

