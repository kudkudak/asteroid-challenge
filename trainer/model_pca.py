MODEL_NAME="model_pca.pkl"
N=10000
train_percentage = 0.95
from sklearn import svm
import cPickle
from data_api import *
import sklearn
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier

train_set_x, train_set_y, test_set_x, test_set_y = \
    get_training_test_matrices_expanded(N=N, oversample_negative=True, generator=generator_fast, add_x_extra=True, train_percentage=train_percentage)

train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
train_set_x = train_set_x[:, 0:train_set_x.shape[1]-ExtraColumns]
test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
test_set_x = test_set_x[:, 0:test_set_x.shape[1]-ExtraColumns]

X_tr, Y_tr, X_tst, Y_st = train_set_x, train_set_y, test_set_x, test_set_y

import os

pca = RandomizedPCA(n_components=2)
if not os.path.exists(MODEL_NAME):

    print "Training on ", train_set_x.shape
    print "Fitting PCA"

    X_tr = pca.fit_transform(X_tr)
    X_tst = pca.transform(X_tst)

    print "Fitted PCA"

    import cPickle
    cPickle.dump(pca, open(MODEL_NAME,"w"))
else:
    pca = cPickle.load(open(MODEL_NAME, "r"))

import matplotlib.pylab as plt

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
