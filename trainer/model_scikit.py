from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm
import cPickle
from data_api import *
import sklearn
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier

print get_example_memory(0)

print "SVM Test.."
X_tr, Y_tr, X_tst, Y_st = get_training_test_matrices_expanded(N=500000, oversample_negative=True, generator=generator_fast, add_x_extra=True)

print "Training on ", X_tr.shape

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


#print "PCA ratios: ", pca.explained_variance_ratio_

clf = KNeighborsClassifier(n_neighbors=3)

clf.fit(X_tr, Y_tr)


tp, tn, fp, fn = _tp_tn_fp_fn(Y_st, clf.predict(X_tst))

print "Accuracy ", (tp+tn)/(tp+tn+fp+fn), "Negative precision ", tn/(tn+fn), "Precision ", tp/(tp+fp)

cPickle.dump(clf, open("rndforest.pkl","w"))

