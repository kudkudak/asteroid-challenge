from sklearn import svm
import cPickle
from data_api import *
import sklearn
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier

print get_example_memory(0)

print "SVM Test.."
X_tr, Y_tr, X_tst, Y_st = get_training_test_matrices_bare()

print "Training on ", X_tr.shape

"""
print "Fitting PCA"

pca = RandomizedPCA(n_components=5)
X_tr = pca.fit_transform(X_tr)
X_tst = pca.transform(X_tst)

import cPickle
cPickle.dump(pca, open("trained_pca.pkl","w"))
"""

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

clf = RandomForestClassifier(n_estimators=24, max_features=250,
                             max_depth=None, min_samples_split=1, random_state=0, n_jobs=6, verbose=5)

clf.fit(X_tr, Y_tr)

print clf.feature_importances_
tp, tn, fp, fn = _tp_tn_fp_fn(Y_st, clf.predict(X_tst))

print "Accuracy ", (tp+tn)/(tp+tn+fp+fn), "Negative precision ", tn/(tn+fn)

cPickle.dump(clf, open("rndforest.pkl","w"))

"""

wclf = linear_model.SGDClassifier(loss='hinge', class_weight='auto')

while True:
    wclf.partial_fit(X_tr, Y_tr, classes=[0,1])
    print "Scoring.."
    print "Accuracy ", wclf.score(X_tst, Y_st)
    print wclf.predict(X_tst)

print "Dumping to file "
"""

