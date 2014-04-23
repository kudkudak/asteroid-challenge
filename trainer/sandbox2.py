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

"""
print "Fitting PCA"

pca = RandomizedPCA(n_components=5)
X_tr = pca.fit_transform(X_tr)
X_tst = pca.transform(X_tst)

import cPickle
cPickle.dump(pca, open("trained_pca.pkl","w"))
"""

#print "PCA ratios: ", pca.explained_variance_ratio_

clf = RandomForestClassifier(n_estimators=18, max_features=128, max_depth=None, min_samples_split=1, random_state=0, n_jobs=6, verbose=5)

clf.fit(X_tr, Y_tr)

print clf.feature_importances_

print clf.score(X_tst, Y_st)


cPickle.dump(clf, open("rndforest.pkl","w"))

"""
print "Training on ", X_tr.shape
wclf = linear_model.SGDClassifier(loss='hinge', class_weight='auto')

while True:
    wclf.partial_fit(X_tr, Y_tr, classes=[0,1])
    print "Scoring.."
    print "Accuracy ", wclf.score(X_tst, Y_st)
    print wclf.predict(X_tst)

print "Dumping to file "
"""

