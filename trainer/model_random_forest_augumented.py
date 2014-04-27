
"""
Accuracy  0.962533333333 Negative precision  0.592991913747 Precision  0.971905119967 - 128 features
Accuracy  0.959333333333 Negative precision  0.713253012048 Precision  0.966335275968 - 128 features

Fixed bug 1

Accuracy  0.962733333333 Negative precision  0.727777777778 Precision  0.971507607192 - diff 8x8 - 128 features
Accuracy  0.960466666667 Negative precision  0.791666666667 Precision  0.966046831956 - diff 8x8 - fixed - 128 features

Added logarithmic scale


ccuracy  0.9467 Negative precision  0.857142857143 Precision  0.947046127591 i - 28 estimators, 16x16, 128 features



"""
MODEL_NAME="rf.pkl"
N=200000

UsePCAKmeans = True
PCAKmeansModel = "model_kmeans_pca.pkl"


from sklearn import svm
import cPickle
from data_api import *
import sklearn
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA
from sklearn.ensemble import RandomForestClassifier

print get_example_memory(0)

print "SVM Test.."
train_set_x, train_set_y, test_set_x, test_set_y = \
    get_training_test_matrices_expanded(N=N, oversample_negative=True, generator=generator_fast, add_x_extra=True)


print "Training on ", train_set_x.shape


if UsePCAKmeans: 
    print "Loading PCA"
    pca, kmeans = cPickle.load(open(PCAKmeansModel, "r"))
    print "Transforming train"
    X_tr = kmeans.transform(X_tr)
    print "Transforming test"
    X_tst = kmeans.transform(X_tst)


X_tr, Y_tr, X_tst, Y_st = train_set_x, train_set_y, test_set_x, test_set_y


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

clf = RandomForestClassifier(n_estimators=7, max_features=64, 
                             max_depth=None, min_samples_split=1, random_state=0, n_jobs=7, verbose=5)

clf.fit(X_tr, Y_tr)

print clf.feature_importances_
tp, tn, fp, fn = _tp_tn_fp_fn(Y_st, clf.predict(X_tst))

print "Accuracy ", (tp+tn)/(tp+tn+fp+fn), "Negative precision ", tn/(tn+fn), "Precision ", tp/(tp+fp)

cPickle.dump(clf, open("rndforest_2.pkl","w"))

"""

wclf = linear_model.SGDClassifier(loss='hinge', class_weight='auto')

while True:
    wclf.partial_fit(X_tr, Y_tr, classes=[0,1])
    print "Scoring.."
    print "Accuracy ", wclf.score(X_tst, Y_st)
    print wclf.predict(X_tst)

print "Dumping to file "
"""

