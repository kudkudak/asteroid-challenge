
"""
Accuracy  0.962533333333 Negative precision  0.592991913747 Precision  0.971905119967 - 128 features
Accuracy  0.959333333333 Negative precision  0.713253012048 Precision  0.966335275968 - 128 features

Fixed bug 1

Accuracy  0.962733333333 Negative precision  0.727777777778 Precision  0.971507607192 - diff 8x8 - 128 features
Accuracy  0.960466666667 Negative precision  0.791666666667 Precision  0.966046831956 - diff 8x8 - fixed - 128 features

Added logarithmic scale


Accuracy  0.9467 Negative precision  0.857142857143 Precision  0.947046127591 i - 28 estimators, 16x16, 128 features

Added KMeans + PCA - high level denoising features (hopefully) ?? - still bug



"""
from sklearn import svm
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
MODEL_NAME="rf.pkl"
N=150000

UsePCAKmeans = True
PCAKmeansModel = "model_kmeans_pca.pkl"
clf = RandomForestClassifier(n_estimators=28, max_features=128,
                             max_depth=None, min_samples_split=1, random_state=0, n_jobs=7, verbose=5)

clf = sklearn.linear_model.SGDClassifier(loss='log', class_weight='auto')
partialFit = True

from sklearn import svm
import cPickle
from data_api import *
import sklearn
from sklearn import linear_model
from sklearn.decomposition import RandomizedPCA

print get_example_memory(0)

print "SVM Test.."
train_set_x, train_set_y, test_set_x, test_set_y = \
    get_training_test_matrices_expanded(N=N, oversample_negative=True, generator=generator_fast, add_x_extra=True)

train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
print "Percentage Positive: ",sum([y for y in train_set_y if y == 1])

print "Training on ", train_set_x.shape

train_set_x_pca, test_set_x_pca = None, None


if UsePCAKmeans: 
    ipixels = ImageChannels*ImageSideFinal*ImageSideFinal
    print "Loading PCA"
    pca, kmeans = cPickle.load(open(PCAKmeansModel, "r"))
    #TODO: add extra columns
    print "Transforming train"
    train_set_x_pca = kmeans.transform(pca.transform(train_set_x[:,0:ipixels]))
    print "Transforming test"
    test_set_x_pca = kmeans.transform(pca.transform(test_set_x[:,0:ipixels]))


X_tr, Y_tr, X_tst, Y_st =np.hstack((train_set_x, train_set_x_pca)), train_set_y, np.hstack((test_set_x, test_set_x_pca)), test_set_y


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
if partialFit:
    for i in xrange(10):
        clf.partial_fit(X_tr, Y_tr, [0,1])
        tp, tn, fp, fn = _tp_tn_fp_fn(Y_st, clf.predict(X_tst))
        print tp, tn, fp, fn
        print "tp", "tn", "fp", "fn"

        print "Accuracy ", (tp+tn)/(tp+tn+fp+fn), "Negative precision ", tn/(tn+fn+0.0001), "Precision ", tp/(tp+fp+0.00001)
        print "True performance ", tn/(fp+tn)


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

