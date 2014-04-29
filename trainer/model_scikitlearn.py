
"""
Accuracy  0.962533333333 Negative precision  0.592991913747 Precision  0.971905119967 - 128 features
Accuracy  0.959333333333 Negative precision  0.713253012048 Precision  0.966335275968 - 128 features

Fixed bug 1

Accuracy  0.962733333333 Negative precision  0.727777777778 Precision  0.971507607192 - diff 8x8 - 128 features
Accuracy  0.960466666667 Negative precision  0.791666666667 Precision  0.966046831956 - diff 8x8 - fixed - 128 features

Added logarithmic scale


Accuracy  0.9467 Negative precision  0.857142857143 Precision  0.947046127591 i - 28 estimators, 16x16, 128 features

Added KMeans + PCA - high level denoising features (hopefully) ?? - still bug

Fixed bug 2

clf = RandomForestClassifier(n_estimators=7, max_features=128,
                             max_depth=None, min_samples_split=1, random_state=0, n_jobs=7, verbose=5)
Accuracy  0.9834 Negative precision  0.769911277312 Precision  0.987081022844
True performance  0.506796116505


Chekcing with 58 estimators and PCA and 58 estimators without PCA


"""
from sklearn import svm
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
MODEL_NAME="rf.pkl"
N=180000

UsePCAKmeans = True
PCAKmeansModel = "model_kmeans_pca.pkl"
clf = sklearn.linear_model.SGDRegressor(verbose=5)
partialFit = False
onlyLast = True
classification = True
clf = sklearn.linear_model.SGDClassifier(loss='log')

clf = RandomForestClassifier(n_estimators=14, max_features=128, max_depth=None, min_samples_split=1, random_state=0, n_jobs=7, verbose=5)


print "====================="
print "Training scikit model with partialFit=",partialFit," onlyLast=", onlyLast, " classification=", classification
print "Training model ",str(clf)
print "===================="


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

# Fit only to the last
if onlyLast:
    test_set_y= test_set_y[:,3] # Try to predict detection on the last only
    train_set_y = train_set_y[:, 3]

train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
print "Percentage Positive: ",sum([y for y in train_set_y if y == 1])

print "Training on ", train_set_x.shape

train_set_x_pca, test_set_x_pca = None, None

X_tr, Y_tr, X_tst, Y_st =train_set_x, train_set_y.astype("int32"),test_set_x, test_set_y.astype("int32")



if UsePCAKmeans: 
    ipixels = ImageChannels*ImageSideFinal*ImageSideFinal
    print "Loading PCA"
    pca, kmeans = cPickle.load(open(PCAKmeansModel, "r"))
    #TODO: add extra columns
    print "Transforming train"
    train_set_x_pca = kmeans.transform(pca.transform(train_set_x[:,0:ipixels]))
    print "Transforming test"
    test_set_x_pca = kmeans.transform(pca.transform(test_set_x[:,0:ipixels]))
    # Add pca variables
    X_tr, Y_tr, X_tst, Y_st =np.hstack((train_set_x, train_set_x_pca)), train_set_y.astype("int32"), np.hstack((test_set_x, test_set_x_pca)), test_set_y.astype("int32")
"""
print "Normalizing"
normalizer = sklearn.preprocessing.Normalizer(norm='l2', copy=False)
normalizer.fit_transform(X_tr)
normalizer.transform(X_tst)
"""

if classification:

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
    else:
        clf.fit(X_tr, Y_tr)
        tp, tn, fp, fn = _tp_tn_fp_fn(Y_st, clf.predict(X_tst))
        print tp, tn, fp, fn
        print "tp", "tn", "fp", "fn"

        print "Accuracy ", (tp+tn)/(tp+tn+fp+fn), "Negative precision ", tn/(tn+fn+0.0001), "Precision ", tp/(tp+fp+0.00001)
        print "True performance ", tn/(fp+tn)

else:
    if partialFit:
        for i in xrange(40):
            print "Partial fit ",i
            clf.partial_fit(X_tr, Y_tr)
            print "Score ", clf.score(X_tst, Y_st)
    else:
        raise "Not implemented" 
    pass
