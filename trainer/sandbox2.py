from sklearn import svm
import cPickle
from data_api import *
import sklearn
from sklearn import linear_model


print "SVM Test.."
X_tr, Y_tr, X_tst, Y_st = get_training_test_matrices_bare(limit_size=1000)
print "Training on ", X_tr.shape
wclf = linear_model.SGDClassifier(loss='hinge', class_weight='auto')

while True:
    wclf.partial_fit(X_tr, Y_tr, classes=[0,1])
    print "Scoring.."
    print "Accuracy ", wclf.score(X_tst, Y_st)

print "Dumping to file "
cPickle.dump(wclf, open("trained_svm.pkl","w"))


