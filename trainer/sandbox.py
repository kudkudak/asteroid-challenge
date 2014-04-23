from sklearn import svm
import cPickle
from data_api import *

print "Number of chunks ", chunks_count
print get_chunk(0)[0][0]
print chunks_positive_ids
print chunks_negative_ids

from visualize import *


show_4_ex(get_chunk(0)[0][0], get_chunk(0)[1][0])


#print "Generating"
#trn, tst = get_training_test_generators_bare(limit_size=1000)
#print "Generated"
#
#for ex, y, det in trn:
#    print ex, y


print "SVM Test.."
X_tr, Y_tr, X_tst, Y_st = get_training_test_matrices_bare()
print "Training on ", X_tr.shape
wclf = svm.SVC(kernel='linear', class_weight={0: 7}, verbose=True)
m = wclf.fit(X_tr, Y_tr)
print "Scoring.."
print "Accuracy ", m.score(X_tst, Y_st)

print "Dumping to file "
cPickle.dump(wclf, open("trained_svm.pkl","w"))
