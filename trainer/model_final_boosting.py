# FInal model - boosted Logisic Regression - nothing fancy but hey - who cares
# :)







from sklearn import svm
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model


##### CONFIG ####
MODEL_NAME="final_boosting.pkl"
#### CONFIG ####



def train_model(train_ids, N): 
    """ Train percentage all set """
    
