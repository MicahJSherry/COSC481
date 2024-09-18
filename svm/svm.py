from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



KERNALS = ['linear', 'poly', 'rbf', 'sigmoid']

svms    = {}
results = {}

for kernal in KERNALS:
    svms[kernal] = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    results[kernal]= svms[kernal].fit()