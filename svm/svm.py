from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load_image import load_images

KERNALS = ['linear', 'poly', 'rbf', 'sigmoid']
num_images = 100
X, y = load_images("./spark22",num_images)


svms    = {}
results = {}

for kernal in KERNALS:
    svms[kernal] = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=kernal))
    results[kernal]= svms[kernal].fit(X,y)
    print(results)