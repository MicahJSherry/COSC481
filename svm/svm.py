from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load_image import load_images

KERNALS = ['rbf']#, 'linear', 'poly',  'sigmoid']
num_images = 100
X, y = load_images("./spark22",num_images)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svms    = {}
results = {}

for kernal in KERNALS:
    svms[kernal] = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=kernal))
    svms[kernal] = svms[kernal].fit(X_train,y_train)
    results[kernal]= svms[kernal].predict(X_test)
    
    print(results)