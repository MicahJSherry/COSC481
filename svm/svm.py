from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, roc_auc_score 
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load_image import load_images
from sklearn.manifold import TSNE

import numpy as np



KERNALS = ['rbf']#, 'linear', 'poly',  'sigmoid']
num_images = 10
X, y = load_images("./spark22",num_images)
tsne = TSNE(n_components=2)

X = tsne.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svms    = {}
results = {}
metrics = {}
for kernal in KERNALS:
    print(kernal)
    svms[kernal] = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=kernal,verbose=2))
    svms[kernal] = svms[kernal].fit(X_train,y_train)
    results[kernal]= svms[kernal].predict(X_test)

    metrics[kernal] = classification_report(y_true=y_test,y_pred=results[kernal])
    print()
    print(kernal)
    print(metrics[kernal])
    
    print()
