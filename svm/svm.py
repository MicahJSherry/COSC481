from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score   

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load_image import load_images
from sklearn.manifold import TSNE

import numpy as np


def calculate_and_save_metrics(y_true, y_pred, filename='metrics.txt'):
  """
  Calculates and saves various evaluation metrics to a file.

  Args:
    y_true: True labels.
    y_pred: Predicted labels.
    filename: Name of the file to save the metrics (default: 'metrics.txt').
  """

  # Calculate metrics
  confusion_mat = confusion_matrix(y_true, y_pred)
  accuracy = accuracy_score(y_true, y_pred)
  precision = precision_score(y_true, y_pred)
  recall = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  auc_roc = roc_auc_score(y_true, y_pred)


  # Save metrics to a file
  with open(filename, 'w') as f:
    f.write("Confusion Matrix:\n")
    f.write(str(confusion_mat))
    f.write("\n\nAccuracy: {:.2f}\n".format(accuracy))
    f.write("Precision: {:.2f}\n".format(precision))
    f.write("Recall: {:.2f}\n".format(recall))
    f.write("F1-score: {:.2f}\n".format(f1))
    f.write("AUC-ROC: {:.2f}\n".format(auc_roc))


KERNALS = ['rbf']#, 'linear', 'poly',  'sigmoid']
num_images = 10
X, y = load_images("./spark22",num_images)
tsne = TSNE(n_components=2)

X = tsne.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svms    = {}
results = {}

for kernal in KERNALS:
    print(kernal)
    svms[kernal] = make_pipeline(StandardScaler(), SVC(gamma='auto', kernel=kernal,verbose=2))
    svms[kernal] = svms[kernal].fit(X_train,y_train)
    results[kernal]= svms[kernal].predict(X_test)
    
    calculate_and_save_metrics(y_test, results[kernal], f"{kernal}_metrics.txt")

