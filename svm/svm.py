from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix,  roc_curve, auc 
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load_image import load_images
from sklearn.manifold import TSNE

import numpy as np

def save_conf_mat(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)

    # Create a heatmap using Seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')   

    plt.ylabel('Actual')

    # Save the image
    plt.savefig(f'{name}_confusion_matrix.png')
    plt.clf()
def save_roc(y_true,y_pred,name):
    # Assuming you have your true labels (y_true) and predicted probabilities (y_score)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2,   
              linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic: {name}')
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig(f'{name}_roc_curve.png')
    plt.clf()


KERNALS = ['rbf']#, 'linear', 'poly',  'sigmoid']
num_images = 1000
X, y = load_images("./spark22",num_images)

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
   
    with open(f"{kernal}_metric.txt", "w") as f:
            # Write some text to the file
            f.write(str(metrics[kernal]))

    save_roc(y_test, results[kernal], kernal)
    save_conf_mat(y_test, results[kernal], kernal)



