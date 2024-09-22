from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from load_image import load_images

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import numpy as np

from xgboost import XGBClassifier

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



num_images = 1000
X, y = load_images("./spark22/train",num_images)
le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

pca = PCA(n_components=16)
X_pca = pca.fit_transform(X)

print("starting Tsne")
tsne = TSNE(n_components=16, method="exact", n_iter=250, verbose=2)
X_tsne = tsne.fit_transform(X_pca)

Xs = {
       # "X": X,
       # "X_pca": X_pca,
        "X_tsne": X_tsne}

for n, x in Xs.items():
    print(n)
    xgb = XGBClassifier()
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)

    metrics = classification_report(y_true=y_test,y_pred=y_pred)
    
    with open(f"xgb_{n}_metrics.txt", "w") as f:
        # Write some text to the file
        f.write(str(metrics))

    save_conf_mat(y_test, y_pred=y_pred, name=f"xgb_{n}")

# Train the classifier









