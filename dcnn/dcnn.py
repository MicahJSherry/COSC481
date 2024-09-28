from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

from sklearn.preprocessing import LabelEncoder

from load_image import load_images
from sklearn.manifold import TSNE

import numpy as np

import tensorflow as tf

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

models = {
    "vgg16": tf.keras.applications.VGG16(),
    "vgg19": tf.keras.applications.VGG19(),
    }

print(models['vgg16'].summary())


num_images = 1000
X, y = load_images("./spark22/train",num_images)

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

