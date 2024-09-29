from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

from sklearn.preprocessing import OneHotEncoder

from load_image import load_images

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential

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
y= np.array(y)
y=y.reshape(-1,1)
print(X.shape)
le = OneHotEncoder(sparse_output=False)
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = 11

model = Sequential([
        ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
        Flatten(),
        Dense(num_classes, activation='softmax'),
        
    ])

model.compile(optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)


