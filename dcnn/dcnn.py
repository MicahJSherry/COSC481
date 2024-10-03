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
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, InceptionV3
from tensorflow.keras.models import Sequential
from datetime import datetime
from alexnet import alexnet

time = datetime.now().strftime("%y-%m-%dT%H-%M")

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


num_images = -1
X, y = load_images("./spark22/train",num_images)
y= np.array(y)
y=y.reshape(-1,1)
print(X.shape)
le = OneHotEncoder(sparse_output=False)
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = 11

resnet50 = Sequential([
        ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
        Flatten(),
        Dense(num_classes, activation='softmax')])


vgg16 = Sequential([
        VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
        Flatten(),
        Dense(num_classes, activation='softmax')])

vgg19 = Sequential([
        VGG19(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
        Flatten(),
        Dense(num_classes, activation='softmax')])

alex = alexnet(num_classes)

models = {"alexnet":alex,
          "resnet50": resnet50,
          "vgg16":vgg16,
          "vgg19":vgg19}

for name, model in models.items():

    model.compile(optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=100)
    
    model.save(f"{name}_{time}.keras") 

    y_pred = model.predict(X_test)
    y_pred = tf.argmax(y_pred,axis=-1)
    y_t = tf.argmax(y_test,axis=-1)
   
    print(y_pred)
    print(y_t)
    results = classification_report(y_true=y_t, y_pred=y_pred)
       
    with open(f"{name}_{time}_metrics.txt", "w") as f:
        # Write some text to the file
        f.write(str(results))
    save_conf_mat(y_t, y_pred, name=f"{name}_{time}")



