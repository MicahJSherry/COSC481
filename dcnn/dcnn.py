import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report

from sklearn.preprocessing import OneHotEncoder

from load_image import load_images

import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, InceptionV3,vgg16, vgg19
from tensorflow.keras.models import Sequential

from datetime import datetime

from alexnet import alexnet
from fusion import *

from tensorflow.keras.optimizers import Adam, Nadam, RMSprop  
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


num_images = 100
X, y = load_images("./spark22/train",num_images)
y= np.array(y)
y=y.reshape(-1,1)
print(X.shape)
le = OneHotEncoder(sparse_output=False)
y = le.fit_transform(y)
X= X/255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = 11

resnet50 = Sequential([
        ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3)),
        Flatten(),
        Dense(4096,activation="relu"),
        Dropout(0.5),
        Dense(4096,activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')])

vgg16_base = VGG16(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
vgg16_base.trainable= False
vgg16_model = Sequential([
        vgg16_base,
        Flatten(),
        Dense(4096,activation="relu"),
        Dropout(0.5),
        Dense(4096,activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')])

vgg19_base = VGG19(include_top=False, weights="imagenet",pooling="max", input_shape=(224, 224, 3))
vgg19_base.trainable= False
vgg19_model = Sequential([
        vgg19_base,
        Flatten(),
        Dense(4096,activation="sigmoid"),
        Dropout(0.5),
        Dense(4096,activation="sigmoid"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")])

googleNet = Sequential([
        InceptionV3(include_top=False, input_shape=(224, 224, 3)),
        Flatten(),
        Dense(4096,activation="relu"),
        Dropout(0.5),
        Dense(4096,activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
        ])

alex = alexnet(num_classes)

models = {#"alexnet":alex,
          #"resnet50": resnet50,
          #"vgg16":vgg16_model,
          #"vgg19":vgg19_model,
          #"googlenet":googleNet,
          #"cat_fusion":build_fusion_model(method="cat", out_size=11)
          #"max_fusion":build_fusion_model(method="max", out_size=11)
          #"min_fusion":build_fusion_model(method="min", out_size=11),
          #"avg_fusion":build_fusion_model(method="avg", out_size=11)
          
          #"mult_fusion":build_fusion_model(method="mult", out_size=11),
          #"sum_fusion":build_fusion_model(method="sum", out_size=11),
          "wave_fusion":build_fusion_model(method="wavelet", out_size=11)
          }

for name, model in models.items():
    #optim = Adam()#learning_rate=0.0005, beta_1=0.9999, beta_2=0.999, epsilon=1e-8)
    optim = RMSprop()


    x      = X_train
    x_test = X_test
    #optim = Nadam(learning_rate=0.001)
    model.compile(optimizer=optim,
        loss="categorical_crossentropy",
        metrics=['accuracy'])
    
    model.fit(x, y_train, epochs=10)
    
    model.save(f"{name}_{time}.keras") 

    y_pred = model.predict(x_test)
    print(y_pred)
    y_pred = tf.argmax(y_pred,axis=-1)
    y_t = tf.argmax(y_test,axis=-1)
    
    print(y_pred.shape)
    print(y_t.shape)
    results = classification_report(y_true=y_t, y_pred=y_pred)
       
    with open(f"{name}_{time}_metrics.txt", "w") as f:
        # Write some text to the file
        f.write(str(results))
    save_conf_mat(y_t, y_pred, name=f"{name}_{time}")



