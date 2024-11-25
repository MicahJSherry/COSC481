# system and bookkeeping imports 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from datetime import datetime

# scientific computing imports  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder
import numpy as np

#tensorflow imports
import tensorflow as tf
from tensorflow.keras.layers import Dense,Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.applications import ResNet50, VGG16, VGG19, InceptionV3,vgg16, vgg19
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Nadam, RMSprop  


# local imports
from fusion import *
from models import model_factory 
from metrics_agg import save_conf_mat
from load_image import load_images




time = datetime.now().strftime("%y-%m-%dT%H-%M") # time for book keeping

num_images = 100 
num_classes = 11
e = 1


X, y = load_images("./spark22/train",num_images)
y= np.array(y)
y=y.reshape(-1,1)
print(X.shape)
le = OneHotEncoder(sparse_output=False)
y = le.fit_transform(y)
X= X/255
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = {"resnet50" : model_factory("resnet50"),
	  "vgg19"    : model_factory("vgg19"),
	  "vgg16"    : model_factory("vgg16"),
	  "googlenet": model_factory("googlenet"),

          #"cat_fusion":build_fusion_model(method="cat", out_size=11),
          #"max_fusion":build_fusion_model(method="max", out_size=11),
          #"min_fusion":build_fusion_model(method="min", out_size=11),
          #"avg_fusion":build_fusion_model(method="avg", out_size=11),
          #"mult_fusion":build_fusion_model(method="mult", out_size=11),
          #"sum_fusion":build_fusion_model(method="sum", out_size=11),
          #"wave_fusion":build_fusion_model(method="wavelet", out_size=11),
          }



for name, model in models.items():
    #optim = Adam()#learning_rate=0.0005, beta_1=0.9999, beta_2=0.999, epsilon=1e-8)
    #optim = Nadam(learning_rate=0.001)
    optim = RMSprop()

    x      = X_train
    x_test = X_test
    model.compile(optimizer=optim,
        loss="categorical_crossentropy",
        metrics=['accuracy'])
    
    model.fit(x, y_train, epochs=e)
    
    model.save(f"models/{name}_{time}.keras") 

    y_pred = model.predict(x_test)
    print(y_pred)
    y_pred = tf.argmax(y_pred,axis=-1)
    y_t = tf.argmax(y_test,axis=-1)
    
    print(y_pred.shape)
    print(y_t.shape)
    results = classification_report(y_true=y_t, y_pred=y_pred)
       
    with open(f"metrics/{name}_{time}_metrics.txt", "w") as f:
        # Write some text to the file
        f.write(str(results))
    save_conf_mat(y_t, y_pred, name=f"metrics/{name}_{time}")



