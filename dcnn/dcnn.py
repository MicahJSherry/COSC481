# system and bookkeeping imports 
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from datetime import datetime
import gc

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
save_model = False
 
num_images = -1 
num_classes = 11
e = 10
metrics_dir = "metrics/CRE"

X, y = load_images("./spark22/train",num_images)
y= np.array(y)
y=y.reshape(-1,1)
print(X.shape)
le = OneHotEncoder(sparse_output=False)
y = le.fit_transform(y)
X= X/255

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


mf = model_factory()

models = [  
    #resnet V1
            #"resnet50",
            #"resnet101",
            #"resnet152",
            
    #resnet V2
            #"resnet50v2" ,
            #"resnet101v2",
            #"resnet152v2",
            
    #EffecientNet V1
            #"EfficientNetB0",
            #"EfficientNetB1",
            #"EfficientNetB2",
            #"EfficientNetB3",
            #"EfficientNetB4",
            #"EfficientNetB5",
            #"EfficientNetB6",
            "EfficientNetB7",

    #EffecientNet V2
            #"EfficientNetB0v2",
            #"EfficientNetB1v2",
            #"EfficientNetB2v2",
            #"EfficientNetB3v2",
             
            #"EfficientNetv2L",
            #"EfficientNetv2M",
            #"EfficientNetv2S",
         
    #other 
            #"googlenet",
            #"vgg19",
            #"vgg16"   
          ]
          



for name in models:
    #optim = Adam()#learning_rate=0.0005, beta_1=0.9999, beta_2=0.999, epsilon=1e-8)
    #optim = Nadam(learning_rate=0.001)
    print("*"*20+f"{name}"+"*"*20)
    
    model = mf.get_model(name)

    optim = RMSprop()

    model.compile(optimizer=optim,
        loss="categorical_crossentropy",
        metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=e)
    if save_model:
        model.save(f"models/{name}_{time}.keras") 

    y_pred = model.predict(X_test)
    print(y_pred)
    
    y_pred = tf.argmax(y_pred,axis=-1)
    y_t = tf.argmax(y_test,axis=-1)
    
    results = classification_report(y_true=y_t, y_pred=y_pred)
       
    with open(f"{metrics_dir}/{name}_{time}_metrics.txt", "w") as f:
        # Write some text to the file
        f.write(str(results))
    save_conf_mat(y_t, y_pred, name=f"{metrics_dir}/{name}_{time}")
    gc.collect()


