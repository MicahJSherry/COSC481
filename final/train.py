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
from models import model_factory 
from utils import save_conf_mat
from load_image import load_images, download_sat_images


time = datetime.now().strftime("%y-%m-%dT%H-%M") # time for book keeping
save_model = False
 
num_images = 500 
num_classes = 11
e = 10
metrics_dir = "metrics"
image_dir = "../dcnn/spark22"
train_dir = f"{image_dir}/train"
test_dir = f"{image_dir}/test"
X_train, y_train = load_images(image_dir,num_images)
y_train= np.array(y_train)
y_train=y_train.reshape(-1,1)
print(X_train.shape)

le = OneHotEncoder(sparse_output=False)
y_train = le.fit_transform(y_train)


X_test, y_test = load_images(test_dir, 100)

y_test = np.array(y_test)
y_test = y_test.reshape(-1,1)
y_test = le.transform(y_test)

X_train = X_train/255
X_test  = X_test/255


mf = model_factory(num_classes=num_classes)

models = [
    "ConvNeXtBase",
    "ConvNeXtLarge",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
    "ConvNeXtXLarge",
    #"DenseNet121",
    #"DenseNet169",
    "DenseNet201",
    #"EfficientNetB0",
    #"EfficientNetB1",
    #"EfficientNetB2",
    #"EfficientNetB3",
    #"EfficientNetB4",
    #"EfficientNetB5",
    #"EfficientNetB6",
    #"EfficientNetB7",
    #"EfficientNetV2B0",
    #"EfficientNetV2B1",
    #"EfficientNetV2B2",
    #"EfficientNetV2B3",
    #"EfficientNetV2L",
    #"EfficientNetV2M",
    #"EfficientNetV2S",
    #"InceptionResNetV2",
    #"InceptionV3",
    #"MobileNet",
    #"MobileNetV2",
    #"MobileNetV3Large",
    #"MobileNetV3Small",
    #"NASNetLarge",
    #"NASNetMobile",
    #"ResNet101",
    #"ResNet101V2",
    #"ResNet152",
    #"ResNet152V2",
    #"ResNet50",
    #"ResNet50V2",
    #"VGG16",
    #"VGG19",
    #"Xception"
    ]
          



for name in models:
    #optim = Adam()#learning_rate=0.0005, beta_1=0.9999, beta_2=0.999, epsilon=1e-8)
    #optim = Nadam(learning_rate=0.001)
    print("*"*20+f"{name}"+"*"*20)
    
    model = mf.get_model(name, train_base=False)
    
    optim = RMSprop()

    model.compile(optimizer=optim,
        loss="categorical_crossentropy",
        metrics=['accuracy'])
    
    model.summary()

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


