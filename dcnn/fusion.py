import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

#import tensorflow_wavelets.Layers.DWT as DWT
from wavetf import WaveTFFactory
wavelet_layer = w = WaveTFFactory().build('db2', dim=1)
"""
model paths
../dcnn/resnet50_24-10-03T00-07.keras
../dcnn/vgg19_24-10-04T16-11.keras
../dcnn/googlenet_24-10-04T18-46.keras  
../dcnn/vgg16_24-10-04T16-11.keras

"""
paths = [
        "models/resnet50_24-10-17T16-00.keras",
        "models/vgg19_24-10-04T16-11.keras",
        "models/googlenet_24-10-17T16-00.keras",
        "models/vgg16_24-10-04T16-11.keras"]

def build_fusion_model(method="cat", out_size=11,model_paths=paths):
    inputs = keras.Input(shape=(224, 224, 3))
    dcnn_out = []
    
    for p in model_paths:
        model = tf.keras.models.load_model(p)
        model.trainable = False
        
        model = tf.keras.Sequential(model.layers[:-1]) 
        print(model.summary()) 
        dcnn  = model(inputs)
        dcnn_out.append(dcnn)
    # lock the models from training 
    
    if method == "cat":
        x = keras.layers.Concatenate()(dcnn_out)
    elif method == "max":
        x = keras.layers.maximum(dcnn_out)
    elif method == "min":
        x = keras.layers.minimum(dcnn_out)
    elif method == "avg":
        x = keras.layers.average(dcnn_out)
    elif method == "mult":
        x = keras.layers.multiply(dcnn_out)
    elif method == "sum":
        x = keras.layers.add(dcnn_out)
    elif method =="wavelet":         
        x = keras.layers.Concatenate()(dcnn_out)
        x = wavelet_layer(x) 
    else: 
        raise Exception(f"fusion method: {method} is not defined")

    x = Dense(4096,     activation="sigmoid")(x)
    x = Dropout(.5)(x)
    x = Dense(4096,     activation="sigmoid")(x)
    x = Dropout(.5)(x)
    x = Dense(4096,     activation="sigmoid")(x)
    x = Dropout(.5)(x)
    outputs = Dense(out_size, activation="softmax")(x)
   
    fusion_model = keras.Model(inputs=inputs, outputs=outputs, name =f"{method}_fusion_model")
    print(fusion_model.summary())

    plot_model(fusion_model, to_file='my_model.png', show_shapes=True, rankdir='TB', expand_nested=False)
    return fusion_model



