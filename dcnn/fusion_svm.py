import os 
import tensorflow as tf
import numpy as np
features_path = "./features/"

from load_image import load_images



def save_features(images_dir="./spark22/train", out_dir= features_path):
    num_images = 10
    X, y = load_images("./spark22/train",num_images)
    print(y)
    
    model_paths = [
        "models/googlenet_24-11-18T16-07.keras",
        "models/resnet50_24-11-18T16-07.keras",
        "models/vgg16_24-11-18T16-07.keras",
        "models/vgg19_24-11-18T16-07.keras"]
    


    for p in model_paths:
        model = tf.keras.models.load_model(p)
        model.trainable = False
        model = tf.keras.Sequential(model.layers[:-1]) 
        features = model.predict(X)

        architecture_name = p.split("/")[1].split("_")[0] 
        print(architecture_name)
        
        for  i in range(len(features)):
            feature = features[i]
            path= f"{features_path}/{architecture_name}/{y[i]}"
            os.makedirs(path,exist_ok= True)
            with open(f'{path}/{i}.npy', 'wb') as f:
                np.save(f, feature)

        
if len(os.listdir(features_path)) == 0:
    save_features()
