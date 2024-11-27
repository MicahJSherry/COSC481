import os 
import cv2
import numpy as np


def load_images(dir,limit=100):
    images = []
    labels = []
    for root, dirs, files in os.walk(dir):
        for file in files[0:limit]:
            if file.endswith('.jpg'):
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path)
                
                image = cv2.resize(image, (224,224))

                images.append(image) 
                labels.append(root.split("/")[-1])
    
    return np.array(images), labels

